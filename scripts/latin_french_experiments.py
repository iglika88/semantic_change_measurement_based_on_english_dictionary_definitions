#!/usr/bin/env python3
"""Latin–French dictionary-based semantic-change experiments.

The script compares Latin and French English-language dictionary definitions
under sixteen settings:

  1–8   normalized cognate mapping (exact lemmas plus conservative verb rules)
  9–16  conservative Latin-to-French machine-translation mapping

Within each mapping family, settings vary definition selection (all/first),
POS selection (all/selected), and Sentence-Transformer model (English/MULTI).

Final data policy
-----------------
* Kaikki/English Wiktionary supplies Latin and French lemmas, POS and English
  definitions.
* The Dickinson Latin frequency list supplies Latin frequency ranks. For the
  conformity correlation, ``frequency_score = -rank`` so larger values mean
  greater frequency.
* Latin polysemy is the number of distinct Latin definitions available under
  the POS policy of the setting.
* Brysbaert et al. supplies English concreteness ratings. Concreteness is the
  pooled mean of all rated English tokens in the definitions selected by each
  setting; cognate and MT mappings use the same calculation.
* Semantic distance is average pairwise cosine distance (APD) between all old
  and new definitions selected by the setting.

Cognate policy
--------------
* Lemmas are Unicode-normalized, lower-cased, stripped of diacritics and
  non-letter characters.
* Normalized lemmas shorter than three characters are excluded by default.
* Entries marked as proper nouns on either side are excluded.
* Exact normalized matches are preferred.
* Latin ``-are`` → French ``-er`` and Latin ``-ire`` → French ``-ir`` are
  accepted only when both entries contain a verb POS. Exact matches always
  take precedence over rule-based matches.

Conservative MT policy
----------------------
* MT output must resolve as a complete one-word or multiword expression to a
  French Kaikki entry; words inside an unmatched phrase are never matched
  independently.
* Surface and optional spaCy-lemmatized expressions are checked.
* Obvious non-lexical targets are removed.
* A French target may receive at most five Latin source lemmas by default.
* Selected-POS MT settings require at least one shared major lexical POS.

Required local inputs
---------------------
--frequency-file     Dickinson Latin frequency list
--concreteness-file  Brysbaert concreteness ratings Excel file

Kaikki Latin and French JSONL files may be supplied locally or downloaded.

Translation options
-------------------
For MT settings, use an existing ``--mt-cache`` whenever possible. Otherwise
choose one of these providers with ``--translation-provider``:

* ``google``: Google Cloud Translation; uses Application Default Credentials.
* ``deepl``: DeepL API; reads ``DEEPL_AUTH_KEY`` from the environment.
* ``libretranslate``: LibreTranslate-compatible HTTP endpoint; optionally reads
  ``LIBRETRANSLATE_API_KEY``.

No credentials are stored in this script or written to output files.

Examples
--------
Run cognate Settings 1 and 8::

    python latin_french_experiments.py \
      --frequency-file latin_frequency_list.txt \
      --concreteness-file Concreteness_ratings.xlsx --settings 1 8

Run MT Settings 9–16 from a translation cache::

    python latin_french_experiments.py \
      --frequency-file latin_frequency_list.txt \
      --concreteness-file Concreteness_ratings.xlsx \
      --settings 9 10 11 12 13 14 15 16 \
      --mt-cache latin_french_mt_translations.csv

Dependencies: numpy, pandas, scipy, requests, openpyxl, sentence-transformers.
Optional MT providers add google-cloud-translate, deepl, or a reachable
LibreTranslate endpoint. spaCy plus ``fr_core_news_sm`` is optional but
recommended for conservative MT lemmatization.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr, wilcoxon

LATIN_KAIKKI_URL = (
    "https://kaikki.org/dictionary/Latin/kaikki.org-dictionary-Latin.jsonl"
)
FRENCH_KAIKKI_URL = (
    "https://kaikki.org/dictionary/French/kaikki.org-dictionary-French.jsonl"
)
EN_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MULTI_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SELECTED_POS = frozenset({"noun", "verb", "adj", "adv"})
ENGLISH_TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)?")
PARENS_RE = re.compile(r"\([^)]*\)")
SPACE_RE = re.compile(r"\s+")
GLOSS_SPLIT_RE = re.compile(r"[;]")
INVALID_MT_TARGETS = frozenset(
    {"", "le", "la", "les", "un", "une", "des", "de", "du", "a", "au", "aux"}
)


@dataclass(frozen=True)
class Setting:
    number: int
    definitions: str
    pos: str
    mapping: str
    model: str

    @property
    def name(self) -> str:
        return (
            f"{'All definitions' if self.definitions == 'all' else 'First definition'} / "
            f"{'all POS' if self.pos == 'all' else 'selected POS'} / "
            f"{self.mapping} / "
            f"{'English model' if self.model == 'english' else 'multilingual model'}"
        )


SETTINGS: dict[int, Setting] = {}
for offset, mapping in ((0, "cognates"), (8, "MT")):
    for first_offset, definitions in ((0, "all"), (4, "first")):
        SETTINGS[offset + first_offset + 1] = Setting(
            offset + first_offset + 1, definitions, "all", mapping, "english"
        )
        SETTINGS[offset + first_offset + 2] = Setting(
            offset + first_offset + 2, definitions, "all", mapping, "multilingual"
        )
        SETTINGS[offset + first_offset + 3] = Setting(
            offset + first_offset + 3, definitions, "selected", mapping, "english"
        )
        SETTINGS[offset + first_offset + 4] = Setting(
            offset + first_offset + 4,
            definitions,
            "selected",
            mapping,
            "multilingual",
        )


@dataclass
class DictionaryData:
    entries: dict[str, dict[str, list[str]]]
    surfaces: dict[str, str]
    expression_aliases: dict[str, str]


class Translator(Protocol):
    def translate_many(self, texts: Sequence[str], batch_size: int) -> list[str]: ...


def dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def clean_definition(value: object, remove_parentheses: bool = True) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    if remove_parentheses:
        text = PARENS_RE.sub("", text)
    return SPACE_RE.sub(" ", text).strip()


def normalize_lemma(value: object, preserve_spaces: bool = False) -> str:
    text = unicodedata.normalize("NFD", str(value or "").strip().lower())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = unicodedata.normalize("NFC", text)
    if preserve_spaces:
        words = ["".join(char for char in word if char.isalpha()) for word in text.split()]
        return " ".join(word for word in words if word)
    return "".join(char for char in text if char.isalpha())


def normalize_pos(value: object) -> str:
    pos = clean_definition(value, False).lower().rstrip(",.;:")
    direct = {
        "n": "noun", "n.": "noun", "noun": "noun",
        "v": "verb", "v.": "verb", "verb": "verb",
        "a": "adj", "a.": "adj", "adj": "adj", "adj.": "adj",
        "adjective": "adj", "adv": "adv", "adv.": "adv", "adverb": "adv",
        "proper noun": "proper_noun", "name": "proper_noun",
        "pron": "pron", "pronoun": "pron", "prep": "prep",
        "preposition": "prep", "conj": "conj", "conjunction": "conj",
        "interj": "interj", "intj": "interj", "article": "article",
        "det": "det", "particle": "particle", "prefix": "prefix",
        "suffix": "suffix", "num": "num", "numeral": "num",
    }
    if pos in direct:
        return direct[pos]
    if re.search(r"\badv(?:erb)?\b", pos):
        return "adv"
    if re.search(r"\badj(?:ective)?\b", pos):
        return "adj"
    if re.search(r"\bverb\b", pos) or re.match(r"^v(?:\.|\s|$)", pos):
        return "verb"
    if re.search(r"\bnoun\b", pos) or re.match(r"^n(?:\.|\s|$)", pos):
        return "noun"
    if "proper" in pos or pos == "name":
        return "proper_noun"
    if "pron" in pos:
        return "pron"
    if "prep" in pos:
        return "prep"
    if "conj" in pos:
        return "conj"
    return pos or "other"


def ensure_resource(path: Path | None, default: Path, url: str, downloads: bool) -> Path:
    target = path or default
    if target.exists() and target.stat().st_size:
        return target
    if not downloads:
        raise FileNotFoundError(f"Required resource not found: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, timeout=300, stream=True) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return target


def looks_like_form(obj: Mapping[str, Any]) -> bool:
    tags = {str(value).lower() for value in obj.get("tags", []) or []}
    if {"form-of", "inflection"} & tags or obj.get("form_of") or obj.get("alt_of"):
        return True
    return any(
        {"form-of", "inflection"}
        & {str(value).lower() for value in sense.get("tags", []) or []}
        or sense.get("form_of")
        or sense.get("alt_of")
        for sense in obj.get("senses", []) or []
    )


def load_kaikki(path: Path, expected_language: str) -> DictionaryData:
    data: defaultdict[str, defaultdict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    surfaces: dict[str, str] = {}
    aliases: dict[str, str] = {}

    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc

            language = obj.get("lang")
            if language not in (None, "", expected_language) or looks_like_form(obj):
                continue

            surface = clean_definition(obj.get("word", ""), False)
            lemma = normalize_lemma(surface)
            expression = normalize_lemma(surface, preserve_spaces=True)
            pos = normalize_pos(obj.get("pos", ""))
            glosses: list[str] = []

            for sense in obj.get("senses", []) or []:
                values = sense.get("glosses") or sense.get("raw_glosses") or []
                for gloss in values:
                    cleaned = clean_definition(gloss).rstrip(".;:")
                    if not cleaned:
                        continue
                    # Preserve comma-separated semantic material; split only
                    # semicolon-delimited annotations/subsenses conservatively.
                    short = GLOSS_SPLIT_RE.split(cleaned, maxsplit=1)[0].strip()
                    if short:
                        glosses.append(short)

            if lemma and glosses:
                surfaces.setdefault(lemma, surface)
                data[lemma][pos].extend(glosses)
                if expression:
                    aliases[expression] = lemma

    entries = {
        lemma: {
            pos: dedupe(definitions)
            for pos, definitions in pos_dict.items()
            if definitions
        }
        for lemma, pos_dict in data.items()
    }
    return DictionaryData(entries=entries, surfaces=surfaces, expression_aliases=aliases)


def load_frequency(path: Path, mode: str = "auto") -> tuple[dict[str, float], dict[str, float]]:
    """Load a Dickinson-style frequency list.

    Returns ``(raw_value, frequency_score)``. For rank-based input,
    ``frequency_score`` is the negative rank. For count/frequency input it is
    the original positive value.
    """
    rows: list[tuple[str, float, int | None]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part for part in re.split(r"[\t,; ]+", line) if part]
            if len(parts) < 2:
                continue

            numeric: list[tuple[int, float]] = []
            textual: list[tuple[int, str]] = []
            for index, part in enumerate(parts):
                try:
                    numeric.append((index, float(part.replace(",", "."))))
                except ValueError:
                    textual.append((index, part))
            if not numeric or not textual:
                continue

            # Dickinson files commonly use rank, lemma, count/frequency. Prefer
            # the first alphabetic field as the lemma and remember rank when it
            # is the first column.
            lemma = normalize_lemma(textual[0][1])
            if not lemma:
                continue
            rank = int(numeric[0][1]) if numeric[0][0] == 0 and numeric[0][1] > 0 else None
            value = numeric[-1][1]
            rows.append((lemma, value, rank))

    if not rows:
        raise ValueError(f"No usable frequency rows found in {path}")

    resolved_mode = mode
    if mode == "auto":
        rank_share = sum(rank is not None for _, _, rank in rows) / len(rows)
        resolved_mode = "rank" if rank_share >= 0.8 else "count"

    raw_values: dict[str, float] = {}
    scores: dict[str, float] = {}
    if resolved_mode == "rank":
        fallback_rank = 1
        for lemma, _, rank in rows:
            effective_rank = float(rank if rank is not None else fallback_rank)
            fallback_rank += 1
            if lemma not in raw_values or effective_rank < raw_values[lemma]:
                raw_values[lemma] = effective_rank
                scores[lemma] = -effective_rank
    elif resolved_mode == "count":
        for lemma, value, _ in rows:
            raw_values[lemma] = raw_values.get(lemma, 0.0) + float(value)
            scores[lemma] = raw_values[lemma]
    else:
        raise ValueError("frequency mode must be auto, rank, or count")
    return raw_values, scores


def load_concreteness(path: Path) -> dict[str, float]:
    dataframe = pd.read_excel(path)
    lower = {str(column).strip().lower(): column for column in dataframe.columns}
    word_column = next(
        (lower[name] for name in ("word", "lemma", "term") if name in lower), None
    )
    score_column = next(
        (
            lower[name]
            for name in ("conc.m", "concreteness", "mean")
            if name in lower
        ),
        None,
    )
    if word_column is None or score_column is None:
        raise ValueError(f"Cannot locate concreteness columns in {list(dataframe.columns)}")

    scores: dict[str, float] = {}
    for _, row in dataframe.iterrows():
        word = str(row[word_column]).strip().lower()
        score = pd.to_numeric(
            str(row[score_column]).replace(",", "."), errors="coerce"
        )
        if word and pd.notna(score):
            scores[word] = float(score)
    return scores


def definitions_concreteness(
    definitions: Sequence[str], lookup: Mapping[str, float]
) -> dict[str, float | int]:
    pooled_scores: list[float] = []
    total_tokens = 0
    matched_tokens = 0
    definitions_with_scores = 0

    for definition in definitions:
        tokens = [token.lower() for token in ENGLISH_TOKEN_RE.findall(definition)]
        ratings = [lookup[token] for token in tokens if token in lookup]
        total_tokens += len(tokens)
        matched_tokens += len(ratings)
        pooled_scores.extend(ratings)
        if ratings:
            definitions_with_scores += 1

    return {
        "mean": float(np.mean(pooled_scores)) if pooled_scores else np.nan,
        "matched_tokens": matched_tokens,
        "all_tokens": total_tokens,
        "coverage": matched_tokens / total_tokens if total_tokens else np.nan,
        "definitions_with_scores": definitions_with_scores,
        "definition_coverage": (
            definitions_with_scores / len(definitions) if definitions else np.nan
        ),
    }


def select_definitions(values: Sequence[str], selection: str) -> list[str]:
    return list(values) if selection == "all" else ([values[0]] if values else [])


def definitions_for(
    lemma: str, pos: str, data: Mapping[str, Mapping[str, Sequence[str]]]
) -> list[str]:
    if lemma not in data:
        return []
    if pos == "all":
        return dedupe(
            definition
            for definitions in data[lemma].values()
            for definition in definitions
        )
    return list(data[lemma].get(pos, []))


def definitions_for_concreteness(
    lemma: str,
    positions: Sequence[str],
    selection: str,
    data: Mapping[str, Mapping[str, Sequence[str]]],
) -> list[str]:
    lemma_data = data.get(lemma, {})
    if not lemma_data:
        return []
    position_set = set(positions)
    if "all" in position_set:
        eligible = dedupe(
            definition
            for definitions in lemma_data.values()
            for definition in definitions
        )
        return select_definitions(eligible, selection)
    if selection == "first":
        for pos, definitions in lemma_data.items():
            if pos in position_set:
                cleaned = dedupe(definitions)
                if cleaned:
                    return [cleaned[0]]
        return []
    return dedupe(
        definition
        for pos, definitions in lemma_data.items()
        if pos in position_set
        for definition in definitions
    )


def embedding_lookup(
    model: Any, texts: Sequence[str], batch_size: int
) -> dict[str, np.ndarray]:
    unique = sorted(set(texts))
    if not unique:
        return {}
    matrix = model.encode(
        unique,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return dict(zip(unique, matrix))


def apd(
    old: Sequence[str], new: Sequence[str], lookup: Mapping[str, np.ndarray]
) -> float:
    left = np.vstack([lookup[text] for text in old])
    right = np.vstack([lookup[text] for text in new])
    return float(np.mean(1.0 - left @ right.T))


def safe_spearman(
    x: Sequence[float], y: Sequence[float]
) -> tuple[float, float, int]:
    dataframe = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(dataframe) < 3 or dataframe.x.nunique() < 2 or dataframe.y.nunique() < 2:
        return np.nan, np.nan, len(dataframe)
    result = spearmanr(dataframe.x, dataframe.y)
    return float(result.statistic), float(result.pvalue), len(dataframe)


def safe_wilcoxon(
    old: Sequence[float], new: Sequence[float]
) -> tuple[float, float, int]:
    dataframe = pd.DataFrame({"old": old, "new": new}).dropna()
    if len(dataframe) < 2:
        return np.nan, np.nan, len(dataframe)
    differences = dataframe.new - dataframe.old
    if np.allclose(differences, 0):
        return 0.0, 1.0, len(dataframe)
    result = wilcoxon(
        dataframe.new,
        dataframe.old,
        alternative="less",
        zero_method="wilcox",
    )
    return float(result.statistic), float(result.pvalue), len(dataframe)


def symbol(p_value: float, expected: bool) -> str:
    if pd.isna(p_value) or not expected or p_value >= 0.05:
        return "✗"
    if p_value < 0.001:
        return "✓**"
    if p_value < 0.01:
        return "✓*"
    return "✓"


def has_proper_noun(
    lemma: str, data: Mapping[str, Mapping[str, Sequence[str]]]
) -> bool:
    return "proper_noun" in data.get(lemma, {})


def latin_verb_candidates(lemma: str) -> list[str]:
    candidates: list[str] = []
    if lemma.endswith("are") and len(lemma) > 4:
        candidates.append(lemma[:-3] + "er")
    if lemma.endswith("ire") and len(lemma) > 4:
        candidates.append(lemma[:-3] + "ir")
    return candidates


def eligible_lemma_pair(
    latin_lemma: str,
    french_lemma: str,
    latin: Mapping[str, Mapping[str, Sequence[str]]],
    french: Mapping[str, Mapping[str, Sequence[str]]],
    min_length: int,
    exclude_proper_nouns: bool,
) -> bool:
    if len(latin_lemma) < min_length or len(french_lemma) < min_length:
        return False
    if exclude_proper_nouns and (
        has_proper_noun(latin_lemma, latin) or has_proper_noun(french_lemma, french)
    ):
        return False
    return True


def build_cognate_mapping(
    setting: Setting,
    latin: Mapping[str, Mapping[str, Sequence[str]]],
    french: Mapping[str, Mapping[str, Sequence[str]]],
    min_length: int,
    exclude_proper_nouns: bool,
    use_verb_rules: bool,
) -> list[dict[str, str]]:
    chosen: dict[str, tuple[str, str]] = {}

    for lemma in sorted(set(latin) & set(french)):
        if eligible_lemma_pair(
            lemma, lemma, latin, french, min_length, exclude_proper_nouns
        ):
            chosen[lemma] = (lemma, "exact")

    if use_verb_rules:
        for latin_lemma in sorted(latin):
            if latin_lemma in chosen or "verb" not in latin[latin_lemma]:
                continue
            for french_lemma in latin_verb_candidates(latin_lemma):
                if french_lemma not in french or "verb" not in french[french_lemma]:
                    continue
                if eligible_lemma_pair(
                    latin_lemma,
                    french_lemma,
                    latin,
                    french,
                    min_length,
                    exclude_proper_nouns,
                ):
                    chosen[latin_lemma] = (french_lemma, "verb_rule")
                    break

    rows: list[dict[str, str]] = []
    for latin_lemma, (french_lemma, match_type) in sorted(chosen.items()):
        if setting.pos == "all":
            rows.append(
                {
                    "latin_lemma": latin_lemma,
                    "french_lemma": french_lemma,
                    "pos": "all",
                    "match_type": match_type,
                }
            )
        else:
            shared = (
                set(latin[latin_lemma]) & set(french[french_lemma]) & SELECTED_POS
            )
            for pos in sorted(shared):
                rows.append(
                    {
                        "latin_lemma": latin_lemma,
                        "french_lemma": french_lemma,
                        "pos": pos,
                        "match_type": match_type,
                    }
                )
    return rows


class GoogleTranslator:
    def __init__(self, project: str | None = None) -> None:
        try:
            from google.cloud import translate_v2 as translate
        except ImportError as exc:
            raise ImportError("Install google-cloud-translate for provider=google") from exc
        self.client = translate.Client(project=project)

    def translate_many(self, texts: Sequence[str], batch_size: int) -> list[str]:
        output: list[str] = []
        for start in range(0, len(texts), batch_size):
            result = self.client.translate(
                list(texts[start : start + batch_size]),
                source_language="la",
                target_language="fr",
                format_="text",
            )
            if isinstance(result, Mapping):
                result = [result]
            output.extend(str(item["translatedText"]) for item in result)
        return output


class DeepLTranslator:
    def __init__(self) -> None:
        auth_key = os.getenv("DEEPL_AUTH_KEY")
        if not auth_key:
            raise EnvironmentError("DEEPL_AUTH_KEY is required for provider=deepl")
        try:
            import deepl
        except ImportError as exc:
            raise ImportError("Install deepl for provider=deepl") from exc
        self.client = deepl.Translator(auth_key)

    def translate_many(self, texts: Sequence[str], batch_size: int) -> list[str]:
        output: list[str] = []
        for start in range(0, len(texts), batch_size):
            # DeepL does not officially expose Latin as a source language.
            # Omitting source_lang lets the service detect it when supported.
            results = self.client.translate_text(
                list(texts[start : start + batch_size]), target_lang="FR"
            )
            if not isinstance(results, list):
                results = [results]
            output.extend(result.text for result in results)
        return output


class LibreTranslateTranslator:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint.rstrip("/") + "/translate"
        self.api_key = os.getenv("LIBRETRANSLATE_API_KEY", "")

    def translate_many(self, texts: Sequence[str], batch_size: int) -> list[str]:
        output: list[str] = []
        for text in texts:
            payload = {
                "q": text,
                "source": "la",
                "target": "fr",
                "format": "text",
            }
            if self.api_key:
                payload["api_key"] = self.api_key
            response = requests.post(self.endpoint, json=payload, timeout=120)
            response.raise_for_status()
            output.append(str(response.json()["translatedText"]))
        return output


def build_translator(args: argparse.Namespace) -> Translator:
    if args.translation_provider == "google":
        return GoogleTranslator(args.google_project)
    if args.translation_provider == "deepl":
        return DeepLTranslator()
    if args.translation_provider == "libretranslate":
        return LibreTranslateTranslator(args.libretranslate_url)
    raise ValueError(
        "MT settings require --mt-cache or --translation-provider "
        "google/deepl/libretranslate"
    )


def translate_lemmas(
    lemmas: Sequence[str],
    surfaces: Mapping[str, str],
    translator: Translator,
    batch_size: int,
) -> pd.DataFrame:
    source_forms = [surfaces.get(lemma, lemma) for lemma in lemmas]
    translated = translator.translate_many(source_forms, batch_size)
    if len(translated) != len(lemmas):
        raise RuntimeError("Translation provider returned an unexpected number of rows")
    return pd.DataFrame(
        {
            "latin_lemma": list(lemmas),
            "source_form": source_forms,
            "mt_output": translated,
        }
    )


def load_spacy_french(optional: bool = True) -> Any | None:
    try:
        import spacy
    except ImportError:
        if optional:
            return None
        raise
    try:
        return spacy.load("fr_core_news_sm", disable=["ner", "parser"])
    except OSError:
        if optional:
            return None
        raise OSError(
            "Install the French spaCy model: python -m spacy download fr_core_news_sm"
        )


def conservative_mt_mapping(
    translations: pd.DataFrame,
    latin: Mapping[str, Mapping[str, Sequence[str]]],
    french_data: DictionaryData,
    nlp: Any | None,
    max_fanout: int,
    min_length: int,
    exclude_proper_nouns: bool,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    cleaned_outputs = [
        clean_definition(value, False).strip(
            " \t\n\r.,;:!?·—–-()[]{}«»“”\"'"
        )
        for value in translations.mt_output
    ]
    docs = list(nlp.pipe(cleaned_outputs, batch_size=128)) if nlp is not None else [None] * len(cleaned_outputs)

    for row, cleaned, doc in zip(
        translations.itertuples(index=False), cleaned_outputs, docs
    ):
        surface_expression = normalize_lemma(cleaned, preserve_spaces=True)
        lemma_expression = ""
        lexical_tokens = 0
        if doc is not None:
            tokens = [token for token in doc if token.is_alpha]
            lexical_tokens = len(tokens)
            lemma_expression = " ".join(
                normalize_lemma(token.lemma_ or token.text)
                for token in tokens
                if normalize_lemma(token.lemma_ or token.text)
            )
        else:
            lexical_tokens = len(surface_expression.split())

        target = french_data.expression_aliases.get(surface_expression, "")
        if not target and lemma_expression:
            target = french_data.expression_aliases.get(lemma_expression, "")
        if not target:
            compact_surface = surface_expression.replace(" ", "")
            compact_lemma = lemma_expression.replace(" ", "")
            if compact_surface in french_data.entries:
                target = compact_surface
            elif compact_lemma in french_data.entries:
                target = compact_lemma

        if target in INVALID_MT_TARGETS or not target:
            target = ""
        if target and not eligible_lemma_pair(
            row.latin_lemma,
            target,
            latin,
            french_data.entries,
            min_length,
            exclude_proper_nouns,
        ):
            target = ""

        source_pos = set(latin.get(row.latin_lemma, {})) & SELECTED_POS
        target_pos = (
            set(french_data.entries.get(target, {})) & SELECTED_POS if target else set()
        )
        shared = source_pos & target_pos
        records.append(
            {
                "latin_lemma": row.latin_lemma,
                "source_form": row.source_form,
                "mt_output": row.mt_output,
                "accepted_target_lemma": target,
                "normalized_surface_expression": surface_expression,
                "lemmatized_expression": lemma_expression,
                "n_lexical_tokens": lexical_tokens,
                "source_selected_pos": " | ".join(sorted(source_pos)),
                "target_selected_pos": " | ".join(sorted(target_pos)),
                "shared_selected_pos": " | ".join(sorted(shared)),
            }
        )

    dataframe = pd.DataFrame(records)
    dataframe = dataframe[dataframe.accepted_target_lemma.ne("")].copy()
    counts = dataframe.accepted_target_lemma.value_counts()
    dataframe["target_fanout"] = dataframe.accepted_target_lemma.map(counts)
    dataframe = dataframe[dataframe.target_fanout <= max_fanout]
    return dataframe.drop_duplicates("latin_lemma").reset_index(drop=True)


def run_setting(
    setting: Setting,
    mapping_rows: Sequence[Mapping[str, str]],
    latin: Mapping[str, Mapping[str, Sequence[str]]],
    french: Mapping[str, Mapping[str, Sequence[str]]],
    frequency_raw: Mapping[str, float],
    frequency_score: Mapping[str, float],
    concreteness_lookup: Mapping[str, float],
    embeddings: Mapping[str, np.ndarray],
    model_name: str,
    output_dir: Path,
) -> dict[str, Any]:
    pair_rows: list[dict[str, Any]] = []

    for mapping in mapping_rows:
        old_lemma = mapping["latin_lemma"]
        new_lemma = mapping["french_lemma"]
        pos = mapping["pos"]
        old_all = definitions_for(old_lemma, pos, latin)
        new_all = definitions_for(new_lemma, pos, french)
        old_defs = select_definitions(old_all, setting.definitions)
        new_defs = select_definitions(new_all, setting.definitions)
        if not old_defs or not new_defs:
            continue

        old_pair_concreteness = definitions_concreteness(
            old_defs, concreteness_lookup
        )
        new_pair_concreteness = definitions_concreteness(
            new_defs, concreteness_lookup
        )
        pair_rows.append(
            {
                "setting": setting.number,
                "setting_name": setting.name,
                "model": model_name,
                "mapping_setting": setting.mapping,
                "latin_lemma": old_lemma,
                "french_lemma": new_lemma,
                "pos": pos,
                "match_type": mapping.get("match_type", setting.mapping),
                "semantic_distance": apd(old_defs, new_defs, embeddings),
                "frequency_raw": frequency_raw.get(old_lemma, np.nan),
                "frequency_score": frequency_score.get(old_lemma, np.nan),
                "latin_polysemy": len(dedupe(old_all)),
                "latin_concreteness": old_pair_concreteness["mean"],
                "french_concreteness": new_pair_concreteness["mean"],
                "latin_concreteness_matched_tokens": old_pair_concreteness[
                    "matched_tokens"
                ],
                "latin_concreteness_all_tokens": old_pair_concreteness["all_tokens"],
                "latin_concreteness_coverage": old_pair_concreteness["coverage"],
                "french_concreteness_matched_tokens": new_pair_concreteness[
                    "matched_tokens"
                ],
                "french_concreteness_all_tokens": new_pair_concreteness["all_tokens"],
                "french_concreteness_coverage": new_pair_concreteness["coverage"],
                "latin_definitions": " || ".join(old_defs),
                "french_definitions": " || ".join(new_defs),
            }
        )

    pair_dataframe = pd.DataFrame(pair_rows)
    if pair_dataframe.empty:
        raise ValueError(f"Setting {setting.number} produced no usable pairs")

    lemma_rows: list[dict[str, Any]] = []
    for old_lemma, group in pair_dataframe.groupby("latin_lemma", sort=True):
        new_lemma = str(group["french_lemma"].iloc[0])
        positions = group["pos"].astype(str).tolist()
        old_concreteness_defs = definitions_for_concreteness(
            old_lemma, positions, setting.definitions, latin
        )
        new_concreteness_defs = definitions_for_concreteness(
            new_lemma, positions, setting.definitions, french
        )
        old_concreteness = definitions_concreteness(
            old_concreteness_defs, concreteness_lookup
        )
        new_concreteness = definitions_concreteness(
            new_concreteness_defs, concreteness_lookup
        )

        polysemy_positions = set(positions)
        if "all" in polysemy_positions:
            polysemy_defs = definitions_for(old_lemma, "all", latin)
        else:
            polysemy_defs = dedupe(
                definition
                for pos in polysemy_positions
                for definition in definitions_for(old_lemma, pos, latin)
            )

        lemma_rows.append(
            {
                "latin_lemma": old_lemma,
                "semantic_distance": float(group.semantic_distance.mean()),
                "frequency_raw": group.frequency_raw.iloc[0],
                "frequency_score": group.frequency_score.iloc[0],
                "latin_polysemy": len(polysemy_defs),
                "latin_concreteness": old_concreteness["mean"],
                "french_concreteness": new_concreteness["mean"],
                "latin_concreteness_matched_tokens": old_concreteness[
                    "matched_tokens"
                ],
                "latin_concreteness_all_tokens": old_concreteness["all_tokens"],
                "latin_concreteness_coverage": old_concreteness["coverage"],
                "french_concreteness_matched_tokens": new_concreteness[
                    "matched_tokens"
                ],
                "french_concreteness_all_tokens": new_concreteness["all_tokens"],
                "french_concreteness_coverage": new_concreteness["coverage"],
                "latin_concreteness_definitions": " || ".join(
                    old_concreteness_defs
                ),
                "french_concreteness_definitions": " || ".join(
                    new_concreteness_defs
                ),
                "french_lemmas": " | ".join(sorted(set(group.french_lemma))),
                "positions": " | ".join(sorted(set(positions))),
                "match_types": " | ".join(sorted(set(group.match_type))),
            }
        )

    lemma_dataframe = pd.DataFrame(lemma_rows)
    lemma_dataframe["concreteness_change"] = (
        lemma_dataframe.french_concreteness - lemma_dataframe.latin_concreteness
    )

    frequency_rho, frequency_p, frequency_n = safe_spearman(
        lemma_dataframe.frequency_score, lemma_dataframe.semantic_distance
    )
    polysemy_rho, polysemy_p, polysemy_n = safe_spearman(
        lemma_dataframe.latin_polysemy, lemma_dataframe.semantic_distance
    )
    concreteness_stat, concreteness_p, concreteness_n = safe_wilcoxon(
        lemma_dataframe.latin_concreteness, lemma_dataframe.french_concreteness
    )
    mean_change = float(lemma_dataframe.concreteness_change.mean())
    median_change = float(lemma_dataframe.concreteness_change.median())
    percent_more_abstract = float(
        100 * (lemma_dataframe.concreteness_change < 0).mean()
    )

    summary = {
        "setting": setting.number,
        "setting_name": setting.name,
        "model": model_name,
        "mapping_setting": setting.mapping,
        "shared_pairs": len(pair_dataframe),
        "shared_lemmas": len(lemma_dataframe),
        "frequency_n": frequency_n,
        "frequency_rho": frequency_rho,
        "frequency_p": frequency_p,
        "frequency_result": symbol(frequency_p, frequency_rho < 0),
        "polysemy_n": polysemy_n,
        "polysemy_rho": polysemy_rho,
        "polysemy_p": polysemy_p,
        "polysemy_result": symbol(polysemy_p, polysemy_rho > 0),
        "concreteness_n": concreteness_n,
        "latin_concreteness_mean": lemma_dataframe.latin_concreteness.mean(),
        "french_concreteness_mean": lemma_dataframe.french_concreteness.mean(),
        "mean_concreteness_change": mean_change,
        "median_concreteness_change": median_change,
        "percent_more_abstract": percent_more_abstract,
        "latin_concreteness_coverage_mean": (
            lemma_dataframe.latin_concreteness_coverage.mean()
        ),
        "french_concreteness_coverage_mean": (
            lemma_dataframe.french_concreteness_coverage.mean()
        ),
        "concreteness_wilcoxon": concreteness_stat,
        "concreteness_p": concreteness_p,
        "concreteness_result": symbol(concreteness_p, mean_change < 0),
    }

    pair_dataframe.to_csv(
        output_dir / f"latin_french_setting_{setting.number}_pair_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lemma_dataframe.to_csv(
        output_dir / f"latin_french_setting_{setting.number}_lemma_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(
        f"\nSETTING {setting.number}: {setting.name}"
        f"\nlemmas={len(lemma_dataframe):,} "
        f"frequency={frequency_rho:.4g}/{frequency_p:.4g} "
        f"polysemy={polysemy_rho:.4g}/{polysemy_p:.4g} "
        f"concreteness p={concreteness_p:.4g}"
    )
    return summary


def parse_settings(values: Sequence[str]) -> list[int]:
    if not values or values == ["all"]:
        return sorted(SETTINGS)
    selected = sorted(set(int(value) for value in values))
    invalid = [value for value in selected if value not in SETTINGS]
    if invalid:
        raise ValueError(f"Settings must be 1–16; invalid: {invalid}")
    return selected


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Run Latin-to-French dictionary semantic-change experiments."
    )
    argument_parser.add_argument("--latin-file", type=Path)
    argument_parser.add_argument("--french-file", type=Path)
    argument_parser.add_argument("--frequency-file", type=Path, required=True)
    argument_parser.add_argument("--concreteness-file", type=Path, required=True)
    argument_parser.add_argument(
        "--output-dir", type=Path, default=Path("latin_french_results")
    )
    argument_parser.add_argument("--no-download", action="store_true")
    argument_parser.add_argument("--settings", nargs="+", default=["all"])
    argument_parser.add_argument("--english-model", default=EN_MODEL)
    argument_parser.add_argument("--multilingual-model", default=MULTI_MODEL)
    argument_parser.add_argument("--batch-size", type=int, default=128)
    argument_parser.add_argument("--mt-batch-size", type=int, default=50)
    argument_parser.add_argument("--max-target-fanout", type=int, default=5)
    argument_parser.add_argument("--minimum-lemma-length", type=int, default=3)
    argument_parser.add_argument(
        "--include-proper-nouns",
        action="store_true",
        help="Do not apply the finalized proper-noun exclusion.",
    )
    argument_parser.add_argument(
        "--disable-verb-rules",
        action="store_true",
        help="Use exact cognate matches only; disable -are/-er and -ire/-ir rules.",
    )
    argument_parser.add_argument(
        "--frequency-mode", choices=["auto", "rank", "count"], default="auto"
    )
    argument_parser.add_argument(
        "--mt-cache",
        type=Path,
        help="CSV with latin_lemma, source_form and mt_output columns.",
    )
    argument_parser.add_argument(
        "--translation-provider",
        choices=["none", "google", "deepl", "libretranslate"],
        default="none",
    )
    argument_parser.add_argument("--google-project")
    argument_parser.add_argument(
        "--libretranslate-url", default="http://localhost:5000"
    )
    return argument_parser


def main() -> None:
    args = parser().parse_args()
    selected = parse_settings(args.settings)

    for path, label in (
        (args.frequency_file, "Frequency file"),
        (args.concreteness_file, "Concreteness file"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    resources = args.output_dir / "resources"
    resources.mkdir(exist_ok=True)
    latin_path = ensure_resource(
        args.latin_file,
        resources / "kaikki.org-dictionary-Latin.jsonl",
        LATIN_KAIKKI_URL,
        not args.no_download,
    )
    french_path = ensure_resource(
        args.french_file,
        resources / "kaikki.org-dictionary-French.jsonl",
        FRENCH_KAIKKI_URL,
        not args.no_download,
    )

    latin_data = load_kaikki(latin_path, "Latin")
    french_data = load_kaikki(french_path, "French")
    frequency_raw, frequency_score = load_frequency(
        args.frequency_file, args.frequency_mode
    )
    concreteness = load_concreteness(args.concreteness_file)

    audit_rows = []
    for lemma in sorted(latin_data.entries):
        definitions = definitions_for(lemma, "all", latin_data.entries)
        audit_rows.append(
            {
                "latin_lemma": lemma,
                "surface": latin_data.surfaces.get(lemma, lemma),
                "positions": " | ".join(sorted(latin_data.entries[lemma])),
                "polysemy": len(definitions),
                "frequency_raw": frequency_raw.get(lemma, np.nan),
                "frequency_score": frequency_score.get(lemma, np.nan),
                "proper_noun": has_proper_noun(lemma, latin_data.entries),
            }
        )
    pd.DataFrame(audit_rows).to_csv(
        args.output_dir / "latin_french_preparation_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mt_needed = any(SETTINGS[number].mapping == "MT" for number in selected)
    mt_dataframe: pd.DataFrame | None = None
    if mt_needed:
        if args.mt_cache:
            if not args.mt_cache.is_file():
                raise FileNotFoundError(f"MT cache not found: {args.mt_cache}")
            translations = pd.read_csv(args.mt_cache)
        else:
            translator = build_translator(args)
            translations = translate_lemmas(
                sorted(latin_data.entries),
                latin_data.surfaces,
                translator,
                args.mt_batch_size,
            )
            translations.to_csv(
                args.output_dir / "latin_french_mt_translations.csv",
                index=False,
                encoding="utf-8-sig",
            )

        required = {"latin_lemma", "source_form", "mt_output"}
        if not required.issubset(translations.columns):
            raise ValueError(f"MT cache must contain {sorted(required)}")
        mt_dataframe = conservative_mt_mapping(
            translations,
            latin_data.entries,
            french_data,
            load_spacy_french(optional=True),
            args.max_target_fanout,
            args.minimum_lemma_length,
            not args.include_proper_nouns,
        )
        mt_dataframe.to_csv(
            args.output_dir / "latin_french_mt_conservative_mapping.csv",
            index=False,
            encoding="utf-8-sig",
        )

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("Install sentence-transformers") from exc

    models: dict[str, Any] = {}
    summaries: list[dict[str, Any]] = []
    for number in selected:
        setting = SETTINGS[number]
        if setting.mapping == "cognates":
            mappings = build_cognate_mapping(
                setting,
                latin_data.entries,
                french_data.entries,
                args.minimum_lemma_length,
                not args.include_proper_nouns,
                not args.disable_verb_rules,
            )
        else:
            assert mt_dataframe is not None
            mappings = []
            for row in mt_dataframe.itertuples(index=False):
                if setting.pos == "all":
                    mappings.append(
                        {
                            "latin_lemma": row.latin_lemma,
                            "french_lemma": row.accepted_target_lemma,
                            "pos": "all",
                            "match_type": "MT",
                        }
                    )
                else:
                    for pos in str(row.shared_selected_pos).split(" | "):
                        if pos in SELECTED_POS:
                            mappings.append(
                                {
                                    "latin_lemma": row.latin_lemma,
                                    "french_lemma": row.accepted_target_lemma,
                                    "pos": pos,
                                    "match_type": "MT",
                                }
                            )

        texts: list[str] = []
        for mapping in mappings:
            texts += select_definitions(
                definitions_for(
                    mapping["latin_lemma"], mapping["pos"], latin_data.entries
                ),
                setting.definitions,
            )
            texts += select_definitions(
                definitions_for(
                    mapping["french_lemma"], mapping["pos"], french_data.entries
                ),
                setting.definitions,
            )

        model_name = (
            args.english_model
            if setting.model == "english"
            else args.multilingual_model
        )
        if setting.model not in models:
            models[setting.model] = SentenceTransformer(model_name)
        embeddings = embedding_lookup(
            models[setting.model], texts, args.batch_size
        )
        summaries.append(
            run_setting(
                setting,
                mappings,
                latin_data.entries,
                french_data.entries,
                frequency_raw,
                frequency_score,
                concreteness,
                embeddings,
                model_name,
                args.output_dir,
            )
        )

    pd.DataFrame(summaries).sort_values("setting").to_csv(
        args.output_dir / "latin_french_settings_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    main()
