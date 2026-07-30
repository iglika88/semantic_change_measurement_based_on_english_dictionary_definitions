#!/usr/bin/env python3
"""Greek dictionary-based semantic-change experiments.

The script compares Biblical/Koine Greek and Modern Greek dictionary
definitions under sixteen settings:

  1-8   normalized identical-lemma ("cognate") mapping
  9-16  conservative Ancient-to-Modern Greek machine-translation mapping

Within each mapping family, settings vary definition selection (all/first),
POS selection (all/selected), and Sentence-Transformer model (English/MULTI).

Final data policy
-----------------
* Abbott-Smith supplies Biblical lemmas, English definitions and polysemy.
* MorphGNT SBLGNT supplies Biblical frequency and lexical POS.
* Kaikki/Wiktionary supplies Modern Greek lemmas, POS and English definitions.
* Brysbaert et al. supplies English concreteness ratings.
* MT uses ``ilsp/m2m100-1.2B-ag-mg-full-ft`` by default.

Conservative MT policy
----------------------
The complete MT output is lemmatized with spaCy. A mapping is accepted only
when the complete single-word or multiword surface/lemmatized expression is a
Kaikki entry. Words inside an unmatched phrase are never matched separately.
Obvious non-lexical targets are removed, and a Modern target may receive at
most five Biblical source lemmas. Selected-POS MT settings additionally require
at least one shared major lexical POS.

Required local inputs
---------------------
--modern-file        Kaikki Greek JSONL dictionary
--concreteness-file  Brysbaert concreteness ratings Excel file

Abbott-Smith and MorphGNT may be supplied locally or downloaded automatically.

Examples
--------
Run cognate Settings 1 and 8::

    python greek_experiments.py --modern-file Greek.jsonl \
      --concreteness-file Concreteness_ratings.xlsx --settings 1 8

Run MT Settings 9-16::

    python greek_experiments.py --modern-file Greek.jsonl \
      --concreteness-file Concreteness_ratings.xlsx --settings 9 10 11 12 13 14 15 16

Dependencies: numpy, pandas, scipy, lxml, requests, openpyxl,
sentence-transformers, transformers; MT settings additionally require torch,
spacy, and the ``el_core_news_sm`` spaCy model.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import requests
from lxml import etree
from scipy.stats import spearmanr, wilcoxon

ABBOTT_URL = "https://raw.githubusercontent.com/translatable-exegetical-tools/Abbott-Smith/master/abbott-smith.tei.xml"
MORPHGNT_URL = "https://codeload.github.com/morphgnt/sblgnt/zip/refs/heads/master"
EN_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MULTI_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MT_MODEL = "ilsp/m2m100-1.2B-ag-mg-full-ft"
SELECTED_POS = frozenset({"noun", "verb", "adj", "adv"})
ENGLISH_TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)?")
MODERN_SPLIT_RE = re.compile(r"[,;]")
PARENS_RE = re.compile(r"\([^)]*\)")
SPACE_RE = re.compile(r"\s+")
INVALID_MT_TARGETS = frozenset({"", "ο", "η", "το", "οι", "τα", "ένας", "μία", "ένα"})
MORPHGNT_POS_MAP = {
    "N-": "noun", "V-": "verb", "A-": "adj", "D-": "adv",
    "P-": "prep", "C-": "conj", "I-": "interj", "X-": "particle",
    "RA": "article", "RD": "pron", "RI": "pron", "RP": "pron", "RR": "pron",
}


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
            f"{self.mapping} / {'English model' if self.model == 'english' else 'multilingual model'}"
        )


SETTINGS: dict[int, Setting] = {}
for offset, mapping in ((0, "cognates"), (8, "MT")):
    for first_offset, definitions in ((0, "all"), (4, "first")):
        SETTINGS[offset + first_offset + 1] = Setting(offset + first_offset + 1, definitions, "all", mapping, "english")
        SETTINGS[offset + first_offset + 2] = Setting(offset + first_offset + 2, definitions, "all", mapping, "multilingual")
        SETTINGS[offset + first_offset + 3] = Setting(offset + first_offset + 3, definitions, "selected", mapping, "english")
        SETTINGS[offset + first_offset + 4] = Setting(offset + first_offset + 4, definitions, "selected", mapping, "multilingual")


def dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def clean_definition(value: object, remove_parentheses: bool = True) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    if remove_parentheses:
        text = PARENS_RE.sub("", text)
    return SPACE_RE.sub(" ", text).strip()


def normalize_greek(value: object, preserve_spaces: bool = False) -> str:
    text = unicodedata.normalize("NFD", str(value or "").strip().lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = unicodedata.normalize("NFC", text).replace("ς", "σ")
    if preserve_spaces:
        words = ["".join(c for c in word if c.isalpha()) for word in text.split()]
        return " ".join(word for word in words if word)
    return "".join(c for c in text if c.isalpha())


def normalize_pos(value: object) -> str:
    pos = clean_definition(value, False).lower().rstrip(",.;:")
    direct = {
        "n": "noun", "n.": "noun", "noun": "noun", "substantive": "noun",
        "v": "verb", "v.": "verb", "verb": "verb",
        "a": "adj", "a.": "adj", "adj": "adj", "adj.": "adj", "adjective": "adj",
        "adv": "adv", "adv.": "adv", "adverb": "adv",
        "pron": "pron", "pronoun": "pron", "prep": "prep", "preposition": "prep",
        "conj": "conj", "conjunction": "conj", "interj": "interj",
        "num": "num", "article": "article", "det": "det", "particle": "particle",
        "proper noun": "proper_noun", "name": "proper_noun",
    }
    if pos in direct:
        return direct[pos]
    if re.search(r"\badv(?:erb)?\b", pos): return "adv"
    if re.search(r"\badj(?:ective)?\b", pos): return "adj"
    if re.search(r"\bverb\b", pos) or re.match(r"^v(?:\.|\s|$)", pos): return "verb"
    if re.search(r"\bnoun\b", pos) or re.match(r"^n(?:\.|\s|$)", pos): return "noun"
    if "pron" in pos: return "pron"
    if "prep" in pos: return "prep"
    if "conj" in pos: return "conj"
    return pos or "other"


def ensure_resource(path: Path | None, default: Path, url: str, downloads: bool) -> Path:
    target = path or default
    if target.exists() and target.stat().st_size:
        return target
    if not downloads:
        raise FileNotFoundError(f"Required resource not found: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=240)
    response.raise_for_status()
    target.write_bytes(response.content)
    return target


def load_abbott(path: Path) -> tuple[dict[str, list[str]], dict[str, int], dict[str, str]]:
    tree = etree.parse(str(path))
    definitions: defaultdict[str, list[str]] = defaultdict(list)
    surface: dict[str, str] = {}
    for entry in tree.xpath("//*[local-name()='entry']"):
        orths = [clean_definition(x) for x in entry.xpath(".//*[local-name()='orth']/text()") if clean_definition(x)]
        if not orths:
            orths = [clean_definition(x) for x in entry.xpath(".//*[local-name()='foreign']/text()") if clean_definition(x)]
        if not orths:
            continue
        raw = orths[0]
        lemma = normalize_greek(raw)
        glosses = dedupe(clean_definition(x).rstrip(".;:") for x in entry.xpath(".//*[local-name()='sense']/*[local-name()='gloss']/text()"))
        if lemma and glosses:
            surface.setdefault(lemma, raw)
            definitions[lemma].extend(glosses)
    final = {lemma: dedupe(glosses) for lemma, glosses in definitions.items()}
    return final, {lemma: len(glosses) for lemma, glosses in final.items()}, surface


def looks_like_form(obj: Mapping[str, Any]) -> bool:
    tags = {str(x).lower() for x in obj.get("tags", []) or []}
    if {"form-of", "inflection"} & tags or obj.get("form_of") or obj.get("alt_of"):
        return True
    return any(
        {"form-of", "inflection"} & {str(x).lower() for x in sense.get("tags", []) or []}
        or sense.get("form_of") or sense.get("alt_of")
        for sense in obj.get("senses", []) or []
    )


def load_modern(path: Path) -> dict[str, dict[str, list[str]]]:
    data: defaultdict[str, defaultdict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}") from exc
            if obj.get("lang") not in (None, "", "Greek") or looks_like_form(obj):
                continue
            lemma = normalize_greek(obj.get("word", ""))
            pos = normalize_pos(obj.get("pos", ""))
            glosses: list[str] = []
            for sense in obj.get("senses", []) or []:
                values = sense.get("glosses") or sense.get("raw_glosses") or []
                for gloss in values:
                    short = MODERN_SPLIT_RE.split(clean_definition(gloss), maxsplit=1)[0].rstrip(".;:")
                    if short:
                        glosses.append(short)
            if lemma and glosses:
                data[lemma][pos].extend(glosses)
    return {lemma: {pos: dedupe(vals) for pos, vals in posdict.items() if vals} for lemma, posdict in data.items()}


def load_morphgnt(zip_path: Path, extraction_dir: Path) -> tuple[dict[str, int], dict[str, set[str]]]:
    if not extraction_dir.exists() or not list(extraction_dir.rglob("*-morphgnt.txt")):
        extraction_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extraction_dir)
    frequency: Counter[str] = Counter()
    pos_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for file in extraction_dir.rglob("*-morphgnt.txt"):
        with file.open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) != 7:
                    continue
                lemma = normalize_greek(parts[6])
                if not lemma:
                    continue
                frequency[lemma] += 1
                broad = MORPHGNT_POS_MAP.get(parts[1])
                if broad:
                    pos_counts[lemma][broad] += 1
    return dict(frequency), {lemma: set(counts) for lemma, counts in pos_counts.items()}


def attach_morphgnt_pos(
    biblical_defs: Mapping[str, Sequence[str]],
    lemma_pos: Mapping[str, set[str]],
) -> dict[str, dict[str, list[str]]]:
    return {
        lemma: {pos: list(defs) for pos in sorted(lemma_pos.get(lemma, set()))}
        for lemma, defs in biblical_defs.items() if lemma_pos.get(lemma)
    }


def flatten(data: Mapping[str, Mapping[str, Sequence[str]]]) -> dict[str, list[str]]:
    return {lemma: dedupe(d for values in posdict.values() for d in values) for lemma, posdict in data.items()}


def load_concreteness(path: Path) -> dict[str, float]:
    df = pd.read_excel(path)
    lower = {str(c).strip().lower(): c for c in df.columns}
    word_col = next((lower[x] for x in ("word", "lemma", "term") if x in lower), None)
    score_col = next((lower[x] for x in ("conc.m", "concreteness", "mean") if x in lower), None)
    if word_col is None or score_col is None:
        raise ValueError(f"Cannot locate concreteness columns in {list(df.columns)}")
    scores: dict[str, float] = {}
    for _, row in df.iterrows():
        word = str(row[word_col]).strip().lower()
        score = pd.to_numeric(str(row[score_col]).replace(",", "."), errors="coerce")
        if word and pd.notna(score):
            scores[word] = float(score)
    return scores


def concreteness(definitions: Sequence[str], lookup: Mapping[str, float], definition_balanced: bool) -> dict[str, float | int]:
    definition_scores: list[float] = []
    pooled_scores: list[float] = []
    total = matched = 0
    for definition in definitions:
        tokens = [x.lower() for x in ENGLISH_TOKEN_RE.findall(definition)]
        values = [lookup[x] for x in tokens if x in lookup]
        total += len(tokens); matched += len(values); pooled_scores.extend(values)
        if values:
            definition_scores.append(float(np.mean(values)))
    score_values = definition_scores if definition_balanced else pooled_scores
    return {
        "mean": float(np.mean(score_values)) if score_values else np.nan,
        "matched_tokens": matched, "all_tokens": total,
        "coverage": matched / max(1, total),
        "definitions_with_scores": len(definition_scores),
        "definition_coverage": len(definition_scores) / max(1, len(definitions)),
    }


def select_definitions(values: Sequence[str], selection: str) -> list[str]:
    return list(values) if selection == "all" else ([values[0]] if values else [])


def embedding_lookup(model: Any, texts: Sequence[str], batch_size: int) -> dict[str, np.ndarray]:
    unique = sorted(set(texts))
    matrix = model.encode(unique, batch_size=batch_size, show_progress_bar=True,
                          convert_to_numpy=True, normalize_embeddings=True)
    return dict(zip(unique, matrix))


def apd(old: Sequence[str], new: Sequence[str], lookup: Mapping[str, np.ndarray]) -> float:
    left = np.vstack([lookup[x] for x in old]); right = np.vstack([lookup[x] for x in new])
    return float(np.mean(1.0 - left @ right.T))


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float, float, int]:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 3 or df.x.nunique() < 2 or df.y.nunique() < 2:
        return np.nan, np.nan, len(df)
    result = spearmanr(df.x, df.y)
    return float(result.statistic), float(result.pvalue), len(df)


def safe_wilcoxon(old: Sequence[float], new: Sequence[float]) -> tuple[float, float, int]:
    df = pd.DataFrame({"old": old, "new": new}).dropna()
    if len(df) < 2: return np.nan, np.nan, len(df)
    if np.allclose(df.new - df.old, 0): return 0.0, 1.0, len(df)
    result = wilcoxon(df.new, df.old, alternative="less", zero_method="wilcox")
    return float(result.statistic), float(result.pvalue), len(df)


def symbol(p: float, expected: bool) -> str:
    if pd.isna(p) or not expected or p >= .05: return "✗"
    if p < .001: return "✓**"
    if p < .01: return "✓*"
    return "✓"


def build_cognate_mapping(setting: Setting, bib: Mapping[str, Mapping[str, Sequence[str]]], modern: Mapping[str, Mapping[str, Sequence[str]]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for lemma in sorted(set(bib) & set(modern)):
        if setting.pos == "all":
            rows.append({"biblical_lemma": lemma, "modern_lemma": lemma, "pos": "all"})
        else:
            for pos in sorted(set(bib[lemma]) & set(modern[lemma]) & SELECTED_POS):
                rows.append({"biblical_lemma": lemma, "modern_lemma": lemma, "pos": pos})
    return rows


def load_spacy_greek() -> Any:
    try:
        import spacy
    except ImportError as exc:
        raise ImportError("MT settings require spaCy and el_core_news_sm") from exc
    try:
        return spacy.load("el_core_news_sm", disable=["ner", "parser"])
    except OSError as exc:
        raise OSError("Install the Greek spaCy model: python -m spacy download el_core_news_sm") from exc


def translate_lemmas(lemmas: Sequence[str], surface: Mapping[str, str], model_id: str, batch_size: int) -> pd.DataFrame:
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("MT settings require torch and transformers") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    rows: list[dict[str, str]] = []
    source_texts = [surface.get(lemma, lemma) for lemma in lemmas]
    for start in range(0, len(lemmas), batch_size):
        batch_lemmas = lemmas[start:start + batch_size]
        batch_texts = source_texts[start:start + batch_size]
        encoded = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            generated = model.generate(**encoded, max_new_tokens=32)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        rows.extend({"biblical_lemma": lemma, "source_form": source, "mt_output": output}
                    for lemma, source, output in zip(batch_lemmas, batch_texts, decoded))
    return pd.DataFrame(rows)


def conservative_mt_mapping(
    translations: pd.DataFrame,
    bib: Mapping[str, Mapping[str, Sequence[str]]],
    modern: Mapping[str, Mapping[str, Sequence[str]]],
    nlp: Any,
    max_fanout: int,
) -> pd.DataFrame:
    modern_expression_lookup: dict[str, str] = {}
    for canonical in modern:
        # Kaikki keys are canonicalized without spaces for the experiment, but
        # MT matching must preserve phrase boundaries and accept only a whole
        # dictionary expression. Reconstruct a phrase-aware inventory from the
        # original normalized key when possible; single words map directly.
        modern_expression_lookup[canonical] = canonical
    # Add phrase-preserving forms from Kaikki surface lemmas stored as keys is
    # impossible after canonicalization, so load_modern also exposes aliases
    # under the private _expression_aliases attribute when available.
    modern_expression_lookup.update(getattr(modern, "_expression_aliases", {}))
    records: list[dict[str, Any]] = []
    cleaned = [clean_definition(x, False).strip(" \t\n\r.,;:!?··—–-()[]{}«»“”\"'") for x in translations.mt_output]
    docs = nlp.pipe(cleaned, batch_size=128)
    pos_map = {"NOUN": "noun", "VERB": "verb", "AUX": "verb", "ADJ": "adj", "ADV": "adv"}
    for row, doc in zip(translations.itertuples(index=False), docs):
        tokens = [token for token in doc if token.is_alpha]
        surface_expr = " ".join(normalize_greek(token.text) for token in tokens if normalize_greek(token.text))
        lemma_expr = " ".join(normalize_greek(token.lemma_ or token.text) for token in tokens if normalize_greek(token.lemma_ or token.text))
        surface_key = surface_expr.replace(" ", "")
        lemma_key = lemma_expr.replace(" ", "")
        target = surface_key if surface_key in modern else lemma_key if lemma_key in modern else ""
        if target in INVALID_MT_TARGETS:
            target = ""
        source_pos = set(bib.get(row.biblical_lemma, {})) & SELECTED_POS
        target_pos = set(modern.get(target, {})) & SELECTED_POS if target else set()
        shared = source_pos & target_pos
        records.append({
            "biblical_lemma": row.biblical_lemma, "source_form": row.source_form,
            "mt_output": row.mt_output, "accepted_target_lemma": target,
            "n_lexical_tokens": len(tokens),
            "lemmatised_expression": lemma_expr,
            "source_selected_pos": " | ".join(sorted(source_pos)),
            "target_selected_pos": " | ".join(sorted(target_pos)),
            "shared_selected_pos": " | ".join(sorted(shared)),
        })
    df = pd.DataFrame(records)
    df = df[df.accepted_target_lemma.ne("")].copy()
    counts = df.accepted_target_lemma.value_counts()
    df["target_fanout"] = df.accepted_target_lemma.map(counts)
    return df[df.target_fanout <= max_fanout].drop_duplicates("biblical_lemma").reset_index(drop=True)


def definitions_for(lemma: str, pos: str, data: Mapping[str, Mapping[str, Sequence[str]]]) -> list[str]:
    if lemma not in data: return []
    if pos == "all": return dedupe(x for values in data[lemma].values() for x in values)
    return list(data[lemma].get(pos, []))


def run_setting(
    setting: Setting,
    mapping_rows: Sequence[Mapping[str, str]],
    bib: Mapping[str, Mapping[str, Sequence[str]]],
    modern: Mapping[str, Mapping[str, Sequence[str]]],
    frequency: Mapping[str, int], polysemy: Mapping[str, int],
    conc_lookup: Mapping[str, float], embeddings: Mapping[str, np.ndarray],
    model_name: str, output_dir: Path,
) -> dict[str, Any]:
    pair_rows: list[dict[str, Any]] = []
    for mapping in mapping_rows:
        old_lemma = mapping["biblical_lemma"]; new_lemma = mapping["modern_lemma"]; pos = mapping["pos"]
        old_defs = select_definitions(definitions_for(old_lemma, pos, bib), setting.definitions)
        new_defs = select_definitions(definitions_for(new_lemma, pos, modern), setting.definitions)
        if not old_defs or not new_defs: continue
        balanced = setting.mapping == "cognates"
        old_c = concreteness(old_defs, conc_lookup, balanced); new_c = concreteness(new_defs, conc_lookup, balanced)
        pair_rows.append({
            "setting": setting.number, "setting_name": setting.name, "model": model_name,
            "mapping_setting": setting.mapping, "biblical_lemma": old_lemma,
            "modern_lemma": new_lemma, "pos": pos,
            "semantic_distance": apd(old_defs, new_defs, embeddings),
            "frequency": frequency.get(old_lemma, np.nan), "biblical_polysemy": polysemy.get(old_lemma, np.nan),
            "old_concreteness": old_c["mean"], "new_concreteness": new_c["mean"],
            "biblical_definitions": " || ".join(old_defs), "modern_definitions": " || ".join(new_defs),
        })
    pair_df = pd.DataFrame(pair_rows)
    if pair_df.empty: raise ValueError(f"Setting {setting.number} produced no usable pairs")
    lemma_df = pair_df.groupby("biblical_lemma", as_index=False).agg(
        semantic_distance=("semantic_distance", "mean"), frequency=("frequency", "first"),
        biblical_polysemy=("biblical_polysemy", "first"), old_concreteness=("old_concreteness", "mean"),
        new_concreteness=("new_concreteness", "mean"), modern_lemmas=("modern_lemma", lambda x: " | ".join(sorted(set(x)))),
    )
    lemma_df["concreteness_change"] = lemma_df.new_concreteness - lemma_df.old_concreteness
    frho, fp, fn = safe_spearman(lemma_df.frequency, lemma_df.semantic_distance)
    prho, pp, pn = safe_spearman(lemma_df.biblical_polysemy, lemma_df.semantic_distance)
    stat, cp, cn = safe_wilcoxon(lemma_df.old_concreteness, lemma_df.new_concreteness)
    mean_change = lemma_df.concreteness_change.mean()
    summary = {
        "setting": setting.number, "setting_name": setting.name, "model": model_name,
        "mapping_setting": setting.mapping, "shared_pairs": len(pair_df), "shared_lemmas": len(lemma_df),
        "frequency_n": fn, "frequency_rho": frho, "frequency_p": fp, "frequency_result": symbol(fp, frho < 0),
        "polysemy_n": pn, "polysemy_rho": prho, "polysemy_p": pp, "polysemy_result": symbol(pp, prho > 0),
        "concreteness_n": cn, "old_concreteness_mean": lemma_df.old_concreteness.mean(),
        "new_concreteness_mean": lemma_df.new_concreteness.mean(), "mean_concreteness_change": mean_change,
        "concreteness_wilcoxon": stat, "concreteness_p": cp, "concreteness_result": symbol(cp, mean_change < 0),
    }
    pair_df.to_csv(output_dir / f"greek_setting_{setting.number}_pair_results.csv", index=False, encoding="utf-8-sig")
    lemma_df.to_csv(output_dir / f"greek_setting_{setting.number}_lemma_results.csv", index=False, encoding="utf-8-sig")
    print(f"\nSETTING {setting.number}: {setting.name}\nlemmas={len(lemma_df):,} frequency={frho:.4g}/{fp:.4g} polysemy={prho:.4g}/{pp:.4g} concreteness p={cp:.4g}")
    return summary


def parse_settings(values: Sequence[str]) -> list[int]:
    if not values or values == ["all"]: return sorted(SETTINGS)
    selected = sorted(set(int(x) for x in values))
    invalid = [x for x in selected if x not in SETTINGS]
    if invalid: raise ValueError(f"Settings must be 1-16; invalid: {invalid}")
    return selected


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Biblical-to-Modern Greek dictionary experiments.")
    p.add_argument("--modern-file", type=Path, required=True)
    p.add_argument("--concreteness-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("greek_results"))
    p.add_argument("--abbott-file", type=Path)
    p.add_argument("--morphgnt-zip", type=Path)
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--settings", nargs="+", default=["all"])
    p.add_argument("--english-model", default=EN_MODEL)
    p.add_argument("--multilingual-model", default=MULTI_MODEL)
    p.add_argument("--mt-model", default=MT_MODEL)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--mt-batch-size", type=int, default=32)
    p.add_argument("--max-target-fanout", type=int, default=5)
    p.add_argument("--mt-cache", type=Path, help="Optional CSV cache with biblical_lemma, source_form, mt_output.")
    return p


def main() -> None:
    args = parser().parse_args()
    selected = parse_settings(args.settings)
    for path, label in ((args.modern_file, "Modern dictionary"), (args.concreteness_file, "Concreteness file")):
        if not path.is_file(): raise FileNotFoundError(f"{label} not found: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resources = args.output_dir / "resources"; resources.mkdir(exist_ok=True)
    abbott = ensure_resource(args.abbott_file, resources / "abbott-smith.tei.xml", ABBOTT_URL, not args.no_download)
    morph_zip = ensure_resource(args.morphgnt_zip, resources / "sblgnt-master.zip", MORPHGNT_URL, not args.no_download)

    abbott_defs, polysemy, surfaces = load_abbott(abbott)
    modern = load_modern(args.modern_file)
    frequency, morph_pos = load_morphgnt(morph_zip, resources / "morphgnt")
    biblical = attach_morphgnt_pos(abbott_defs, morph_pos)
    conc = load_concreteness(args.concreteness_file)

    pd.DataFrame([{"lemma": l, "frequency": frequency.get(l, np.nan), "pos": " | ".join(sorted(morph_pos.get(l, set()))), "polysemy": polysemy.get(l)} for l in sorted(biblical)]).to_csv(args.output_dir / "greek_preparation_audit.csv", index=False, encoding="utf-8-sig")

    mt_needed = any(SETTINGS[n].mapping == "MT" for n in selected)
    mt_df: pd.DataFrame | None = None
    if mt_needed:
        if args.mt_cache and args.mt_cache.exists():
            translations = pd.read_csv(args.mt_cache)
        else:
            translations = translate_lemmas(sorted(biblical), surfaces, args.mt_model, args.mt_batch_size)
            translations.to_csv(args.output_dir / "greek_mt_translations.csv", index=False, encoding="utf-8-sig")
        required = {"biblical_lemma", "source_form", "mt_output"}
        if not required.issubset(translations.columns): raise ValueError(f"MT cache must contain {sorted(required)}")
        mt_df = conservative_mt_mapping(translations, biblical, modern, load_spacy_greek(), args.max_target_fanout)
        mt_df.to_csv(args.output_dir / "greek_mt_conservative_mapping.csv", index=False, encoding="utf-8-sig")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("Install sentence-transformers") from exc
    models: dict[str, Any] = {}
    summaries: list[dict[str, Any]] = []
    for number in selected:
        setting = SETTINGS[number]
        if setting.mapping == "cognates":
            mappings = build_cognate_mapping(setting, biblical, modern)
        else:
            assert mt_df is not None
            mappings = []
            for row in mt_df.itertuples(index=False):
                if setting.pos == "all":
                    mappings.append({"biblical_lemma": row.biblical_lemma, "modern_lemma": row.accepted_target_lemma, "pos": "all"})
                else:
                    for pos in str(row.shared_selected_pos).split(" | "):
                        if pos in SELECTED_POS:
                            mappings.append({"biblical_lemma": row.biblical_lemma, "modern_lemma": row.accepted_target_lemma, "pos": pos})
        texts: list[str] = []
        for mapping in mappings:
            texts += select_definitions(definitions_for(mapping["biblical_lemma"], mapping["pos"], biblical), setting.definitions)
            texts += select_definitions(definitions_for(mapping["modern_lemma"], mapping["pos"], modern), setting.definitions)
        model_name = args.english_model if setting.model == "english" else args.multilingual_model
        if setting.model not in models:
            models[setting.model] = SentenceTransformer(model_name)
        lookup = embedding_lookup(models[setting.model], texts, args.batch_size)
        summaries.append(run_setting(setting, mappings, biblical, modern, frequency, polysemy, conc, lookup, model_name, args.output_dir))
    pd.DataFrame(summaries).sort_values("setting").to_csv(args.output_dir / "greek_settings_summary.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
