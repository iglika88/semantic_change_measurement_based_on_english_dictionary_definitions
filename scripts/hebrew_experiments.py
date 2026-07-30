#!/usr/bin/env python3
"""
Hebrew dictionary-based semantic change experiments.

This script compares Biblical Hebrew and Modern Hebrew dictionary definitions
using Sentence-Transformer embeddings. It supports the eight experimental
settings used in the accompanying study:

    1. all definitions / all POS / English model
    2. all definitions / all POS / multilingual model
    3. all definitions / selected POS / English model
    4. all definitions / selected POS / multilingual model
    5. first definition / all POS / English model
    6. first definition / all POS / multilingual model
    7. first definition / selected POS / English model
    8. first definition / selected POS / multilingual model

For each selected setting, the script:

1. loads the Brown–Driver–Briggs and LexicalIndex XML files;
2. loads a Kaikki/Wiktionary-derived Modern Hebrew JSONL dictionary;
3. removes senses explicitly labelled as Biblical Hebrew;
4. identifies shared Hebrew lemma–POS pairs;
5. reconstructs Biblical Hebrew lemma frequencies from the OSHB corpus;
6. calculates Biblical polysemy as the number of unique Biblical definitions;
7. calculates definition-balanced concreteness scores using the Brysbaert
   concreteness ratings;
8. calculates average pairwise cosine distance between Biblical and Modern
   definitions;
9. tests the Law of Conformity, Law of Innovation, and the predicted movement
   toward abstractness;
10. saves detailed pair-level, lemma-level, audit, diagnostic, and summary CSV
    files.

Required local inputs
---------------------
The user must provide:

--modern-file
    Path to the Kaikki Hebrew JSONL dictionary.

--concreteness-file
    Path to the Brysbaert concreteness ratings Excel file.

The BDB XML, LexicalIndex XML, and morphhb corpus ZIP may either be supplied
locally or downloaded automatically from their default public repositories.

Example
-------
Run Settings 1 and 4:

    python hebrew_experiments.py \
        --modern-file /path/to/kaikki-hebrew.jsonl \
        --concreteness-file /path/to/Concreteness_ratings.xlsx \
        --output-dir ./hebrew_results \
        --settings 1 4

Run all eight settings:

    python hebrew_experiments.py \
        --modern-file /path/to/kaikki-hebrew.jsonl \
        --concreteness-file /path/to/Concreteness_ratings.xlsx \
        --output-dir ./hebrew_results \
        --settings all

Dependencies
------------
numpy
pandas
scipy
lxml
requests
openpyxl
sentence-transformers
transformers
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import requests
from lxml import etree
from scipy.stats import spearmanr, wilcoxon
if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Default remote resources and model names
# ---------------------------------------------------------------------------

DEFAULT_BDB_URL = (
    "https://raw.githubusercontent.com/"
    "openscriptures/HebrewLexicon/master/BrownDriverBriggs.xml"
)

DEFAULT_LEXICAL_INDEX_URL = (
    "https://raw.githubusercontent.com/"
    "openscriptures/HebrewLexicon/master/LexicalIndex.xml"
)

DEFAULT_MORPHHB_URL = (
    "https://codeload.github.com/"
    "openscriptures/morphhb/zip/refs/heads/master"
)

DEFAULT_ENGLISH_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MULTILINGUAL_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

SELECTED_POS = frozenset({"noun", "verb", "adjective", "adverb"})

XML_NAMESPACE = {
    "mhb": "http://openscriptures.github.com/morphhb/namespace"
}


# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentSetting:
    number: int
    definition_selection: str
    pos_selection: str
    model_key: str

    @property
    def name(self) -> str:
        model_label = (
            "English model"
            if self.model_key == "english"
            else "multilingual model"
        )
        definition_label = (
            "All definitions"
            if self.definition_selection == "all"
            else "First definition"
        )
        pos_label = (
            "all POS"
            if self.pos_selection == "all"
            else "selected POS"
        )
        return (
            f"{definition_label} / {pos_label} / "
            f"cognates / {model_label}"
        )


SETTINGS: dict[int, ExperimentSetting] = {
    1: ExperimentSetting(1, "all", "all", "english"),
    2: ExperimentSetting(2, "all", "all", "multilingual"),
    3: ExperimentSetting(3, "all", "selected", "english"),
    4: ExperimentSetting(4, "all", "selected", "multilingual"),
    5: ExperimentSetting(5, "first", "all", "english"),
    6: ExperimentSetting(6, "first", "all", "multilingual"),
    7: ExperimentSetting(7, "first", "selected", "english"),
    8: ExperimentSetting(8, "first", "selected", "multilingual"),
}


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

NIQQUD_RE = re.compile(r"[\u0591-\u05C7]")
PARENS_RE = re.compile(r"\([^)]*\)")
SPACE_RE = re.compile(r"\s+")
ENGLISH_TOKEN_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")


def strip_niqqud(text: object) -> str:
    return NIQQUD_RE.sub("", str(text or ""))


def normalise_lemma(text: object) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = strip_niqqud(value)
    return SPACE_RE.sub(" ", value).strip()


def clean_definition(text: object, remove_parentheses: bool = False) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    if remove_parentheses:
        value = PARENS_RE.sub("", value)
    return SPACE_RE.sub(" ", value).strip()


def dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)

    return result


def flatten_strings(value: object) -> list[str]:
    """Flatten nested lists and dictionaries into searchable strings."""
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, Mapping):
        result: list[str] = []
        for key, subvalue in value.items():
            result.append(str(key))
            result.extend(flatten_strings(subvalue))
        return result

    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            result.extend(flatten_strings(item))
        return result

    return [str(value)]


def localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


# ---------------------------------------------------------------------------
# POS mappings
# ---------------------------------------------------------------------------

MODERN_POS_MAP = {
    "noun": "noun",
    "name": "noun",
    "proper-noun": "noun",
    "proper noun": "noun",
    "num": "noun",
    "number": "noun",
    "root": "noun",
    "verb": "verb",
    "adj": "adjective",
    "adjective": "adjective",
    "adv": "adverb",
    "adverb": "adverb",
}

BIBLICAL_NON_POS_TAGS = {
    "m", "f", "mf", "pl", "sg", "du",
    "mpl", "fpl", "ms", "fs", "mp", "fp",
    "1s", "1pl", "2ms", "2fs", "2mpl",
    "2fpl", "3s", "3mpl", "3fpl",
    "1", "2", "3",
}


def map_modern_pos(raw_pos: object) -> str:
    return MODERN_POS_MAP.get(
        str(raw_pos or "").strip().lower(),
        "other",
    )


def canonicalise_biblical_pos_tag(tag: object) -> str:
    value = str(tag or "").strip().lower()
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"\.+", ".", value)
    return value.strip(".")


def map_biblical_pos(raw_tag: object) -> str:
    tag = canonicalise_biblical_pos_tag(raw_tag)

    if not tag or tag in BIBLICAL_NON_POS_TAGS:
        return "other"

    if (
        tag in {"n", "np", "subst", "noun", "nom.gent", "n.decl"}
        or tag.startswith("n.")
        or tag.startswith("npr")
        or tag.startswith("n.pr")
    ):
        return "noun"

    if tag == "v" or tag.startswith("v.") or tag.startswith("vb"):
        return "verb"

    if (
        tag in {"a", "ag", "ao"}
        or tag.startswith("a.")
        or tag.startswith("adj")
    ):
        return "adjective"

    if tag == "adv" or tag.startswith("adv"):
        return "adverb"

    return "other"


# ---------------------------------------------------------------------------
# Modern Hebrew Biblical-sense filtering
# ---------------------------------------------------------------------------

EXPLICIT_BIBLICAL_TAGS = {
    "biblical",
    "biblical-hebrew",
    "biblical hebrew",
}

EXPLICIT_BIBLICAL_CATEGORIES = {
    "biblical hebrew",
    "biblical hebrew lemmas",
    "biblical characters",
    "biblical figures",
}

INITIAL_QUALIFIER_RE = re.compile(r"^\s*\(([^)]*)\)", re.IGNORECASE)


def normalise_label(label: object) -> str:
    value = str(label or "").strip().lower().replace("_", "-")
    return re.sub(r"\s+", " ", value)


def qualifier_is_biblical(raw_gloss: object) -> tuple[bool, str | None]:
    """
    Detect a Biblical usage qualifier only at the start of a raw gloss.

    Plain definition content mentioning the Bible or Tanakh is not treated as
    sufficient evidence that the dictionary sense itself is Biblical.
    """
    match = INITIAL_QUALIFIER_RE.match(str(raw_gloss or "").strip())
    if not match:
        return False, None

    qualifier = normalise_label(match.group(1))

    if "post-biblical" in qualifier or "post biblical" in qualifier:
        return False, None

    if (
        re.search(r"\bbiblical\b", qualifier)
        or "biblical-hebrew" in qualifier
        or "biblical hebrew" in qualifier
    ):
        return True, qualifier

    return False, None


def biblical_sense_evidence(sense: Mapping[str, object]) -> list[str]:
    """
    Return structured evidence that explicitly labels a Modern Hebrew sense
    as Biblical Hebrew.
    """
    evidence: list[str] = []

    tags = (
        flatten_strings(sense.get("tags"))
        + flatten_strings(sense.get("raw_tags"))
        + flatten_strings(sense.get("labels"))
    )

    for tag in tags:
        normalised = normalise_label(tag)
        if "post-biblical" in normalised or "post biblical" in normalised:
            continue
        if normalised in EXPLICIT_BIBLICAL_TAGS:
            evidence.append(f"tag: {tag}")

    for topic in flatten_strings(sense.get("topics")):
        if normalise_label(topic) in EXPLICIT_BIBLICAL_TAGS:
            evidence.append(f"topic: {topic}")

    for category in flatten_strings(sense.get("categories")):
        if normalise_label(category) in EXPLICIT_BIBLICAL_CATEGORIES:
            evidence.append(f"category: {category}")

    qualifiers = (
        flatten_strings(sense.get("qualifier"))
        + flatten_strings(sense.get("qualifiers"))
    )

    for qualifier in qualifiers:
        normalised = normalise_label(qualifier)
        if "post-biblical" in normalised or "post biblical" in normalised:
            continue
        if (
            normalised in EXPLICIT_BIBLICAL_TAGS
            or re.search(r"\bbiblical\b", normalised)
        ):
            evidence.append(f"qualifier: {qualifier}")

    for raw_gloss in flatten_strings(sense.get("raw_glosses")):
        matched, qualifier = qualifier_is_biblical(raw_gloss)
        if matched:
            evidence.append(f"initial qualifier: {qualifier}")

    return dedupe(evidence)


# ---------------------------------------------------------------------------
# Resource handling
# ---------------------------------------------------------------------------

def download_if_needed(
    url: str,
    output_path: Path,
    timeout: int = 180,
) -> None:
    """Download a resource unless a non-empty local copy already exists."""
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Using existing resource: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {output_path.name}...")

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    output_path.write_bytes(response.content)

    print(
        f"Saved {output_path} "
        f"({output_path.stat().st_size:,} bytes)"
    )


def ensure_resource(
    supplied_path: Path | None,
    default_path: Path,
    url: str,
    allow_downloads: bool,
) -> Path:
    path = supplied_path or default_path

    if path.exists() and path.stat().st_size > 0:
        return path

    if not allow_downloads:
        raise FileNotFoundError(
            f"Required resource not found: {path}. "
            "Supply the corresponding command-line path or permit downloads."
        )

    download_if_needed(url, path)
    return path


# ---------------------------------------------------------------------------
# Dictionary loading
# ---------------------------------------------------------------------------

BibData = dict[str, dict[str, list[str]]]


def load_biblical_dictionary(
    bdb_path: Path,
    lexical_index_path: Path,
) -> tuple[BibData, etree._ElementTree]:
    print("\nLoading Biblical Hebrew dictionary...")

    bdb_xml = etree.parse(str(bdb_path))
    lexical_index_xml = etree.parse(str(lexical_index_path))

    data: defaultdict[str, defaultdict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for lexical_entry in lexical_index_xml.xpath(
        "//mhb:entry",
        namespaces=XML_NAMESPACE,
    ):
        word_nodes = lexical_entry.xpath(
            ".//mhb:w",
            namespaces=XML_NAMESPACE,
        )

        if not word_nodes:
            continue

        lemma = normalise_lemma(word_nodes[0].text or "")
        if not lemma:
            continue

        bdb_nodes = []

        for xref in lexical_entry.xpath(
            ".//mhb:xref",
            namespaces=XML_NAMESPACE,
        ):
            bdb_id = xref.attrib.get("bdb")
            if not bdb_id:
                continue

            bdb_nodes.extend(
                bdb_xml.xpath(
                    f"//mhb:entry[@id='{bdb_id}']",
                    namespaces=XML_NAMESPACE,
                )
            )

        for bdb_entry in bdb_nodes:
            raw_pos_tags = [
                (node.text or "").strip()
                for node in bdb_entry.xpath(
                    ".//mhb:pos",
                    namespaces=XML_NAMESPACE,
                )
                if (node.text or "").strip()
            ]

            mapped_pos = [
                pos
                for pos in dedupe(map_biblical_pos(tag) for tag in raw_pos_tags)
                if pos != "other"
            ] or ["other"]

            definitions = dedupe(
                clean_definition(node.text or "", remove_parentheses=True)
                for node in bdb_entry.xpath(
                    ".//mhb:def",
                    namespaces=XML_NAMESPACE,
                )
            )

            for pos in mapped_pos:
                data[lemma][pos].extend(definitions)

    final_data: BibData = {}
    for lemma, pos_dict in data.items():
        final_data[lemma] = {
            pos: dedupe(definitions)
            for pos, definitions in pos_dict.items()
            if definitions
        }

    definition_count = sum(
        len(definitions)
        for pos_dict in final_data.values()
        for definitions in pos_dict.values()
    )

    print(f"Biblical lemmas loaded: {len(final_data):,}")
    print(f"Biblical definitions loaded: {definition_count:,}")

    return final_data, lexical_index_xml


def load_modern_dictionary(
    modern_file: Path,
) -> tuple[BibData, pd.DataFrame, pd.DataFrame]:
    print("\nLoading Modern Hebrew dictionary...")

    data: defaultdict[str, defaultdict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    removed_records: list[dict[str, object]] = []
    retained_records: list[dict[str, object]] = []

    entries_read = 0
    senses_seen = 0
    senses_removed = 0
    senses_retained = 0

    with modern_file.open("r", encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            if not line.strip():
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {modern_file}"
                ) from exc

            if obj.get("lang") != "Hebrew":
                continue

            entries_read += 1
            lemma = normalise_lemma(obj.get("word", ""))
            if not lemma:
                continue

            pos = map_modern_pos(obj.get("pos", ""))

            for sense_index, sense in enumerate(obj.get("senses", [])):
                senses_seen += 1
                evidence = biblical_sense_evidence(sense)

                glosses = dedupe(
                    clean_definition(gloss)
                    for gloss in (sense.get("glosses") or [])
                )

                if evidence:
                    senses_removed += 1
                    removed_records.append(
                        {
                            "lemma": lemma,
                            "pos": pos,
                            "sense_index": sense_index,
                            "evidence": " | ".join(evidence),
                            "glosses": " || ".join(glosses),
                            "raw_glosses": " || ".join(
                                flatten_strings(sense.get("raw_glosses"))
                            ),
                            "tags": " | ".join(
                                flatten_strings(sense.get("tags"))
                            ),
                            "categories": " | ".join(
                                flatten_strings(sense.get("categories"))
                            ),
                            "topics": " | ".join(
                                flatten_strings(sense.get("topics"))
                            ),
                        }
                    )
                    continue

                if not glosses:
                    continue

                senses_retained += 1
                data[lemma][pos].extend(glosses)
                retained_records.append(
                    {
                        "lemma": lemma,
                        "pos": pos,
                        "sense_index": sense_index,
                        "glosses": " || ".join(glosses),
                    }
                )

    final_data: BibData = {}
    for lemma, pos_dict in data.items():
        final_data[lemma] = {
            pos: dedupe(definitions)
            for pos, definitions in pos_dict.items()
            if definitions
        }

    definition_count = sum(
        len(definitions)
        for pos_dict in final_data.values()
        for definitions in pos_dict.values()
    )

    print(f"Modern entries read: {entries_read:,}")
    print(f"Modern senses inspected: {senses_seen:,}")
    print(f"Explicitly Biblical senses removed: {senses_removed:,}")
    print(f"Modern senses retained: {senses_retained:,}")
    print(f"Modern lemmas retained: {len(final_data):,}")
    print(f"Modern definitions retained: {definition_count:,}")

    return (
        final_data,
        pd.DataFrame(removed_records),
        pd.DataFrame(retained_records),
    )


def build_shared_pairs(
    biblical_data: BibData,
    modern_data: BibData,
) -> list[tuple[str, str]]:
    shared_pairs: list[tuple[str, str]] = []

    for lemma, biblical_pos_dict in biblical_data.items():
        if lemma not in modern_data:
            continue

        for pos in sorted(set(biblical_pos_dict) & set(modern_data[lemma])):
            if biblical_pos_dict[pos] and modern_data[lemma][pos]:
                shared_pairs.append((lemma, pos))

    return sorted(set(shared_pairs))


# ---------------------------------------------------------------------------
# OSHB frequency reconstruction
# ---------------------------------------------------------------------------

VALID_NUMERIC_CODE_RE = re.compile(r"^\d+[a-z]?$", re.IGNORECASE)

OSHB_GRAMMATICAL_LEMMA_MAP = {
    "b": "ב",
    "c": "ו",
    "d": "ה",
    "k": "כ",
    "l": "ל",
    "m": "מן",
}


def canonical_numeric_code(value: object) -> str:
    """
    Convert OSHB and LexicalIndex code variants to a common form.

    Examples:
        H1121     -> 1121
        1121 a    -> 1121a
        1177+     -> 1177
        c/m/6529  -> 6529
    """
    if value is None:
        return ""

    code = str(value).strip().lower()
    code = code.split("/")[-1]
    code = re.sub(r"^h", "", code)
    code = re.sub(r"\s+", "", code)
    code = code.rstrip("+")

    return code if VALID_NUMERIC_CODE_RE.fullmatch(code) else ""


def canonical_oshb_code(value: object) -> str:
    raw_value = str(value or "").strip().lower()

    if re.fullmatch(r"[a-z]", raw_value):
        return raw_value

    return canonical_numeric_code(raw_value)


def count_oshb_codes(morphhb_zip: Path) -> Counter[str]:
    print("\nCounting OSHB lemma occurrences...")

    code_frequency: Counter[str] = Counter()

    with zipfile.ZipFile(morphhb_zip, "r") as archive:
        xml_names = [
            name
            for name in archive.namelist()
            if "/wlc/" in name and name.lower().endswith(".xml")
        ]

        if not xml_names:
            raise ValueError(
                f"No /wlc/*.xml files were found inside {morphhb_zip}"
            )

        print(f"OSHB corpus XML files: {len(xml_names):,}")

        for name in xml_names:
            with archive.open(name) as file_handle:
                context = etree.iterparse(
                    file_handle,
                    events=("end",),
                    recover=True,
                )

                for _, element in context:
                    if localname(element.tag) == "w":
                        code = canonical_oshb_code(element.get("lemma"))
                        if code:
                            code_frequency[code] += 1

                    element.clear()

    print(f"Distinct OSHB lemma identifiers: {len(code_frequency):,}")
    return code_frequency


def build_lexical_code_maps(
    lexical_index_xml: etree._ElementTree,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    code_to_headwords: defaultdict[str, set[str]] = defaultdict(set)

    for lexical_entry in lexical_index_xml.xpath(
        "//mhb:entry",
        namespaces=XML_NAMESPACE,
    ):
        word_nodes = lexical_entry.xpath(
            "./mhb:w | .//mhb:w",
            namespaces=XML_NAMESPACE,
        )

        if not word_nodes:
            continue

        primary_headword = normalise_lemma(word_nodes[0].text or "")
        if not primary_headword:
            continue

        for element in lexical_entry.iter():
            for attribute_name, attribute_value in element.attrib.items():
                name = attribute_name.lower()

                if "strong" not in name and "aug" not in name:
                    continue

                code = canonical_numeric_code(attribute_value)
                if code:
                    code_to_headwords[code].add(primary_headword)

    unique_map: dict[str, str] = {}
    ambiguous_map: dict[str, list[str]] = {}

    for code, headwords in code_to_headwords.items():
        if len(headwords) == 1:
            unique_map[code] = next(iter(headwords))
        else:
            ambiguous_map[code] = sorted(headwords)

    print(f"Valid numeric identifiers: {len(code_to_headwords):,}")
    print(f"Uniquely mapped identifiers: {len(unique_map):,}")
    print(f"Ambiguous identifiers: {len(ambiguous_map):,}")

    return unique_map, ambiguous_map


def reconstruct_frequencies(
    code_frequency: Counter[str],
    unique_code_map: Mapping[str, str],
    ambiguous_code_map: Mapping[str, Sequence[str]],
) -> tuple[Counter[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frequencies: Counter[str] = Counter()
    unmapped_rows: list[dict[str, object]] = []
    ambiguous_rows: list[dict[str, object]] = []
    grammatical_rows: list[dict[str, object]] = []

    for raw_code, count in code_frequency.items():
        code = canonical_oshb_code(raw_code)

        if code in OSHB_GRAMMATICAL_LEMMA_MAP:
            lemma = OSHB_GRAMMATICAL_LEMMA_MAP[code]
            frequencies[lemma] += count
            grammatical_rows.append(
                {"code": code, "lemma": lemma, "count": count}
            )
            continue

        if not code:
            unmapped_rows.append(
                {
                    "raw_code": raw_code,
                    "canonical_code": "",
                    "count": count,
                    "reason": "unsupported code format",
                }
            )
            continue

        if code in unique_code_map:
            frequencies[unique_code_map[code]] += count
        elif code in ambiguous_code_map:
            ambiguous_rows.append(
                {
                    "code": code,
                    "count": count,
                    "possible_headwords": " | ".join(
                        ambiguous_code_map[code]
                    ),
                }
            )
        else:
            unmapped_rows.append(
                {
                    "raw_code": raw_code,
                    "canonical_code": code,
                    "count": count,
                    "reason": "not found in LexicalIndex",
                }
            )

    print(f"Headwords with reconstructed frequency: {len(frequencies):,}")
    print(f"Unmapped OSHB identifiers: {len(unmapped_rows):,}")
    print(f"Ambiguous corpus identifiers excluded: {len(ambiguous_rows):,}")

    return (
        frequencies,
        pd.DataFrame(unmapped_rows),
        pd.DataFrame(ambiguous_rows),
        pd.DataFrame(grammatical_rows),
    )


# ---------------------------------------------------------------------------
# Polysemy and concreteness
# ---------------------------------------------------------------------------

def build_biblical_polysemy(
    biblical_data: BibData,
) -> dict[str, int]:
    return {
        lemma: len(
            dedupe(
                definition
                for definitions in pos_dict.values()
                for definition in definitions
            )
        )
        for lemma, pos_dict in biblical_data.items()
    }


def load_concreteness_lookup(concreteness_file: Path) -> dict[str, float]:
    print("\nLoading Brysbaert concreteness ratings...")

    dataframe = pd.read_excel(concreteness_file)
    required_columns = {"Word", "Conc.M"}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "The concreteness file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe["Word_normalised"] = (
        dataframe["Word"].astype(str).str.strip().str.lower()
    )

    dataframe["Conc.M"] = pd.to_numeric(
        dataframe["Conc.M"]
        .astype(str)
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )

    lookup = (
        dataframe
        .dropna(subset=["Word_normalised", "Conc.M"])
        .drop_duplicates(subset=["Word_normalised"], keep="first")
        .set_index("Word_normalised")["Conc.M"]
        .astype(float)
        .to_dict()
    )

    print(f"Concreteness ratings loaded: {len(lookup):,}")
    return lookup


def definition_tokens(definition: str) -> list[str]:
    return [
        token.lower()
        for token in ENGLISH_TOKEN_RE.findall(definition or "")
    ]


def definitions_concreteness(
    definitions: Sequence[str],
    concreteness_lookup: Mapping[str, float],
) -> dict[str, float | int]:
    """
    Calculate entry-level concreteness with equal definition weighting.

    A mean is calculated separately for every definition with at least one
    recognised token. The entry score is the arithmetic mean of those
    definition-level means. Longer definitions therefore do not receive more
    weight merely because they contain more tokens.
    """
    definition_scores: list[float] = []
    total_tokens = 0
    matched_tokens = 0

    for definition in definitions:
        tokens = definition_tokens(definition)
        total_tokens += len(tokens)

        token_scores = [
            float(concreteness_lookup[token])
            for token in tokens
            if token in concreteness_lookup
        ]

        matched_tokens += len(token_scores)

        if token_scores:
            definition_scores.append(float(np.mean(token_scores)))

    definitions_with_scores = len(definition_scores)

    return {
        "mean": (
            float(np.mean(definition_scores))
            if definition_scores
            else np.nan
        ),
        "matched_tokens": matched_tokens,
        "all_tokens": total_tokens,
        "coverage": matched_tokens / max(1, total_tokens),
        "definitions_total": len(definitions),
        "definitions_with_scores": definitions_with_scores,
        "definition_coverage": (
            definitions_with_scores / max(1, len(definitions))
        ),
    }


# ---------------------------------------------------------------------------
# Embeddings and statistics
# ---------------------------------------------------------------------------

def select_definitions(
    definitions: Sequence[str],
    selection: str,
) -> list[str]:
    if selection == "all":
        return list(definitions)

    if selection == "first":
        return [definitions[0]] if definitions else []

    raise ValueError(f"Unknown definition selection: {selection}")


def collect_unique_definitions(
    setting_pairs: Sequence[tuple[str, str]],
    biblical_data: BibData,
    modern_data: BibData,
    definition_selection: str,
) -> list[str]:
    definitions: set[str] = set()

    for lemma, pos in setting_pairs:
        definitions.update(
            select_definitions(
                biblical_data[lemma][pos],
                definition_selection,
            )
        )
        definitions.update(
            select_definitions(
                modern_data[lemma][pos],
                definition_selection,
            )
        )

    return sorted(definitions)


def build_embedding_lookup(
    model: Any,
    definitions: Sequence[str],
    batch_size: int,
) -> dict[str, np.ndarray]:
    if not definitions:
        raise ValueError("No definitions are available for embedding.")

    embeddings = model.encode(
        list(definitions),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return {
        definition: embedding
        for definition, embedding in zip(definitions, embeddings)
    }


def average_pairwise_cosine_distance(
    old_definitions: Sequence[str],
    new_definitions: Sequence[str],
    embedding_lookup: Mapping[str, np.ndarray],
) -> float:
    """
    Calculate average pairwise cosine distance (APD).

    Embeddings are L2-normalised, so cosine distance equals one minus the dot
    product. In first-definition settings, APD reduces to the cosine distance
    between the first Biblical and first Modern definition.
    """
    old_embeddings = np.vstack(
        [embedding_lookup[definition] for definition in old_definitions]
    )
    new_embeddings = np.vstack(
        [embedding_lookup[definition] for definition in new_definitions]
    )

    return float(
        np.mean(1.0 - (old_embeddings @ new_embeddings.T))
    )


def safe_spearman(
    x: Sequence[float],
    y: Sequence[float],
) -> dict[str, float | int]:
    valid = pd.DataFrame({"x": x, "y": y}).dropna()

    if (
        len(valid) < 3
        or valid["x"].nunique() < 2
        or valid["y"].nunique() < 2
    ):
        return {
            "rho": np.nan,
            "p_value": np.nan,
            "n": len(valid),
        }

    result = spearmanr(valid["x"], valid["y"])

    return {
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
        "n": len(valid),
    }


def safe_one_sided_wilcoxon(
    old_values: Sequence[float],
    new_values: Sequence[float],
) -> dict[str, float | int]:
    valid = pd.DataFrame(
        {"old": old_values, "new": new_values}
    ).dropna()

    if len(valid) < 2:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "n": len(valid),
        }

    differences = valid["new"] - valid["old"]

    if np.allclose(differences, 0):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "n": len(valid),
        }

    result = wilcoxon(
        valid["new"],
        valid["old"],
        alternative="less",
        zero_method="wilcox",
    )

    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n": len(valid),
    }


def significance_symbol(
    p_value: float,
    expected_direction: bool,
) -> str:
    if (
        pd.isna(p_value)
        or not expected_direction
        or p_value >= 0.05
    ):
        return "✗"

    if p_value < 0.001:
        return "✓**"

    if p_value < 0.01:
        return "✓*"

    return "✓"


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------

def pairs_for_setting(
    shared_pairs: Sequence[tuple[str, str]],
    pos_selection: str,
) -> list[tuple[str, str]]:
    if pos_selection == "all":
        return list(shared_pairs)

    if pos_selection == "selected":
        return [
            pair
            for pair in shared_pairs
            if pair[1] in SELECTED_POS
        ]

    raise ValueError(f"Unknown POS selection: {pos_selection}")


def run_setting(
    setting: ExperimentSetting,
    setting_pairs: Sequence[tuple[str, str]],
    biblical_data: BibData,
    modern_data: BibData,
    frequency: Mapping[str, int],
    biblical_polysemy: Mapping[str, int],
    concreteness_lookup: Mapping[str, float],
    embedding_lookup: Mapping[str, np.ndarray],
    model_name: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    print("\n" + "=" * 78)
    print(f"SETTING {setting.number}: {setting.name}")
    print("=" * 78)

    pair_rows: list[dict[str, object]] = []

    for lemma, pos in setting_pairs:
        old_definitions = select_definitions(
            biblical_data[lemma][pos],
            setting.definition_selection,
        )
        new_definitions = select_definitions(
            modern_data[lemma][pos],
            setting.definition_selection,
        )

        if not old_definitions or not new_definitions:
            continue

        old_concreteness = definitions_concreteness(
            old_definitions,
            concreteness_lookup,
        )
        new_concreteness = definitions_concreteness(
            new_definitions,
            concreteness_lookup,
        )

        old_mean = float(old_concreteness["mean"])
        new_mean = float(new_concreteness["mean"])

        pair_rows.append(
            {
                "setting": setting.number,
                "setting_name": setting.name,
                "model": model_name,
                "definition_setting": setting.definition_selection,
                "pos_setting": setting.pos_selection,
                "mapping_setting": "cognates",
                "lemma": lemma,
                "pos": pos,
                "n_biblical_definitions": len(old_definitions),
                "n_modern_definitions": len(new_definitions),
                "semantic_distance": average_pairwise_cosine_distance(
                    old_definitions,
                    new_definitions,
                    embedding_lookup,
                ),
                "frequency": frequency.get(lemma, np.nan),
                "biblical_polysemy": biblical_polysemy.get(
                    lemma,
                    np.nan,
                ),
                "old_concreteness": old_mean,
                "new_concreteness": new_mean,
                "concreteness_change": (
                    new_mean - old_mean
                    if not pd.isna(old_mean) and not pd.isna(new_mean)
                    else np.nan
                ),
                "old_concreteness_matched_tokens": old_concreteness[
                    "matched_tokens"
                ],
                "new_concreteness_matched_tokens": new_concreteness[
                    "matched_tokens"
                ],
                "old_concreteness_token_coverage": old_concreteness[
                    "coverage"
                ],
                "new_concreteness_token_coverage": new_concreteness[
                    "coverage"
                ],
                "old_concreteness_definition_coverage": old_concreteness[
                    "definition_coverage"
                ],
                "new_concreteness_definition_coverage": new_concreteness[
                    "definition_coverage"
                ],
                "biblical_definitions": " || ".join(old_definitions),
                "modern_definitions": " || ".join(new_definitions),
            }
        )

    pair_df = pd.DataFrame(pair_rows)

    if pair_df.empty:
        raise ValueError(
            f"Setting {setting.number} produced no usable lemma–POS pairs."
        )

    lemma_df = (
        pair_df
        .groupby("lemma", as_index=False)
        .agg(
            semantic_distance=("semantic_distance", "mean"),
            frequency=("frequency", "first"),
            biblical_polysemy=("biblical_polysemy", "first"),
            old_concreteness=("old_concreteness", "mean"),
            new_concreteness=("new_concreteness", "mean"),
            shared_pos_count=("pos", "nunique"),
            shared_pos=(
                "pos",
                lambda values: " | ".join(sorted(set(values))),
            ),
        )
    )

    lemma_df["concreteness_change"] = (
        lemma_df["new_concreteness"]
        - lemma_df["old_concreteness"]
    )

    conformity = safe_spearman(
        lemma_df["frequency"],
        lemma_df["semantic_distance"],
    )
    conformity_symbol = significance_symbol(
        float(conformity["p_value"]),
        not pd.isna(conformity["rho"]) and float(conformity["rho"]) < 0,
    )

    innovation = safe_spearman(
        lemma_df["biblical_polysemy"],
        lemma_df["semantic_distance"],
    )
    innovation_symbol = significance_symbol(
        float(innovation["p_value"]),
        not pd.isna(innovation["rho"]) and float(innovation["rho"]) > 0,
    )

    concreteness_valid = lemma_df.dropna(
        subset=["old_concreteness", "new_concreteness"]
    ).copy()

    concreteness_test = safe_one_sided_wilcoxon(
        concreteness_valid["old_concreteness"],
        concreteness_valid["new_concreteness"],
    )

    mean_old = concreteness_valid["old_concreteness"].mean()
    mean_new = concreteness_valid["new_concreteness"].mean()
    mean_change = concreteness_valid["concreteness_change"].mean()
    median_change = concreteness_valid["concreteness_change"].median()
    percent_more_abstract = (
        100
        * (concreteness_valid["concreteness_change"] < 0).mean()
        if len(concreteness_valid)
        else np.nan
    )

    concreteness_symbol = significance_symbol(
        float(concreteness_test["p_value"]),
        not pd.isna(mean_change) and mean_change < 0,
    )

    print(f"Model: {model_name}")
    print(f"Shared lemma + POS pairs: {len(pair_df):,}")
    print(f"Unique shared lemmas: {len(lemma_df):,}")

    print("\nLaw of Conformity")
    print(f"  n: {conformity['n']:,}")
    print(f"  Spearman rho: {conformity['rho']}")
    print(f"  p-value: {conformity['p_value']}")
    print(f"  result: {conformity_symbol}")

    print("\nLaw of Innovation")
    print(f"  n: {innovation['n']:,}")
    print(f"  Spearman rho: {innovation['rho']}")
    print(f"  p-value: {innovation['p_value']}")
    print(f"  result: {innovation_symbol}")

    print("\nConcreteness / abstractness")
    print(f"  n: {concreteness_test['n']:,}")
    print(f"  mean Biblical concreteness: {mean_old}")
    print(f"  mean Modern concreteness: {mean_new}")
    print(f"  mean change (Modern - Biblical): {mean_change}")
    print(f"  median change: {median_change}")
    print(f"  percent moving toward abstractness: {percent_more_abstract}")
    print(f"  Wilcoxon statistic: {concreteness_test['statistic']}")
    print(f"  one-sided p-value: {concreteness_test['p_value']}")
    print(f"  result: {concreteness_symbol}")

    pair_path = output_dir / (
        f"hebrew_setting_{setting.number}_pair_results.csv"
    )
    lemma_path = output_dir / (
        f"hebrew_setting_{setting.number}_lemma_results.csv"
    )

    pair_df.to_csv(pair_path, index=False, encoding="utf-8-sig")
    lemma_df.to_csv(lemma_path, index=False, encoding="utf-8-sig")

    summary = {
        "setting": setting.number,
        "setting_name": setting.name,
        "model": model_name,
        "definition_setting": setting.definition_selection,
        "pos_setting": setting.pos_selection,
        "mapping_setting": "cognates",
        "included_pos": (
            "noun | verb | adjective | adverb | other"
            if setting.pos_selection == "all"
            else "noun | verb | adjective | adverb"
        ),
        "shared_lemma_pos_pairs": len(pair_df),
        "shared_lemmas": len(lemma_df),
        "frequency_n": conformity["n"],
        "frequency_rho": conformity["rho"],
        "frequency_p": conformity["p_value"],
        "frequency_result": conformity_symbol,
        "polysemy_n": innovation["n"],
        "polysemy_rho": innovation["rho"],
        "polysemy_p": innovation["p_value"],
        "polysemy_result": innovation_symbol,
        "concreteness_n": concreteness_test["n"],
        "old_concreteness_mean": mean_old,
        "new_concreteness_mean": mean_new,
        "mean_concreteness_change": mean_change,
        "median_concreteness_change": median_change,
        "percent_more_abstract": percent_more_abstract,
        "concreteness_wilcoxon_statistic": concreteness_test[
            "statistic"
        ],
        "concreteness_p": concreteness_test["p_value"],
        "concreteness_result": concreteness_symbol,
    }

    return pair_df, lemma_df, summary


# ---------------------------------------------------------------------------
# Diagnostics and output
# ---------------------------------------------------------------------------

def save_preparation_outputs(
    output_dir: Path,
    removed_df: pd.DataFrame,
    retained_df: pd.DataFrame,
    shared_pairs: Sequence[tuple[str, str]],
    unmapped_frequency_df: pd.DataFrame,
    ambiguous_frequency_df: pd.DataFrame,
    grammatical_frequency_df: pd.DataFrame,
    biblical_data: BibData,
    modern_data: BibData,
    concreteness_lookup: Mapping[str, float],
) -> None:
    removed_df.to_csv(
        output_dir / "hebrew_removed_biblical_senses.csv",
        index=False,
        encoding="utf-8-sig",
    )
    retained_df.to_csv(
        output_dir / "hebrew_retained_modern_senses.csv",
        index=False,
        encoding="utf-8-sig",
    )

    shared_pair_df = pd.DataFrame(
        shared_pairs,
        columns=["lemma", "pos"],
    )
    shared_pair_df.to_csv(
        output_dir / "hebrew_shared_lemma_pos_pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    unmapped_frequency_df.to_csv(
        output_dir / "hebrew_frequency_unmapped_codes.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ambiguous_frequency_df.to_csv(
        output_dir / "hebrew_frequency_ambiguous_codes.csv",
        index=False,
        encoding="utf-8-sig",
    )
    grammatical_frequency_df.to_csv(
        output_dir / "hebrew_frequency_grammatical_codes.csv",
        index=False,
        encoding="utf-8-sig",
    )

    diagnostic_rows: list[dict[str, object]] = []

    for lemma, pos in shared_pairs:
        old_result = definitions_concreteness(
            biblical_data[lemma][pos],
            concreteness_lookup,
        )
        new_result = definitions_concreteness(
            modern_data[lemma][pos],
            concreteness_lookup,
        )

        diagnostic_rows.append(
            {
                "lemma": lemma,
                "pos": pos,
                "old_concreteness": old_result["mean"],
                "new_concreteness": new_result["mean"],
                "old_token_coverage": old_result["coverage"],
                "new_token_coverage": new_result["coverage"],
                "old_definition_coverage": old_result[
                    "definition_coverage"
                ],
                "new_definition_coverage": new_result[
                    "definition_coverage"
                ],
                "old_definitions_total": old_result[
                    "definitions_total"
                ],
                "new_definitions_total": new_result[
                    "definitions_total"
                ],
                "old_definitions_with_scores": old_result[
                    "definitions_with_scores"
                ],
                "new_definitions_with_scores": new_result[
                    "definitions_with_scores"
                ],
            }
        )

    pd.DataFrame(diagnostic_rows).to_csv(
        output_dir / "hebrew_concreteness_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def parse_settings(values: Sequence[str]) -> list[int]:
    if not values or values == ["all"]:
        return sorted(SETTINGS)

    parsed: list[int] = []

    for value in values:
        try:
            setting_number = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid setting {value!r}; use 1-8 or 'all'."
            ) from exc

        if setting_number not in SETTINGS:
            raise argparse.ArgumentTypeError(
                f"Invalid setting {setting_number}; use 1-8."
            )

        parsed.append(setting_number)

    return sorted(set(parsed))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run dictionary-based Biblical-to-Modern Hebrew "
            "semantic change experiments."
        )
    )

    parser.add_argument(
        "--modern-file",
        type=Path,
        required=True,
        help="Path to the Kaikki Hebrew JSONL dictionary.",
    )
    parser.add_argument(
        "--concreteness-file",
        type=Path,
        required=True,
        help="Path to the Brysbaert concreteness ratings Excel file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hebrew_results"),
        help="Directory for downloaded resources and result CSV files.",
    )

    parser.add_argument(
        "--bdb-file",
        type=Path,
        help=(
            "Optional local BrownDriverBriggs.xml path. "
            "Downloaded into --output-dir/resources when omitted."
        ),
    )
    parser.add_argument(
        "--lexical-index-file",
        type=Path,
        help=(
            "Optional local LexicalIndex.xml path. "
            "Downloaded into --output-dir/resources when omitted."
        ),
    )
    parser.add_argument(
        "--morphhb-zip",
        type=Path,
        help=(
            "Optional local morphhb ZIP path. "
            "Downloaded into --output-dir/resources when omitted."
        ),
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download missing BDB, LexicalIndex, or morphhb resources.",
    )

    parser.add_argument(
        "--settings",
        nargs="+",
        default=["all"],
        metavar="SETTING",
        help="One or more setting numbers (1-8), or 'all'.",
    )
    parser.add_argument(
        "--english-model",
        default=DEFAULT_ENGLISH_MODEL,
        help="Sentence-Transformer model for English-model settings.",
    )
    parser.add_argument(
        "--multilingual-model",
        default=DEFAULT_MULTILINGUAL_MODEL,
        help="Sentence-Transformer model for multilingual settings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--download-timeout",
        type=int,
        default=180,
        help="Resource download timeout in seconds.",
    )

    return parser


def validate_input_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    selected_setting_numbers = parse_settings(args.settings)

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")

    validate_input_file(args.modern_file, "Modern Hebrew dictionary")
    validate_input_file(args.concreteness_file, "Concreteness ratings file")

    output_dir: Path = args.output_dir
    resource_dir = output_dir / "resources"
    output_dir.mkdir(parents=True, exist_ok=True)
    resource_dir.mkdir(parents=True, exist_ok=True)

    allow_downloads = not args.no_download

    bdb_path = ensure_resource(
        args.bdb_file,
        resource_dir / "BrownDriverBriggs.xml",
        DEFAULT_BDB_URL,
        allow_downloads,
    )
    lexical_index_path = ensure_resource(
        args.lexical_index_file,
        resource_dir / "LexicalIndex.xml",
        DEFAULT_LEXICAL_INDEX_URL,
        allow_downloads,
    )
    morphhb_zip = ensure_resource(
        args.morphhb_zip,
        resource_dir / "morphhb-master.zip",
        DEFAULT_MORPHHB_URL,
        allow_downloads,
    )

    biblical_data, lexical_index_xml = load_biblical_dictionary(
        bdb_path,
        lexical_index_path,
    )
    modern_data, removed_df, retained_df = load_modern_dictionary(
        args.modern_file
    )

    shared_pairs = build_shared_pairs(biblical_data, modern_data)
    shared_lemmas = sorted({lemma for lemma, _ in shared_pairs})

    if not shared_pairs:
        raise ValueError(
            "No shared lemma–POS pairs were found between the dictionaries."
        )

    print(f"\nShared lemma + POS pairs: {len(shared_pairs):,}")
    print(f"Unique shared lemmas: {len(shared_lemmas):,}")

    code_frequency = count_oshb_codes(morphhb_zip)
    unique_code_map, ambiguous_code_map = build_lexical_code_maps(
        lexical_index_xml
    )

    (
        hebrew_frequency,
        unmapped_frequency_df,
        ambiguous_frequency_df,
        grammatical_frequency_df,
    ) = reconstruct_frequencies(
        code_frequency,
        unique_code_map,
        ambiguous_code_map,
    )

    frequency_coverage = sum(
        lemma in hebrew_frequency for lemma in shared_lemmas
    )
    print(
        "Shared lemmas with usable frequency: "
        f"{frequency_coverage:,} / {len(shared_lemmas):,} "
        f"({100 * frequency_coverage / max(1, len(shared_lemmas)):.2f}%)"
    )

    biblical_polysemy = build_biblical_polysemy(biblical_data)
    concreteness_lookup = load_concreteness_lookup(
        args.concreteness_file
    )

    save_preparation_outputs(
        output_dir=output_dir,
        removed_df=removed_df,
        retained_df=retained_df,
        shared_pairs=shared_pairs,
        unmapped_frequency_df=unmapped_frequency_df,
        ambiguous_frequency_df=ambiguous_frequency_df,
        grammatical_frequency_df=grammatical_frequency_df,
        biblical_data=biblical_data,
        modern_data=modern_data,
        concreteness_lookup=concreteness_lookup,
    )

    model_names = {
        "english": args.english_model,
        "multilingual": args.multilingual_model,
    }

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required to run the experiments. "
            "Install the dependencies listed in the module docstring."
        ) from exc

    loaded_models: dict[str, Any] = {}
    embedding_cache: dict[
        tuple[str, str, str],
        dict[str, np.ndarray],
    ] = {}

    summaries: list[dict[str, object]] = []

    for setting_number in selected_setting_numbers:
        setting = SETTINGS[setting_number]
        setting_pairs = pairs_for_setting(
            shared_pairs,
            setting.pos_selection,
        )

        model_key = setting.model_key
        model_name = model_names[model_key]

        if model_key not in loaded_models:
            print(f"\nLoading embedding model: {model_name}")
            loaded_models[model_key] = SentenceTransformer(model_name)

        cache_key = (
            model_key,
            setting.definition_selection,
            setting.pos_selection,
        )

        if cache_key not in embedding_cache:
            definitions = collect_unique_definitions(
                setting_pairs,
                biblical_data,
                modern_data,
                setting.definition_selection,
            )

            print(
                f"\nEncoding {len(definitions):,} unique definitions "
                f"for Setting {setting.number}..."
            )

            embedding_cache[cache_key] = build_embedding_lookup(
                loaded_models[model_key],
                definitions,
                args.batch_size,
            )

        _, _, summary = run_setting(
            setting=setting,
            setting_pairs=setting_pairs,
            biblical_data=biblical_data,
            modern_data=modern_data,
            frequency=hebrew_frequency,
            biblical_polysemy=biblical_polysemy,
            concreteness_lookup=concreteness_lookup,
            embedding_lookup=embedding_cache[cache_key],
            model_name=model_name,
            output_dir=output_dir,
        )
        summaries.append(summary)

    summary_df = (
        pd.DataFrame(summaries)
        .sort_values("setting")
        .reset_index(drop=True)
    )

    summary_path = output_dir / "hebrew_settings_summary.csv"
    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 78)
    print("COMPLETED SETTINGS")
    print("=" * 78)
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExecution interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
