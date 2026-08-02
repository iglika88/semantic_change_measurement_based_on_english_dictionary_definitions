#!/usr/bin/env python3
"""Compare Latin–French and Latin–Italian semantic distances across 16 settings.

Prerequisites
-------------
Run ``latin_french_experiments.py`` and ``latin_italian_experiments.py`` before
this script. Each experiment must produce its sixteen per-setting lemma-result
files, normally named::

    latin_french_setting_1_lemma_results.csv
    ...
    latin_french_setting_16_lemma_results.csv

and::

    latin_italian_setting_1_lemma_results.csv
    ...
    latin_italian_setting_16_lemma_results.csv

Pass the two result directories with ``--french-results-dir`` and
``--italian-results-dir``. The comparison script does not rerun dictionary
loading, translation, embedding, or semantic-change experiments.

Scientific comparison
---------------------
For each setting, the script reports:

1. A global comparison of the mean lemma-level semantic distance in each
   language pair.
2. A paired comparison restricted to Latin lemmas available in both pairs.

The principal comparison is the paired common-lemma analysis. Its difference is
``Latin–Italian minus Latin–French``; therefore, a negative value supports the
hypothesis that Italian is semantically closer to Latin than French is.

The paired analysis includes a one-sided Wilcoxon signed-rank test, a percentile
bootstrap 95% confidence interval for the paired mean difference, paired effect
size ``dz``, and the percentage of common lemmas for which Italian is closer.

Examples
--------
Run all settings::

    python latin_french_latin_italian_comparison.py \
      --french-results-dir latin_french_results \
      --italian-results-dir latin_italian_results

Run selected settings::

    python latin_french_latin_italian_comparison.py \
      --french-results-dir latin_french_results \
      --italian-results-dir latin_italian_results \
      --settings 1 2 9 10

Dependencies: numpy, pandas, scipy.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


@dataclass(frozen=True)
class Setting:
    number: int
    definitions: str
    pos: str
    mapping: str
    model: str

    @property
    def name(self) -> str:
        definition = "all def." if self.definitions == "all" else "first def."
        pos = "all POS" if self.pos == "all" else "selected POS"
        mapping = "cogn." if self.mapping == "cognates" else "MT"
        model = "EN" if self.model == "english" else "MULTI"
        return f"{definition}/{pos}/{mapping}/{model}"


SETTINGS: dict[int, Setting] = {}
for offset, mapping in ((0, "cognates"), (8, "MT")):
    for first_offset, definitions in ((0, "all"), (4, "first")):
        SETTINGS[offset + first_offset + 1] = Setting(
            offset + first_offset + 1, definitions, "all", mapping, "english"
        )
        SETTINGS[offset + first_offset + 2] = Setting(
            offset + first_offset + 2,
            definitions,
            "all",
            mapping,
            "multilingual",
        )
        SETTINGS[offset + first_offset + 3] = Setting(
            offset + first_offset + 3,
            definitions,
            "selected",
            mapping,
            "english",
        )
        SETTINGS[offset + first_offset + 4] = Setting(
            offset + first_offset + 4,
            definitions,
            "selected",
            mapping,
            "multilingual",
        )

LEMMA_COLUMNS = ("latin_lemma", "normalized_lemma", "normalised_lemma", "lemma")
DISTANCE_COLUMNS = (
    "semantic_distance",
    "semantic_change_apd",
    "semantic_change",
    "apd",
    "mean_distance",
    "mean_cosine_distance",
    "cosine_distance",
    "distance",
)


def normalize_column_name(value: object) -> str:
    return re.sub(r"_+", "_", str(value).strip().lower().replace("-", "_"))


def normalize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.rename(columns={c: normalize_column_name(c) for c in dataframe.columns})


def first_column(dataframe: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    return next((candidate for candidate in candidates if candidate in dataframe.columns), None)


def parse_settings(values: Sequence[str]) -> list[int]:
    if not values or values == ["all"]:
        return sorted(SETTINGS)
    selected = sorted(set(int(value) for value in values))
    invalid = [number for number in selected if number not in SETTINGS]
    if invalid:
        raise ValueError(f"Settings must be between 1 and 16; invalid: {invalid}")
    return selected


def locate_result_file(directory: Path, pair: str, setting: int) -> Path:
    """Locate one per-setting lemma-level file using preferred names first."""
    preferred = [
        directory / f"latin_{pair}_setting_{setting}_lemma_results.csv",
        directory / f"latin_{pair}_setting_{setting:02d}_lemma_results.csv",
    ]
    for path in preferred:
        if path.is_file():
            return path

    patterns = [
        f"**/latin_{pair}_setting_{setting}_lemma_results.csv",
        f"**/latin_{pair}_setting_{setting:02d}_lemma_results.csv",
        f"**/*{pair}*setting*{setting}*lemma*results*.csv",
    ]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(path for path in directory.glob(pattern) if path.is_file())
    matches = sorted(set(matches))
    if not matches:
        raise FileNotFoundError(
            f"No lemma-level result file found for Latin–{pair.title()}, "
            f"setting {setting}, below {directory}. Run the corresponding "
            "experiment script first or pass the correct results directory."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple candidate files found for Latin–{pair.title()}, setting {setting}:\n"
            + "\n".join(f"  - {path}" for path in matches)
        )
    return matches[0]


def load_lemma_results(path: Path, setting: int, pair_label: str) -> pd.DataFrame:
    dataframe = normalize_columns(pd.read_csv(path))
    lemma_column = first_column(dataframe, LEMMA_COLUMNS)
    distance_column = first_column(dataframe, DISTANCE_COLUMNS)
    if lemma_column is None:
        raise ValueError(f"No Latin lemma column found in {path}; columns={list(dataframe.columns)}")
    if distance_column is None:
        raise ValueError(f"No semantic-distance column found in {path}; columns={list(dataframe.columns)}")

    if "setting" in dataframe.columns:
        observed = pd.to_numeric(dataframe["setting"], errors="coerce").dropna().unique()
        if len(observed) and set(observed.astype(int)) != {setting}:
            raise ValueError(
                f"File {path} contains setting values {observed.tolist()}, expected {setting}."
            )

    standardized = pd.DataFrame(
        {
            "setting": setting,
            "pair": pair_label,
            "latin_lemma": (
                dataframe[lemma_column].fillna("").astype(str).str.strip().str.lower()
            ),
            "distance": pd.to_numeric(dataframe[distance_column], errors="coerce"),
            "source_file": str(path),
        }
    )
    standardized = standardized[
        standardized.latin_lemma.ne("")
        & standardized.latin_lemma.ne("nan")
        & standardized.distance.notna()
        & np.isfinite(standardized.distance)
    ].copy()
    if standardized.empty:
        raise ValueError(f"No usable lemma-level distances found in {path}")

    # The experiment scripts should already output one row per Latin lemma.
    # Aggregate defensively in case an older result file contains duplicates.
    return (
        standardized.groupby("latin_lemma", as_index=False)
        .agg(
            distance=("distance", "mean"),
            source_rows=("distance", "size"),
            source_file=("source_file", "first"),
        )
        .assign(setting=setting, pair=pair_label)
    )


def paired_wilcoxon_less(italian: Sequence[float], french: Sequence[float]) -> tuple[float, float, int]:
    differences = np.asarray(italian, dtype=float) - np.asarray(french, dtype=float)
    differences = differences[np.isfinite(differences)]
    if len(differences) < 3:
        return np.nan, np.nan, len(differences)
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 0.0, 1.0, len(differences)
    result = wilcoxon(nonzero, alternative="less", zero_method="wilcox")
    return float(result.statistic), float(result.pvalue), len(differences)


def bootstrap_mean_difference(
    italian: Sequence[float],
    french: Sequence[float],
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    differences = np.asarray(italian, dtype=float) - np.asarray(french, dtype=float)
    differences = differences[np.isfinite(differences)]
    if len(differences) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=float)
    chunk_size = 250
    completed = 0
    while completed < iterations:
        size = min(chunk_size, iterations - completed)
        indices = rng.integers(0, len(differences), size=(size, len(differences)))
        means[completed : completed + size] = differences[indices].mean(axis=1)
        completed += size
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def paired_dz(italian: Sequence[float], french: Sequence[float]) -> float:
    differences = np.asarray(italian, dtype=float) - np.asarray(french, dtype=float)
    differences = differences[np.isfinite(differences)]
    if len(differences) < 2:
        return np.nan
    standard_deviation = differences.std(ddof=1)
    if np.isclose(standard_deviation, 0):
        return 0.0
    return float(differences.mean() / standard_deviation)


def result_symbol(difference: float, p_value: float) -> str:
    if not np.isfinite(difference) or not np.isfinite(p_value) or difference >= 0 or p_value >= 0.05:
        return "✗"
    if p_value < 0.001:
        return "**"
    if p_value < 0.01:
        return "*"
    return "✓"


def compare_setting(
    setting: Setting,
    french: pd.DataFrame,
    italian: pd.DataFrame,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    french_global_mean = float(french.distance.mean())
    italian_global_mean = float(italian.distance.mean())
    global_difference = italian_global_mean - french_global_mean

    paired = french[["latin_lemma", "distance", "source_rows"]].rename(
        columns={"distance": "french_distance", "source_rows": "french_source_rows"}
    ).merge(
        italian[["latin_lemma", "distance", "source_rows"]].rename(
            columns={"distance": "italian_distance", "source_rows": "italian_source_rows"}
        ),
        on="latin_lemma",
        how="inner",
        validate="one_to_one",
    )
    if paired.empty:
        raise ValueError(f"Setting {setting.number} has no common Latin lemmas")

    paired["difference_italian_minus_french"] = (
        paired.italian_distance - paired.french_distance
    )
    paired["italian_closer"] = paired.difference_italian_minus_french < 0
    paired["french_closer"] = paired.difference_italian_minus_french > 0
    paired["equal_distance"] = paired.difference_italian_minus_french == 0
    paired.insert(0, "setting", setting.number)
    paired.insert(1, "setting_name", setting.name)

    french_values = paired.french_distance.to_numpy()
    italian_values = paired.italian_distance.to_numpy()
    paired_difference = float((italian_values - french_values).mean())
    statistic, p_value, test_n = paired_wilcoxon_less(italian_values, french_values)
    ci_lower, ci_upper = bootstrap_mean_difference(
        italian_values,
        french_values,
        bootstrap_iterations,
        seed + setting.number,
    )

    summary: dict[str, object] = {
        "setting": setting.number,
        "setting_name": setting.name,
        "definition_mode": setting.definitions,
        "pos_mode": setting.pos,
        "mapping_mode": setting.mapping,
        "model": setting.model,
        "latin_french_lemmas": int(french.latin_lemma.nunique()),
        "latin_italian_lemmas": int(italian.latin_lemma.nunique()),
        "latin_french_global_mean": french_global_mean,
        "latin_italian_global_mean": italian_global_mean,
        "global_difference_li_minus_lf": global_difference,
        "global_direction": "Italian closer" if global_difference < 0 else "French closer",
        "global_criterion": "✓" if global_difference < 0 else "✗",
        "paired_common_lemmas": len(paired),
        "paired_latin_french_mean": float(french_values.mean()),
        "paired_latin_italian_mean": float(italian_values.mean()),
        "paired_mean_difference_li_minus_lf": paired_difference,
        "paired_median_difference_li_minus_lf": float(
            paired.difference_italian_minus_french.median()
        ),
        "bootstrap_95_ci_lower": ci_lower,
        "bootstrap_95_ci_upper": ci_upper,
        "bootstrap_ci_entirely_negative": bool(np.isfinite(ci_upper) and ci_upper < 0),
        "italian_closer_percentage": float(100 * paired.italian_closer.mean()),
        "french_closer_percentage": float(100 * paired.french_closer.mean()),
        "equal_percentage": float(100 * paired.equal_distance.mean()),
        "paired_effect_size_dz": paired_dz(italian_values, french_values),
        "wilcoxon_n": test_n,
        "wilcoxon_statistic": statistic,
        "wilcoxon_one_sided_p": p_value,
        "paired_result": result_symbol(paired_difference, p_value),
    }
    return summary, paired


def format_number(value: object, digits: int = 5) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def format_p(value: object) -> str:
    if pd.isna(value):
        return "N/A"
    number = float(value)
    return f"{number:.3e}" if number < 0.001 else f"{number:.4f}"


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description=(
            "Compare the lemma-level outputs of the Latin–French and "
            "Latin–Italian semantic-change experiment scripts."
        )
    )
    argument_parser.add_argument("--french-results-dir", type=Path, required=True)
    argument_parser.add_argument("--italian-results-dir", type=Path, required=True)
    argument_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("latin_french_latin_italian_comparison_results"),
    )
    argument_parser.add_argument("--settings", nargs="+", default=["all"])
    argument_parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    argument_parser.add_argument("--random-seed", type=int, default=42)
    return argument_parser


def main() -> None:
    args = parser().parse_args()
    selected_settings = parse_settings(args.settings)
    if args.bootstrap_iterations < 100:
        raise ValueError("--bootstrap-iterations must be at least 100")
    for path, label in (
        (args.french_results_dir, "French results directory"),
        (args.italian_results_dir, "Italian results directory"),
    ):
        if not path.is_dir():
            raise NotADirectoryError(f"{label} not found: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_directory = args.output_dir / "paired_common_lemmas"
    detail_directory.mkdir(exist_ok=True)

    summaries: list[dict[str, object]] = []
    input_manifest: list[dict[str, object]] = []

    for number in selected_settings:
        setting = SETTINGS[number]
        french_path = locate_result_file(args.french_results_dir, "french", number)
        italian_path = locate_result_file(args.italian_results_dir, "italian", number)
        french = load_lemma_results(french_path, number, "Latin–French")
        italian = load_lemma_results(italian_path, number, "Latin–Italian")
        summary, paired = compare_setting(
            setting,
            french,
            italian,
            args.bootstrap_iterations,
            args.random_seed,
        )
        summaries.append(summary)
        input_manifest.extend(
            [
                {
                    "setting": number,
                    "pair": "Latin–French",
                    "input_file": str(french_path.resolve()),
                    "usable_lemmas": len(french),
                },
                {
                    "setting": number,
                    "pair": "Latin–Italian",
                    "input_file": str(italian_path.resolve()),
                    "usable_lemmas": len(italian),
                },
            ]
        )
        paired.to_csv(
            detail_directory / f"setting_{number:02d}_paired_common_latin_lemmas.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(
            f"Setting {number:2d} | {setting.name} | common={len(paired):,} | "
            f"LI-LF={format_number(summary['paired_mean_difference_li_minus_lf'], 6)} | "
            f"p={format_p(summary['wilcoxon_one_sided_p'])} | "
            f"{summary['paired_result']}"
        )

    summary_dataframe = pd.DataFrame(summaries).sort_values("setting")
    summary_path = args.output_dir / "latin_french_latin_italian_comparison_summary.csv"
    summary_dataframe.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(input_manifest).to_csv(
        args.output_dir / "input_file_manifest.csv", index=False, encoding="utf-8-sig"
    )

    global_expected = int((summary_dataframe.global_difference_li_minus_lf < 0).sum())
    paired_expected = int(
        (summary_dataframe.paired_mean_difference_li_minus_lf < 0).sum()
    )
    paired_significant = int(summary_dataframe.paired_result.isin(["✓", "*", "**"]).sum())

    print("\nOverall result")
    print(f"  Italian closer by global mean: {global_expected}/{len(summary_dataframe)}")
    print(f"  Italian closer by paired mean: {paired_expected}/{len(summary_dataframe)}")
    print(
        "  Italian significantly closer by paired Wilcoxon: "
        f"{paired_significant}/{len(summary_dataframe)}"
    )
    print(f"\nSummary saved to: {summary_path}")
    print(f"Paired detail files saved to: {detail_directory}")


if __name__ == "__main__":
    main()
