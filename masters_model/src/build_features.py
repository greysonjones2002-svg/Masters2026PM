from __future__ import annotations

import pandas as pd
import numpy as np


def safe_minmax(series: pd.Series, invert: bool = False) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().all():
        return pd.Series(np.zeros(len(series)), index=series.index)
    mn, mx = s.min(), s.max()
    if mx == mn:
        out = pd.Series(np.ones(len(series)) * 0.5, index=series.index)
    else:
        out = (s - mn) / (mx - mn)
    if invert:
        out = 1 - out
    return out.fillna(out.mean() if not np.isnan(out.mean()) else 0.5)


def load_and_merge(data_dir: str = "data") -> pd.DataFrame:
    categories = pd.read_csv(f"{data_dir}/masters_categories.csv")
    field = pd.read_csv(f"{data_dir}/masters_field.csv")
    owgr = pd.read_csv(f"{data_dir}/owgr_current.csv")
    sgwr = pd.read_csv(f"{data_dir}/owgr_sgwr.csv")
    pga = pd.read_csv(f"{data_dir}/pga_stats.csv")
    aug = pd.read_csv(f"{data_dir}/augusta_history.csv")

    df = categories.merge(field, on="player", how="left")
    df = df.merge(owgr, on="player", how="left")
    df = df.merge(sgwr, on="player", how="left")
    df = df.merge(pga, on="player", how="left")
    df = df.merge(aug, on="player", how="left")

    # Fill rookie Augusta history neutrally instead of punishing with absurd placeholders.
    df["masters_starts"] = df["masters_starts"].fillna(0)
    df["masters_cuts_made"] = df["masters_cuts_made"].fillna(0)
    df["masters_top10s"] = df["masters_top10s"].fillna(0)
    df["masters_best_finish"] = df["masters_best_finish"].replace(999, np.nan)
    df["masters_avg_finish"] = df["masters_avg_finish"].replace(999, np.nan)
    df["masters_scoring_avg"] = df["masters_scoring_avg"].replace(999, np.nan)

    return df


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Core normalized signals
    out["f_owgr"] = safe_minmax(out["owgr_rank"], invert=True)
    out["f_owgr_points"] = safe_minmax(out["owgr_points"])
    out["f_sgwr"] = safe_minmax(out["sgwr"])
    out["f_sg_total"] = safe_minmax(out["sg_total"])
    out["f_sg_ott"] = safe_minmax(out["sg_off_tee"])
    out["f_sg_app"] = safe_minmax(out["sg_approach"])
    out["f_sg_arg"] = safe_minmax(out["sg_around_green"])
    out["f_sg_putt"] = safe_minmax(out["sg_putting"])
    out["f_scoring"] = safe_minmax(out["scoring_avg"], invert=True)

    # Augusta/course-history features
    out["aug_cut_rate"] = np.where(
        out["masters_starts"] > 0,
        out["masters_cuts_made"] / out["masters_starts"],
        0.50,
    )
    out["aug_top10_rate"] = np.where(
        out["masters_starts"] > 0,
        out["masters_top10s"] / out["masters_starts"],
        0.00,
    )
    out["f_aug_cut_rate"] = safe_minmax(out["aug_cut_rate"])
    out["f_aug_top10_rate"] = safe_minmax(out["aug_top10_rate"])
    out["f_aug_avg_finish"] = safe_minmax(out["masters_avg_finish"], invert=True)
    out["f_aug_scoring_avg"] = safe_minmax(out["masters_scoring_avg"], invert=True)

    # Rookie penalty is mild, not fatal.
    out["rookie_penalty"] = np.where(out["is_first_timer"] == 1, 0.03, 0.0)

    # Composite predicted "fantasy quality" score
    out["model_score"] = (
        0.12 * out["f_owgr"]
        + 0.08 * out["f_owgr_points"]
        + 0.12 * out["f_sgwr"]
        + 0.15 * out["f_sg_total"]
        + 0.12 * out["f_sg_app"]
        + 0.08 * out["f_sg_ott"]
        + 0.06 * out["f_sg_arg"]
        + 0.06 * out["f_sg_putt"]
        + 0.08 * out["f_scoring"]
        + 0.06 * out["f_aug_cut_rate"]
        + 0.04 * out["f_aug_top10_rate"]
        + 0.02 * out["f_aug_avg_finish"]
        + 0.02 * out["f_aug_scoring_avg"]
        - out["rookie_penalty"]
    )

    # Convert to projected total-to-par for the tournament.
    # Lower is better because fantasy uses score-to-par totals.
    # Tune this intercept if you want more or less aggressive projections.
    out["projected_to_par"] = -4.0 - 12.0 * out["model_score"]

    return out


if __name__ == "__main__":
    df = load_and_merge()
    df = add_model_features(df)
    df.to_csv("data/model_input_scored.csv", index=False)
    print(
        df[["player", "category", "model_score", "projected_to_par"]]
        .sort_values("projected_to_par")
        .to_string(index=False)
    )
