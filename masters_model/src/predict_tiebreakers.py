from __future__ import annotations

import pandas as pd

from src.build_features import load_and_merge, add_model_features


def predict_winning_score(df: pd.DataFrame) -> int:
    # Lower to-par is better. Use the best projected player, then shrink toward
    # a plausible Augusta winning range.
    best = df["projected_to_par"].min()
    # Blend best player projection with a prior centered around -11.
    pred = 0.65 * best + 0.35 * (-11.0)
    return int(round(pred))


def predict_total_putts(
    df: pd.DataFrame, field_size: int | None = None, cut_size: int = 50
) -> int:
    # Simple field-level estimate.
    # Thurs/Fri: all players play 2 rounds.
    # Sat/Sun: cut_size players play 2 rounds.
    if field_size is None:
        field_size = df["player"].nunique()

    total_rounds = field_size * 2 + cut_size * 2

    # Estimate avg putts/round from putting strength:
    # good putters slightly below 29, weaker fields above.
    # Since PGA stat file may not cover all invitees perfectly, use conservative baseline.
    baseline = 29.1

    if "sg_putting" in df.columns:
        s = pd.to_numeric(df["sg_putting"], errors="coerce")
        if s.notna().sum() > 5:
            # Better field putting lowers total putts slightly.
            field_putting = s.mean()
            baseline = 29.1 - 0.15 * field_putting

    projected = total_rounds * baseline
    return int(round(projected))


if __name__ == "__main__":
    df = load_and_merge()
    df = add_model_features(df)

    winning_score = predict_winning_score(df)
    total_putts = predict_total_putts(df)

    print(f"Predicted winning score to par: {winning_score}")
    print(f"Predicted total field putts: {total_putts}")
