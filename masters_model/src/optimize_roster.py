from __future__ import annotations

import pandas as pd
import pulp

from src.build_features import load_and_merge, add_model_features


def optimize_roster(data_dir: str = "data") -> pd.DataFrame:
    df = load_and_merge(data_dir)
    df = add_model_features(df)

    categories = sorted(df["category"].dropna().unique().tolist())

    prob = pulp.LpProblem("masters_fantasy_optimizer", pulp.LpMinimize)

    x = {
        i: pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat="Binary")
        for i in df.index
    }

    # Objective: minimize projected total to par
    prob += pulp.lpSum(df.loc[i, "projected_to_par"] * x[i] for i in df.index)

    # Exactly one pick per category
    for cat in categories:
        idx = df.index[df["category"] == cat].tolist()
        prob += pulp.lpSum(x[i] for i in idx) == 1, f"one_pick_{cat}"

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Solver failed: {pulp.LpStatus[status]}")

    chosen = [i for i in df.index if x[i].value() == 1]
    roster = df.loc[chosen, ["player", "category", "projected_to_par", "model_score"]].copy()
    roster = roster.sort_values(["category", "projected_to_par"])
    return roster


if __name__ == "__main__":
    roster = optimize_roster()
    print("\nOPTIMAL ROSTER\n")
    print(roster.to_string(index=False))
    print(f"\nProjected roster total to par: {roster['projected_to_par'].sum():.2f}")
