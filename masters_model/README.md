# Masters Fantasy Optimizer

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Populate data

Fill these CSVs:

- data/masters_categories.csv
- data/masters_field.csv
- data/owgr_current.csv
- data/owgr_sgwr.csv
- data/pga_stats.csv
- data/augusta_history.csv

## Run scripts

```bash
python src/build_features.py
python src/optimize_roster.py
python src/predict_tiebreakers.py
```

## Run UI

```bash
streamlit run ui/app.py
```

In the UI you can either:

- use bundled sample CSVs from `masters_model/data/`, or
- upload your own six CSVs (same filenames as listed above).

## Notes

- The optimizer chooses exactly 1 player from each fantasy category.
- Lower projected total-to-par is better.
- Update the CSVs immediately before lock.
