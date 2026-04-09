from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from src.build_features import add_model_features, load_and_merge
from src.optimize_roster import optimize_roster
from src.predict_tiebreakers import predict_total_putts, predict_winning_score


st.set_page_config(page_title="Masters Fantasy Optimizer", layout="wide")

DATA_FILES = [
    "masters_categories.csv",
    "masters_field.csv",
    "owgr_current.csv",
    "owgr_sgwr.csv",
    "pga_stats.csv",
    "augusta_history.csv",
]


@st.cache_data(show_spinner=False)
def _load_dataframes_from_bytes(file_bytes_map: dict[str, bytes]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name, raw in file_bytes_map.items():
        out[name] = pd.read_csv(io.BytesIO(raw))
    return out


def _write_uploaded_to_temp(uploaded: dict[str, pd.DataFrame], temp_dir: Path) -> None:
    temp_dir.mkdir(parents=True, exist_ok=True)
    for name, df in uploaded.items():
        df.to_csv(temp_dir / name, index=False)


def _load_example_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(data_dir / name) for name in DATA_FILES}


def main() -> None:
    st.title("Masters Fantasy Optimizer UI")
    st.caption(
        "Upload fresh CSVs right before lock, or use the included sample files to test the pipeline."
    )

    root_dir = Path(__file__).resolve().parents[1]
    sample_data_dir = root_dir / "data"
    temp_upload_dir = root_dir / "data" / "_uploaded"

    source_mode = st.radio(
        "Data source",
        options=["Use bundled sample CSVs", "Upload my own CSVs"],
        horizontal=True,
    )

    uploaded_dataframes: dict[str, pd.DataFrame]
    if source_mode == "Upload my own CSVs":
        st.subheader("Upload required files")
        st.write("Upload all six files using these exact names:")
        st.code("\n".join(DATA_FILES))

        uploads = {
            name: st.file_uploader(name, type=["csv"], key=name) for name in DATA_FILES
        }

        if all(uploads.values()):
            raw_map = {name: uploads[name].getvalue() for name in DATA_FILES}
            uploaded_dataframes = _load_dataframes_from_bytes(raw_map)
            _write_uploaded_to_temp(uploaded_dataframes, temp_upload_dir)
            active_data_dir = temp_upload_dir
            st.success("Uploaded files loaded and staged for modeling.")
        else:
            st.info("Upload all six files to run the model.")
            return
    else:
        uploaded_dataframes = _load_example_data(sample_data_dir)
        active_data_dir = sample_data_dir
        st.success("Using bundled sample CSVs from masters_model/data.")

    with st.expander("Data preview", expanded=False):
        preview_name = st.selectbox("Choose a table", DATA_FILES)
        st.dataframe(uploaded_dataframes[preview_name], use_container_width=True)

    if st.button("Run feature engineering", type="primary"):
        scored = add_model_features(load_and_merge(str(active_data_dir)))
        st.subheader("Scored player input")
        st.dataframe(
            scored[
                ["player", "category", "model_score", "projected_to_par"]
            ].sort_values("projected_to_par"),
            use_container_width=True,
        )

        st.download_button(
            "Download model_input_scored.csv",
            data=scored.to_csv(index=False).encode("utf-8"),
            file_name="model_input_scored.csv",
            mime="text/csv",
        )

        roster = optimize_roster(str(active_data_dir))
        st.subheader("Optimal roster")
        st.dataframe(roster, use_container_width=True)

        total_to_par = roster["projected_to_par"].sum()
        st.metric("Projected roster total to par", f"{total_to_par:.2f}")

        winning_score = predict_winning_score(scored)
        total_putts = predict_total_putts(scored)

        col1, col2 = st.columns(2)
        col1.metric("Predicted winning score (to par)", winning_score)
        col2.metric("Predicted total field putts", total_putts)


if __name__ == "__main__":
    main()
