from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Gym Progress Dashboard", layout="wide")


DEFAULT_FILES = {
    "weight": "weight.csv",
    "nutrition": "cronometer.csv",
    "workouts": "strong.csv",
}

EXAMPLE_FILES = {
    "weight": Path(r"C:\Users\xavie\OneDrive\Desktop\weightloss\weight.csv"),
    "nutrition": Path(r"C:\Users\xavie\OneDrive\Desktop\weightloss\cronometer.csv"),
    "workouts": Path(r"C:\Users\xavie\OneDrive\Desktop\weightloss\strong.csv"),
}

COMPOUND_MOVEMENTS = {
    "Bench Press": ["bench press", "bench press barbell", "flat barbell bench press"],
    "Squat": ["squat", "squat barbell", "back squat", "barbell squat"],
    "Deadlift": ["deadlift", "deadlift barbell", "conventional deadlift", "barbell deadlift"],
}

TIMEFRAME_DAYS = {
    "Week": 7,
    "Month": 30,
    "All Time": None,
}


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    return cleaned


def normalize_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
    )


def find_column(df: pd.DataFrame, options: list[str]) -> str | None:
    normalized = {normalize_name(col): col for col in df.columns}
    for option in options:
        if normalize_name(option) in normalized:
            return normalized[normalize_name(option)]
    return None


def load_csv(upload, fallback_name: str) -> pd.DataFrame | None:
    if upload is not None:
        return clean_columns(pd.read_csv(upload))

    fallback_path = Path(fallback_name)
    if fallback_path.exists():
        return clean_columns(pd.read_csv(fallback_path))

    example_path = EXAMPLE_FILES.get(
        "weight" if fallback_name == DEFAULT_FILES["weight"] else
        "nutrition" if fallback_name == DEFAULT_FILES["nutrition"] else
        "workouts"
    )
    if example_path and example_path.exists():
        return clean_columns(pd.read_csv(example_path))

    return None


def prepare_weight(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    date_col = find_column(df, ["Date", "Day"])
    weight_col = find_column(df, ["Weight", "Body Weight", "Weight (lb)", "Weight (kg)"])

    if not date_col or not weight_col:
        return pd.DataFrame(), "Weight data needs a date column and a weight column."

    prepared = df[[date_col, weight_col]].copy()
    prepared.columns = ["Date", "Weight"]
    prepared["Date"] = pd.to_datetime(prepared["Date"], errors="coerce")
    prepared["Weight"] = pd.to_numeric(prepared["Weight"], errors="coerce")
    prepared = prepared.dropna().sort_values("Date")

    if prepared.empty:
        return pd.DataFrame(), "Weight data could not be parsed."

    daily = (
        prepared.groupby(prepared["Date"].dt.floor("D"), as_index=False)["Weight"]
        .mean()
        .rename(columns={"Date": "Date"})
    )
    daily["7 Day Avg"] = daily["Weight"].rolling(7, min_periods=1).mean()
    return daily, None


def prepare_nutrition(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    date_col = find_column(df, ["Date", "Day"])
    calorie_col = find_column(df, ["Calories", "Energy (kcal)", "Energy"])
    protein_col = find_column(df, ["Protein (g)", "Protein", "Protein g"])
    carb_col = find_column(df, ["Carbs (g)", "Net Carbs (g)", "Carbohydrates (g)", "Carbs"])
    fat_col = find_column(df, ["Fat (g)", "Fat"])

    if not date_col or not calorie_col:
        return pd.DataFrame(), "Nutrition data needs at least date and calories columns."

    columns = [date_col, calorie_col]
    rename_map = {date_col: "Date", calorie_col: "Calories"}

    if protein_col:
        columns.append(protein_col)
        rename_map[protein_col] = "Protein"
    if carb_col:
        columns.append(carb_col)
        rename_map[carb_col] = "Carbs"
    if fat_col:
        columns.append(fat_col)
        rename_map[fat_col] = "Fat"

    prepared = df[columns].copy().rename(columns=rename_map)
    prepared["Date"] = pd.to_datetime(prepared["Date"], errors="coerce")

    for col in ["Calories", "Protein", "Carbs", "Fat"]:
        if col in prepared.columns:
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce")

    prepared = prepared.dropna(subset=["Date", "Calories"]).sort_values("Date")

    if prepared.empty:
        return pd.DataFrame(), "Nutrition data could not be parsed."

    daily = prepared.groupby(prepared["Date"].dt.floor("D"), as_index=False).mean(numeric_only=True)
    daily["Calorie Avg"] = daily["Calories"].rolling(7, min_periods=1).mean()
    if "Protein" in daily.columns:
        daily["Protein Avg"] = daily["Protein"].rolling(7, min_periods=1).mean()
    return daily, None


def prepare_workouts(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    date_col = find_column(df, ["Date"])
    exercise_col = find_column(df, ["Exercise Name", "Exercise"])
    weight_col = find_column(df, ["Weight", "Weight (lb)", "Weight (kg)"])
    reps_col = find_column(df, ["Reps", "Rep"])
    set_order_col = find_column(df, ["SetOrder"])

    if not date_col or not exercise_col or not weight_col:
        return pd.DataFrame(), "Workout data needs date, exercise, and weight columns."

    columns = [date_col, exercise_col, weight_col]
    rename_map = {
        date_col: "Date",
        exercise_col: "Exercise",
        weight_col: "Weight",
    }

    if reps_col:
        columns.append(reps_col)
        rename_map[reps_col] = "Reps"
    if set_order_col:
        columns.append(set_order_col)
        rename_map[set_order_col] = "SetOrder"

    prepared = df[columns].copy().rename(columns=rename_map)
    prepared["Date"] = pd.to_datetime(prepared["Date"], errors="coerce")
    prepared["Weight"] = pd.to_numeric(prepared["Weight"], errors="coerce")
    prepared["Reps"] = pd.to_numeric(prepared.get("Reps", 1), errors="coerce").fillna(1)
    if "SetOrder" in prepared.columns:
        prepared["SetOrder"] = pd.to_numeric(prepared["SetOrder"], errors="coerce")

    prepared = prepared.dropna(subset=["Date", "Exercise", "Weight"]).sort_values("Date")
    prepared["Exercise"] = prepared["Exercise"].astype(str).str.strip()
    prepared["Volume"] = prepared["Weight"] * prepared["Reps"]
    prepared["Estimated 1RM"] = prepared["Weight"] * (1 + (prepared["Reps"] / 30))

    if prepared.empty:
        return pd.DataFrame(), "Workout data could not be parsed."

    return prepared, None


def filter_timeframe(df: pd.DataFrame, days: int | None) -> pd.DataFrame:
    if df.empty or days is None:
        return df
    cutoff = df["Date"].max() - pd.Timedelta(days=days - 1)
    return df[df["Date"] >= cutoff].copy()


def metric_delta_text(current: float, previous: float, suffix: str = "") -> str:
    if pd.isna(previous):
        return "Not enough data"
    return f"{current - previous:+.1f}{suffix}"


def estimate_change(df: pd.DataFrame, value_col: str, days: int | None) -> float | None:
    if df.empty:
        return None
    sliced = filter_timeframe(df, days)
    if len(sliced) < 2:
        return None
    return float(sliced[value_col].iloc[-1] - sliced[value_col].iloc[0])


def build_projection(df: pd.DataFrame, value_col: str, periods: int, freq: str = "D") -> pd.DataFrame:
    if df.empty or len(df) < 2:
        return pd.DataFrame()

    ordered = df.sort_values("Date").copy()
    ordered["DaysFromStart"] = (ordered["Date"] - ordered["Date"].min()).dt.days.astype(float)

    if ordered["DaysFromStart"].nunique() < 2:
        return pd.DataFrame()

    slope, intercept = np.polyfit(ordered["DaysFromStart"], ordered[value_col], 1)
    future_offsets = np.arange(1, periods + 1, dtype=float)
    last_offset = ordered["DaysFromStart"].iloc[-1]
    future_days = last_offset + future_offsets
    future_dates = pd.date_range(ordered["Date"].max() + pd.Timedelta(days=1), periods=periods, freq=freq)

    projected = pd.DataFrame(
        {
            "Date": future_dates,
            value_col: intercept + slope * future_days,
            "Series": "Projection",
        }
    )
    return projected


def add_actual_and_projection(fig: go.Figure, actual_df: pd.DataFrame, projection_df: pd.DataFrame, value_col: str, actual_name: str) -> go.Figure:
    fig.add_trace(
        go.Scatter(
            x=actual_df["Date"],
            y=actual_df[value_col],
            mode="lines+markers",
            name=actual_name,
            line=dict(color="#0f766e", width=3),
        )
    )
    if not projection_df.empty:
        fig.add_trace(
            go.Scatter(
                x=projection_df["Date"],
                y=projection_df[value_col],
                mode="lines",
                name="Projection",
                line=dict(color="#f97316", width=3, dash="dash"),
            )
        )
    fig.update_layout(
        template="plotly_white",
        legend_title_text="",
        margin=dict(l=12, r=12, t=60, b=12),
        hovermode="x unified",
    )
    return fig


def find_compound_data(workouts: pd.DataFrame, aliases: list[str]) -> pd.DataFrame:
    normalized_aliases = {normalize_name(alias) for alias in aliases}
    normalized_exercise = workouts["Exercise"].map(normalize_name)
    mask = normalized_exercise.apply(
        lambda exercise_name: any(alias in exercise_name for alias in normalized_aliases)
    )
    return workouts[mask].copy()


def build_compound_summary(workouts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    for label, aliases in COMPOUND_MOVEMENTS.items():
        movement_data = find_compound_data(workouts, aliases)
        if movement_data.empty:
            continue

        best_by_day = movement_data.groupby("Date", as_index=False)["Estimated 1RM"].max()
        forecast = build_projection(best_by_day, "Estimated 1RM", periods=30)
        projected_value = forecast["Estimated 1RM"].iloc[-1] if not forecast.empty else np.nan

        rows.append(
            {
                "Lift": label,
                "Current e1RM": best_by_day["Estimated 1RM"].iloc[-1],
                "30-Day Projection": projected_value,
                "Sessions": movement_data["Date"].dt.date.nunique(),
            }
        )
    return pd.DataFrame(rows)


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(20, 184, 166, 0.12), transparent 32%),
            radial-gradient(circle at top right, rgba(249, 115, 22, 0.12), transparent 28%);
    }

    [data-theme="light"] .stApp {
        background-image:
            radial-gradient(circle at top left, rgba(20, 184, 166, 0.12), transparent 32%),
            radial-gradient(circle at top right, rgba(249, 115, 22, 0.12), transparent 28%),
            linear-gradient(180deg, #f4fbf9 0%, #fffaf4 100%);
    }

    [data-theme="dark"] .stApp {
        background-image:
            radial-gradient(circle at top left, rgba(20, 184, 166, 0.08), transparent 32%),
            radial-gradient(circle at top right, rgba(249, 115, 22, 0.08), transparent 28%),
            linear-gradient(180deg, #0f1117 0%, #1a1a2e 100%);
    }

    div[data-testid="stMetric"] {
        border-radius: 18px;
        padding: 0.75rem 1rem;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.05);
    }

    [data-theme="light"] div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(15, 118, 110, 0.12);
    }

    [data-theme="dark"] div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(20, 184, 166, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("Gym Progress Dashboard")
st.caption("See your lifting, diet, and bodyweight trends together, with simple projections based on your historical rate of change.")

with st.sidebar:
    st.header("Data Sources")
    st.write("Upload files here or place them in the project folder.")
    weight_upload = st.file_uploader("Weight CSV", type="csv", key="weight")
    nutrition_upload = st.file_uploader("Cronometer CSV", type="csv", key="nutrition")
    workouts_upload = st.file_uploader("Strong CSV", type="csv", key="workouts")
    timeframe_label = st.selectbox("Default chart range", list(TIMEFRAME_DAYS.keys()), index=1)
    forecast_days = st.slider("Projection length (days)", min_value=14, max_value=90, value=30, step=7)
    st.caption("Default filenames: `weight.csv`, `cronometer.csv`, and `strong.csv`.")
    st.caption("Your sample desktop exports are also used automatically when available.")


weight_raw = load_csv(weight_upload, DEFAULT_FILES["weight"])
nutrition_raw = load_csv(nutrition_upload, DEFAULT_FILES["nutrition"])
workouts_raw = load_csv(workouts_upload, DEFAULT_FILES["workouts"])

weight_data, weight_error = prepare_weight(weight_raw) if weight_raw is not None else (pd.DataFrame(), "Weight file not found.")
nutrition_data, nutrition_error = prepare_nutrition(nutrition_raw) if nutrition_raw is not None else (pd.DataFrame(), "Nutrition file not found.")
workout_data, workout_error = prepare_workouts(workouts_raw) if workouts_raw is not None else (pd.DataFrame(), "Workout file not found.")

selected_days = TIMEFRAME_DAYS[timeframe_label]
weight_view = filter_timeframe(weight_data, selected_days)
nutrition_view = filter_timeframe(nutrition_data, selected_days)

st.subheader("Weight Change")
wc1, wc2, wc3, wc4 = st.columns(4)

with wc1:
    week_val = estimate_change(weight_data, "Weight", 7)
    st.metric("Week", "Not enough data" if week_val is None else f"{week_val:+.1f}")

with wc2:
    month_val = estimate_change(weight_data, "Weight", 30)
    st.metric("Month", "Not enough data" if month_val is None else f"{month_val:+.1f}")

with wc3:
   all_val = estimate_change(weight_data, "Weight", None)
   st.metric("All Time", "Not enough data" if all_val is None else f"{all_val:+.1f}")

with wc4:
    st.empty()

top_col1, top_col2, top_col3, top_col4 = st.columns(4)

with top_col1:
    if not weight_data.empty:
        st.metric("Current Weight", f"{weight_data['Weight'].iloc[-1]:.1f}", metric_delta_text(weight_data["Weight"].iloc[-1], weight_data["Weight"].iloc[0], ""))
    else:
        st.metric("Current Weight", "No data")

with top_col2:
    if not nutrition_data.empty:
        current_avg = nutrition_data["Calories"].tail(7).mean()
        previous_avg = nutrition_data["Calories"].tail(14).head(7).mean()
        st.metric("Calories Avg (7d)", f"{current_avg:.0f}", metric_delta_text(current_avg, previous_avg))
    else:
        st.metric("Calories Avg (7d)", "No data")

with top_col3:
    if not workout_data.empty:
        st.metric("Training Days", workout_data["Date"].dt.date.nunique())
    else:
        st.metric("Training Days", "No data")

with top_col4:
    if not workout_data.empty:
        recent_volume = workout_data[workout_data["Date"] >= (workout_data["Date"].max() - pd.Timedelta(days=6))]["Volume"].sum()
        st.metric("Volume (7d)", f"{recent_volume:,.0f}")
    else:
        st.metric("Volume (7d)", "No data")

if weight_error and weight_data.empty:
    st.warning(weight_error)
else:
    goal_weight = st.number_input(
        "Goal weight",
        min_value=0.0,
        value=float(weight_data["Weight"].iloc[-1]),
        step=1.0,
    )
    weight_projection = build_projection(weight_data, "Weight", periods=forecast_days)
    weight_fig = add_actual_and_projection(
        go.Figure(),
        weight_view if not weight_view.empty else weight_data,
        weight_projection,
        "Weight",
        "Actual Weight",
    )
    weight_fig.add_trace(
        go.Scatter(
            x=weight_data["Date"],
            y=weight_data["7 Day Avg"],
            mode="lines",
            name="7-Day Avg",
            line=dict(color="#1d4ed8", width=2),
        )
    )
    weight_fig.add_hline(y=goal_weight, line_dash="dot", line_color="#15803d", annotation_text="Goal")
    weight_fig.update_layout(title=f"Bodyweight Trend ({timeframe_label})", yaxis_title="Weight")
    st.plotly_chart(weight_fig, use_container_width=True)


st.subheader("Diet")
if nutrition_error and nutrition_data.empty:
    st.warning(nutrition_error)
else:
    diet_col1, diet_col2 = st.columns([2, 1])

    with diet_col1:
        calories_projection = build_projection(nutrition_data, "Calories", periods=forecast_days)
        calories_fig = add_actual_and_projection(
            go.Figure(),
            nutrition_view if not nutrition_view.empty else nutrition_data,
            calories_projection,
            "Calories",
            "Calories",
        )
        calories_fig.add_trace(
            go.Scatter(
                x=nutrition_data["Date"],
                y=nutrition_data["Calorie Avg"],
                mode="lines",
                name="7-Day Avg",
                line=dict(color="#9333ea", width=2),
            )
        )
        calories_fig.update_layout(title=f"Calories ({timeframe_label})", yaxis_title="Calories")
        st.plotly_chart(calories_fig, use_container_width=True)

    with diet_col2:
        latest_calories = nutrition_data["Calories"].iloc[-1]
        week_calorie_change = estimate_change(nutrition_data, "Calories", 7)
        month_calorie_change = estimate_change(nutrition_data, "Calories", 30)
        st.metric("Latest Calories", f"{latest_calories:.0f}")
        st.metric("Week Change", "Not enough data" if week_calorie_change is None else f"{week_calorie_change:+.0f}")
        st.metric("Month Change", "Not enough data" if month_calorie_change is None else f"{month_calorie_change:+.0f}")
        if "Protein" in nutrition_data.columns:
            st.metric("Protein Avg (7d)", f"{nutrition_data['Protein'].tail(7).mean():.0f} g")

    macro_source = nutrition_view if not nutrition_view.empty else nutrition_data
    macro_cols = [col for col in ["Protein", "Carbs", "Fat"] if col in macro_source.columns]
    if macro_cols:
        macro_fig = px.line(
            macro_source,
            x="Date",
            y=macro_cols,
            template="plotly_white",
            title=f"Macro Trends ({timeframe_label})",
        )
        macro_fig.update_layout(margin=dict(l=12, r=12, t=60, b=12), hovermode="x unified")
        st.plotly_chart(macro_fig, use_container_width=True)


st.subheader("Compound Lift Projections")
if workout_error and workout_data.empty:
    st.warning(workout_error)
else:
    summary = build_compound_summary(workout_data)
    if summary.empty:
        st.info("No matching bench press, squat, or deadlift entries were found in the workout export yet.")
    else:
        metric_cols = st.columns(max(len(summary), 1))
        for column, row in zip(metric_cols, summary.to_dict("records")):
            with column:
                delta_text = "Not enough data" if pd.isna(row["30-Day Projection"]) else f"{row['30-Day Projection'] - row['Current e1RM']:+.1f}"
                st.metric(row["Lift"], f"{row['Current e1RM']:.1f} e1RM", delta_text)

        chart_cols = st.columns(3)
        for chart_col, (lift_name, aliases) in zip(chart_cols, COMPOUND_MOVEMENTS.items()):
            with chart_col:
                movement_data = find_compound_data(workout_data, aliases)
                if movement_data.empty:
                    st.info(f"No {lift_name.lower()} data found.")
                    continue

                best_by_day = movement_data.groupby("Date", as_index=False)["Estimated 1RM"].max()
                lift_projection = build_projection(best_by_day, "Estimated 1RM", periods=forecast_days)
                lift_fig = add_actual_and_projection(go.Figure(), best_by_day, lift_projection, "Estimated 1RM", lift_name)
                lift_fig.update_layout(title=f"{lift_name} e1RM", yaxis_title="Estimated 1RM")
                st.plotly_chart(lift_fig, use_container_width=True)

        exercise_options = sorted(workout_data["Exercise"].dropna().unique())
        selected_exercise = st.selectbox("Exercise details", exercise_options)
        exercise_data = workout_data[workout_data["Exercise"] == selected_exercise].copy()

        detail_col1, detail_col2 = st.columns(2)
        with detail_col1:
            max_by_day = exercise_data.groupby("Date", as_index=False)["Weight"].max()
            detail_fig = px.line(
                max_by_day,
                x="Date",
                y="Weight",
                markers=True,
                template="plotly_white",
                title=f"{selected_exercise} Top Set Weight",
            )
            detail_fig.update_layout(margin=dict(l=12, r=12, t=60, b=12), hovermode="x unified")
            st.plotly_chart(detail_fig, use_container_width=True)

        with detail_col2:
            volume_by_day = exercise_data.groupby("Date", as_index=False)["Volume"].sum()
            volume_fig = px.bar(
                volume_by_day,
                x="Date",
                y="Volume",
                template="plotly_white",
                title=f"{selected_exercise} Volume",
            )
            volume_fig.update_layout(margin=dict(l=12, r=12, t=60, b=12))
            st.plotly_chart(volume_fig, use_container_width=True)

        st.dataframe(
            exercise_data.sort_values("Date", ascending=False).head(25),
            use_container_width=True,
        )


with st.expander("How the predictions work"):
    st.write(
        "These projections use a simple line fitted to your historical trend. They are useful for seeing direction and pace, not as guaranteed outcomes."
    )
    st.write(
        "For compound lifts, the dashboard estimates 1RM from your logged weight and reps, then projects that trend forward."
    )
