# Import Path class to handle filesystem paths in a platform-independent way
from pathlib import Path

# Import commonly used Python libraries
import numpy as np            # For numerical operations, arrays, math functions
import pandas as pd           # For working with tables of data (DataFrames)
import plotly.express as px   # For simple plotting and charts
import plotly.graph_objects as go  # For advanced/custom plotting
import streamlit as st        # For building web apps easily

# Configure Streamlit page properties: title in browser tab and layout style
st.set_page_config(page_title="Gym Progress Dashboard", layout="wide")

# Default filenames for user's personal data CSV files
DEFAULT_FILES = {
    "weight": "weight.csv",       # File storing weight data
    "nutrition": "cronometer.csv",# File storing nutrition data
    "workouts": "strong.csv",     # File storing workout logs
}

# Example file paths to use if personal files are missing (Windows paths here)
EXAMPLE_FILES = {
    "weight": Path(r"C:\Users\xavie\OneDrive\Desktop\weightloss\weight.csv"),
    "nutrition": Path(r"C:\Users\xavie\OneDrive\Desktop\weightloss\cronometer.csv"),
    "workouts": Path(r"C:\Users\xavie\OneDrive\Desktop\weightloss\strong.csv"),
}

# Map of main compound exercises to different ways they might be written in CSVs
COMPOUND_MOVEMENTS = {
    "Bench Press": ["bench press", "bench press barbell", "flat barbell bench press"],
    "Squat": ["squat", "squat barbell", "back squat", "barbell squat"],
    "Deadlift": ["deadlift", "deadlift barbell", "conventional deadlift", "barbell deadlift"],
}

# Define timeframes in days for filtering data
TIMEFRAME_DAYS = {
    "Week": 7,
    "Month": 30,
    "All Time": None,  # None means no limit
}

# Filename for sleep tracking CSV
SLEEP_FILE = "sleep.csv"

# Function to load sleep data from CSV if it exists, otherwise return empty DataFrame
def load_sleep() -> pd.DataFrame:
    path = Path(SLEEP_FILE)  # Convert filename to Path object
    if path.exists():        # Check if file exists
        return pd.read_csv(path, parse_dates=["Date"])  # Load CSV and parse Date column
    return pd.DataFrame(columns=["Date", "Hours"])  # Return empty table with correct columns

# Function to save a new sleep entry or update an existing one
def save_sleep(date: str, hours: float) -> None:
    df = load_sleep()                     # Load existing sleep data
    df["Date"] = pd.to_datetime(df["Date"])  # Make sure Date column is datetime type
    new_row = pd.DataFrame([{"Date": pd.to_datetime(date), "Hours": hours}])  # Create new entry
    # Remove any old entry for the same date and add the new one
    df = pd.concat([df[df["Date"] != pd.to_datetime(date)], new_row])
    df.sort_values("Date").to_csv(SLEEP_FILE, index=False)  # Sort by date and save CSV

# Function to clean DataFrame column names: remove extra spaces
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()  # Make a copy to avoid modifying original
    cleaned.columns = [str(col).strip() for col in cleaned.columns]  # Remove spaces from column names
    return cleaned

# Function to normalize text for easier matching: lowercase, remove spaces/underscores/special characters
def normalize_name(name: str) -> str:
    return (
        str(name)           # Convert to string
        .strip()            # Remove leading/trailing whitespace
        .lower()            # Make lowercase
        .replace("_", " ")  # Replace underscores with space
        .replace("-", " ")  # Replace hyphens with space
        .replace("(", "")   # Remove left parentheses
        .replace(")", "")   # Remove right parentheses
        .replace(".", "")   # Remove periods
    )

# Function to find a matching column in a DataFrame from a list of possible names
def find_column(df: pd.DataFrame, options: list[str]) -> str | None:
    normalized = {normalize_name(col): col for col in df.columns}  # Map normalized column names to original
    for option in options:
        if normalize_name(option) in normalized:  # Check if normalized option exists
            return normalized[normalize_name(option)]  # Return original column name
    return None  # Return None if no match found

# Function to load a CSV from upload or fallback paths
def load_csv(upload, fallback_name: str) -> pd.DataFrame | None:
    if upload is not None:  # If user uploaded a file, use it
        return clean_columns(pd.read_csv(upload))

    fallback_path = Path(fallback_name)  # Convert fallback filename to Path
    if fallback_path.exists():           # If fallback file exists, use it
        return clean_columns(pd.read_csv(fallback_path))

    # Otherwise, use example file
    example_path = EXAMPLE_FILES.get(
        "weight" if fallback_name == DEFAULT_FILES["weight"] else
        "nutrition" if fallback_name == DEFAULT_FILES["nutrition"] else
        "workouts"
    )
    if example_path and example_path.exists():
        return clean_columns(pd.read_csv(example_path))

    return None  # If no file found, return None


# Function to prepare weight data for analysis
def prepare_weight(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    # Find columns for date and weight using possible names
    date_col = find_column(df, ["Date", "Day"])
    weight_col = find_column(df, ["Weight", "Body Weight", "Weight (lb)", "Weight (kg)"])

    # If either is missing, return empty DataFrame with error message
    if not date_col or not weight_col:
        return pd.DataFrame(), "Weight data needs a date column and a weight column."

    prepared = df[[date_col, weight_col]].copy()  # Keep only date and weight columns
    prepared.columns = ["Date", "Weight"]         # Rename columns to standard names
    prepared["Date"] = pd.to_datetime(prepared["Date"], errors="coerce")  # Convert to datetime
    prepared["Weight"] = pd.to_numeric(prepared["Weight"], errors="coerce")  # Convert to numbers
    prepared = prepared.dropna().sort_values("Date")  # Remove invalid rows and sort by date

    if prepared.empty:  # If no valid data
        return pd.DataFrame(), "Weight data could not be parsed."

    # Aggregate by day and compute daily mean
    daily = (
        prepared.groupby(prepared["Date"].dt.floor("D"), as_index=False)["Weight"]
        .mean()
        .rename(columns={"Date": "Date"})
    )
    daily["7 Day Avg"] = daily["Weight"].rolling(7, min_periods=1).mean()  # 7-day moving average
    return daily, None  # Return prepared data and no error

# Function to prepare nutrition data for analysis
def prepare_nutrition(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    # Find columns for date and nutrition metrics
    date_col = find_column(df, ["Date", "Day"])
    calorie_col = find_column(df, ["Calories", "Energy (kcal)", "Energy"])
    protein_col = find_column(df, ["Protein (g)", "Protein", "Protein g"])
    carb_col = find_column(df, ["Carbs (g)", "Net Carbs (g)", "Carbohydrates (g)", "Carbs"])
    fat_col = find_column(df, ["Fat (g)", "Fat"])

    # Require at least date and calories
    if not date_col or not calorie_col:
        return pd.DataFrame(), "Nutrition data needs at least date and calories columns."

    columns = [date_col, calorie_col]  # Start with required columns
    rename_map = {date_col: "Date", calorie_col: "Calories"}  # Map to standard names

    # Add optional columns if they exist
    if protein_col:
        columns.append(protein_col)
        rename_map[protein_col] = "Protein"
    if carb_col:
        columns.append(carb_col)
        rename_map[carb_col] = "Carbs"
    if fat_col:
        columns.append(fat_col)
        rename_map[fat_col] = "Fat"

    prepared = df[columns].copy().rename(columns=rename_map)  # Keep and rename columns
    prepared["Date"] = pd.to_datetime(prepared["Date"], errors="coerce")  # Convert dates

    # Convert all numeric columns to numbers
    for col in ["Calories", "Protein", "Carbs", "Fat"]:
        if col in prepared.columns:
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce")

    prepared = prepared.dropna(subset=["Date", "Calories"]).sort_values("Date")  # Drop invalid rows

    if prepared.empty:  # If no valid data
        return pd.DataFrame(), "Nutrition data could not be parsed."

    # Aggregate by day and calculate 7-day averages
    daily = prepared.groupby(prepared["Date"].dt.floor("D"), as_index=False).mean(numeric_only=True)
    daily["Calorie Avg"] = daily["Calories"].rolling(7, min_periods=1).mean()
    if "Protein" in daily.columns:
        daily["Protein Avg"] = daily["Protein"].rolling(7, min_periods=1).mean()
    return daily, None

# Function to estimate 1-rep max (1RM) based on weight lifted and reps
def estimate_1rm(weight: float, reps: float) -> float:
    if reps == 1:  # If only 1 rep, 1RM is the weight itself
        return weight
    elif reps <= 10:  # For 2-10 reps, use Epley formula approximation
        return weight * 36 / (37 - reps)
    else:  # For >10 reps, use linear approximation
        return weight * (1 + reps / 30)

# Function to prepare workout data for analysis
def prepare_workouts(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    # Find required columns
    date_col = find_column(df, ["Date"])
    exercise_col = find_column(df, ["Exercise Name", "Exercise"])
    weight_col = find_column(df, ["Weight", "Weight (lb)", "Weight (kg)"])
    reps_col = find_column(df, ["Reps", "Rep"])
    set_order_col = find_column(df, ["SetOrder"])

    # Require essential columns
    if not date_col or not exercise_col or not weight_col:
        return pd.DataFrame(), "Workout data needs date, exercise, and weight columns."

    columns = [date_col, exercise_col, weight_col]
    rename_map = {
        date_col: "Date",
        exercise_col: "Exercise",
        weight_col: "Weight",
    }

    # Add optional columns if they exist
    if reps_col:
        columns.append(reps_col)
        rename_map[reps_col] = "Reps"
    if set_order_col:
        columns.append(set_order_col)
        rename_map[set_order_col] = "SetOrder"

    prepared = df[columns].copy().rename(columns=rename_map)
    prepared["Date"] = pd.to_datetime(prepared["Date"], errors="coerce")  # Convert dates
    prepared["Weight"] = pd.to_numeric(prepared["Weight"], errors="coerce")  # Convert weights
    prepared["Reps"] = pd.to_numeric(prepared.get("Reps", 1), errors="coerce").fillna(1)  # Convert reps, default 1
    if "SetOrder" in prepared.columns:
        prepared["SetOrder"] = pd.to_numeric(prepared["SetOrder"], errors="coerce")  # Convert set order if exists

    # Drop rows with missing required data and sort by date
    prepared = prepared.dropna(subset=["Date", "Exercise", "Weight"]).sort_values("Date")
    prepared["Exercise"] = prepared["Exercise"].astype(str).str.strip()  # Clean exercise names
    prepared = prepared[prepared["Weight"] >= 20]  # Ignore unrealistically low weights
    prepared["Volume"] = prepared["Weight"] * prepared["Reps"]  # Calculate volume per set
    # Calculate estimated 1RM for sets with <=10 reps
    prepared["Estimated 1RM"] = prepared.apply(lambda r: estimate_1rm(r["Weight"], r["Reps"]) if r["Reps"] <= 10 else np.nan,
    axis=1)

    if prepared.empty:  # If no valid data
        return pd.DataFrame(), "Workout data could not be parsed."

    return prepared, None

# Function to filter data for the last N days
def filter_timeframe(df: pd.DataFrame, days: int | None) -> pd.DataFrame:
    if df.empty or days is None:  # If no data or no limit, return all
        return df
    cutoff = df["Date"].max() - pd.Timedelta(days=days - 1)  # Calculate earliest date to keep
    return df[df["Date"] >= cutoff].copy()  # Return filtered data

# Function to create a text showing difference between current and previous value
def metric_delta_text(current: float, previous: float, suffix: str = "") -> str:
    if pd.isna(previous):  # If previous value is missing
        return "Not enough data"
    return f"{current - previous:+.1f}{suffix}"  # Show delta with + or - sign

# Function to estimate change in a value column over a timeframe
def estimate_change(df: pd.DataFrame, value_col: str, days: int | None) -> float | None:
    if df.empty:  # No data, return None
        return None
    sliced = filter_timeframe(df, days)  # Filter data to timeframe
    if len(sliced) < 2:  # Need at least 2 points to compute change
        return None
    return float(sliced[value_col].iloc[-1] - sliced[value_col].iloc[0])  # Compute difference


def build_projection(df: pd.DataFrame, value_col: str, periods: int, freq: str = "D", floor_slope: bool = False) -> pd.DataFrame:
    if df.empty or len(df) < 2:
        return pd.DataFrame()

    ordered = df.sort_values("Date").copy()
    ordered["DaysFromStart"] = (ordered["Date"] - ordered["Date"].min()).dt.days.astype(float)

    if ordered["DaysFromStart"].nunique() < 2:
        return pd.DataFrame()

    slope, intercept = np.polyfit(ordered["DaysFromStart"], ordered[value_col], 1)
    if floor_slope:
        slope = max(slope, 0)  # only used for lifts — never project downward

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


# Function to build a simple linear projection into the future
def build_projection(df: pd.DataFrame, value_col: str, periods: int, freq: str = "D", floor_slope: bool = False) -> pd.DataFrame:
    # Return empty DataFrame if input is empty or has fewer than 2 rows
    if df.empty or len(df) < 2:
        return pd.DataFrame()

    ordered = df.sort_values("Date").copy()  # Sort data by date
    # Calculate number of days since first date for each row
    ordered["DaysFromStart"] = (ordered["Date"] - ordered["Date"].min()).dt.days.astype(float)

    # If all rows have the same day offset, cannot fit a line
    if ordered["DaysFromStart"].nunique() < 2:
        return pd.DataFrame()

    # Fit a linear regression line (slope and intercept) for value vs days
    slope, intercept = np.polyfit(ordered["DaysFromStart"], ordered[value_col], 1)
    if floor_slope:
        slope = max(slope, 0)  # If floor_slope is True, do not allow negative slope

    # Generate future day offsets for the projection
    future_offsets = np.arange(1, periods + 1, dtype=float)
    last_offset = ordered["DaysFromStart"].iloc[-1]  # Last observed day
    future_days = last_offset + future_offsets  # Compute future days for projection
    # Generate future dates starting from the day after last date
    future_dates = pd.date_range(ordered["Date"].max() + pd.Timedelta(days=1), periods=periods, freq=freq)

    # Create a DataFrame with projected values
    projected = pd.DataFrame(
        {
            "Date": future_dates,                   # Dates of future points
            value_col: intercept + slope * future_days,  # Linear projection values
            "Series": "Projection",                 # Label series as "Projection"
        }
    )
    return projected  # Return projected DataFrame

# Function to add actual data and projected data to a Plotly figure
def add_actual_and_projection(fig: go.Figure, actual_df: pd.DataFrame, projection_df: pd.DataFrame, value_col: str, actual_name: str) -> go.Figure:
    # Add trace for actual data points
    fig.add_trace(
        go.Scatter(
            x=actual_df["Date"],            # X-axis: dates
            y=actual_df[value_col],         # Y-axis: actual values
            mode="lines+markers",           # Show both line and markers
            name=actual_name,               # Name of actual data in legend
            line=dict(color="#0f766e", width=3),  # Style: dark teal line, width 3
        )
    )
    # Add trace for projected data if it exists
    if not projection_df.empty:
        fig.add_trace(
            go.Scatter(
                x=projection_df["Date"],            # X-axis: future dates
                y=projection_df[value_col],         # Y-axis: projected values
                mode="lines",                       # Show only line
                name="Projection",                  # Legend name
                line=dict(color="#f97316", width=3, dash="dash"),  # Orange dashed line
            )
        )
    # Update figure layout for appearance and hover behavior
    fig.update_layout(
        template="plotly_white",  # White background template
        legend_title_text="",     # Remove legend title
        margin=dict(l=12, r=12, t=60, b=12),  # Margins around plot
        hovermode="x unified",    # Show all y-values at same x when hovering
    )
    return fig  # Return updated figure


# Function to find all rows in workouts matching a list of compound exercise aliases
def find_compound_data(workouts: pd.DataFrame, aliases: list[str]) -> pd.DataFrame:
    normalized_aliases = {normalize_name(alias) for alias in aliases}  # Normalize all aliases for comparison
    normalized_exercise = workouts["Exercise"].map(normalize_name)    # Normalize exercise names in data
    # Create a boolean mask where any alias matches the exercise name
    mask = normalized_exercise.apply(
        lambda exercise_name: any(alias in exercise_name for alias in normalized_aliases)
    )
    return workouts[mask].copy()  # Return filtered DataFrame with only matching exercises

# Function to remove outliers from a numeric column using rolling mean and standard deviation
def filter_outliers(df: pd.DataFrame, col: str) -> pd.DataFrame:
    # Compute rolling mean over 10 rows (centered) with at least 3 points
    mean = df[col].rolling(10, min_periods=3, center=True).mean()
    # Compute rolling std dev, fill NaN with overall std dev
    std  = df[col].rolling(10, min_periods=3, center=True).std().fillna(df[col].std())
    # Keep only rows where value is within 2 standard deviations from rolling mean
    return df[(df[col] - mean).abs() <= 2 * std]

# Function to detect fatigue based on recent sleep data
def detect_sleep_fatigue(sleep_df: pd.DataFrame) -> dict | None:
    if sleep_df.empty or len(sleep_df) < 3:  # Need at least 3 records to detect trend
        return None

    sleep_df = sleep_df.copy()  # Avoid modifying original
    sleep_df["Date"] = pd.to_datetime(sleep_df["Date"])  # Ensure Date column is datetime
    sleep_df = sleep_df.sort_values("Date")  # Sort by date

    recent_avg = sleep_df["Hours"].tail(3).mean()  # Average of last 3 days
    baseline   = sleep_df["Hours"].tail(14).mean()  # Average of last 14 days
    deficit    = recent_avg - baseline  # Difference from baseline

    # Performance impact logic based on sleep research
    if recent_avg < 6:
        perf_impact = -3.0
        status = "High Fatigue — under 6h recently"
    elif recent_avg < 7:
        perf_impact = -2.0
        status = "Elevated Fatigue — sleeping less than usual"
    elif recent_avg < 7.5:
        perf_impact = -1.0
        status = "Moderate — aim for 7.5h+"
    elif recent_avg >= 8.5:
        perf_impact = +1.5
        status = "Well Rested — slight performance boost expected"
    else:
        perf_impact = 0.0
        status = "Well Rested"

    # Apply additional penalty if sleep is trending worse than baseline
    if deficit < -1.5:
        perf_impact -= 1.0

    # Return dictionary summarizing sleep fatigue status and metrics
    return {
        "status": status,
        "recent_avg": recent_avg,
        "baseline": baseline,
        "deficit": deficit,
        "perf_impact": perf_impact,
    }

# Function to build a summary table for compound lifts
def build_compound_summary(workouts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | pd.Timestamp]] = []  # List to collect summary rows
    for label, aliases in COMPOUND_MOVEMENTS.items():       # Loop over each compound lift
        movement_data = find_compound_data(workouts, aliases)  # Get all relevant rows
        if movement_data.empty:
            continue  # Skip if no data for this lift

        # Compute the best estimated 1RM per day
        best_by_day = movement_data.groupby("Date", as_index=False)["Estimated 1RM"].max()
        # Filter outliers in 1RM data
        best_by_day = filter_outliers(best_by_day, "Estimated 1RM")
        # Keep only last 60 days for projection
        recent_for_projection = filter_timeframe(best_by_day, 60)
        # Build 30-day projection of estimated 1RM
        forecast = build_projection(recent_for_projection, "Estimated 1RM", periods=30, floor_slope=True)
        current = best_by_day["Estimated 1RM"].iloc[-1]  # Current best 1RM
        projected_raw = forecast["Estimated 1RM"].iloc[-1] if not forecast.empty else np.nan
        # Ensure projection never goes below current value
        projected_value = max(projected_raw, current) if not pd.isna(projected_raw) else np.nan

        # Append summary row for this lift
        rows.append(
            {
                "Lift": label,
                "Current e1RM": recent_for_projection["Estimated 1RM"].max(),  # Current max
                "30-Day Projection": projected_value,                             # Projected max
                "Sessions": movement_data["Date"].dt.date.nunique(),             # Number of workout days
            }
        )
    return pd.DataFrame(rows)  # Convert all rows to a DataFrame


# Add custom CSS to style the Streamlit app
st.markdown(
    """
    <style>
    /* General app background gradients */
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(20, 184, 166, 0.12), transparent 32%),
            radial-gradient(circle at top right, rgba(249, 115, 22, 0.12), transparent 28%);
    }

    /* Light theme background adjustments */
    [data-theme="light"] .stApp {
        background-image:
            radial-gradient(circle at top left, rgba(20, 184, 166, 0.12), transparent 32%),
            radial-gradient(circle at top right, rgba(249, 115, 22, 0.12), transparent 28%),
            linear-gradient(180deg, #f4fbf9 0%, #fffaf4 100%);
    }

    /* Dark theme background adjustments */
    [data-theme="dark"] .stApp {
        background-image:
            radial-gradient(circle at top left, rgba(20, 184, 166, 0.08), transparent 32%),
            radial-gradient(circle at top right, rgba(249, 115, 22, 0.08), transparent 28%),
            linear-gradient(180deg, #0f1117 0%, #1a1a2e 100%);
    }

    /* Style metric cards */
    div[data-testid="stMetric"] {
        border-radius: 18px;                   /* Rounded corners */
        padding: 0.75rem 1rem;                 /* Inner padding */
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.05); /* Soft shadow */
    }

    /* Light theme metric card background */
    [data-theme="light"] div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(15, 118, 110, 0.12);
    }

    /* Dark theme metric card background */
    [data-theme="dark"] div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(20, 184, 166, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,  # Allow raw HTML/CSS
)

# Set the main title of the app
st.title("Gym Progress Dashboard")
# Subtitle describing the purpose of the dashboard
st.caption("See your lifting, diet, and bodyweight trends together, with simple projections based on your historical rate of change.")

# Sidebar layout for file uploads and settings
with st.sidebar:
    st.header("Data Sources")  # Section header
    st.write("Upload files here or place them in the project folder.")
    # File upload widgets for weight, nutrition, and workout CSV files
    weight_upload = st.file_uploader("Weight CSV", type="csv", key="weight")
    nutrition_upload = st.file_uploader("Cronometer CSV", type="csv", key="nutrition")
    workouts_upload = st.file_uploader("Strong CSV", type="csv", key="workouts")
    # Dropdown to select default chart range (week, month, all time)
    timeframe_label = st.selectbox("Default chart range", list(TIMEFRAME_DAYS.keys()), index=1)
    # Slider to select how many days to project into the future
    forecast_days = st.slider("Projection length (days)", min_value=14, max_value=90, value=30, step=7)
    # Captions explaining default filenames
    st.caption("Default filenames: `weight.csv`, `cronometer.csv`, and `strong.csv`.")
    st.caption("Your sample desktop exports are also used automatically when available.")

# Load CSV data using previously defined load_csv function
weight_raw = load_csv(weight_upload, DEFAULT_FILES["weight"])
nutrition_raw = load_csv(nutrition_upload, DEFAULT_FILES["nutrition"])
workouts_raw = load_csv(workouts_upload, DEFAULT_FILES["workouts"])

# Prepare the loaded data for analysis (parsing dates, numeric conversion, cleaning)
weight_data, weight_error = prepare_weight(weight_raw) if weight_raw is not None else (pd.DataFrame(), "Weight file not found.")
nutrition_data, nutrition_error = prepare_nutrition(nutrition_raw) if nutrition_raw is not None else (pd.DataFrame(), "Nutrition file not found.")
workout_data, workout_error = prepare_workouts(workouts_raw) if workouts_raw is not None else (pd.DataFrame(), "Workout file not found.")

# Filter data based on selected timeframe from sidebar
selected_days = TIMEFRAME_DAYS[timeframe_label]
weight_view = filter_timeframe(weight_data, selected_days)
nutrition_view = filter_timeframe(nutrition_data, selected_days)

# Subheader for the weight change section
st.subheader("Weight Change")
# Create 4 columns to display different metrics
wc1, wc2, wc3, wc4 = st.columns(4)

with wc1:
    week_val = estimate_change(weight_data, "Weight", 7)  # Compute 7-day change
    st.metric("Week", "Not enough data" if week_val is None else f"{week_val:+.1f}")

with wc2:
    month_val = estimate_change(weight_data, "Weight", 30)  # Compute 30-day change
    st.metric("Month", "Not enough data" if month_val is None else f"{month_val:+.1f}")

with wc3:
   all_val = estimate_change(weight_data, "Weight", None)  # Compute all-time change
   st.metric("All Time", "Not enough data" if all_val is None else f"{all_val:+.1f}")

with wc4:
    st.empty()  # Placeholder column (empty)

# Create 4 columns to display top-level metrics for weight, nutrition, and workouts
top_col1, top_col2, top_col3, top_col4 = st.columns(4)

with top_col1:
    if not weight_data.empty:
        # Show current weight and delta from first recorded weight
        st.metric(
            "Current Weight",
            f"{weight_data['Weight'].iloc[-1]:.1f}",
            metric_delta_text(weight_data["Weight"].iloc[-1], weight_data["Weight"].iloc[0], "")
        )
    else:
        st.metric("Current Weight", "No data")  # Display placeholder if no data

with top_col2:
    if not nutrition_data.empty:
        # Calculate 7-day average calories and compare to previous 7-day period
        current_avg = nutrition_data["Calories"].tail(7).mean()
        previous_avg = nutrition_data["Calories"].tail(14).head(7).mean()
        st.metric(
            "Calories Avg (7d)",
            f"{current_avg:.0f}",
            metric_delta_text(current_avg, previous_avg)
        )
    else:
        st.metric("Calories Avg (7d)", "No data")

with top_col3:
    if not workout_data.empty:
        # Count number of unique training days
        st.metric("Training Days", workout_data["Date"].dt.date.nunique())
    else:
        st.metric("Training Days", "No data")

with top_col4:
    if not workout_data.empty:
        # Sum the training volume from last 7 days
        recent_volume = workout_data[
            workout_data["Date"] >= (workout_data["Date"].max() - pd.Timedelta(days=6))
        ]["Volume"].sum()
        st.metric("Volume (7d)", f"{recent_volume:,.0f}")
    else:
        st.metric("Volume (7d)", "No data")

# Show warning if weight data couldn't be loaded
if weight_error and weight_data.empty:
    st.warning(weight_error)
else:
    # Input box for goal weight, defaulting to current weight
    goal_weight = st.number_input(
        "Goal weight",
        min_value=0.0,
        value=float(weight_data["Weight"].iloc[-1]),
        step=1.0,
    )
    # Build future weight projection for the specified forecast days
    weight_projection = build_projection(weight_data, "Weight", periods=forecast_days)
    # Create plotly figure combining actual weight and projection
    weight_fig = add_actual_and_projection(
        go.Figure(),
        weight_view if not weight_view.empty else weight_data,
        weight_projection,
        "Weight",
        "Actual Weight",
    )
    # Add line for 7-day rolling average
    weight_fig.add_trace(
        go.Scatter(
            x=weight_data["Date"],
            y=weight_data["7 Day Avg"],
            mode="lines",
            name="7-Day Avg",
            line=dict(color="#1d4ed8", width=2),
        )
    )
    # Add horizontal line for goal weight
    weight_fig.add_hline(y=goal_weight, line_dash="dot", line_color="#15803d", annotation_text="Goal")
    weight_fig.update_layout(title=f"Bodyweight Trend ({timeframe_label})", yaxis_title="Weight")
    st.plotly_chart(weight_fig, use_container_width=True)

# Diet section
st.subheader("Diet")
if nutrition_error and nutrition_data.empty:
    st.warning(nutrition_error)
else:
    # Create two columns: main chart and summary metrics
    diet_col1, diet_col2 = st.columns([2, 1])

    with diet_col1:
        # Build calorie projection
        calories_projection = build_projection(nutrition_data, "Calories", periods=forecast_days)
        # Plot actual calories and projection
        calories_fig = add_actual_and_projection(
            go.Figure(),
            nutrition_view if not nutrition_view.empty else nutrition_data,
            calories_projection,
            "Calories",
            "Calories",
        )
        # Add 7-day rolling average line
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
        # Show latest calories and weekly/monthly changes
        latest_calories = nutrition_data["Calories"].iloc[-1]
        week_calorie_change = estimate_change(nutrition_data, "Calories", 7)
        month_calorie_change = estimate_change(nutrition_data, "Calories", 30)
        st.metric("Latest Calories", f"{latest_calories:.0f}")
        st.metric("Week Change", "Not enough data" if week_calorie_change is None else f"{week_calorie_change:+.0f}")
        st.metric("Month Change", "Not enough data" if month_calorie_change is None else f"{month_calorie_change:+.0f}")
        # Show 7-day average protein if column exists
        if "Protein" in nutrition_data.columns:
            st.metric("Protein Avg (7d)", f"{nutrition_data['Protein'].tail(7).mean():.0f} g")

    # Macro trends chart (Protein, Carbs, Fat) if columns exist
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

# Sleep section header
st.subheader("Sleep")
# Load sleep data from CSV
sleep_data = load_sleep()

# Create two columns: left for input, right for charts
sleep_col1, sleep_col2 = st.columns([1, 2])

with sleep_col1:
    # Input for sleep date and hours
    sleep_date = st.date_input("Date", value=pd.Timestamp.today().date())
    sleep_hours = st.number_input("Hours slept", min_value=0.0, max_value=24.0, value=8.0, step=0.5)
    # Save button: store the sleep data and reload CSV
    if st.button("Save"):
        save_sleep(str(sleep_date), sleep_hours)
        st.success(f"Saved {sleep_hours}h for {sleep_date}")
        sleep_data = load_sleep()

with sleep_col2:
    if not sleep_data.empty:
        # Show 7-day average sleep
        avg_7d = sleep_data.tail(7)["Hours"].mean()
        st.metric("7-Day Avg Sleep", f"{avg_7d:.1f} hrs")
        # Plot sleep trend line chart
        sleep_fig = px.line(
            sleep_data.sort_values("Date"),
            x="Date",
            y="Hours",
            markers=True,  # show data points
            template="plotly_white",
            title="Sleep Trend",
        )
        # Add 8-hour goal reference line
        sleep_fig.add_hline(y=8, line_dash="dot", line_color="#15803d", annotation_text="8h goal")
        sleep_fig.update_layout(margin=dict(l=12, r=12, t=60, b=12))
        st.plotly_chart(sleep_fig, use_container_width=True)

# Sleep fatigue section
st.subheader("Sleep Fatigue")
# Analyze sleep data for fatigue
fatigue_from_sleep = detect_sleep_fatigue(sleep_data)
if fatigue_from_sleep is None:
    # Not enough data warning
    st.info("Log at least 3 nights of sleep to see fatigue status.")
else:
    # Display metrics in 4 columns
    sf1, sf2, sf3, sf4 = st.columns(4)
    with sf1:
        st.metric("Status", fatigue_from_sleep["status"])
    with sf2:
        st.metric("Avg (Last 3 nights)", f"{fatigue_from_sleep['recent_avg']:.1f} hrs")
    with sf3:
        st.metric("vs 2-Week Baseline", f"{fatigue_from_sleep['deficit']:+.1f} hrs")
    with sf4:
        # Show estimated impact on today's lift
        impact = fatigue_from_sleep["perf_impact"]
        label = "Today's Lift Impact"
        if impact > 0:
            st.metric(label, f"+{impact:.1f}%", "above normal")
        elif impact < 0:
            st.metric(label, f"{impact:.1f}%", "below normal")
        else:
            st.metric(label, "±0%", "no impact expected")

# Compound lift projections section
st.subheader("Compound Lift Projections")
if workout_error and workout_data.empty:
    st.warning(workout_error)  # Show warning if workout data missing
else:
    summary = build_compound_summary(workout_data)
    if summary.empty:
        st.info("No matching bench press, squat, or deadlift entries were found in the workout export yet.")
    else:
        # Display top-level e1RM metrics in columns
        metric_cols = st.columns(max(len(summary), 1))
        for column, row in zip(metric_cols, summary.to_dict("records")):
            with column:
                # Show delta between current e1RM and 30-day projection
                delta_text = "Not enough data" if pd.isna(row["30-Day Projection"]) else f"{row['30-Day Projection'] - row['Current e1RM']:+.1f}"
                st.metric(row["Lift"], f"{row['Current e1RM']:.1f} e1RM", delta_text)

        # Display charts for bench press, squat, deadlift
        chart_cols = st.columns(3)
        for chart_col, (lift_name, aliases) in zip(chart_cols, COMPOUND_MOVEMENTS.items()):
            with chart_col:
                movement_data = find_compound_data(workout_data, aliases)
                if movement_data.empty:
                    st.info(f"No {lift_name.lower()} data found.")
                    continue

                # Group by date to get best estimated 1RM
                best_by_day = movement_data.groupby("Date", as_index=False)["Estimated 1RM"].max()
                best_by_day = filter_outliers(best_by_day, "Estimated 1RM")  # Remove outliers
                recent_for_projection = filter_timeframe(best_by_day, 60)  # last 60 days only
                # Build projection for next N days
                lift_projection = build_projection(recent_for_projection, "Estimated 1RM", periods=forecast_days, floor_slope=True)
                # Plot actual vs projected
                lift_fig = add_actual_and_projection(go.Figure(), best_by_day, lift_projection, "Estimated 1RM", lift_name)
                lift_fig.update_layout(title=f"{lift_name} e1RM", yaxis_title="Estimated 1RM")
                st.plotly_chart(lift_fig, use_container_width=True)

        # Select a specific exercise for detailed view
        exercise_options = sorted(workout_data["Exercise"].dropna().unique())
        selected_exercise = st.selectbox("Exercise details", exercise_options)
        exercise_data = workout_data[workout_data["Exercise"] == selected_exercise].copy()

        # Two-column layout for top set weight and volume
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

        # Show raw table of recent 25 exercise entries
        st.dataframe(
            exercise_data.sort_values("Date", ascending=False).head(25),
            use_container_width=True,
        )

# Add an expander section to explain how projections are calculated
with st.expander("How the predictions work"):
    # General explanation of projections
    st.write(
        "These projections use a simple line fitted to your historical trend. "
        "They are useful for seeing direction and pace, not as guaranteed outcomes."
    )
    # Specific explanation for compound lifts
    st.write(
        "For compound lifts, the dashboard estimates 1RM from your logged weight and reps, "
        "then projects that trend forward."
    )
