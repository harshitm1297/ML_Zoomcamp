"""Build pre-date pair features from the public Mixmosa CSV files."""

from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"
PROFILE_COLUMNS = (
    "age",
    "height_inches",
    "avg_physical_attractiveness_rating",
    "agreeableness",
    "conscientiousness",
    "extraversion",
    "neuroticism",
    "openness",
    "behavioral_health_score",
    "get_along_well_with_family_score",
    "num_children_wanted_score",
)
DIFFERENCE_COLUMNS = (
    "age",
    "height_inches",
    "agreeableness",
    "conscientiousness",
    "extraversion",
    "neuroticism",
    "openness",
    "num_children_wanted_score",
)
FEATURE_COLUMNS = (
    tuple(f"male_{column}" for column in PROFILE_COLUMNS)
    + tuple(f"female_{column}" for column in PROFILE_COLUMNS)
    + tuple(f"abs_diff_{column}" for column in DIFFERENCE_COLUMNS)
)
OUTCOME_COLUMNS = {
    "mutual_interest",
    "does_male_like_female",
    "does_female_like_male",
}


def read_source(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    interactions = pd.read_csv(data_dir / "interactions.csv")
    users = pd.read_csv(data_dir / "user_features.csv")
    return interactions, users


def pair_features(male: dict, female: dict) -> pd.DataFrame:
    """Create one inference row from two pre-date profile dictionaries."""
    row = {}
    for side, person in (("male", male), ("female", female)):
        for column in PROFILE_COLUMNS:
            value = person.get(column)
            row[f"{side}_{column}"] = pd.to_numeric(value, errors="coerce")
    for column in DIFFERENCE_COLUMNS:
        a, b = row[f"male_{column}"], row[f"female_{column}"]
        row[f"abs_diff_{column}"] = abs(a - b)
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def build_dataset(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Join pre-date profiles to outcomes; never include outcome columns in X."""
    interactions, users = read_source(data_dir)
    if users.user_id.duplicated().any():
        raise ValueError("user_features.csv has duplicate user_id values")
    if not interactions["mutual_interest"].isin([0, 1]).all():
        raise ValueError("mutual_interest must be binary")

    cols = ["user_id", *PROFILE_COLUMNS]
    male = users[cols].add_prefix("male_")
    female = users[cols].add_prefix("female_")
    joined = interactions.merge(male, on="male_user_id", how="left", validate="many_to_one")
    joined = joined.merge(female, on="female_user_id", how="left", validate="many_to_one")
    # A small number of recorded interactions have no corresponding feature row.
    # They cannot be used for a two-profile predictor.
    joined = joined.dropna(subset=["male_age", "female_age"]).copy()
    for column in DIFFERENCE_COLUMNS:
        joined[f"abs_diff_{column}"] = (
            joined[f"male_{column}"] - joined[f"female_{column}"]
        ).abs()
    return joined


def model_input(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, FEATURE_COLUMNS].copy()
