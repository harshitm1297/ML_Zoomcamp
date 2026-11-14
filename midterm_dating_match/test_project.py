"""Checks the feature contract and the saved model's inference path."""

import pickle
from pathlib import Path

import pandas as pd

from features import FEATURE_COLUMNS, OUTCOME_COLUMNS, PROFILE_COLUMNS, build_dataset, model_input, pair_features
from modeling import split_by_event


def test_features_match_api_record():
    frame = build_dataset()
    row = frame.iloc[0]
    male = {key: row[f"male_{key}"] for key in PROFILE_COLUMNS}
    female = {key: row[f"female_{key}"] for key in PROFILE_COLUMNS}
    api_row = pair_features(male, female).iloc[0]
    training_row = model_input(frame).iloc[0]
    pd.testing.assert_series_equal(api_row, training_row, check_names=False, check_dtype=False)
    assert not OUTCOME_COLUMNS.intersection(FEATURE_COLUMNS)


def test_event_split_and_artifact_prediction():
    frame = build_dataset()
    train, validation, test = split_by_event(frame)
    assert not set(train.event_id) & set(validation.event_id)
    assert not set(train.event_id) & set(test.event_id)
    assert not set(validation.event_id) & set(test.event_id)
    with Path(__file__).with_name("model.bin").open("rb") as f:
        artifact = pickle.load(f)
    prediction = artifact["model"].predict_proba(model_input(test.head(1)))[0, 1]
    assert 0 <= prediction <= 1
