"""Generate the project notebook from executable, reviewable cells."""

from pathlib import Path

import nbformat as nbf


cells = [
    nbf.v4.new_markdown_cell("""# Predicting mutual interest after a first date

**Problem.** Given two Mixmosa participants' information available before an eight-minute date, estimate whether both will say they want to meet again. An event organizer could use the score to prioritize introductions. The score is not a measure of long-term compatibility.

**Data.** The [Mixmosa public repository](https://github.com/dankras/mixmosa-anonymized-data) contains anonymized `interactions.csv` and `user_features.csv`. Its README describes 2,961 dates overall; the published `interactions.csv` has 1,895 rows, of which seven have no corresponding profile in `user_features.csv` and are removed here. The target is `mutual_interest`.

**Prediction time.** Inputs are pre-date profile and questionnaire features. The participants' date decisions, `mutual_interest`, IDs, swipes, and event feedback are excluded from predictors. This dataset covers in-person events with male/female pairs, so results should not be generalized to all dating-app interactions.
"""),
    nbf.v4.new_code_cell("""from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.inspection import permutation_importance

from features import DATA_DIR, FEATURE_COLUMNS, PROFILE_COLUMNS, build_dataset, model_input, read_source
from modeling import compare_models, make_model, score, split_by_event, split_summary
from train import train as run_training

pd.set_option('display.max_columns', 50)
interactions, users = read_source(DATA_DIR)
print('Interactions:', interactions.shape, 'Profiles:', users.shape)
print('Outcome counts:', interactions.mutual_interest.value_counts().to_dict())
display(interactions.drop(columns=['male_user_id', 'female_user_id']).head())"""),
    nbf.v4.new_markdown_cell("""## Data cleaning and feature construction

`build_dataset` joins each interaction to the male and female profile, drops seven interactions without a profile, and creates absolute differences in age, height, personality, and stated children preference. The feature list is explicit, so outcome columns cannot silently enter the model. We keep incomplete questionnaire rows; median imputation and missing-value indicators happen **inside each fitted model pipeline**.
"""),
    nbf.v4.new_code_cell("""frame = build_dataset()
print('Usable rows:', len(frame))
print('Events:', frame.event_id.nunique())
print('Positive rate:', round(frame.mutual_interest.mean(), 4))
print('Duplicate interaction pairs:', frame.duplicated(['event_id', 'male_user_id', 'female_user_id']).sum())
assert not {'mutual_interest', 'does_male_like_female', 'does_female_like_male'} & set(FEATURE_COLUMNS)
display(frame[['event_id', 'male_age', 'female_age', 'abs_diff_age', 'mutual_interest']].head())"""),
    nbf.v4.new_markdown_cell("""## Exploratory analysis

First inspect class balance, feature ranges, and missingness. Missing questionnaire answers are substantial and may be related to who completed the questionnaire, so missingness itself can be informative. The plots are descriptive; they do not establish causality.
"""),
    nbf.v4.new_code_cell("""summary = pd.DataFrame({
    'dtype': model_input(frame).dtypes.astype(str),
    'missing_fraction': model_input(frame).isna().mean(),
    'minimum': model_input(frame).min(),
    'median': model_input(frame).median(),
    'maximum': model_input(frame).max(),
}).sort_values('missing_fraction', ascending=False)
display(summary)
display(frame.groupby('mutual_interest')[['male_age', 'female_age', 'abs_diff_age']].median())

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
frame.mutual_interest.value_counts().sort_index().plot.bar(ax=axes[0], title='Target counts')
summary.missing_fraction.head(15).sort_values().plot.barh(ax=axes[1], title='Most missing features')
frame.groupby('event_id').mutual_interest.mean().plot.hist(ax=axes[2], bins=10, title='Positive rate by event')
plt.tight_layout()
plt.show()"""),
    nbf.v4.new_code_cell("""age_bins = pd.cut(frame.abs_diff_age, bins=[-0.1, 2, 5, 10, 100], labels=['0-2', '3-5', '6-10', '11+'])
age_analysis = frame.groupby(age_bins, observed=True).mutual_interest.agg(['mean', 'size'])
display(age_analysis)
print('Profiles represented in interactions:', len(set(frame.male_user_id) | set(frame.female_user_id)))"""),
    nbf.v4.new_markdown_cell("""## Validation design and model comparison

The same participants can appear in several dates. We hold out whole **events**, remove participant IDs from features, and use five-fold event-grouped out-of-fold predictions on the development events to choose hyperparameters. This tests transfer to other events more honestly than a random row split, though a participant may still occur in more than one event.

**Primary metric:** average precision (AP), because positives are relatively uncommon. We also report ROC AUC, precision, recall, and a confusion table. A dummy prior model establishes a baseline. Logistic regression, decision trees, and random forests are tuned over small, documented grids.
"""),
    nbf.v4.new_code_cell("""train_frame, val_frame, test_frame = split_by_event(frame)
display(pd.DataFrame(split_summary(train_frame, val_frame, test_frame)).T)
development = pd.concat([train_frame, val_frame], ignore_index=True)
comparison = compare_models(development)
display(comparison[['model', 'parameters', 'average_precision', 'roc_auc']])"""),
    nbf.v4.new_markdown_cell("""## Feature importance on held-out development events

Permutation importance measures how much validation AP changes when a feature is shuffled. Correlated features can share importance, and importance does not imply a causal effect. This diagnostic uses only the development split, leaving the final test events untouched.
"""),
    nbf.v4.new_code_cell("""best = comparison.iloc[0]
diagnostic_model = make_model(best['model'], best['parameters'])
diagnostic_model.fit(model_input(train_frame), train_frame.mutual_interest)
importance = permutation_importance(
    diagnostic_model,
    model_input(val_frame),
    val_frame.mutual_interest,
    scoring='average_precision',
    n_repeats=3,
    random_state=42,
)
importance_table = pd.DataFrame({
    'feature': FEATURE_COLUMNS,
    'ap_change_mean': importance.importances_mean,
    'ap_change_std': importance.importances_std,
}).sort_values('ap_change_mean', ascending=False)
display(importance_table.head(15))
importance_table.head(15).sort_values('ap_change_mean').plot.barh(
    x='feature', y='ap_change_mean', legend=False, figsize=(8, 6),
    title='Validation permutation importance (AP change)'
)
plt.tight_layout()
plt.show()"""),
    nbf.v4.new_markdown_cell("""## Final training and untouched test evaluation

`train.py` uses the same code as this notebook: choose the model using grouped development predictions, choose an F1 threshold on those predictions, refit on all development rows, and evaluate once on held-out test events. A weak held-out result is reported as a limitation rather than hidden by selecting a different test split.
"""),
    nbf.v4.new_code_cell("""report = run_training(DATA_DIR, Path('model.bin'), Path('results/metrics.json'))
display(pd.DataFrame({
    'development_out_of_fold': report['validation'],
    'held_out_test': report['test'],
}).T)
print('Selected model:', report['selected_model'], report['selected_parameters'])
print('Decision threshold:', report['threshold'])
print('Model saved:', Path('model.bin').resolve())"""),
    nbf.v4.new_markdown_cell("""## Conclusions and limits

Interpret the held-out test metrics against the positive-class prevalence and dummy baseline. If ROC AUC is near 0.5, this feature set is not reliably ranking mutually interested pairs across events. The project still demonstrates the full modeling and deployment workflow; it must not claim useful matchmaking performance without stronger evidence.

The interaction sample is modest, many optional questionnaire fields are missing, and different events may have different populations. The source records in-person eight-minute dates, not online swipe outcomes. This project deliberately excludes ethnicity, religion, politics, income, drug use, and sexual-history fields. The trained API accepts the two profiles and returns a probability plus a thresholded decision.
"""),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
)
nbf.write(notebook, Path(__file__).with_name("notebook.ipynb"))
