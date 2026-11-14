# Predicting mutual interest after a first date

## Problem and use

This ML Zoomcamp midterm project predicts whether **both** participants in a Mixmosa eight-minute date will say they want to meet again. A hypothetical event organizer could use the predicted probability to prioritize introductions. The prediction is made **before** the date from two participant profiles. It does not predict long-term compatibility, relationship success, or an online swipe outcome.

The [Mixmosa source repository](https://github.com/dankras/mixmosa-anonymized-data) documents 2,961 dates overall. Its published `interactions.csv` has 1,895 rows. Seven interactions cannot be joined to both entries in `user_features.csv`, leaving **1,888 usable labeled pairs** and **259 positive outcomes** (13.7%). The analysis uses the `mutual_interest` outcome. It excludes `does_male_like_female`, `does_female_like_male`, all IDs, swiping, and feedback from predictors because those values reveal outcomes or are unavailable at prediction time.

The source records male/female pairs at in-person events. It does not represent every dating app, orientation, or dating context. The optional profile features have missing values; these are imputed inside each training fold and accompanied by missingness indicators. The project excludes ethnicity, religion, politics, income, drug use, and sexual-history features. It retains age, height, independent photo ratings, personality and a few lifestyle measures; these choices and the profile-based nature of the predictions should be considered when interpreting results.

## Repository contents

| File | Purpose |
| --- | --- |
| `notebook.ipynb` | Executed EDA, feature analysis, model comparison, tuning, and final evaluation |
| `features.py` | Dataset join and identical pair feature construction for training and API requests |
| `modeling.py` | Event-grouped splits, five-fold model comparison, scoring, threshold choice |
| `train.py` | Train selected model and save `model.bin` plus `results/metrics.json` |
| `predict.py` | FastAPI `/health` and `/predict` service |
| `download_data.py` | Retrieve two CSVs from a pinned source commit and verify SHA256 hashes |
| `example_request.json`, `smoke_test.py`, `check_http.py` | API request example and smoke checks; optional JSON deployment evidence |
| `Dockerfile`, `requirements-runtime.txt` | Container image and pinned inference dependencies |
| `deploy_railway.ps1` | Railway deployment, domain creation, and optional public URL check |
| `results/railway_deployment.json`, `results/railway_smoke.json` | Remote Docker build and public HTTP verification evidence |
| `requirements.txt` | Full notebook and development environment |

The source CSVs are intentionally ignored by Git; the download script restores them from a pinned public source. The trained `model.bin`, notebook, metrics, and HTTP smoke result are included as project artifacts.

## Reproduce on Windows PowerShell

Use Python 3.13. From the project directory:

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python download_data.py
python train.py
```

To explore and rerun the notebook:

```powershell
python -m jupyter notebook notebook.ipynb
```

Run the notebook with the project directory as its working directory. `make_notebook.py` can regenerate the notebook source, but regeneration clears its saved outputs; run it only when updating the notebook content.

To start the API locally:

```powershell
python -m uvicorn predict:app --host 127.0.0.1 --port 9696
```

In a second terminal:

```powershell
. .\.venv\Scripts\Activate.ps1
python smoke_test.py
```

The interactive API documentation is at `http://127.0.0.1:9696/docs`. If PowerShell blocks activation scripts, run `Set-ExecutionPolicy -Scope Process RemoteSigned` in that terminal, then activate again. On macOS/Linux, use `source .venv/bin/activate` and the same `python` commands. Run `deactivate` when finished.

## API contract

`POST /predict` accepts a JSON object containing `male` and `female` participant profiles. Each requires `age`; the other profile features are optional and are imputed by the fitted model pipeline. See `example_request.json` for a complete example. The response contains:

```json
{
  "mutual_interest_probability": 0.1583,
  "mutual_interest_prediction": true,
  "threshold": 0.1
}
```

The numeric example is from the checked-in model and example request; a newly trained model can return a slightly different value. `mutual_interest_prediction` applies the threshold selected on development data. Invalid input, such as age 10 or an unknown field, returns HTTP 422. `GET /health` returns a simple readiness response.

## Notebook analysis and model selection

The notebook examines target balance, feature ranges, missingness, age differences, and event-level outcome variation. It also reports permutation feature importance on a held-out **development** event split.

Events, rather than individual rows, are held out for the final test. Within the development events, five-fold `GroupKFold` creates out-of-fold probabilities for each hyperparameter candidate. The primary selection metric is **average precision (AP)**, suitable for the 13.7% positive class. ROC AUC is also reported. Candidate models are a dummy prior predictor, logistic regression with four `C` values, decision trees with depth and leaf-size tuning, and random forests with depth and leaf-size tuning. Preprocessing is fitted within each fold: median imputation, missing indicators, and standardization for logistic regression.

The top out-of-fold AP candidate is refitted on all development events. The decision threshold is chosen from its development out-of-fold predictions by F1; the held-out test events are evaluated only after that choice. See `results/metrics.json` for every candidate and the full confusion counts.

### Observed result

The selected candidate is a random forest. Development out-of-fold AP is approximately **0.180**, versus a development positive rate of **0.130**; development ROC AUC is approximately **0.570**. On six held-out test events, AP is approximately **0.167**, essentially the test positive rate of **0.166**, and ROC AUC is **0.490**. At the development-selected threshold of `0.1`, test precision is about **0.152** and recall about **0.557**.

**This is not evidence of a useful matchmaking model.** The model does not reliably rank mutual-interest outcomes on unseen events. The project documents this result because the dataset is small, many profile answers are missing, people can appear in multiple dates, and event populations can differ. Further work would need more informative pre-date data and a stronger external evaluation. The API demonstrates deployment mechanics, not reliable recommendations.

## Docker

The `Dockerfile` uses Python 3.13 and installs the pinned inference dependencies from `requirements-runtime.txt`. It copies only the service, feature code, and trained model into the image. From this directory:

```powershell
docker build -t mixmosa-midterm .
docker run --rm -p 9696:9696 mixmosa-midterm
```

In another terminal, activate `.venv` and run `python smoke_test.py`, or send the JSON in `example_request.json` to `http://localhost:9696/predict`. `results/api_smoke.json` records a successful local HTTP request against Uvicorn, including an HTTP 422 check for invalid input. The container reads `PORT` when the host provides it and defaults to 9696 locally. Docker is not installed on the development machine, so the local Docker commands were not run. Railway remotely built and started the same Dockerfile; see `results/railway_deployment.json` and the public smoke result.

## Deploy to Railway

Install the [Railway CLI](https://docs.railway.com/cli) (Node.js/npm on Windows) and sign in when prompted. From this project directory in PowerShell:

```powershell
npm install -g @railway/cli
.\deploy_railway.ps1
```

The script runs `railway up -y` to upload this directory. Railway [detects the root Dockerfile](https://docs.railway.com/builds/dockerfiles) and builds the container remotely, so local Docker is not needed for this deployment. The script then runs `railway domain` to create a public `*.up.railway.app` URL. The service listens on Railway's `PORT` variable. In Railway service settings, set **Healthcheck Path** to `/health` as described in the [Railway healthcheck guide](https://docs.railway.com/deployments/healthchecks). A Railway account and any charges are the deployer's responsibility.

The live deployment is **[midtermdatingmatch-production.up.railway.app](https://midtermdatingmatch-production.up.railway.app)**. The following command checks a real HTTP health response, a prediction, and a rejected invalid request. The evidence file records the URL, time, status codes, and response:

```powershell
python smoke_test.py --url https://midtermdatingmatch-production.up.railway.app --evidence results/railway_smoke.json
```

On subsequent deployments, `.\deploy_railway.ps1 -PublicUrl https://midtermdatingmatch-production.up.railway.app` also runs this check. The [Railway CLI documentation](https://docs.railway.com/cli/up) describes `railway up` and its account sign-in flow. On 2026-11-14, Railway reported deployment `fd47f9cc-c531-4bd9-bebc-53c65b07df0d` as `SUCCESS` with builder `DOCKERFILE`; the public `/health` and `/predict` requests returned HTTP 200, and an invalid age returned HTTP 422. See `results/railway_deployment.json` and `results/railway_smoke.json`.

Before submitting to the course, complete the course's required three peer reviews. The dataset download is pinned to source commit `48676f40689fe5f41faa9d875e18320880725633`; `download_data.py` checks both SHA256 hashes. Loading `model.bin` uses Python pickle, so only load the artifact from a trusted source.
