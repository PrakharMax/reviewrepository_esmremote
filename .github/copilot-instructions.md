## Purpose
Short, actionable guidance for an AI coding agent working in this repository (fetching IAQ + physiological data).

## Big picture
- **Top-level script:** [my_esm_acquisition/main.py](my_esm_acquisition/main.py) is a runnable script that refreshes a Netatmo OAuth token, fetches IAQ measurements and writes a CSV.
- **Reusable client:** [my_esm_acquisition/utils/netatmo_client.py](my_esm_acquisition/utils/netatmo_client.py) contains a `NetatmoClient` class used to call Netatmo endpoints (preferred for re-use). Note: a similar class exists inline in `main.py` (duplicate implementation).
- **Module layout:** `src/` is intended for ingest/orchestration modules (many files currently placeholders). `config/` stores secrets and participant metadata. `data_lake/` is where fetched data should be organized (FMI, Netatmo, physiological).

## Core flows & why
- OAuth flow: call `/oauth2/token` with `refresh_token` → receive access token (see `NetatmoClient.get_access_token`).
- Data fetch: call `/api/getmeasure` with `device_id`, `scale`, and `date_begin` to retrieve IAQ timeseries. `date_begin` is used to request historical data in epoch seconds.
- Output: CSV files are written by `main.py` as `IAQ_User_{id}.csv` in the current working directory.

## How to run & debug (practical)
- Install runtime deps (used in code): `requests`, `python-dotenv`. Run from the `my_esm_acquisition` directory to ensure relative config paths work.
  - Example:
    - `cd my_esm_acquisition`
    - `python main.py`
- `main.py` constructs the env path with `config/api_keys.env` (look for variables named like `Cozie_Aalto_{user}_client_ID`); ensure the file exists and contains correct names. Running from another CWD will break the relative lookup.
- To test token fetch and API call: set `user_id` at the top of `main.py` and ensure `Cozie_Aalto_{user}_refresh_token` and `Cozie_Aalto_{user}_MAC_ID` are present in `config/api_keys.env`.

## Project-specific conventions and patterns
- Env naming: credentials use the pattern `Cozie_Aalto_{user}_...` (see `main.py`). Search `config/api_keys.env` for actual keys.
- Duplicate code: There are two `NetatmoClient` implementations (one in `utils/`, one copy in `main.py`). Prefer importing the `utils/netatmo_client.py` client when refactoring or adding tests.
- Placement: `utils/` contains small reusable helpers; `src/` should host project modules and orchestrators. New features should add module tests under a `tests/` folder (not present yet).

## Integrations & external dependencies
- External APIs: Netatmo OAuth and `/api/getmeasure` endpoints. Network errors should be handled by catching `requests` exceptions and using `response.raise_for_status()` as currently done.
- Config: `config/api_keys.env` (sensitive) and `config/participants.yaml` (participant metadata). `api_keys.env` is loaded explicitly with `load_dotenv(dotenv_path=env_path)` in `main.py`.

## Things to watch for / guidance for PRs
- Preserve the env-loading pattern or convert to a centralized config loader (if you change it, update every script that relies on `BASE_DIR` resolution).
- Consolidate the Netatmo client into `utils/netatmo_client.py` and replace the inline class in `main.py` to avoid drift.
- Prefer explicit date handling: `fetch_measurements(..., date_begin)` expects epoch seconds. Add utility helpers for time-window calculations if adding features.

## References
- Script/runner: [my_esm_acquisition/main.py](my_esm_acquisition/main.py)
- Reusable client: [my_esm_acquisition/utils/netatmo_client.py](my_esm_acquisition/utils/netatmo_client.py)
- Config and participants: [my_esm_acquisition/config/api_keys.env](my_esm_acquisition/config/api_keys.env), [my_esm_acquisition/config/participants.yaml](my_esm_acquisition/config/participants.yaml)

---
If anything above is unclear or you want me to include more examples (unit-test patterns, CI commands, or a refactor PR consolidating the Netatmo client), tell me which area to expand.
