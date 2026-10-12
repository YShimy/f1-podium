# F1 Podium Prediction (CSEN903)

Predict, for each driver in each race, whether they finish on the podium, using only information available after
qualifying. See `CLAUDE.md` for the project rules (leakage rules, data rules, conventions).

- `src/f1podium/`: reusable code (loading, validation, cleaning, EDA).
- `notebooks/milestone1.ipynb`: the run-all notebook. It is **generated**: edit `scripts/build_notebook.py`, never the `.ipynb`.
- `tests/`: `python -m pytest`.

## Kaggle packaging

The notebook must run on Kaggle, where the repo is not available. `src/` is shipped as a Kaggle dataset:

1. Commit all changes (the script refuses to run on a dirty git working tree).
2. Run `python scripts/package_kaggle.py`. It writes `dist/f1podium_src_<shorthash>.zip`, containing `src/` and a
   `src/VERSION` file with the full commit hash.
3. Upload the zip as a **new version** of the Kaggle dataset, and attach the dataset to the notebook.

**After any change under `src/`, re-run `package_kaggle.py` and re-upload as a new dataset version.**
The notebook's Setup cell prints the commit hash of the code it loaded. **Before submission, verify that this printed
hash matches the final commit** (`git rev-parse HEAD`).
