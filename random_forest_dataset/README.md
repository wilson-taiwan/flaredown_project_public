Random forest dataset artifacts
===============================

What I created
---------------
This folder contains the train/test datasets prepared for Random Forest forecasting.
I created stratified test samples with exactly 1500 instances (when available) and
wrote the matching train sets using the remaining windows.

Files written (per condition)
-----------------------------
- random_forest_depression_train.csv  — training set for depression (all train-user windows)
- random_forest_depression_test.csv   — test set for depression (exactly 1500 sampled rows)
- random_forest_anxiety_train.csv     — training set for anxiety (all train-user windows)
- random_forest_anxiety_test.csv      — test set for anxiety (exactly 1500 sampled rows)

Where these come from
---------------------
Source files used:
- cleaned_data/rf_depression_forecast.csv
- cleaned_data/rf_anxiety_forecast.csv

Process summary
---------------
1. Read the full windowed CSV for each condition.
2. Perform a user-level 80/20 split (scikit-learn `train_test_split` on unique user IDs) to avoid data leakage.
   - All windows from a given user are assigned to either the training or test partition; there is no user overlap.
3. From the windows belonging to test-users, sample exactly TARGET_TEST_SIZE rows (1500 by default) using
   proportional (stratified) sampling by `user_seq_id`:
   - Each test user is sampled roughly in proportion to their number of windows in the test pool.
   - Each user is guaranteed at least one sampled window when possible.
   - Rounding may cause slight over/undershoot; the implementation trims or fills from the remaining pool to reach
     exactly the target size.
4. Save the sampled test set and the remaining train set to the `random_forest_dataset/<condition>/` directory.

Why this change
----------------
Previously the code truncated the test set with `.head(1500)`, which selected the first 1,500 rows in CSV order.
That approach was biased (it covered only ~21% of test users in prior runs). The new approach keeps comparisons fair
and ensures broad user coverage while keeping the test size constrained for runtime/cost reasons.

How to change behavior
----------------------
- To change the target test size, edit `TARGET_TEST_SIZE` near the top of
  `scripts/create_rf_datasets.py` and re-run the script:

```bash
python scripts/create_rf_datasets.py
```

- To use the full test set (no sampling), set `TARGET_TEST_SIZE` to a value >= available test windows for that
  condition (or modify the script to skip sampling when you want all windows).

- Randomness is controlled by `RNG` in the script. Use a fixed seed for reproducible samples.

Notes and follow-ups
--------------------
- After regenerating the test set you should rebuild any downstream artifacts that depend on the test instances,
  such as RAG embeddings (`*_embedded.json`) and LLM-RAG experiments, to ensure they use the corrected sample.
- The script created the files with user-requested names; if you'd like them placed directly at the repo root or with a
  different naming scheme, I can update that quickly.

Contact
-------
If anything about the sampling or naming needs to change (e.g., target size -> 1000, or minimum test-user count), let me
know and I will update the script and re-run.
