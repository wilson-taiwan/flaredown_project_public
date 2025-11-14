"""
Create stratified 1500-instance test sets for Random Forest forecasting.
Writes outputs to:
  - random_forest_dataset/depression/rf_depression_forecast_train.csv
  - random_forest_dataset/depression/rf_depression_forecast_test.csv
  - random_forest_dataset/anxiety/rf_anxiety_forecast_train.csv
  - random_forest_dataset/anxiety/rf_anxiety_forecast_test.csv

Usage:
    python scripts/create_rf_datasets.py

This script samples the test set proportionally by user from the user-level test split
(to avoid leakage). If the available test windows are fewer than TARGET_TEST_SIZE,
all are used.
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parents[1]
CLEANED = BASE / "cleaned_data"
OUT_DIR = BASE / "random_forest_dataset"
TARGET_TEST_SIZE = 1500
RNG = 42


def make_dirs(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def stratified_sample_test(df, target_size, random_state=42):
    """Given df where df['user_seq_id'] exists, sample `target_size` rows from df
    proportionally across users, guaranteeing at least 1 row per user when possible.
    Returns sampled_df (exactly target_size rows unless df has fewer rows).
    """
    if len(df) <= target_size:
        return df.copy()

    sample_frac = target_size / len(df)
    rng = np.random.RandomState(random_state)

    parts = []
    for uid, g in df.groupby('user_seq_id'):
        n_samples = max(1, int(round(len(g) * sample_frac)))
        if n_samples >= len(g):
            parts.append(g)
        else:
            parts.append(g.sample(n=n_samples, random_state=random_state))

    sampled = pd.concat(parts, ignore_index=True)

    # If we've overshot due to rounding, trim randomly to target_size
    if len(sampled) > target_size:
        sampled = sampled.sample(n=target_size, random_state=random_state).reset_index(drop=True)
        return sampled

    # If undershot, sample additional rows from remaining pool
    if len(sampled) < target_size:
        needed = target_size - len(sampled)
        # identify remaining rows not yet selected
        sampled_idx = set()
        # Try to use an index column if present, otherwise use positional uniqueness
        if '_orig_index' in df.columns:
            sampled_idx = set(sampled['_orig_index'].tolist())
            remaining = df[~df['_orig_index'].isin(sampled_idx)]
        else:
            # fall back to using the dataframe index
            remaining = df.loc[~df.index.isin(sampled.index)]

        if len(remaining) <= needed:
            sampled = pd.concat([sampled, remaining], ignore_index=True)
        else:
            sampled = pd.concat([sampled, remaining.sample(n=needed, random_state=random_state)], ignore_index=True)

    # Final trim if somehow still off
    if len(sampled) > target_size:
        sampled = sampled.sample(n=target_size, random_state=random_state)

    return sampled.reset_index(drop=True)


def process_file(filename, out_subdir, target_test_size=TARGET_TEST_SIZE, random_state=RNG):
    path = CLEANED / filename
    if not path.exists():
        print(f"ERROR: source file not found: {path}")
        return

    df = pd.read_csv(path)
    # Keep a copy of original index to allow re-sampling if needed
    df['_orig_index'] = np.arange(len(df))

    users = df['user_seq_id'].unique()
    train_users, test_users = train_test_split(users, test_size=0.2, random_state=random_state)

    df_train = df[df['user_seq_id'].isin(train_users)].copy()
    df_test_all = df[df['user_seq_id'].isin(test_users)].copy()

    print(f"\nProcessing {filename}:")
    print(f"  total windows: {len(df)}")
    print(f"  users total: {len(users)}")
    print(f"  test-users (20% split): {len(test_users)} -> test windows available: {len(df_test_all)}")

    if len(df_test_all) <= target_test_size:
        df_test = df_test_all.copy()
        print(f"  using all available test windows ({len(df_test)}) since <= target {target_test_size}")
    else:
        df_test = stratified_sample_test(df_test_all, target_test_size, random_state=random_state)
        print(f"  sampled test windows: {len(df_test)}")

    # Remove helper column before saving
    df_train = df_train.drop(columns=['_orig_index'], errors='ignore')
    df_test = df_test.drop(columns=['_orig_index'], errors='ignore')

    # Write outputs
    out_path = OUT_DIR / out_subdir
    make_dirs(out_path)

    # use a clear, stable filename format requested by the user:
    # random_forest_{condition}_train.csv and random_forest_{condition}_test.csv
    condition = out_subdir.lower()
    train_file = out_path / f"random_forest_{condition}_train.csv"
    test_file = out_path / f"random_forest_{condition}_test.csv"

    df_train.to_csv(train_file, index=False)
    df_test.to_csv(test_file, index=False)

    print(f"  -> train written: {train_file} ({len(df_train)} rows, {df_train['user_seq_id'].nunique()} users)")
    print(f"  -> test written:  {test_file} ({len(df_test)} rows, {df_test['user_seq_id'].nunique()} users)")

    return {
        'total_windows': len(df),
        'users_total': len(users),
        'test_users': len(test_users),
        'test_windows_available': len(df_test_all),
        'test_sampled_rows': len(df_test),
        'test_sampled_users': df_test['user_seq_id'].nunique(),
        'train_rows': len(df_train),
        'train_users': df_train['user_seq_id'].nunique(),
        'train_file': str(train_file),
        'test_file': str(test_file),
    }


if __name__ == '__main__':
    results = {}
    # Depression
    results['depression'] = process_file('rf_depression_forecast.csv', 'depression')
    # Anxiety
    results['anxiety'] = process_file('rf_anxiety_forecast.csv', 'anxiety')

    print('\nSummary:')
    for k, v in results.items():
        if not v:
            continue
        print(f"\n{k.upper()}")
        print(f"  total_windows: {v['total_windows']}")
        print(f"  users_total: {v['users_total']}")
        print(f"  test_windows_available: {v['test_windows_available']}")
        print(f"  test_sampled_rows: {v['test_sampled_rows']}")
        print(f"  test_sampled_users: {v['test_sampled_users']}")
        print(f"  train_rows: {v['train_rows']}")
        print(f"  train_users: {v['train_users']}")

    print('\nDone.')
