"""
DTW-only forecasting baseline for depression/anxiety

Purpose:
- Run Dynamic Time Warping (DTW) retrieval on the primary-condition days 1-10
  against a historical RAG text database and produce short-term forecasts
  (days 11-20) using the top-k matched cases.

Notes:
- This is intentionally minimal and self-contained so it can act as a reproducible
  baseline for comparison with the existing DTW+LLM pipeline.
- The RAG text database is expected at: llm_dataset/<condition>/rag_<condition>_forecast_text_database.json
- The test CSV is expected at: llm_dataset/<condition>/llm_<condition>_test.csv

Usage example:
  python llm_output/train_dtw_only.py --condition depression --topk 10 --radius 2

"""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from datetime import datetime
import math
import importlib.util

# Load the main RAG module by path so this script can be executed directly
# (avoids requiring llm_output to be a Python package)
rag_path = Path(__file__).resolve().parents[1] / 'llm_output' / 'train_llm_rag_with_dtw.py'
if not rag_path.exists():
    # Fallback to project-root llm_output path (if run from different cwd)
    rag_path = Path.cwd() / 'llm_output' / 'train_llm_rag_with_dtw.py'

spec = importlib.util.spec_from_file_location('train_llm_rag_with_dtw', str(rag_path))
rag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag)


def compute_dtw_distance(series1, series2, radius=2):
    """Compute a univariate DTW distance using fastdtw.

    Returns a float (lower = more similar). On error returns +inf.
    """
    try:
        s1 = np.array(series1).reshape(-1, 1)
        s2 = np.array(series2).reshape(-1, 1)
        distance, _ = fastdtw(s1, s2, radius=radius, dist=euclidean)
        return float(distance)
    except Exception as e:
        print(f"   ⚠️  DTW error: {e}")
        return float('inf')


def extract_history_timeseries_from_db_record(rec, condition):
    """Robustly extract days 1-10 time series and days 11-20 future series from a DB record.

    Expected shapes: lists of length 10. The function attempts a few common key names
    used in this repository. Returns (history_list_or_None, future_list_or_None).
    """
    meta = None
    if isinstance(rec, dict):
        meta = rec.get('metadata') or rec

    if not isinstance(meta, dict):
        return None, None

    hist_keys = [f'{condition}_days_1_10', f'{condition}_day_1_to_10', f'{condition}_days_1-10']
    fut_keys = [f'{condition}_days_11_20', f'{condition}_days_11_to_20', f'{condition}_days_11-20', f'{condition}_future_11_20']

    hist = None
    fut = None
    for k in hist_keys:
        if k in meta:
            hist = meta[k]
            break

    for k in fut_keys:
        if k in meta:
            fut = meta[k]
            break

    # Sometimes the DB stores flattened keys like 'day1'...'day10' or 'depression_day1'...
    if hist is None:
        maybe = []
        for i in range(1, 11):
            k = f'{condition}_day{i}'
            if k in meta:
                maybe.append(meta[k])
            else:
                maybe = []
                break
        if len(maybe) == 10:
            hist = maybe

    if fut is None:
        maybe = []
        for i in range(11, 21):
            k = f'target_{condition}_day{i}'
            if k in meta:
                maybe.append(meta[k])
            else:
                maybe = []
                break
        if len(maybe) == 10:
            fut = maybe

    # Validate shapes
    if isinstance(hist, list) and len(hist) == 10:
        hist = [float(x) for x in hist]
    else:
        hist = None

    if isinstance(fut, list) and len(fut) == 10:
        fut = [float(x) for x in fut]
    else:
        fut = None

    return hist, fut


def inverse_distance_weights(distances, eps=1e-6):
    """Convert an array of distances to normalized inverse-distance weights.

    We add a small epsilon to avoid division by zero.
    """
    d = np.array(distances, dtype=float)
    # Replace inf with large number
    d = np.where(np.isfinite(d), d, 1e12)
    inv = 1.0 / (d + eps)
    if inv.sum() == 0 or not np.isfinite(inv.sum()):
        # fallback to uniform
        return np.ones_like(inv) / len(inv)
    return inv / inv.sum()


def predict_from_matches(matches, topk):
    """Given list of (rec, distance), return predicted next-10-days (list) or None.

    This does inverse-distance weighted average across available matched futures.
    """
    # Collect available futures
    futs = []
    distances = []
    for rec, dist in matches:
        if rec.get('__future__') is not None:
            futs.append(rec['__future__'])
            distances.append(dist)

    if len(futs) == 0:
        return None

    # Limit to topk (they should already be topk but enforce)
    futs = futs[:topk]
    distances = distances[:topk]

    weights = inverse_distance_weights(distances)
    arr = np.array(futs, dtype=float)  # shape (n,10)
    weighted = (weights.reshape(-1, 1) * arr).sum(axis=0)
    return weighted.tolist()


def main():
    p = argparse.ArgumentParser(description='DTW-only forecasting baseline')
    p.add_argument('--condition', choices=['depression', 'anxiety'], default='anxiety')
    p.add_argument('--topk', type=int, default=25)
    p.add_argument('--radius', type=int, default=2, help='fastdtw radius')
    p.add_argument('--max-db', type=int, default=None, help='If set, sample this many DB records for speed')
    p.add_argument('--out-dir', type=str, default=None)
    # By default mirror train_llm behavior: use multivariate DTW and prefiltering
    p.add_argument('--no-adaptive-weights', action='store_true', dest='no_adaptive_weights',
                   help='Disable automatic adaptive-weight computation from available training CSVs')
    # Mirror train_llm: primary-only DTW by default; allow opt-in multivariate with explicit flag
    p.add_argument('--use-multivariate', action='store_true', dest='use_multivariate', default=False,
                   help='Enable multivariate DTW (includes comorbid symptom series). Default: disabled (primary-only)')
    p.add_argument('--no-prefilter', action='store_false', dest='use_prefilter', default=True,
                   help='Disable fast Euclidean prefilter before DTW')
    p.add_argument('--prefilter-size', type=int, default=500, help='Number of candidates to keep after prefiltering')
    args = p.parse_args()

    base = Path.cwd()
    cond = args.condition
    test_csv = base / 'llm_dataset' / cond / f'llm_{cond}_test.csv'
    db_json = base / 'llm_dataset' / cond / f'rag_{cond}_forecast_text_database.json'
    train_csv = base / 'llm_dataset' / cond / f'llm_{cond}_train.csv'

    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")
    if not db_json.exists():
        raise FileNotFoundError(f"RAG text database not found: {db_json}")

    print(f"Loading test CSV: {test_csv}")
    df_test = pd.read_csv(test_csv)
    print(f"Loaded {len(df_test)} test rows")

    # Validate test columns presence for history (days 1-10)
    hist_cols = [f'{cond}_day{i}' for i in range(1, 11)]
    for c in hist_cols:
        if c not in df_test.columns:
            raise ValueError(f"Required column missing in test CSV: {c}")

    # Load DB (try simple load; if it fails due to size, advise the user)
    print(f"Loading RAG text DB (this may be large): {db_json}")
    try:
        with open(db_json, 'r') as f:
            db = json.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load DB with json.load(): {e}\nIf the file is huge, consider using --max-db to sample or pre-build a smaller DB file."
        )

    print(f"DB loaded: {len(db)} records (raw)")

    # Optionally sample DB for speed
    if args.max_db is not None:
        db = db[:args.max_db]
        print(f"Sampled DB to first {len(db)} records (--max-db)")

    # Attempt to compute adaptive weights from training CSVs (mirror train_llm behavior)
    base_weights = None
    if not args.no_adaptive_weights:
        train_candidates = [
            base / 'llm_dataset' / cond / f'llm_{cond}_train.csv',
            base / 'random_forest_dataset' / cond / f'random_forest_{cond}_train.csv',
            base / 'random_forest_dataset' / f'rf_{cond}_train.csv',
            base / 'random_forest_dataset' / f'{cond}_train.csv'
        ]
        for tp in train_candidates:
            try:
                if tp.exists():
                    print(f"Found training CSV for adaptive weights at: {tp}")
                    train_df = pd.read_csv(tp)
                    base_weights = rag.compute_adaptive_weights_from_df(train_df, condition=cond, min_primary_weight=0.35)
                    print(f"Using adaptive weights: {base_weights}")
                    break
            except Exception as e:
                print(f"   ⚠️  Could not compute adaptive weights from {tp}: {e}")
        if base_weights is None:
            print("No training CSV found or failed to compute adaptive weights; proceeding with default fixed weights.")

    # Keep DB as loaded and let the imported find_similar_patients_dtw handle prefiltering
    text_db = db
    print(f"Using text DB with {len(text_db)} records")

    results = []
    # Ensure the imported module uses the correct CONDITION for prefiltering logic
    rag.CONDITION = cond
    # Build a forecast pipeline that mirrors train_llm_rag_with_dtw's forecast_with_dtw
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    def extract_future_from_record(record):
        tdict = record.get('metadata', {}) or {}
        block = tdict.get(f'target_{cond}_days_11_20') or tdict.get(f'{cond}_days_11_20') or tdict.get('target_days_11_20')
        vals = None
        if isinstance(block, dict):
            vals = [block.get(f'day_{d}') for d in range(11, 21)]
        elif isinstance(block, (list, tuple)):
            if len(block) >= 10:
                vals = list(block[:10])

        if vals is None:
            nested = tdict.get('patient_data') or tdict.get('metadata') or {}
            if isinstance(nested, dict):
                block2 = nested.get(f'target_{cond}_days_11_20') or nested.get(f'{cond}_days_11_20')
                if isinstance(block2, dict):
                    vals = [block2.get(f'day_{d}') for d in range(11, 21)]

        if vals and len(vals) == 10:
            out = []
            for v in vals:
                try:
                    out.append(float(v))
                except Exception:
                    out.append(np.nan)
            return out
        return None

    def process_single_patient_dtw_only(idx, row, text_db, top_k, max_patient_retries=2, weights=None):
        # Build test_patient dict (same as _process_patient_core)
        test_patient = {
            'age': row.get('age', 'Unknown'),
            'sex': row.get('sex', 'Unknown'),
            'country': row.get('country', 'Unknown'),
            f'{cond}_days_1_10': { f'day_{i}': row[f'{cond}_day{i}'] for i in range(1, 11)}
        }
        try:
            test_patient['_orig_index'] = int(idx)
        except Exception:
            test_patient['_orig_index'] = idx

        # contextual features
        key_features = ['anxiety_mean_10d', 'fatigue_mean_10d', 'pain_mean_10d', 'sadness_mean_10d',
                        'stress_mean_10d', 'irritability_mean_10d', 'cognitive_dysfunction_mean_10d']
        contextual_features = {}
        for feature in key_features:
            if feature in row.index:
                value = row.get(feature)
                if pd.notna(value):
                    contextual_features[feature] = value
        test_patient['contextual_features'] = contextual_features

        # Build query_features same as train_llm
        query_features = {cond: [test_patient[f'{cond}_days_1_10'][f'day_{i}'] for i in range(1, 11)]}
        symptom_map = {
            'anxiety_mean_10d': 'anxiety',
            'fatigue_mean_10d': 'fatigue',
            'pain_mean_10d': 'pain',
            'sadness_mean_10d': 'sadness',
            'stress_mean_10d': 'stress',
            'irritability_mean_10d': 'irritability',
            'cognitive_dysfunction_mean_10d': 'cognitive_dysfunction'
        }
        for feature_key, symptom_name in symptom_map.items():
            if feature_key in contextual_features and pd.notna(contextual_features[feature_key]):
                query_features[symptom_name] = [contextual_features[feature_key]] * 10

        # compute per-patient weights
        base_for_patient = weights if weights is not None else base_weights
        per_patient_weights = rag.compute_per_patient_weights(test_patient, base_weights=base_for_patient, min_primary_weight=0.35)

        # retrieve similar patients (mirror train_llm: use_multivariate=False by default)
        similar_patients = rag.find_similar_patients_dtw(query_features, text_db, top_k=top_k,
                                                        use_multivariate=bool(args.use_multivariate),
                                                        use_fast_prefilter=bool(args.use_prefilter),
                                                        prefilter_size=int(args.prefilter_size),
                                                        weights=per_patient_weights,
                                                        ensure_unique_patients=True)

        # Aggregate matched patients' futures using similarity weights
        sims = []
        futures = []
        for rec, sim in similar_patients:
            fut = extract_future_from_record(rec)
            if fut is not None:
                sims.append(sim)
                futures.append(fut)

        if len(futures) == 0:
            return {
                'idx': int(idx),
                'predicted': [np.nan]*10,
                'actual': [row.get(f'target_{cond}_day{i}', np.nan) for i in range(11,21)],
                'topk_similarities': [s for _, s in similar_patients]
            }

        sims_arr = np.array(sims, dtype=float)
        # normalize
        w = sims_arr / sims_arr.sum()
        fut_arr = np.array(futures, dtype=float)  # shape (n_matches, 10)
        weighted = (w.reshape(-1,1) * fut_arr).sum(axis=0)

        return {
            'idx': int(idx),
            'predicted': weighted.tolist(),
            'actual': [float(row.get(f'target_{cond}_day{i}', np.nan)) for i in range(11,21)],
            'topk_similarities': [s for _, s in similar_patients]
        }

    # Run parallel forecasting like forecast_with_dtw
    import os
    TOP_K = args.topk
    # Use a fixed number of workers (16) as requested. If you later want
    # to make this configurable, add a --workers CLI flag and honor it here.
    N_WORKERS = 16
    print(f"Processing {len(df_test)} test patients with {N_WORKERS} parallel workers...")
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        future_to_idx = {executor.submit(process_single_patient_dtw_only, idx, row, text_db, TOP_K, 2, base_weights): idx
                         for idx, row in df_test.iterrows()}
        with tqdm(total=len(future_to_idx), desc="Forecasting DTW-only") as pbar:
            for future in as_completed(future_to_idx):
                try:
                    r = future.result()
                    if r:
                        results.append(r)
                except Exception as e:
                    tqdm.write(f"Error processing patient: {e}")
                pbar.update(1)

    # Save results and mirror train_llm outputs under llm_output/<condition>/dtw-only-<timestamp>/
    base = Path.cwd()
    timestamp = datetime.now().strftime('%m%d%y-%H%M')
    model_sanitized = 'dtw-only'
    run_name = f"{model_sanitized}-{timestamp}"
    run_dir = base / 'llm_output' / cond / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Set imported module globals so its helpers write into the same run dir/shape
    rag.CONDITION = cond
    rag.RESULTS_DIR = run_dir

    # Flatten results into DataFrame matching train_llm expected column names
    rows = []
    for r in results:
        test_idx = r.get('idx', r.get('test_index'))
        base_row = {}

        # predicted values -> 'predicted_day_11'..'predicted_day_20'
        preds = r.get('predicted')
        if preds is not None:
            for i, v in enumerate(preds, start=11):
                try:
                    base_row[f'predicted_day_{i}'] = float(v) if not (v is None or (isinstance(v, float) and np.isnan(v))) else np.nan
                except Exception:
                    base_row[f'predicted_day_{i}'] = np.nan
        else:
            for i in range(11, 21):
                base_row[f'predicted_day_{i}'] = np.nan

        # actual values -> 'actual_day_11'..'actual_day_20'
        actuals = r.get('actual') or r.get('ground_truth_days_11_20')
        if actuals is not None:
            for i, v in enumerate(actuals, start=11):
                try:
                    base_row[f'actual_day_{i}'] = float(v) if not (v is None or (isinstance(v, float) and np.isnan(v))) else np.nan
                except Exception:
                    base_row[f'actual_day_{i}'] = np.nan
        else:
            for i in range(11, 21):
                base_row[f'actual_day_{i}'] = np.nan

        # top-k similarities (optional)
        sims = r.get('topk_similarities') or []
        for k_i, s in enumerate(sims, start=1):
            try:
                base_row[f'top{k_i}_sim'] = float(s)
            except Exception:
                base_row[f'top{k_i}_sim'] = np.nan

        # preserve index
        base_row['test_index'] = int(test_idx) if test_idx is not None else None
        rows.append(base_row)

    results_df = pd.DataFrame(rows)

    # Save predictions CSV in the same naming convention as train_llm
    predictions_path = run_dir / 'predictions.csv'
    results_df.to_csv(predictions_path, index=False)
    print(f"Saved predictions to {predictions_path}")

    # Evaluate predictions using the same evaluator from train_llm
    # The evaluator expects columns like 'predicted_day_11' and 'actual_day_11'
    metrics = rag.evaluate_predictions(results_df)

    # Attempt to find a corresponding LLM run to compare against
    def find_llm_run_dir(base_dir, condition, model_prefix='meta-llama-3.1-8b-instruct-full-run-1'):
        base_p = Path(base_dir) / 'llm_output' / condition
        if not base_p.exists():
            return None
        # find directories that start with the model_prefix
        candidates = [p for p in base_p.iterdir() if p.is_dir() and p.name.startswith(model_prefix)]
        if not candidates:
            return None
        # choose newest by mtime
        chosen = max(candidates, key=lambda p: p.stat().st_mtime)
        return chosen

    llm_run = find_llm_run_dir(base, cond, model_prefix='llama-3.1-8b-instruct-full-run-1')
    llm_metrics = None
    if llm_run is None:
        # Write placeholder note so user knows where to put LLM run outputs for comparison
        note = run_dir / 'LLM_COMPARISON_PLACEHOLDER.txt'
        with open(note, 'w') as f:
            f.write('LLM run folder not found. To enable DTW-only vs LLM comparison, place the LLM run folder (named starting with "llama-3.1-8b-instruct-full-run-1") under llm_output/{cond}/ and rerun this script or re-run the visualization step.')
        print(f"LLM run folder with prefix 'llama-3.1-8b-instruct-full-run-1' not found under llm_output/{cond}; wrote placeholder: {note}")
    else:
        try:
            eval_path = llm_run / 'evaluation_metrics.json'
            if eval_path.exists():
                with open(eval_path, 'r') as f:
                    llm_metrics = json.load(f)
                print(f"Loaded LLM run metrics from: {eval_path}")
            else:
                print(f"Found LLM run dir {llm_run} but evaluation_metrics.json not present; skipping LLM comparison.")
        except Exception as e:
            print(f"Error loading LLM run metrics: {e}; skipping LLM comparison.")

    # Create visualizations (mirroring train_llm but comparing to LLM run if available)
    def create_visualizations_local(results_df_local, metrics_local, llm_metrics_local=None):
        fig_dir = run_dir / 'figures'
        fig_dir.mkdir(exist_ok=True)

        # 1. Predicted vs Actual scatter
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        pred_cols = [f'predicted_day_{i}' for i in range(11, 21)]
        act_cols = [f'actual_day_{i}' for i in range(11, 21)]
        y_pred = results_df_local[pred_cols].values.flatten()
        y_true = results_df_local[act_cols].values.flatten()
        plt.scatter(y_true, y_pred, alpha=0.5, s=20)
        plt.plot([0, 4], [0, 4], 'r--', label='Perfect Prediction')
        plt.xlabel(f'Actual {cond.capitalize()} Level', fontsize=12)
        plt.ylabel(f'Predicted {cond.capitalize()} Level', fontsize=12)
        plt.title(f'DTW-only: Predicted vs Actual {cond.capitalize()} Levels', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / 'predicted_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('✅ Saved: predicted_vs_actual.png')

        # 2. Per-day MAE
        import numpy as _np
        plt.figure(figsize=(12, 6))
        day_metrics = metrics_local['per_day']
        days = [dm['day'] for dm in day_metrics]
        maes = [dm['mae'] for dm in day_metrics]
        plt.bar(days, maes, color='steelblue', alpha=0.7)
        plt.xlabel('Day', fontsize=12)
        plt.ylabel('Mean Absolute Error', fontsize=12)
        plt.title(f'DTW-only: MAE by Forecast Day ({cond.capitalize()})', fontsize=14, fontweight='bold')
        plt.xticks(days)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / 'mae_by_day.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('✅ Saved: mae_by_day.png')

        # 3. Error distribution
        plt.figure(figsize=(10, 6))
        errors = y_pred - y_true
        plt.hist(errors, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        plt.xlabel('Prediction Error (Predicted - Actual)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'DTW-only: Prediction Error Distribution ({cond.capitalize()})', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('✅ Saved: error_distribution.png')

        # 4. Comparison with LLM per-day MAE if available
        try:
            if llm_metrics_local and 'per_day' in llm_metrics_local:
                rf_maes = [d.get('mae', _np.nan) for d in llm_metrics_local['per_day']]
                x = _np.arange(11, 21)
                width = 0.35
                plt.figure(figsize=(12, 6))
                plt.bar(x - width/2, maes, width, label='DTW-only', color='steelblue')
                plt.bar(x + width/2, rf_maes, width, label='LLM (ref)', color='orange', alpha=0.9)
                plt.xlabel('Day', fontsize=12)
                plt.ylabel('Mean Absolute Error', fontsize=12)
                plt.title(f'MAE by Day: DTW-only vs LLM ({cond.capitalize()})', fontsize=14, fontweight='bold')
                plt.xticks(x)
                plt.legend()
                plt.grid(True, axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(fig_dir / 'mae_comparison_llm.png', dpi=300, bbox_inches='tight')
                plt.close()
                print('✅ Saved: mae_comparison_llm.png')
            else:
                print('No LLM metrics available for per-day comparison; skipped LLM comparison plot.')
        except Exception as e:
            print(f'   ⚠️  Could not create LLM comparison plot: {e}')

    create_visualizations_local(results_df, metrics, llm_metrics)

    # Save results and summary using train_llm helper for consistency
    extra_info = {
        'top_k': int(TOP_K),
        'max_patients': int(len(df_test)),
        'n_workers': int(N_WORKERS),
        'text_db': str(db_json),
        'test_data': str(test_csv),
        'run_name': run_name,
        'adaptive_weights': base_weights
    }

    try:
        rag.save_results(results_df, metrics, extra_info=extra_info)
    except Exception as e:
        print(f"Warning: rag.save_results failed: {e}; basic files already written to {run_dir}")


if __name__ == '__main__':
    main()
