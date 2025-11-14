"""
DTW-based RAG forecasting for depression/anxiety (quick reference)

What this script does
- Uses Dynamic Time Warping (DTW) on the primary condition time series
    (days 1-10) to retrieve temporally similar historical windows.
    - Augments an LLM prompt with the retrieved DTW cases and additional
    embedding-RAG (text-based) matches built from aggregated/contextual features
    (e.g., anxiety_mean_10d, fatigue_mean_10d). Numeric/vector-based RAG
    support has been removed; embedding-RAG is the supported retrieval augmentation.
- The LLM is asked to predict the target condition for days 11-20 and must
    return a JSON block with day_11...day_20 plus a short thinking block.

Design choices and rationale
- DTW is used primarily on the primary condition (depression or anxiety) to compare
    temporal shape/phase. This is where DTW has the most value.
    Contextual/comorbid signals are handled via embedding-RAG (text-based). Embedding-RAG is disabled by default; enable it at runtime with the `--use-embedding-rag` CLI flag.
- Adaptive global weights (computed from training CSVs when available) and
    small per-patient weight adjustments are available to tune retrieval
    importance across symptoms.

Quick usage
-- Run with defaults (embedding-RAG disabled, numeric vector-RAG disabled):
    python llm_output/train_llm_rag_with_dtw.py

- Note: numeric/vector-RAG options were removed. Use embedding-RAG only.

- Precompute test-query embeddings (one-time; faster repeated runs):
    python llm_output/train_llm_rag_with_dtw.py --precompute-query-embeddings

- Change the target condition (depression/anxiety):
    python llm_output/train_llm_rag_with_dtw.py --condition anxiety

Key files / outputs
- Input text DB: llm_dataset/<condition>/rag_<condition>_forecast_text_database.json
- Test CSV:   llm_dataset/<condition>/llm_<condition>_test.csv
- Embedded DB: llm_dataset/<condition>/rag_<condition>_forecast_embedded.json
- Vector DB (persisted): llm_dataset/<condition>/vector_db_<condition>.pkl
- Precomputed queries: llm_dataset/<condition>/rag_<condition>_test_embedded.json
- Outputs (per-run): llm_output/<condition>/<model>-TIMESTAMP/
        - predictions.csv
        - evaluation_metrics.json
        - training_summary.txt
        - figures/

Notes and recommended experiments
- The script uses primary-only DTW (not multivariate) for temporal matching;
    embedding-RAG handles contextual similarity when explicitly enabled. To evaluate retrieval design,
    compare two settings: (A) primary-only DTW + embedding-RAG (enable with --use-embedding-rag) and
    (B) primary-only DTW only.
- Adaptive weights are computed from training CSVs (if present). They are
    interpretable but heuristic; consider cross-validation or a learned
    weighting model if you need more robust importance estimates.
- Multivariate DTW is available as a function but not exposed via CLI; the current
    implementation focuses on primary-condition DTW for temporal matching.

Current defaults:
- DTW: Primary condition only (use_multivariate=False)
- Embedding-RAG: Disabled by default (USE_EMBEDDING_RAG=False). Enable with `--use-embedding-rag`.
    (numeric/vector-RAG support removed)
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
import argparse
import requests
from tqdm import tqdm
import time
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from config import OPENROUTER_API_KEY
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
import importlib.util
import sys
import platform
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr, skew
from copy import deepcopy
import re
import pickle

# Runtime condition (can be 'depression' or 'anxiety')
CONDITION = 'depression'  # default, can be overridden via CLI
VALID_CONDITIONS = ['depression', 'anxiety']

# Paths (will be updated in __main__ based on CONDITION)
TEXT_DB_PATH = None
TEST_DATA_PATH = None
# RESULTS_DIR will be set per-run in the __main__ block (pointing to llm_output/<condition>/<run_name>)
# Avoid creating directories at import time so running or importing the module doesn't produce side-effects.
RESULTS_DIR = None
# Global adaptive weights (computed at runtime in __main__ if training data is available)
ADAPTIVE_WEIGHTS = None
# Embedding-RAG globals
EMBEDDED_DB_CACHE = {}
# Disable embedding-RAG by default per request (will skip text-embedding retrieval/append)
USE_EMBEDDING_RAG = False
# Toggle whether to request the model to include explicit reasoning in its response
# This will be added to the API payload as the `reasoning` parameter when present.
# Set to True to enable reasoning by default for all LLM calls.
ENABLE_REASONING = True
# Toggle inclusion of the demographic & clinical distribution summary (TOP-N DTW matches)
# When False the distribution block will be omitted from prompts. Set to True to enable.
INCLUDE_DEMOGRAPHIC_DISTRIBUTION = False
_openai_client = None
PRECOMPUTED_QUERY_EMBS = {}

# Simple API rate-limiting and debug globals (used by call_openrouter_api)
_api_lock = threading.Lock()
_last_api_call_time = 0.0
_api_call_count = 0
_debug_api_calls = True
_min_interval_between_calls = 0.5  # seconds between API calls
_printed_api_examples = 0  # how many full request/response examples we've printed
# Toggle to show appended embedding-RAG match messages (previously named vector-RAG)
_print_vector_append_logs = False
# Debug flag retained (logs insertion method for the embedding-RAG block)
_debug_vector_insert = False
# Collect the first N API examples (request + response) for atomic printing after the call
_api_examples = []  # list of dicts: {'call_index':int,'prompt':str,'weights':..., 'response':str}
_api_examples_lock = threading.Lock()
_api_examples_to_capture = 3
_api_examples_collected = 0
_api_examples_reserved = 0
# After the first N captured API examples, capture one example every this many API calls
# (one patient == one API call in the normal pipeline).
_api_example_periodic_interval = 250


def _sanitize_model_name(model_name: str) -> str:
    """Sanitize model name to be used in filesystem (replace slashes and illegal chars)."""
    # replace / and spaces with - and remove other problematic chars
    sanitized = re.sub(r'[^A-Za-z0-9._-]+', '-', model_name)
    return sanitized

# Model Configuration
LLM_MODEL = "meta-llama/llama-3.3-70b-instruct"

# Reusable, concise guiding principles to prepend to forecasting prompts.
# Keep these as soft constraints that the LLM should apply when producing
# short-term forecasts. Kept terse so prompts remain clear and scannable.
# ORIGINAL_PROMPT_GUIDING_FORECASTING_PRINCIPLES (kept here as a commented backup so it can be restored easily)
"""
**GUIDING FORECASTING PRINCIPLES (soft constraints):**
1. Temporal continuity: Day 11 should be close to Day 10 (typical change ±1.0). Use Day 10 as the main anchor.
2. Gradual change: Prefer daily changes ≲0.5 points; avoid unrealistic sudden jumps when possible.
3. Trend inertia: Recent momentum (days 8–10) often continues short-term unless strong contrary evidence.
4. Regression to the mean: Extreme values may drift toward patient baseline unless similar cases show sustained extremes.
5. Evidence-based: Align predictions with matched historical cases and the patient's observed volatility.
6. Empirical distribution: A numeric empirical distribution (from matched cases) is provided below the retrieved examples — use its center, spread, and any modes to guide your central forecast and uncertainty.
   - Weight distribution statistics by case similarity: higher-similarity matches should pull the forecast closer to their outcomes.
   - If the empirical distribution is multimodal, reflect that in your uncertainty (mention multiple plausible modes).
   - If distributional guidance conflicts with recent patient momentum, prioritize high-similarity cases but briefly state which signal you followed and why.
7. Evidence-to-output link: When returning the JSON forecast, include a short rationale referencing the empirical distribution (mean/spread or mode) and the top matching cases that influenced the prediction.
"""

# Minimal, unconstrained guiding text (active):
PROMPT_GUIDING_FORECASTING_PRINCIPLES = (
    "You are a clinical forecasting specialist. Use the provided patient data (days 1–10), the retrieved DTW historical cases, and any empirical summaries included in the prompt to produce short-term forecasts.\n"
    "You may adopt any reasoning approach you prefer and use your pre-trained clinical knowledge and judgement to improve predictions — there are no imposed restrictions on method or allowable deviations from the data.\n"
    "Treat the empirical distribution and DTW cases as informative inputs, not binding rules. Give concise, actionable rationale for your final choices.\n"
)


def load_text_database():
    """Load the text-based patient database (no embeddings needed for DTW-based RAG)"""
    print("Loading patient text database for DTW-based RAG...")
    with open(TEXT_DB_PATH, 'r') as f:
        text_db = json.load(f)
    print(f"✅ Loaded {len(text_db)} patient records")
    return text_db


def load_test_data():
    """Load test dataset"""
    print("\nLoading test data...")
    df_test = pd.read_csv(TEST_DATA_PATH)
    print(f"✅ Loaded {len(df_test)} test patients")
    return df_test


def validate_test_dataframe(df):
    """Validate that the test DataFrame contains required columns for the selected CONDITION.

    Raises:
        ValueError: if required columns are missing or dataframe is empty.
    """
    # Required columns: days 1-10 and target days 11-20
    dep_days_1_10 = [f'{CONDITION}_day{i}' for i in range(1, 11)]
    dep_days_11_20 = [f'target_{CONDITION}_day{i}' for i in range(11, 21)]
    required = dep_days_1_10 + dep_days_11_20

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Test data missing required columns for condition '{CONDITION}': {missing}\n"
            f"Expected columns like {dep_days_1_10[:3]} ... and {dep_days_11_20[:3]} ...\n"
            f"Check that the CSV at {TEST_DATA_PATH} contains the correct columns or that --condition is set correctly."
        )

    if df.empty:
        raise ValueError(f"Test data at {TEST_DATA_PATH} is empty")

    print(f"✅ Test data validation passed: contains required columns for '{CONDITION}' and {len(df)} rows")


def validate_text_db(text_db, sample_size: int = 10):
    """Basic validation for the text database JSON structure.

    This checks that the text_db is a non-empty list and that at least some
    records contain the expected metadata key for the selected CONDITION.
    """
    if not isinstance(text_db, list) or len(text_db) == 0:
        raise ValueError(f"Text DB at {TEXT_DB_PATH} is empty or not a list")

    db_days_key = f"{CONDITION}_days_1_10"
    checked = min(sample_size, len(text_db))
    found = 0
    for rec in text_db[:checked]:
        if isinstance(rec, dict) and 'metadata' in rec and db_days_key in rec['metadata']:
            found += 1

    if found == 0:
        # Warn rather than crash because DB might store time series differently,
        # but inform the user that retrieval may be affected.
        print(f"⚠️  Warning: None of the first {checked} records contained '{db_days_key}' in metadata.")
        print("   Retrieval may fail or produce poor matches. Inspect the text DB format.")
    else:
        print(f"✅ Text DB validation passed: found '{db_days_key}' in {found}/{checked} sampled records")


def compute_dtw_distance(series1, series2, radius=2):
    """
    Compute Dynamic Time Warping distance between two time series.
    
    Args:
        series1: First time series (list or array)
        series2: Second time series (list or array)
    radius: FastDTW radius constraint (lower = faster, default: 2)
    
    Returns:
        DTW distance (lower = more similar)
    """
    try:
        # Convert to numpy arrays and reshape for fastdtw
        s1 = np.array(series1).reshape(-1, 1)
        s2 = np.array(series2).reshape(-1, 1)
        
        # Compute DTW distance with radius constraint for speed
        distance, path = fastdtw(s1, s2, radius=radius, dist=euclidean)
        return distance
    except Exception as e:
        print(f"   ⚠️  DTW computation error: {e}")
        return float('inf')  # Return infinite distance on error


def compute_multivariate_dtw_distance(query_features, db_features, weights=None):
    """
    Compute multi-variate DTW distance across multiple symptom time series.
    
    This considers the full patient trajectory including depression and comorbid symptoms,
    providing a more comprehensive similarity measure than depression alone.
    
    Args:
        query_features: Dict mapping symptom names to time series (e.g., {'depression': [...], 'anxiety': [...]})
        db_features: Dict mapping symptom names to time series for database patient
        weights: Dict mapping symptom names to importance weights (default: equal weights)
    
    Returns:
        Weighted multi-variate DTW distance (lower = more similar)
    """
    if weights is None:
        # Default weights: depression is primary (50%), comorbidities share remaining (50%)
        weights = {
            'depression': 0.50,
            'anxiety': 0.10,
            'fatigue': 0.08,
            'pain': 0.08,
            'sadness': 0.08,
            'stress': 0.08,
            'irritability': 0.04,
            'cognitive_dysfunction': 0.04
        }
    
    total_distance = 0.0
    total_weight = 0.0
    
    for symptom, query_series in query_features.items():
        if symptom in db_features and symptom in weights:
            # Get corresponding database series
            db_series = db_features[symptom]
            
            # Only compute if both series are valid (not None/empty)
            if query_series and db_series and len(query_series) == len(db_series):
                # Compute DTW for this symptom
                distance = compute_dtw_distance(query_series, db_series)
                
                # Add weighted distance
                weight = weights[symptom]
                total_distance += weight * distance
                total_weight += weight
    
    # Normalize by total weight (in case some symptoms were missing)
    if total_weight > 0:
        return total_distance / total_weight
    else:
        return float('inf')  # No valid comparisons possible


def compute_adaptive_weights_from_df(df, condition=CONDITION, min_primary_weight=0.3, floor=0.01):
    """
    Compute adaptive symptom weights from a training DataFrame.

    Strategy:
      - For the primary condition (e.g., depression) use mean(days1-10).
      - For contextual/comorbid features (e.g., anxiety_mean_10d) use that column directly.
      - Compute absolute Pearson correlation between each feature and the target mean (days11-20).
      - Convert correlations to non-negative scores, apply a floor, normalize to sum to 1.
      - Ensure primary condition has at least `min_primary_weight` by boosting and renormalizing.

    Returns a dict mapping symptom names used by compute_multivariate_dtw_distance to weights.
    """
    import numpy as _np

    sympt_features = {
        'depression': [f'{condition}_day{i}' for i in range(1, 11)],
        'anxiety': ['anxiety_mean_10d'],
        'fatigue': ['fatigue_mean_10d'],
        'pain': ['pain_mean_10d'],
        'sadness': ['sadness_mean_10d'],
        'stress': ['stress_mean_10d'],
        'irritability': ['irritability_mean_10d'],
        'cognitive_dysfunction': ['cognitive_dysfunction_mean_10d']
    }

    # Target: mean of days 11-20
    target_cols = [f'target_{condition}_day{i}' for i in range(11, 21)]
    df_clean = df.dropna(subset=target_cols, how='any')
    if df_clean.empty:
        raise ValueError('Cannot compute adaptive weights: no rows with complete target days in provided training data')

    target_mean = df_clean[target_cols].mean(axis=1)

    scores = {}
    for name, cols in sympt_features.items():
        try:
            if name == 'depression':
                vals = df_clean[cols].mean(axis=1)
            else:
                col = cols[0]
                if col in df_clean.columns:
                    vals = df_clean[col].fillna(df_clean[col].mean())
                else:
                    vals = None

            if vals is None or vals.std() == 0 or target_mean.std() == 0:
                scores[name] = 0.0
            else:
                corr, _ = pearsonr(vals, target_mean)
                scores[name] = abs(corr) if not _np.isnan(corr) else 0.0
        except Exception:
            scores[name] = 0.0

    # Apply floor and normalize
    keys = list(sympt_features.keys())
    vec = _np.array([max(floor, scores.get(k, 0.0)) for k in keys], dtype=float)
    if vec.sum() <= 0:
        vec = _np.ones_like(vec) / len(vec)
    else:
        vec = vec / vec.sum()

    # Ensure primary has at least min_primary_weight
    primary_idx = keys.index('depression')
    if vec[primary_idx] < min_primary_weight:
        vec[primary_idx] = min_primary_weight
        others_sum = vec.sum() - vec[primary_idx]
        if others_sum <= 0:
            rem = 1.0 - min_primary_weight
            for i in range(len(vec)):
                if i != primary_idx:
                    vec[i] = rem / (len(vec) - 1)
        else:
            scale = (1.0 - vec[primary_idx]) / others_sum
            for i in range(len(vec)):
                if i != primary_idx:
                    vec[i] *= scale

    weights = {k: float(vec[i]) for i, k in enumerate(keys)}
    return weights


def compute_per_patient_weights(test_patient, base_weights=None, min_primary_weight=0.3, boost_scale=0.35):
    """
    Produce per-patient adaptive weights by adjusting a base weight vector according
    to the patient's contextual symptom severities.

    Strategy (lightweight and deterministic):
      - Start from `base_weights` (if provided) or the default static weights.
      - For each contextual feature (e.g. anxiety_mean_10d) compute a small "boost"
        proportional to severity (normalized to [0,1] by dividing by 4.0).
      - Add boosts to the matching symptom weights, then renormalize to sum to 1.
      - Ensure the primary condition ('depression') retains at least `min_primary_weight`.

    This is inexpensive and avoids retraining; it's intended as a sensible per-patient
    re-weighting heuristic rather than a learned mapping.
    """
    # Default base if none provided
    default_base = {
        'depression': 0.50,
        'anxiety': 0.10,
        'fatigue': 0.08,
        'pain': 0.08,
        'sadness': 0.08,
        'stress': 0.08,
        'irritability': 0.04,
        'cognitive_dysfunction': 0.04
    }

    base = deepcopy(base_weights) if base_weights else deepcopy(default_base)

    # Map contextual feature keys (from CSV / DB) to symptom keys used in weights
    symptom_map = {
        'anxiety_mean_10d': 'anxiety',
        'fatigue_mean_10d': 'fatigue',
        'pain_mean_10d': 'pain',
        'sadness_mean_10d': 'sadness',
        'stress_mean_10d': 'stress',
        'irritability_mean_10d': 'irritability',
        'cognitive_dysfunction_mean_10d': 'cognitive_dysfunction'
    }

    contextual = test_patient.get('contextual_features', {}) or {}

    # Compute additive boosts (bounded) based on severity [0-4]
    boosts = {k: 0.0 for k in base.keys()}
    total_boost = 0.0
    for feat, sym in symptom_map.items():
        if feat in contextual and pd.notna(contextual[feat]):
            try:
                val = float(contextual[feat])
            except Exception:
                val = 0.0
            # clamp to 0-4 and normalize
            val = max(0.0, min(4.0, val))
            norm = val / 4.0
            b = boost_scale * norm
            boosts[sym] += b
            total_boost += b

    # Apply boosts to base weights, then normalize
    weights = {}
    for k in base:
        weights[k] = base.get(k, 0.0) + boosts.get(k, 0.0)

    s = sum(weights.values())
    if s <= 0:
        weights = deepcopy(default_base)
        s = sum(weights.values())

    for k in list(weights.keys()):
        weights[k] = float(weights[k] / s)

    # Enforce minimum primary weight and renormalize others if needed
    if weights.get('depression', 0.0) < min_primary_weight:
        weights['depression'] = float(min_primary_weight)
        others = [k for k in weights.keys() if k != 'depression']
        others_sum = sum(weights[k] for k in others)
        if others_sum > 0:
            scale = (1.0 - weights['depression']) / others_sum
            for k in others:
                weights[k] = float(weights[k] * scale)
        else:
            # evenly distribute remainder
            rem = 1.0 - weights['depression']
            for k in others:
                weights[k] = float(rem / len(others))

    return weights


def find_similar_patients_dtw(query_features, text_db, top_k=3, use_multivariate=True, 
                              use_fast_prefilter=True, prefilter_size=500, weights=None,
                              ensure_unique_patients=True):
    """
    Find top-k most similar patients using DTW on symptom time series (days 1-10).
    This is the RETRIEVAL component of DTW-based RAG.
    
    OPTIMIZATION: Uses 2-stage filtering for speed:
    1. Fast pre-filter: Euclidean distance on depression → top 500 candidates
      2. Precise filter: Multi-variate DTW on candidates → top k results
    
    Args:
        query_features: Dict mapping symptom names to time series
                       e.g., {'depression': [...], 'anxiety': [...], 'fatigue': [...]}
        text_db: List of patient records from text database
    top_k: Number of similar patients to retrieve (default reduced to 3)
        use_multivariate: If True, use multi-variate DTW across all symptoms (default: True)
                         If False, use only depression trajectory
        use_fast_prefilter: If True, use Euclidean pre-filtering (much faster)
    prefilter_size: Number of candidates to consider for DTW (default: 500)
    
    Returns:
        List of (record, similarity_score) tuples (lower DTW distance = higher similarity)
    """
    
    # STAGE 1: Fast pre-filtering with Euclidean distance (optional but recommended)
    if use_fast_prefilter and len(text_db) > prefilter_size:
        # Primary symptom series (e.g., 'depression' or 'anxiety')
        query_primary = query_features.get(CONDITION, [])
        euclidean_distances = []
        
        for record in text_db:
            # Database stores primary symptom under e.g. 'depression_days_1_10' or 'anxiety_days_1_10'
            db_days_key = f"{CONDITION}_days_1_10"
            dep_1_10_dict = record['metadata'].get(db_days_key, {})
            db_depression_series = [dep_1_10_dict.get(f'day_{i}', 0.0) for i in range(1, 11)]
            
            # Fast Euclidean distance
            euc_dist = np.sqrt(np.sum((np.array(query_primary) - np.array(db_depression_series))**2))
            euclidean_distances.append((record, euc_dist))
        
        # Sort by Euclidean distance and take top candidates
        euclidean_distances.sort(key=lambda x: x[1])
        candidates = [record for record, _ in euclidean_distances[:prefilter_size]]
    else:
        candidates = text_db
    
    # STAGE 2: Precise DTW filtering on candidates only
    dtw_distances = []
    
    for record in candidates:
        # Extract primary-condition days 1-10 from database record
        db_days_key = f"{CONDITION}_days_1_10"
        dep_1_10_dict = record['metadata'].get(db_days_key, {})
        db_depression_series = [dep_1_10_dict.get(f'day_{i}', 0.0) for i in range(1, 11)]
        
        if use_multivariate:
            # Build multi-variate feature dict for database patient
            # Primary feature key should match CONDITION variable
            db_features = {CONDITION: db_depression_series}

            # Extract comorbid symptom time series from database if available
            # The database stores contextual features as aggregated values, but we'll use them
            contextual = record['metadata'].get('contextual_features', {})

            # Map contextual features to time series (using mean values as proxy)
            # Note: If database has actual time series, extract those instead
            symptom_map = {
                'anxiety_mean_10d': 'anxiety',
                'fatigue_mean_10d': 'fatigue',
                'pain_mean_10d': 'pain',
                'sadness_mean_10d': 'sadness',
                'stress_mean_10d': 'stress',
                'irritability_mean_10d': 'irritability',
                'cognitive_dysfunction_mean_10d': 'cognitive_dysfunction'
            }

            for db_key, symptom_name in symptom_map.items():
                if db_key in contextual and pd.notna(contextual[db_key]):
                    # Create constant time series from mean value
                    # This is a simplification; ideally we'd have actual time series
                    db_features[symptom_name] = [contextual[db_key]] * 10

            # Compute multi-variate DTW distance (pass adaptive weights if provided)
            distance = compute_multivariate_dtw_distance(query_features, db_features, weights=weights)
        else:
            # Use only primary-condition trajectory
            query_primary = query_features.get(CONDITION, [])
            distance = compute_dtw_distance(query_primary, db_depression_series)
        
        # Convert distance to similarity score (0-1 range, higher = more similar)
        # Using exponential decay: similarity = e^(-distance/scale)
        # Scale factor controls how quickly similarity decreases with distance
        scale = 5.0  # Adjust based on typical DTW distances
        similarity = np.exp(-distance / scale)
        
        dtw_distances.append((record, similarity, distance))
    
    # Sort by distance (ascending = most similar first)
    dtw_distances.sort(key=lambda x: x[2])
    
    # If deduplication requested, select best window per patient id
    if True:
        ensure = True
    else:
        ensure = False

    if ensure_unique_patients:
        seen = {}
        ordered = []
        for record, sim, dist in dtw_distances:
            meta = record.get('metadata', {}) or {}
            pid = None
            pmeta = meta.get('patient_metadata') if isinstance(meta.get('patient_metadata'), dict) else None
            if pmeta:
                pid = pmeta.get('patient_id') or pmeta.get('id') or pid
            pid = pid or meta.get('patient_id') or meta.get('user_seq_id') or record.get('source') or None
            pid_key = pid if pid is not None else f"_rec_{id(record)}"
            if pid_key not in seen:
                seen[pid_key] = (record, sim, dist)
                ordered.append((record, sim, dist))
            if len(ordered) >= top_k:
                break
        return [(r, s) for r, s, _ in ordered[:top_k]]

    # No dedupe: return top-k with (record, similarity_score) format
    return [(record, similarity) for record, similarity, _ in dtw_distances[:top_k]]


def create_dtw_prompt(test_patient, similar_patients, top_k=5):
    """
    Create enhanced prompt using DTW-matched similar patients.
    This is the AUGMENTATION component of DTW-based RAG.
    
    Args:
        test_patient: Dictionary with test patient data
        similar_patients: List of (record, similarity) tuples from DTW matching
        top_k: Number of similar patients to include
    
    Returns:
        Formatted prompt string with retrieved context
    """
    # Extract and analyze target patient's primary-condition pattern
    dep_values = [test_patient[f'{CONDITION}_days_1_10'][f'day_{i}'] for i in range(1, 11)]
    baseline_avg = np.mean(dep_values[:3])  # First 3 days
    mid_avg = np.mean(dep_values[3:7])      # Middle days 4-7
    recent_avg = np.mean(dep_values[-3:])   # Last 3 days
    overall_avg = np.mean(dep_values)
    trend = recent_avg - baseline_avg
    volatility = np.std(dep_values)
    
    # Calculate momentum (is trend accelerating or decelerating?)
    early_trend = mid_avg - baseline_avg
    late_trend = recent_avg - mid_avg
    momentum = late_trend - early_trend
    
    # Last observed value (critical for continuity)
    last_value = dep_values[-1]
    
    # Determine trend description
    if trend > 0.5:
        trend_desc = "worsening (increasing severity)"
    elif trend < -0.5:
        trend_desc = "improving (decreasing severity)"
    else:
        trend_desc = "stable (minimal change)"
    
    # Determine volatility description
    if volatility > 1.0:
        volatility_desc = f"High (σ={volatility:.2f})"
    elif volatility > 0.5:
        volatility_desc = f"Moderate (σ={volatility:.2f})"
    else:
        volatility_desc = f"Low (σ={volatility:.2f})"
    
    # Momentum description
    if abs(momentum) > 0.3:
        momentum_desc = "accelerating" if momentum > 0 else "decelerating"
    else:
        momentum_desc = "steady"
    
    cond_label = CONDITION.capitalize()

    prompt = f"""You are a clinical forecasting specialist. Your task is to predict {cond_label} severity for days 11-20 based on days 1-10, similar patient trajectories, and your own pre-trained internal clinical knowledge.

{PROMPT_GUIDING_FORECASTING_PRINCIPLES}

RESPONSE FORMAT (MUST FOLLOW EXACTLY):
- You MUST output only two blocks in this exact order: a thinking block and a JSON block.
- THINKING block: must be wrapped between <<<THINKING>>> and <<<END_THINKING>>> on their own lines.
    Inside the thinking block include, in order, these tags on their own lines: <dtw_reasoning>...</dtw_reasoning>, <draft>...</draft>, <clinical_adjustment>...</clinical_adjustment>, then a single line with the exact text: End thought
- JSON block: must be valid JSON wrapped between <<<JSON>>> and <<<END_JSON>>> on their own lines. Only include keys "day_11".."day_20" with numeric values in [0,4]. No other text allowed.
- Markers and tags are case-sensitive and must appear exactly as shown, each on its own line with no extra characters or commentary.
- You MUST use the specified format with the THINKING and JSON blocks.

***IMPORTANT: STRUCTURED RESPONSE DIVIDER***: Do NOT emit raw internal chain-of-thought. If you performed internal, private reasoning while composing your answer, do NOT reproduce that internal stream verbatim.

Instead, when you are ready to produce the machine-readable output, begin your response with the exact divider line below (on its own line, with no additional characters or commentary):

--- BEGIN STRUCTURED RESPONSE ---

Immediately after that divider, provide the required <<<THINKING>>> block followed by the <<<JSON>>> block described below. No other text may appear before the divider. The <<<THINKING>>> block should be a concise, structured SUMMARY of your reasoning (not an uncensored transcript of internal thoughts).

Example minimal response shape (you must follow this shape exactly):
<<<THINKING>>>
<dtw_reasoning>
...short bullets...
</dtw_reasoning>
<draft>
day_11: 2.0
...
day_20: 1.4
</draft>
<clinical_adjustment>
...brief lines describing adjustments...
</clinical_adjustment>
End thought
<<<END_THINKING>>>

<<<JSON>>>
{{"day_11": 2.0, "day_12": 2.0, "day_13": 2.0, "day_14": 1.9, "day_15": 1.8, "day_16": 1.7, "day_17": 1.6, "day_18": 1.5, "day_19": 1.5, "day_20": 1.4}}
<<<END_JSON>>>

**SCALE**: 0=None, 1=Mild, 2=Moderate, 3=Severe, 4=Very Severe

**TARGET PATIENT** (predict days 11-20):

{cond_label} Trajectory (Days 1-10):
{', '.join([f'D{i}:{dep_values[i-1]:.1f}' for i in range(1, 11)])}

**Key Statistics:**
• Last Value (Day 10): {last_value:.2f} (use as the main anchor for Day 11)
• Baseline (D1-3): {baseline_avg:.2f}
• Recent (D8-10): {recent_avg:.2f}
• Overall Average: {overall_avg:.2f}
• 10-Day Trend: {trend_desc} (Δ={trend:+.2f})
• Momentum: {momentum_desc} ({momentum:+.2f})
• Volatility: {volatility_desc}
"""
    
    # Add demographics and other pre-target clinical features to remind the LLM
    age = test_patient.get('age', 'Unknown')
    sex = test_patient.get('sex', 'Unknown')
    country = test_patient.get('country', 'Unknown')
    clinical_info = test_patient.get('clinical_features', {}) or {}

    prompt += "\n**DEMOGRAPHICS & CLINICAL FEATURES (pre-target):**\n"
    prompt += f"• Age: {age}\n"
    prompt += f"• Sex: {sex}\n"
    prompt += f"• Country: {country}\n"

    # Add any additional pre-target clinical columns (e.g., meds, treatments, diagnoses)
    if clinical_info:
        prompt += "\n**Other clinical fields provided (pre-target):**\n"
        # Keep ordering deterministic for readability
        for k in sorted(clinical_info.keys()):
            try:
                v = clinical_info[k]
                # Skip duplicate demographic keys if present
                if k in ('age', 'sex', 'country'):
                    continue
                # Shorten long strings for prompt readability
                if isinstance(v, str) and len(v) > 200:
                    v_short = v[:200] + '...'
                else:
                    v_short = v
                prompt += f"• {k}: {v_short}\n"
            except Exception:
                continue

    # Add target patient's contextual features with severity interpretation
    target_contextual = test_patient.get('contextual_features', {})
    if target_contextual:
        prompt += "\n**Comorbid Symptoms** (Days 1-10 avg):\n"
        
        symptom_map = {
            'anxiety_mean_10d': 'Anxiety',
            'fatigue_mean_10d': 'Fatigue',
            'pain_mean_10d': 'Pain',
            'sadness_mean_10d': 'Sadness',
            'stress_mean_10d': 'Stress',
            'irritability_mean_10d': 'Irritability',
            'cognitive_dysfunction_mean_10d': 'Cognitive Issues'
        }
        
        high_symptoms = []
        for feature, label in symptom_map.items():
            value = target_contextual.get(feature)
            if value is not None and pd.notna(value):
                if value >= 2.0:  # Show moderate+ symptoms
                    high_symptoms.append(f"{label}={value:.1f}")
        
        if high_symptoms:
            prompt += "• " + ", ".join(high_symptoms) + f" (may sustain/worsen {CONDITION})\n"
        else:
            prompt += "• No severe comorbid symptoms\n"
    
    # Limit DTW examples to up to 3 (reduce appended historical cases to the top 3 most similar)
    dtw_k = min(3, top_k)
    prompt += f"""
**SIMILAR HISTORICAL CASES — DTW (top {dtw_k})** (Ranked by DTW similarity):
Context: The following patient cases were retrieved using Dynamic Time Warping (DTW) to match similar temporal patterns in the primary symptom trajectory over days 1-10, focusing on phase and shape resemblance

"""

    # Add similar patients with enhanced trajectory analysis (DTW matches)
    for i, (record, similarity) in enumerate(similar_patients[:dtw_k], 1):
        meta = record['metadata'].get('patient_metadata', {})
        dep_1_10_dict = record['metadata'].get(f'{CONDITION}_days_1_10', {})
        dep_11_20_dict = record['metadata'].get(f'target_{CONDITION}_days_11_20', {})
        
        # Convert to lists for analysis
        sim_dep_1_10 = [dep_1_10_dict.get(f'day_{j}', 0.0) for j in range(1, 11)]
        sim_dep_11_20 = [dep_11_20_dict.get(f'day_{j}', 0.0) for j in range(11, 21)]
        
        # Calculate trajectory
        sim_baseline = np.mean(sim_dep_1_10[:3])
        sim_recent = np.mean(sim_dep_1_10[-3:])
        sim_trend = sim_recent - sim_baseline
        sim_last = sim_dep_1_10[-1]
        
        # Outcome analysis
        outcome_avg = np.mean(sim_dep_11_20)
        outcome_first = sim_dep_11_20[0]  # Day 11
        outcome_last = sim_dep_11_20[-1]   # Day 20
        outcome_change = outcome_last - outcome_first
        
        # Trajectory description
        if sim_trend > 0.5:
            initial_pattern = "↑worsening"
        elif sim_trend < -0.5:
            initial_pattern = "↓improving"
        else:
            initial_pattern = "→stable"
            
        if outcome_change > 0.5:
            outcome_pattern = "↑worsened"
        elif outcome_change < -0.5:
            outcome_pattern = "↓improved"
        else:
            outcome_pattern = "→stable"
        
        prompt += f"""Case {i} [{similarity:.1%} match]:
  D1-10: {', '.join([f'{v:.1f}' for v in sim_dep_1_10])} [{initial_pattern}, trend={sim_trend:+.1f}]
  D11-20: {', '.join([f'{v:.1f}' for v in sim_dep_11_20])} [Outcome: {outcome_pattern}, avg={outcome_avg:.1f}]
"""
        # Include demographics for the similar historical case when available
        try:
            dem_items = []
            # patient_metadata block may contain demographic keys; also check top-level metadata
            sim_age = meta.get('age') if isinstance(meta, dict) else None
            if sim_age is None:
                sim_age = record['metadata'].get('age')
            sim_sex = meta.get('sex') if isinstance(meta, dict) else None
            if sim_sex is None:
                sim_sex = record['metadata'].get('sex')
            sim_country = meta.get('country') if isinstance(meta, dict) else None
            if sim_country is None:
                sim_country = record['metadata'].get('country')

            if sim_age is not None and str(sim_age).strip() != '':
                dem_items.append(f"Age={sim_age}")
            if sim_sex is not None and str(sim_sex).strip() != '':
                dem_items.append(f"Sex={sim_sex}")
            if sim_country is not None and str(sim_country).strip() != '':
                dem_items.append(f"Country={sim_country}")

            if dem_items:
                prompt += "  Demographics: " + ", ".join(dem_items) + "\n"
        except Exception:
            pass
        # Include comorbid/contextual features for this DTW case when available
        # Only include contextual fields that are positive/non-zero or non-empty.
        contextual = record['metadata'].get('contextual_features', {}) or {}
        if contextual:
            ctx_items = []
            # Keys to hide from prompts (internal IDs, window boundaries)
            skip_ctx_keys = {'user_seq_id', 'window_end_date', 'window_start_date'}
            for k, v in contextual.items():
                # Skip empty/None and internal keys
                if v is None or k in skip_ctx_keys:
                    continue
                try:
                    fv = float(v)
                    # Include only positive/non-zero numeric context
                    if fv != 0.0 and not np.isnan(fv):
                        if fv > 0:
                            ctx_items.append(f"{k}={fv:.2f}")
                        else:
                            # include negative values as well if they exist (rare)
                            ctx_items.append(f"{k}={fv:.2f}")
                    else:
                        continue
                except Exception:
                    s = str(v).strip()
                    if s:
                        ctx_items.append(f"{k}={s}")
            if ctx_items:
                prompt += "  Comorbid: " + ", ".join(ctx_items) + "\n"
        # Ensure there is a blank line after each case for readability in the prompt
        prompt += "\n"
        # After Case 3, append empirical distribution summary across the top-25 DTW matches
        try:
            if i == 3:
                # Collect days 11-20 scores from the top-N similar patients (N=25 or fewer if not available)
                top_n = min(25, len(similar_patients))
                collected = []
                for rec, sim2 in similar_patients[:top_n]:
                    tdict = rec.get('metadata', {}) or {}
                    # Attempt to find the target days block under common keys
                    block = tdict.get(f'target_{CONDITION}_days_11_20') or tdict.get(f'{CONDITION}_days_11_20') or tdict.get('target_days_11_20')
                    vals = None
                    if isinstance(block, dict):
                        vals = [block.get(f'day_{d}') for d in range(11, 21)]
                    elif isinstance(block, list) or isinstance(block, tuple):
                        # If stored as list-like, assume length >= 10 and take first 10 entries for days 11-20
                        if len(block) >= 10:
                            vals = list(block[:10])
                    # If block not found at top-level, check nested metadata key used elsewhere
                    if vals is None:
                        nested = rec.get('metadata', {}).get('patient_data') or rec.get('metadata', {}).get('metadata') or {}
                        if isinstance(nested, dict):
                            block2 = nested.get(f'target_{CONDITION}_days_11_20') or nested.get(f'{CONDITION}_days_11_20')
                            if isinstance(block2, dict):
                                vals = [block2.get(f'day_{d}') for d in range(11, 21)]

                    # Only include if we successfully extracted 10 numeric-ish values
                    if vals and len(vals) == 10:
                        # Convert to floats where possible, keep NaN for missing
                        row = []
                        for v in vals:
                            try:
                                row.append(float(v))
                            except Exception:
                                row.append(np.nan)
                        collected.append(row)

                if collected:
                    arr = np.array(collected, dtype=float)  # shape (n_patients, 10)
                    per_day_stats = []
                    histograms = []
                    for col_idx in range(arr.shape[1]):
                        col = arr[:, col_idx]
                        # drop NaNs
                        col_valid = col[~np.isnan(col)]
                        if col_valid.size == 0:
                            mean = median = iqr = p10 = p25 = p75 = p90 = skewness = np.nan
                            hist_counts = {b: 0 for b in range(0, 5)}
                            modality = 'unknown'
                            skew_desc = 'n/a'
                        else:
                            mean = float(np.mean(col_valid))
                            median = float(np.median(col_valid))
                            p10 = float(np.percentile(col_valid, 10))
                            p25 = float(np.percentile(col_valid, 25))
                            p75 = float(np.percentile(col_valid, 75))
                            p90 = float(np.percentile(col_valid, 90))
                            iqr = p75 - p25
                            # histogram aggregated into integer bins 0..4 by rounding to nearest integer
                            rounded = np.clip(np.rint(col_valid).astype(int), 0, 4)
                            hist_counts = {b: int((rounded == b).sum()) for b in range(0, 5)}

                            # Skewness: positive => right-skewed (long right tail), negative => left-skewed
                            try:
                                skewness = float(skew(col_valid))
                            except Exception:
                                skewness = float(np.nan)

                            if np.isnan(skewness):
                                skew_desc = 'n/a'
                            elif skewness > 0.3:
                                skew_desc = 'right-skewed'
                            elif skewness < -0.3:
                                skew_desc = 'left-skewed'
                            else:
                                skew_desc = 'approximately symmetric'

                            # Simple modality heuristic on histogram counts: count local peaks
                            try:
                                # use 7 bins across 0-4 range for modest granularity
                                bins = np.linspace(0, 4, 8)
                                counts, _ = np.histogram(col_valid, bins=bins)
                                peaks = 0
                                for k in range(1, len(counts)-1):
                                    if counts[k] > counts[k-1] and counts[k] > counts[k+1]:
                                        peaks += 1
                                modality = 'multimodal' if peaks >= 2 else 'unimodal'
                            except Exception:
                                modality = 'unknown'

                        per_day_stats.append({
                            'mean': mean,
                            'median': median,
                            'iqr': iqr,
                            'p10': p10,
                            'p25': p25,
                            'p75': p75,
                            'p90': p90,
                            'skew': skewness,
                            'skew_desc': skew_desc,
                            'shape': modality
                        })
                        histograms.append(hist_counts)

                    # Append a readable summary to the prompt
                    prompt += "**EMPIRICAL DISTRIBUTION FROM TOP DTW MATCHES (days 11-20, top {} matches):**\n".format(top_n)
                    prompt += "Per-day statistics (Day, mean, 10th, 25th, 75th, 90th, IQR, skew, shape, histogram counts for bins 0-4):\n"
                    for j, day in enumerate(range(11, 21)):
                        stats = per_day_stats[j]
                        hist = histograms[j]
                        hist_str = ', '.join([f"{k}:{hist[k]}" for k in sorted(hist.keys())])
                        # Use safe formatting that handles NaNs
                        def _fmt(x):
                            try:
                                if x is None or (isinstance(x, float) and np.isnan(x)):
                                    return 'n/a'
                                return f"{x:.2f}"
                            except Exception:
                                return str(x)

                        prompt += (f"• Day {day}: mean={_fmt(stats.get('mean'))}, "
                                   f"10th={_fmt(stats.get('p10'))}, 25th={_fmt(stats.get('p25'))}, "
                                   f"75th={_fmt(stats.get('p75'))}, 90th={_fmt(stats.get('p90'))}, "
                                   f"IQR={_fmt(stats.get('iqr'))}, skew={_fmt(stats.get('skew'))} "
                                   f"({stats.get('skew_desc')}), shape={stats.get('shape')}; "
                                   f"hist={{ {hist_str} }}\n")
                    prompt += "\n"
        except Exception:
            # If anything fails here, don't break prompt generation; skip the empirical block
            pass
    # Optionally add demographic & clinical distribution summary across the TOP-N DTW matches,
    # stratified by trajectory type (recovery, worsening, stable).
    # Controlled by global flag INCLUDE_DEMOGRAPHIC_DISTRIBUTION (default: False).
    if globals().get('INCLUDE_DEMOGRAPHIC_DISTRIBUTION'):
        try:
            top_n = min(25, len(similar_patients))
            records = [r for r, _ in similar_patients[:top_n]]

            def _extract_target_series(rec):
                meta = rec.get('metadata', {}) if isinstance(rec, dict) else {}
                # Try nested dict like '{cond}_days_11_20'
                block = meta.get(f'target_{CONDITION}_days_11_20') or meta.get(f'{CONDITION}_days_11_20') or meta.get('target_days_11_20')
                if isinstance(block, dict):
                    vals = [block.get(f'day_{d}') for d in range(11, 21)]
                    if all(v is not None for v in vals):
                        return [float(v) for v in vals]
                # Try flat keys inside metadata
                vals = []
                for d in range(11, 21):
                    k = f'target_{CONDITION}_day{d}'
                    v = meta.get(k)
                    if v is None:
                        # Try alternate key
                        k2 = f'{CONDITION}_day{d}'
                        v = meta.get(k2)
                    vals.append(v)
                if any(v is None for v in vals):
                    return None
                return [float(v) for v in vals]

            def _extract_baseline_and_day10(rec):
                meta = rec.get('metadata', {}) if isinstance(rec, dict) else {}
                # Try nested primary days 1-10 dict
                block = meta.get(f'{CONDITION}_days_1_10')
                if isinstance(block, dict):
                    baseline = [block.get(f'day_{i}') for i in range(1, 4)]
                    day10 = block.get('day_10')
                    if all(v is not None for v in baseline) and day10 is not None:
                        return ([float(v) for v in baseline], float(day10))
                # Try flat keys
                baseline = []
                for i in range(1, 4):
                    k = f'{CONDITION}_day{i}'
                    baseline.append(meta.get(k))
                day10 = meta.get(f'{CONDITION}_day10')
                if any(v is None for v in baseline) or day10 is None:
                    return (None, None)
                return ([float(v) for v in baseline], float(day10))

            from collections import Counter

            # Dynamically discover comorbidity/contextual keys from the top-N records
            skip_ctx_keys = {'user_seq_id', 'window_end_date', 'window_start_date',
                             'age', 'sex', 'country', 'patient_metadata', 'patient_meta', 'metadata'}
            candidate_keys = set()
            for rec in records:
                meta = rec.get('metadata', {}) or {}
                # Prefer contextual_features dict if present
                ctx = None
                if isinstance(meta, dict):
                    ctx = meta.get('contextual_features') or meta.get('contextual')
                if isinstance(ctx, dict):
                    for k in ctx.keys():
                        kl = k.lower() if isinstance(k, str) else ''
                        if kl.find('window') != -1:
                            continue
                        if k not in skip_ctx_keys:
                            candidate_keys.add(k)
                # Also include top-level meta keys that look like symptom means
                if isinstance(meta, dict):
                    for k in meta.keys():
                        kl = k.lower() if isinstance(k, str) else ''
                        if kl.find('window') != -1:
                            continue
                        if k in skip_ctx_keys:
                            continue
                        if k.endswith('_mean_10d') or '_mean_' in k or k.endswith('_mean'):
                            candidate_keys.add(k)

            # Fallback to a small known list if nothing discovered
            if not candidate_keys:
                candidate_keys = set([
                    'anxiety_mean_10d', 'fatigue_mean_10d', 'pain_mean_10d',
                    'sadness_mean_10d', 'stress_mean_10d', 'irritability_mean_10d',
                    'cognitive_dysfunction_mean_10d'
                ])

            RECOVERY_TH = -0.5
            WORSENING_TH = 0.5

            groups = {'recovery': [], 'worsening': [], 'stable': []}
            for rec in records:
                targ = _extract_target_series(rec)
                baseline_vals, day10 = _extract_baseline_and_day10(rec)
                if targ is None or day10 is None:
                    continue
                mean_target = float(np.nanmean(targ))
                delta = mean_target - float(day10)
                if delta <= RECOVERY_TH:
                    traj = 'recovery'
                elif delta >= WORSENING_TH:
                    traj = 'worsening'
                else:
                    traj = 'stable'

                meta = rec.get('metadata', {}) or {}
                # Some DB records nest demographics under metadata['patient_metadata']
                patient_meta = {}
                if isinstance(meta, dict):
                    patient_meta = meta.get('patient_metadata') or meta.get('patient_meta') or {}
                row = rec.get('row', {}) or {}

                # Robustly search common places/keys for age and sex/gender
                age = None
                sex = None
                # 1) row dict (if present)
                if isinstance(row, dict) and row:
                    age = row.get('age') or row.get('Age') or row.get('age_years')
                    sex = row.get('sex') or row.get('Sex') or row.get('gender')
                # 2) nested patient_metadata
                if (age is None or sex is None) and isinstance(patient_meta, dict):
                    if age is None:
                        age = patient_meta.get('age') or patient_meta.get('Age')
                    if sex is None:
                        sex = patient_meta.get('sex') or patient_meta.get('Sex') or patient_meta.get('gender')
                # 3) top-level metadata
                if (age is None or sex is None) and isinstance(meta, dict):
                    if age is None:
                        age = meta.get('age') or meta.get('Age')
                    if sex is None:
                        sex = meta.get('sex') or meta.get('Sex') or meta.get('gender')

                # Normalize non-numeric/empty age to None
                try:
                    if age is not None and str(age).strip() != '':
                        # only convert when looks numeric
                        age_val = float(age)
                        age = age_val
                    else:
                        age = None
                except Exception:
                    age = None
                if sex is not None:
                    sex = str(sex).strip()
                    if sex == '':
                        sex = None

                # collect all available contextual/comorbidity info for this record
                contextual = {}
                ctx = meta.get('contextual_features') or meta.get('contextual') or {}
                if isinstance(ctx, dict) and ctx:
                    # copy contextual keys but exclude any window-related keys
                    contextual = {kk: vv for kk, vv in ctx.items() if 'window' not in (kk.lower() if isinstance(kk, str) else '')}
                else:
                    # fallback: include any metadata keys that look like symptom means
                    for kk, vv in meta.items():
                        kkl = kk.lower() if isinstance(kk, str) else ''
                        if ('window' in kkl) or kk in ('age', 'sex', 'country', 'patient_metadata', 'patient_meta', 'metadata', 'user_seq_id', 'window_start_date', 'window_end_date'):
                            continue
                        # include keys that contain 'mean' or '_mean_' as likely comorbidities
                        if ('mean' in kk) or kk.endswith('_score') or kk.endswith('_mean'):
                            contextual[kk] = vv

                groups[traj].append({'age': age, 'sex': sex, 'baseline_mean': (np.mean(baseline_vals) if baseline_vals else None), 'contextual': contextual})

            # Format block
            lines = []
            lines.append('\n**DEMOGRAPHIC & CLINICAL DISTRIBUTION AMONG TOP-%d DTW MATCHES (STRATIFIED BY TRAJECTORY)**\n' % top_n)
            lines.append('Assumptions: trajectory type defined by mean(days11-20) - Day10 change: <= %.2f => recovery, >= %.2f => worsening, else stable.\n' % (RECOVERY_TH, WORSENING_TH))
            for g in ['recovery', 'worsening', 'stable']:
                grp = groups[g]
                n = len(grp)
                lines.append(f'-- {g.upper()} (N={n}) --')
                if n == 0:
                    lines.append('  (no cases)\n')
                    continue
                ages = [float(x['age']) for x in grp if x.get('age') is not None]
                sexes = [x['sex'] for x in grp if x.get('sex')]
                baselines = [x['baseline_mean'] for x in grp if x.get('baseline_mean') is not None]
                if ages:
                    lines.append(f'  Age — mean: {np.mean(ages):.1f}, sd: {np.std(ages):.1f}, <30: {sum(a<30 for a in ages)}, 30-50: {sum((a>=30 and a<=50) for a in ages)}, >50: {sum(a>50 for a in ages)}')
                else:
                    lines.append('  Age — not available')
                if sexes:
                    c = Counter(sexes)
                    lines.append('  Sex — ' + ', '.join([f"{k}:{v}" for k, v in c.items()]))
                else:
                    lines.append('  Sex — not available')
                if baselines:
                    lines.append(f'  Baseline (D1-3) — mean: {np.mean(baselines):.2f}, sd: {np.std(baselines):.2f}')
                else:
                    lines.append('  Baseline (D1-3) — not available')
                # comorbidity prevalence: compute for all discovered contextual keys
                com_stats = []
                for k in sorted(candidate_keys):
                    vals = []
                    for x in grp:
                        ctx = x.get('contextual') or {}
                        v = None
                        if isinstance(ctx, dict):
                            v = ctx.get(k)
                        # allow falling back to group-level dict entries
                        if v is None:
                            # maybe stored as top-level baseline in metadata; skip here
                            v = None
                        if v is not None:
                            try:
                                vals.append(float(v))
                            except Exception:
                                pass
                    if vals:
                        preval = 100.0 * sum(1 for v in vals if v >= 1.0) / len(vals)
                        meanv = float(np.mean(vals))
                        # detect binary-like variables (only 0/1 values)
                        unique_vals = set([int(v) for v in np.unique(np.array(vals)) if not np.isnan(v)])
                        is_binary = unique_vals.issubset({0, 1})
                        com_stats.append((k, preval, meanv, is_binary))

                # Present comorbidities as percentages only (no means). If none present, show 'None'.
                if com_stats:
                    # Sort by prevalence descending
                    com_stats.sort(key=lambda t: t[1], reverse=True)
                    top3 = com_stats[:3]
                    # If the top comorbidities all have 0% prevalence, treat as None
                    if all(p <= 0 for (_, p, *_) in top3):
                        lines.append('  Comorbidities — None')
                    else:
                        com_lines = []
                        for k, p, *_ in top3:
                            label = k.replace('_mean_10d', '').replace('_', ' ').capitalize()
                            com_lines.append(f"{label}: {p:.0f}%")
                        lines.append('  Top comorbidities — ' + '; '.join(com_lines))
                else:
                    # No comorbidity data discovered for this group
                    lines.append('  Comorbidities — None')
                lines.append('')

            prompt += '\n'.join(lines)
        except Exception:
            prompt += '\n**DEMOGRAPHIC & CLINICAL DISTRIBUTION AMONG TOP MATCHES:** not available\n'
    else:
        # Distribution summary disabled; do not append anything to keep prompts compact
        pass
    # Placeholder where additional embedding-RAG matches (text-based) will be inserted.
    # If embedding-RAG is disabled, leave an empty line instead of the placeholder so prompts remain clean.
    try:
        if globals().get('USE_EMBEDDING_RAG'):
            prompt += "\n{VECTOR_RAG_PLACEHOLDER}\n"
        else:
            prompt += "\n"  # embedding-RAG disabled; no additional block
    except Exception:
        prompt += "\n"
    
        prompt += f"""
**YOUR FORECASTING TASK — DRAFT, THEN ADJUST:**

You will produce both an initial draft forecast and a final adjusted forecast for days 11–20.

1) Draft step: Based only on the information provided here (target patient days 1–10, the DTW-matched historical cases listed above, and the empirical distribution block), produce a concise reasoning/thinking block and then DRAFT numeric predictions for day_11 through day_20. Place this draft reasoning in the thinking block (see output markers below).

2) Adjustment step (use pretrained, internal knowledge): After drafting the initial projection, re-evaluate and ADJUST those 10 predictions using your pre-trained, internal clinical knowledge, external judgment, and any domain heuristics you consider appropriate. DTW and empirical statistics are informative but not sufficient; you must use your broader clinical or epidemiological knowledge to refine the final values.

Definitions & how the data are presented:
- Empirical distribution: the prompt includes, where available, per-day statistics computed from top DTW matches (mean, median, 10th/25th/75th/90th percentiles, IQR, skew, and histogram counts rounded to integer bins 0..4). Treat this as a statistical guide — useful for centering uncertainty and identifying modes — but you are free to weight or deviate from it when your judgement or other evidence indicates.
- DTW cases: concrete historical examples showing days 1–10 and their observed days 11–20 outcomes; use them as illustrative outcomes to inform likely trajectories.
- How to use these inputs: combine the draft (data-driven) projection, the empirical distribution, case-level outcomes, and your own knowledge to produce the final adjusted forecast.

No methodological restrictions: You may use any reasoning style or external-domain heuristics. There are no imposed limits on how much you may deviate from the empirical distribution or DTW-derived suggestions.

OUTPUT REQUIREMENTS (THINK+JSON) — strict, machine-parseable format:

***IMPORTANT: STRUCTURED RESPONSE DIVIDER***: Do NOT emit raw internal chain-of-thought. If you performed internal, private reasoning while composing your answer, do NOT reproduce that internal stream verbatim.

Instead, when you are ready to produce the machine-readable output, begin your response with the exact divider line below (on its own line, with no additional characters or commentary):

--- BEGIN STRUCTURED RESPONSE ---

Immediately after that divider, provide the required <<<THINKING>>> block followed by the <<<JSON>>> block described below. No other text may appear before the divider. The <<<THINKING>>> block should be a concise, structured SUMMARY of your reasoning (not an uncensored transcript of internal thoughts).

1) Thinking block (required, must appear BEFORE the JSON):
    - Wrap the entire thinking block exactly between <<<THINKING>>> and <<<END_THINKING>>> (these markers must appear on their own lines).
    - The thinking block must contain these three tagged sections in this exact order:
      a) <dtw_reasoning>...</dtw_reasoning>
          - Analyze and interpret the DTW cases and empirical distribution. Prepare to draft your initial predictions based on these signals.
      b) <draft>...</draft>
          - A plain mapping of draft numeric predictions for day_11..day_20, one per line, in the form: day_11: 2.0
          - Allowed numeric format: integer or decimal (e.g., 2 or 2.0). Values must be within [0, 4].
      c) <clinical_adjustment>...</clinical_adjustment>
          - Explain WHY and HOW you will adjust the draft using your own pretrained clinical knowledge. Do not adjust your draft based on the DTW cases and empirical distributions--it should be based on your own internal knowledge only.
    - After the three sections, include a single line exactly containing: End thought

2) Final JSON block (required, must appear AFTER the thinking block):
    - Wrap the JSON exactly between <<<JSON>>> and <<<END_JSON>>> (markers on their own lines).
    - The JSON must be valid JSON (parsable by standard JSON parsers). No trailing commas.
    - Required keys: "day_11" .. "day_20" (inclusive). Values must be numbers in [0,4]. Example: "day_11": 2.0
    - Do NOT include any other top-level keys or explanatory text inside the JSON block.

3) Strict output rules (these enable reliable automated parsing):
    - Do NOT include any text or characters outside the specified markers (<<<THINKING>>>, <<<END_THINKING>>>, <<<JSON>>>, <<<END_JSON>>>). The entire LLM response should only contain the thinking block and the JSON block.
    - Markers and tags are case-sensitive and must match exactly.
    - Maintain the exact order: <<<THINKING>>> -> <dtw_reasoning> -> <draft> -> <clinical_adjustment> -> End thought -> <<<END_THINKING>>> -> <<<JSON>>> -> JSON -> <<<END_JSON>>>.
    - Use plain ASCII for the markers and tags to avoid encoding issues.

4) Minimal template (the parser expects this shape; also, remember if you are using internal thoughts, they must be inside the thinking block, even if you have to repeat it):
<<<THINKING>>>
<dtw_reasoning>
...short bullets...
</dtw_reasoning>
<draft>
day_11: 2.0
day_12: 2.0
...
day_20: 1.4
</draft>
<clinical_adjustment>
...brief lines describing adjustments...
</clinical_adjustment>
End thought
<<<END_THINKING>>>

<<<JSON>>>
{
  "day_11": 2.0,
  "day_12": 2.0,
  "day_13": 2.0,
  "day_14": 1.9,
  "day_15": 1.8,
  "day_16": 1.7,
  "day_17": 1.6,
  "day_18": 1.5,
  "day_19": 1.5,
  "day_20": 1.4
}
<<<END_JSON>>>

"""

    # Append the example blocks as a plain string (NOT an f-string) so that
    # literal braces in the JSON example aren't interpreted by Python's f-string
    # parser. This avoids errors like "Invalid format specifier ... for object of type 'str'".
    example_block = '''
    
    --- BEGIN STRUCTURED RESPONSE ---
    
    <<<THINKING>>>
    <dtw_reasoning>
    [PLACEHOLDER: Summarizing and interpreting DTW signals and empirical distribution]
    - [PLACEHOLDER: top-case analysis]
    - [PLACEHOLDER: empirical per-day pattern]
    - [PLACEHOLDER: brief plan / anchor statement]
    </dtw_reasoning>

    <draft>
    day_11: <D11>
    day_12: <D12>
    day_13: <D13>
    day_14: <D14>
    day_15: <D15>
    day_16: <D16>
    day_17: <D17>
    day_18: <D18>
    day_19: <D19>
    day_20: <D20>
    </draft>

    <clinical_adjustment>
    [PLACEHOLDER: brief clinical rationale for adjustments; describe internal knowledge used and concepts previously learned]
    - [PLACEHOLDER: rationale 1]
    - [PLACEHOLDER: rationale 2]
    </clinical_adjustment>

    End thought
    <<<END_THINKING>>>

    <<<JSON>>>
    {
        "day_11": "<D11>",
        "day_12": "<D12>",
        "day_13": "<D13>",
        "day_14": "<D14>",
        "day_15": "<D15>",
        "day_16": "<D16>",
        "day_17": "<D17>",
        "day_18": "<D18>",
        "day_19": "<D19>",
        "day_20": "<D20>"
    }
    <<<END_JSON>>>

        '''
    prompt += example_block

    return prompt


##########################
# Vector RAG helpers
##########################
def _get_openai_client():
    """Lazily initialize OpenAI client using config or env var OPENAI_API_KEY."""
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    key = None
    try:
        # try top-level config first
        from config import OPENAI_API_KEY as _k
        key = _k
    except Exception:
        # try llm_output/config.py by path
        cfg_path = Path('llm_output') / 'config.py'
        if cfg_path.exists():
            try:
                spec = importlib.util.spec_from_file_location('llm_output_config', str(cfg_path))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                key = getattr(mod, 'OPENAI_API_KEY', None)
            except Exception:
                key = None

    if not key:
        key = os.environ.get('OPENAI_API_KEY')

    if not key:
        raise RuntimeError('OPENAI_API_KEY not found for embedding-based RAG. Set config.OPENAI_API_KEY or llm_output/config.py or env OPENAI_API_KEY')

    if OpenAI is None:
        raise RuntimeError('openai package not available; install openai (pip install openai) to use embedding-based RAG')

    _openai_client = OpenAI(api_key=key)
    return _openai_client


def load_embedded_db(condition):
    """Load precomputed embedded JSON under llm_dataset/<condition> and cache results.

    Returns (entries, embeddings_np) where entries is the loaded JSON list and
    embeddings_np is an (N, D) float np.array normalized to unit length.
    """
    global EMBEDDED_DB_CACHE
    if condition in EMBEDDED_DB_CACHE:
        return EMBEDDED_DB_CACHE[condition]

    p = Path('/Users/wilsonyeh/flaredown_project') / 'llm_dataset' / condition / f'rag_{condition}_forecast_embedded.json'
    if not p.exists():
        raise FileNotFoundError(f'Embedded DB not found at {p}')

    with open(p, 'r') as f:
        data = json.load(f)

    embs = np.array([r.get('embedding') for r in data], dtype=float)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / (norms + 1e-12)

    EMBEDDED_DB_CACHE[condition] = (data, embs)
    print(f"✅ Loaded embedded DB for {condition}: {len(data)} records, dim={embs.shape[1]}")
    return data, embs


def _build_text_from_patient(test_patient):
    """Serialize a test_patient dict into the same text form used to build the embedded DB."""
    # Demographics
    text = ""
    age = test_patient.get('age', '')
    country = test_patient.get('country', '')
    sex = test_patient.get('sex', '')
    text += f"Demographics: Age {age}, Country: {country}, Sex: {sex}\n\n"
    text += f"{CONDITION.capitalize()} levels for days 1-10:\n"
    for i in range(1, 11):
        v = test_patient.get(f'{CONDITION}_day{i}') or test_patient.get(f'{CONDITION}_days_1_10', {}).get(f'day_{i}') if isinstance(test_patient.get(f'{CONDITION}_days_1_10'), dict) else test_patient.get(f'{CONDITION}_day{i}', '')
        text += f"  day_{i}: {v}\n"

    text += "\nOther symptoms and conditions (days 1-10):\n"
    skip = set(['age', 'country', 'sex', 'user_seq_id', 'window_start_date', 'window_end_date'])
    # include contextual_features and clinical_features
    for k, v in (test_patient.get('contextual_features') or {}).items():
        if k in skip:
            continue
        if v is None:
            continue
        text += f"  {k}: {v}\n"
    for k, v in (test_patient.get('clinical_features') or {}).items():
        if k in skip:
            continue
        if v is None:
            continue
        text += f"  {k}: {v}\n"
    return text


def _print_api_example(call_index, prompt, weights, response_text, reasoning=None):
    """Atomically print a captured API prompt+response example.

    This function holds the examples lock to avoid interleaving other example
    prints from concurrent threads.
    """
    global _printed_api_examples
    try:
        with _api_examples_lock:
            sep = '=' * 70
            print(f"\n{sep}", flush=True)
            print(f"🔍 API Call #{call_index} (example output)", flush=True)
            print(f"{sep}", flush=True)
            print("PROMPT SENT TO LLM:", flush=True)
            # Print a trimmed prompt header then the full prompt for diagnostics
            try:
                print(prompt, flush=True)
            except Exception:
                try:
                    print(str(prompt)[:2000], flush=True)
                except Exception:
                    print("<unprintable prompt>", flush=True)

            if weights is not None:
                try:
                    print("ADAPTIVE WEIGHTS USED FOR THIS CALL:", flush=True)
                    print(weights, flush=True)
                except Exception:
                    print("ADAPTIVE WEIGHTS: <unprintable>", flush=True)
            # Print whether reasoning was requested in the payload (structured payload)
            try:
                if reasoning is not None:
                    try:
                        print("REASONING PAYLOAD:", flush=True)
                        print(json.dumps(reasoning, indent=2), flush=True)
                    except Exception:
                        print("REASONING:", reasoning, flush=True)
            except Exception:
                pass
            print(f"{sep}\n", flush=True)
            print(f"\n{sep}", flush=True)
            print(f"📥 API Response #{call_index} (example output)", flush=True)
            print(f"{sep}", flush=True)
            try:
                print("RESPONSE FROM LLM:", flush=True)
                print(response_text, flush=True)
            except Exception:
                try:
                    print(str(response_text)[:2000], flush=True)
                except Exception:
                    print("<unprintable response>", flush=True)
            print(f"{sep}\n", flush=True)

            # Keep compatibility with any existing checks on _printed_api_examples
            try:
                _printed_api_examples = max(_printed_api_examples, _api_examples_collected)
            except Exception:
                pass
            # Ensure any buffered stdout is flushed so logs appear immediately
            try:
                import sys
                sys.stdout.flush()
            except Exception:
                pass
    except Exception:
        # Never raise from diagnostic printing
        pass


def retrieve_embedding_rag(test_patient, embedded_entries, embedded_vecs, top_k=3):
    """Retrieve top-k embedding matches for a test_patient using OpenAI embeddings for the query text.

    Returns list of (entry_like, similarity) where entry_like matches the shape expected by append_vector_matches_to_prompt
    (i.e., has 'row' dict).
    """
    # Allow using a precomputed query embedding if available (test row index mapping)
    global PRECOMPUTED_QUERY_EMBS
    qemb = None
    q_index = test_patient.get('_orig_index') if isinstance(test_patient, dict) else None
    if q_index is not None and q_index in PRECOMPUTED_QUERY_EMBS:
        qemb = PRECOMPUTED_QUERY_EMBS[q_index]
    else:
        client = _get_openai_client()
        qtext = _build_text_from_patient(test_patient)
        resp = client.embeddings.create(input=[qtext], model='text-embedding-3-small', encoding_format='float')
        qemb = np.array(resp.data[0].embedding, dtype=float)
        qemb = qemb / (np.linalg.norm(qemb) + 1e-12)

    sims = embedded_vecs.dot(qemb)
    idxs = np.argsort(sims)[-top_k:][::-1]
    results = []
    for i in idxs:
        e = embedded_entries[int(i)]
        # normalize output entry to shape expected by append_vector_matches_to_prompt
        entry_like = {'row': e.get('metadata', {}), 'source': e.get('source', f'embedded_index_{i}')}
        results.append((entry_like, float(sims[int(i)])))
    return results


def precompute_test_query_embeddings(condition, test_csv_path=None, out_path=None, batch_size=64):
    """Precompute embeddings for all rows in the test CSV and save to JSON.

    Each saved item will contain: {'index': original_index, 'text': text, 'row': row_dict, 'embedding': [...]}
    """
    client = _get_openai_client()
    base = Path('/Users/wilsonyeh/flaredown_project')
    test_csv = Path(test_csv_path) if test_csv_path else base / 'llm_dataset' / condition / f'llm_{condition}_test.csv'
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found at {test_csv}")

    df = pd.read_csv(test_csv)
    items = []
    texts = []
    index_map = []

    print(f"Precomputing embeddings for {len(df)} test rows from {test_csv} (batch_size={batch_size})")
    for idx, row in df.iterrows():
        # Build a test_patient dict similar to processing path so _build_text_from_patient works
        tp = {
            'age': row.get('age', ''),
            'sex': row.get('sex', ''),
            'country': row.get('country', ''),
            f'{condition}_days_1_10': {f'day_{i}': row.get(f'{condition}_day{i}') for i in range(1, 11)},
            'contextual_features': {}
        }
        # include contextual features if present
        key_features = ['anxiety_mean_10d', 'fatigue_mean_10d', 'pain_mean_10d', 'sadness_mean_10d',
                       'stress_mean_10d', 'irritability_mean_10d', 'cognitive_dysfunction_mean_10d']
        for k in key_features:
            if k in row.keys() and pd.notna(row.get(k)):
                tp['contextual_features'][k] = row.get(k)

        text = _build_text_from_patient(tp)
        texts.append(text)
        index_map.append(int(idx))

    out_items = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(input=batch, model='text-embedding-3-small', encoding_format='float')
        for j, r in enumerate(resp.data):
            emb = list(map(float, r.embedding))
            orig_idx = index_map[i + j]
            out_items.append({'index': orig_idx, 'text': batch[j], 'embedding': emb, 'row_index': orig_idx})

    out_file = Path(out_path) if out_path else base / 'llm_dataset' / condition / f'rag_{condition}_test_embedded.json'
    with open(out_file, 'w') as f:
        json.dump(out_items, f)

    print(f"✅ Wrote {len(out_items)} precomputed test embeddings to {out_file}")
    return out_file


def load_precomputed_query_embeddings(condition, in_path=None):
    """Load precomputed test query embeddings into PRECOMPUTED_QUERY_EMBS (index -> np.array(normalized))."""
    global PRECOMPUTED_QUERY_EMBS
    base = Path('/Users/wilsonyeh/flaredown_project')
    p = Path(in_path) if in_path else base / 'llm_dataset' / condition / f'rag_{condition}_test_embedded.json'
    if not p.exists():
        raise FileNotFoundError(f"Precomputed query embedding file not found at {p}")
    with open(p, 'r') as f:
        data = json.load(f)
    PRECOMPUTED_QUERY_EMBS = {}
    for item in data:
        idx = int(item.get('index') or item.get('row_index'))
        emb = np.array(item['embedding'], dtype=float)
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        PRECOMPUTED_QUERY_EMBS[idx] = emb
    print(f"✅ Loaded {len(PRECOMPUTED_QUERY_EMBS)} precomputed query embeddings for {condition} from {p}")
    return PRECOMPUTED_QUERY_EMBS


def append_vector_matches_to_prompt(prompt, vector_matches, top_k=3):
    """
    Append a small provenance block with embedding-RAG (text-based) matches to the prompt.
    vector_matches: list of (entry, similarity)
    """
    # Build a standardized block with up to top_k embedding-RAG cases formatted
    # similarly to the DTW 'Case' blocks for readability.
    lines = []
    k_count = min(len(vector_matches), top_k)
    # Preface embedding-based cases with a short explanatory sentence
    # Use the exact user-specified preface for embedding cases
    try:
        emb_header = f"**ADDITIONAL {k_count} HISTORICAL CASES: Embedding-RAG matches (top {k_count}/{k_count}):**"
        emb_context = (
            f"Context: These {k_count} additional cases were selected via embedding-based retrieval augmented generation (RAG), "
            f"matching contextual and comorbid feature similarity in the vector embedding space to identify patients with overlapping symptom profiles\n\n"
        )
    except Exception:
        emb_header = f"**ADDITIONAL CONTEXT: Embedding-RAG matches (top {k_count}/{len(vector_matches)}):**"
        emb_context = ""

    lines.append(emb_header)
    lines.append(emb_context)
    for i, (entry, sim) in enumerate(vector_matches[:k_count], 1):
        # entry is expected to be {'vector':..., 'row': {...}, 'source': path}
        row = entry.get('row', {}) if isinstance(entry, dict) else {}

        # Build depression snippet (D1-10)
        sim_dep_1_10 = []
        for j in range(1, 11):
            col = f'{CONDITION}_day{j}'
            if col in row:
                try:
                    sim_dep_1_10.append(float(row[col]))
                except Exception:
                    try:
                        sim_dep_1_10.append(float(str(row[col]).strip()))
                    except Exception:
                        sim_dep_1_10.append(0.0)
            else:
                sim_dep_1_10.append(0.0)

        # Outcome snippet if present
        sim_dep_11_20 = []
        for j in range(11, 21):
            tcol = f'target_{CONDITION}_day{j}'
            if tcol in row:
                try:
                    sim_dep_11_20.append(float(row[tcol]))
                except Exception:
                    try:
                        sim_dep_11_20.append(float(str(row[tcol]).strip()))
                    except Exception:
                        sim_dep_11_20.append(0.0)
            else:
                sim_dep_11_20.append(0.0)

        # Trajectory and outcome summaries
        try:
            sim_baseline = np.mean(sim_dep_1_10[:3])
            sim_recent = np.mean(sim_dep_1_10[-3:])
            sim_trend = sim_recent - sim_baseline
        except Exception:
            sim_trend = 0.0

        try:
            outcome_avg = np.mean(sim_dep_11_20) if any(sim_dep_11_20) else 0.0
            outcome_first = sim_dep_11_20[0]
            outcome_last = sim_dep_11_20[-1]
            outcome_change = outcome_last - outcome_first
        except Exception:
            outcome_avg = outcome_first = outcome_last = outcome_change = 0.0

        if sim_trend > 0.5:
            initial_pattern = "↑worsening"
        elif sim_trend < -0.5:
            initial_pattern = "↓improving"
        else:
            initial_pattern = "→stable"

        if outcome_change > 0.5:
            outcome_pattern = "↑worsened"
        elif outcome_change < -0.5:
            outcome_pattern = "↓improved"
        else:
            outcome_pattern = "→stable"

        # Demographics from row if available
        dem_items = []
        sim_age = row.get('age') or row.get('Age')
        sim_sex = row.get('sex') or row.get('Sex')
        sim_country = row.get('country') or row.get('Country')
        if sim_age is not None and str(sim_age).strip() != '':
            dem_items.append(f"Age={sim_age}")
        if sim_sex is not None and str(sim_sex).strip() != '':
            dem_items.append(f"Sex={sim_sex}")
        if sim_country is not None and str(sim_country).strip() != '':
            dem_items.append(f"Country={sim_country}")

        # Comorbid/contextual fields
        ctx_items = []
        skip_keys = set([f'{CONDITION}_day{i}' for i in range(1, 11)]) | set([f'target_{CONDITION}_day{i}' for i in range(11, 21)])
        skip_keys.update({'patient_id', 'id', 'source', 'user_seq_id', 'window_start_date', 'window_end_date'})
        for k in sorted(row.keys()):
            if k in skip_keys:
                continue
            v = row.get(k)
            if v is None:
                continue
            try:
                fv = float(v)
                if not np.isnan(fv) and fv != 0.0:
                    ctx_items.append(f"{k}={fv:.2f}")
            except Exception:
                s = str(v).strip()
                if s:
                    ctx_items.append(f"{k}={s[:120]}")

        # Build formatted case block similar to DTW cases
        dep_line = ', '.join([f"{v:.1f}" for v in sim_dep_1_10])
        out_line = ', '.join([f"{v:.1f}" for v in sim_dep_11_20]) if any(sim_dep_11_20) else None

        header = f"EmbedCase {i} [{sim:.1%} match]:"
        case_block = f"  D1-10: {dep_line} [{initial_pattern}, trend={sim_trend:+.1f}]\n"
        if out_line:
            case_block += f"  D11-20: {out_line} [Outcome: {outcome_pattern}, avg={outcome_avg:.1f}]\n"
        if dem_items:
            case_block += "  Demographics: " + ", ".join(dem_items) + "\n"
        if ctx_items:
            case_block += "  Comorbid: " + ", ".join(ctx_items) + "\n"

        lines.append(header + "\n" + case_block + "\n")

    block = '\n'.join(lines)

    # If caller created the VECTOR_RAG placeholder in the prompt, replace it there; otherwise append at end
    placeholder = '{VECTOR_RAG_PLACEHOLDER}'
    insertion_method = None
    result = None
    if placeholder in prompt:
        result = prompt.replace(placeholder, block)
        insertion_method = 'replaced_placeholder'
    else:
        # If the placeholder is missing (unexpected), insert the block before
        # the main task/example section so the vector context still appears
        # directly after the DTW-ranked cases. This guards against races or
        # older prompt formats where the placeholder wasn't present.
        insert_at = prompt.find('\n**YOUR FORECASTING TASK')
        if insert_at != -1:
            result = prompt[:insert_at] + '\n' + block + '\n' + prompt[insert_at:]
            insertion_method = 'inserted_before_task'
        else:
            # Fallback: append at end
            result = prompt + '\n' + block
            insertion_method = 'appended_at_end'

    # Optional debug logging to help trace where vector block was placed at runtime.
    try:
        if _debug_vector_insert:
            print(f"[DEBUG] vector insert method: {insertion_method}")
            # Show a short snippet around the insertion point to help diagnostics
            try:
                idx = result.find(block)
                start = max(0, idx - 120)
                end = min(len(result), idx + 120)
                snippet = result[start:end]
                print('[DEBUG] snippet around insertion:\n' + snippet)
            except Exception:
                pass
    except NameError:
        pass

    # Embedding-RAG runtime printing is intentionally suppressed to avoid noisy logs.
    # Detailed embedding diagnostics can be enabled via debug flags if needed.
    pass

    return result


def call_openrouter_api(prompt, model=None, max_retries=3, weights=None, reasoning=None):
    """
    Call OpenRouter API with simple rate limiting and retry logic.
    Returns response text or None on failure.
    """
    global _api_lock, _last_api_call_time, _api_call_count, _debug_api_calls
    global _printed_api_examples

    if model is None:
        model = LLM_MODEL

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 10000
    }
    # Decide whether to request reasoning explicitly. If caller passed None, fall back to global ENABLE_REASONING.
    try:
        # Resolve boolean intent (caller may pass None/True/False)
        if reasoning is None:
            enabled_flag = bool(globals().get('ENABLE_REASONING', False))
        else:
            enabled_flag = bool(reasoning)

        # Only include the structured reasoning object in the request payload
        # when reasoning is explicitly enabled. If reasoning is disabled we
        # omit the `reasoning` key entirely (do not send an explicit disabled
        # reasoning framework) to avoid triggering API behavior for an
        # explicit disabled field.
        if enabled_flag:
            reasoning_payload = {
                'enabled': True,
                'effort': 'high',
                "exclude": True,
                'exclusions': []
            }
            payload['reasoning'] = reasoning_payload
        else:
            # Ensure we don't accidentally include a reasoning key when disabled.
            try:
                payload.pop('reasoning', None)
            except Exception:
                # Best-effort: if payload isn't a dict or pop fails, ignore.
                pass
    except Exception:
        # Don't fail API call if reasoning payload construction encounters an issue.
        # As a safe fallback, ensure we do NOT add a `reasoning` key to the
        # outgoing payload (omit the field entirely rather than sending an
        # explicit disabled payload).
        try:
            payload.pop('reasoning', None)
        except Exception:
            pass

    for attempt in range(max_retries):
        try:
            with _api_lock:
                current_time = time.time()
                time_since_last_call = current_time - _last_api_call_time
                if time_since_last_call < _min_interval_between_calls:
                    time.sleep(_min_interval_between_calls - time_since_last_call)
                _last_api_call_time = time.time()

                _api_call_count += 1
                # Reserve whether we'll capture this call's prompt+response.
                # Capture enough examples so that the TOTAL of collected+reserved
                # reaches `_api_examples_to_capture`. This is more robust than
                # relying solely on call index when some earlier reservations may
                # have failed to produce a collected example. Also allow periodic
                # captures every `_api_example_periodic_interval` calls.
                global _api_examples_reserved, _api_examples_collected, _api_example_periodic_interval
                should_capture = False
                call_index = _api_call_count
                try:
                    # If we still need more captured examples (account for already reserved slots)
                    need_more = (_api_examples_collected + _api_examples_reserved) < _api_examples_to_capture
                    if need_more:
                        _api_examples_reserved += 1
                        should_capture = True
                    elif _api_example_periodic_interval and call_index % _api_example_periodic_interval == 0:
                        # Periodic capture (one patient example every N patients)
                        _api_examples_reserved += 1
                        should_capture = True
                except Exception:
                    # If anything goes wrong deciding capture, fall back to not capturing
                    should_capture = False

            # Send request (we avoid printing the prompt before the call so
            # that request+response can be printed together atomically AFTER
            # the response is received to avoid interleaving with other logs).
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            response_text = response.json()['choices'][0]['message']['content']

            # If we reserved a capture slot, store and print the example atomically
            if should_capture:
                try:
                    # Use a safe lookup for the optional reasoning payload which
                    # may not be defined when reasoning is disabled. Fall back to
                    # the explicit `reasoning` parameter or None.
                    rp = None
                    try:
                        rp = locals().get('reasoning_payload', None)
                    except Exception:
                        rp = None
                    if rp is None:
                        # fall back to any reasoning passed by the caller
                        rp = reasoning if 'reasoning' in locals() else None

                    with _api_examples_lock:
                        _api_examples.append({'call_index': call_index, 'prompt': prompt, 'weights': weights, 'reasoning': rp, 'response': response_text})
                        # Mark as collected (one fewer reserved; one more collected)
                        _api_examples_collected += 1
                        _api_examples_reserved = max(0, _api_examples_reserved - 1)
                    # Print the example block atomically (no mid-block interleaving)
                    _print_api_example(call_index, prompt, weights, response_text, reasoning=rp)
                except Exception:
                    # Best-effort printing — don't let this break the API flow
                    pass

            return response_text

        except Exception as e:
            # Try to extract a requests.Response if available so we can show
            # status code and response body/json returned by the API for debugging.
            resp = None
            try:
                if isinstance(e, requests.exceptions.RequestException) and getattr(e, 'response', None) is not None:
                    resp = e.response
            except Exception:
                resp = None

            # Also fall back to any local `response` variable (if request succeeded but parsing failed)
            if resp is None and 'response' in locals() and locals().get('response') is not None:
                resp = locals().get('response')

            status = getattr(resp, 'status_code', None) if resp is not None else None
            body = None
            json_body = None
            if resp is not None:
                try:
                    json_body = resp.json()
                except Exception:
                    json_body = None
                try:
                    body = resp.text
                except Exception:
                    body = None

            # Compose a compact error details string for logging
            details_parts = [repr(e)]
            if status is not None:
                details_parts.append(f"status_code={status}")
            if json_body is not None:
                try:
                    details_parts.append(f"json_body={json.dumps(json_body, indent=2)[:2000]}")
                except Exception:
                    details_parts.append(f"json_body=<unserializable>")
            elif body is not None:
                details_parts.append(f"body={body[:2000]}")

            details = " | ".join(details_parts)

            # If this was a requests.RequestException, allow retrying per the original logic.
            if isinstance(e, requests.exceptions.RequestException):
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"   ⚠️  API error (attempt {attempt+1}/{max_retries}), retrying in {wait_time}s... Details: {details}")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"   ❌ API call failed after {max_retries} attempts: {details}")
                    return None
            else:
                # Non-request exception (e.g., JSON parsing / key error). Don't retry; surface details.
                print(f"   ❌ API call error (non-Request exception): {details}")
                return None

    return None


def parse_llm_response(response_text):
    """
    Parse LLM response to extract day 11-20 predictions.
    
    Returns:
        Dictionary with day_11 through day_20 predictions, or None if parsing fails
    """
    try:
        if not response_text or not isinstance(response_text, str):
            return None

        # 1) Prefer explicit markers <<<JSON>>>(.*?)<<<END_JSON>>> for strict parsing
        marker_match = re.search(r'<<<JSON>>>(.*?)<<<END_JSON>>>', response_text, re.DOTALL | re.IGNORECASE)
        json_text = None

        if marker_match:
            json_text = marker_match.group(1).strip()
            # If the marked block contains extra text, try to extract the first {...} inside it
            inner_match = re.search(r'\{.*\}', json_text, re.DOTALL)
            if inner_match:
                json_text = inner_match.group(0)

        else:
            # 2) Fallback: look for a top-level JSON object anywhere in the response
            fallback_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if fallback_match:
                json_text = fallback_match.group(0)

        if not json_text:
            # Nothing that looks like JSON was found
            return None

        # Ensure thinking block is present and extract it (thinking is REQUIRED)
        thinking_match = re.search(r'<<<THINKING>>>(.*?)<<<END_THINKING>>>', response_text, re.DOTALL | re.IGNORECASE)
        if not thinking_match:
            print("   ⚠️  Missing required thinking block <<<THINKING>>>...<<<END_THINKING>>>; rejecting response")
            return None
        thinking_text = thinking_match.group(1).strip()

        # Attempt to load JSON
        predictions = json.loads(json_text)

        # Validate required day keys
        required_days = [f'day_{i}' for i in range(11, 21)]
        if not all(day in predictions for day in required_days):
            # Missing required keys — treat as parse failure
            print("   ⚠️  Parsed JSON missing required day keys; rejecting response")
            return None

        # Normalize and clip values to 0-4 as floats
        for day in required_days:
            try:
                val = float(predictions[day])
            except Exception:
                print(f"   ⚠️  Could not convert value for {day} to float; rejecting response")
                return None
            predictions[day] = max(0.0, min(4.0, val))

        return {
            'predictions': predictions,
            'thinking': thinking_text
        }

    except json.JSONDecodeError as e:
        print(f"   ⚠️  JSON decode error: {e}")
        return None
    except Exception as e:
        print(f"   ⚠️  Error parsing response strictly: {e}")
        return None


def _process_patient_core(idx, row, text_db, top_k, weights=None):
    """
    Core patient processing logic using DTW-based RAG (called by retry wrapper).
    
    RAG Pipeline:
      1. RETRIEVAL: Extract depression time series and find similar patients via DTW
      2. AUGMENTATION: Create prompt with retrieved similar patient trajectories
      3. GENERATION: LLM generates forecasts based on augmented context
    
    Returns:
        Dictionary with results, or None if failed
    """
    # Prepare patient data with contextual features
    test_patient = {
        'age': row.get('age', 'Unknown'),
        'sex': row.get('sex', 'Unknown'),
        'country': row.get('country', 'Unknown'),
        f'{CONDITION}_days_1_10': { f'day_{i}': row[f'{CONDITION}_day{i}'] for i in range(1, 11)}
    }
    # Preserve original DataFrame index so we can lookup precomputed query embeddings (if available)
    try:
        test_patient['_orig_index'] = int(idx)
    except Exception:
        test_patient['_orig_index'] = idx
    
    # Extract contextual features (same as Random Forest uses)
    key_features = ['anxiety_mean_10d', 'fatigue_mean_10d', 'pain_mean_10d', 'sadness_mean_10d', 
                   'stress_mean_10d', 'irritability_mean_10d', 'cognitive_dysfunction_mean_10d']
    contextual_features = {}
    for feature in key_features:
        if feature in row.index:
            value = row.get(feature)
            if pd.notna(value):
                contextual_features[feature] = value
    
    test_patient['contextual_features'] = contextual_features

    # Collect additional pre-target clinical columns (demographics already included above)
    # These are any columns present on the row that are NOT the primary days 1-10
    # and NOT the target days 11-20. We also skip known trivial fields like patient_id.
    clinical_features = {}
    try:
        row_cols = set(row.index.tolist()) if hasattr(row, 'index') else set()
        pre_target_days = {f'{CONDITION}_day{i}' for i in range(1, 11)}
        target_days = {f'target_{CONDITION}_day{i}' for i in range(11, 21)}
        # Skip internal/identifier fields and core demographics
        skip_keys = {'patient_id', 'age', 'sex', 'country', 'user_seq_id', 'window_end_date', 'window_start_date'}

        candidate_cols = sorted(list(row_cols - pre_target_days - target_days - skip_keys))
        for col in candidate_cols:
            # Skip contextual features already captured
            if col in contextual_features:
                continue
            try:
                val = row.get(col)
            except Exception:
                continue
            # Only include columns with non-missing, non-zero values
            if val is None:
                continue
            # Skip pandas NA
            try:
                if pd.isna(val):
                    continue
            except Exception:
                pass

            # If numeric and exactly zero, skip it (user requested filtering zeros)
            try:
                fval = float(val)
                if fval == 0.0:
                    continue
            except Exception:
                # Non-numeric values: skip empty strings
                if isinstance(val, str) and val.strip() == '':
                    continue

            clinical_features[col] = val
    except Exception:
        clinical_features = {}

    test_patient['clinical_features'] = clinical_features
    
    # Build multi-variate feature dictionary for DTW matching (RETRIEVAL step)
    query_features = {
        CONDITION: [test_patient[f'{CONDITION}_days_1_10'][f'day_{i}'] for i in range(1, 11)]
    }
    
    # Add comorbid symptom time series (using mean values as constant series)
    # Note: If we had actual daily time series for these, we'd use those instead
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
            # Create constant time series from mean value
            query_features[symptom_name] = [contextual_features[feature_key]] * 10
    
    # Compute per-patient adaptive weights (derive from run-level `weights` or ADAPTIVE_WEIGHTS)
    base_for_patient = weights if weights is not None else ADAPTIVE_WEIGHTS
    per_patient_weights = compute_per_patient_weights(test_patient, base_weights=base_for_patient, min_primary_weight=0.35)

    # Find similar patients using DTW on the PRIMARY condition only (no API calls needed!).
    # Comorbid/contextual similarity is handled via embedding-RAG (text-based) augmentation.
    similar_patients = find_similar_patients_dtw(query_features, text_db, top_k, use_multivariate=False, weights=per_patient_weights)
    
    # Create DTW-based RAG prompt (AUGMENTATION step)
    prompt = create_dtw_prompt(test_patient, similar_patients, top_k)
    
    # Handle embedding-RAG (text-based)
    vector_matches = None
    if USE_EMBEDDING_RAG and embedded_entries is not None and embedded_vecs is not None:
        # Attempt retrieval with a few retries to be robust to transient API/network issues.
        vector_matches = None
        embedding_error = None
        embedding_attempts = 3
        for emb_attempt in range(embedding_attempts):
            try:
                vector_matches = retrieve_embedding_rag(test_patient, embedded_entries, embedded_vecs, top_k=3)
                # Successful retrieval (may be empty list)
                embedding_error = None
                break
            except Exception as e:
                embedding_error = str(e)
                vector_matches = None
                # If debug/logging enabled, emit a short retry note
                try:
                    if globals().get('_debug_vector_insert') or globals().get('_print_vector_append_logs'):
                        print(f"   ⚠️  Embedding-RAG retrieval attempt {emb_attempt+1}/{embedding_attempts} failed for patient {idx}: {e}")
                except Exception:
                    pass
                time.sleep(0.5)
        # Record any error for downstream reporting/analysis
        if embedding_error:
            # Keep a short, non-verbose warning unless debug mode is enabled
            try:
                if globals().get('_debug_vector_insert') or globals().get('_print_vector_append_logs'):
                    print(f"   ⚠️  Embedding-RAG retrieval ultimately failed for patient {idx}: {embedding_error}")
                else:
                    # Minimal one-line notice suppressed in normal runs
                    pass
            except Exception:
                pass
    
    # Append embedding-RAG matches to prompt if available
    embedding_matches_count = 0
    embedding_top_sim = None
    if vector_matches and USE_EMBEDDING_RAG:
        try:
            embedding_matches_count = len(vector_matches)
            embedding_top_sim = float(max(sim for _, sim in vector_matches))
        except Exception:
            embedding_matches_count = len(vector_matches) if vector_matches else 0
            embedding_top_sim = None

        # Threshold below which we treat matches as uninformative
        if embedding_top_sim is not None and embedding_top_sim <= 0.01:
            try:
                if globals().get('_debug_vector_insert') or globals().get('_print_vector_append_logs'):
                    print(f"ℹ️  Embedding-RAG: top cosine similarity {embedding_top_sim:.3f} <= 0.01 for patient {idx}; skipping append (matches not informative).")
            except Exception:
                pass
        else:
            prompt = append_vector_matches_to_prompt(prompt, vector_matches, top_k=3)
            try:
                if globals().get('_debug_vector_insert') or globals().get('_print_vector_append_logs'):
                    print(f"✅ Appended {len(vector_matches)} embedding-RAG matches (top_sim={embedding_top_sim:.3f}) for patient {idx}")
            except Exception:
                pass
    
    # Numeric vector-RAG has been removed - only embedding-RAG is supported
    # This section is intentionally left empty to make it clear that vector-RAG
    # functionality has been completely removed from the codebase.
    
    # Get LLM prediction (has built-in retry logic) (GENERATION step)
    # Pass per-patient adaptive weights so the API debug logger can show which weights were used
    response = call_openrouter_api(prompt, weights=per_patient_weights)
    
    # Check if LLM call failed
    if response is None:
        return None  # LLM API failed after retries
    
    # Parse LLM response (now returns {'predictions':..., 'rationale':...})
    parsed = parse_llm_response(response)

    if parsed is None:
        return None  # Parsing failed (will trigger patient-level retry)

    predictions = parsed.get('predictions')
    thinking_text = parsed.get('thinking')
    
    # Store results
    result = {
        'patient_id': idx,
        'age': test_patient['age'],
        'sex': test_patient['sex'],
        'country': test_patient['country']
    }
    
    # Add predictions and ground truth
    for i in range(11, 21):
        result[f'predicted_day_{i}'] = predictions[f'day_{i}']
        result[f'actual_day_{i}'] = row[f'target_{CONDITION}_day{i}']

    # Include the concise thinking summary provided by the LLM
    result['thinking'] = thinking_text
    # Include per-patient adaptive weights for auditing (one column per symptom)
    try:
        for wk, wv in per_patient_weights.items():
            result[f'weight_{wk}'] = float(wv)
    except Exception:
        # If weights are missing or malformed, skip adding them but don't fail the patient
        pass
    # Include concise embedding-RAG metadata for higher-level summary/aggregation
    try:
        result['embedding_matches_count'] = int(embedding_matches_count) if 'embedding_matches_count' in locals() and embedding_matches_count is not None else 0
    except Exception:
        result['embedding_matches_count'] = 0
    try:
        result['embedding_top_sim'] = float(embedding_top_sim) if 'embedding_top_sim' in locals() and embedding_top_sim is not None else None
    except Exception:
        result['embedding_top_sim'] = None
    try:
        result['embedding_error'] = str(embedding_error) if 'embedding_error' in locals() and embedding_error is not None else None
    except Exception:
        result['embedding_error'] = None

    return result


def process_single_patient(idx, row, text_db, top_k, max_patient_retries=3, weights=None):
    """
    Process a single patient with full retry logic using DTW-based RAG (for parallel execution).
    
    If response parsing fails, retries the ENTIRE RAG pipeline:
      1. RETRIEVAL: DTW matching to find similar patients
      2. AUGMENTATION: Prompt creation with retrieved context
      3. GENERATION: LLM call and parsing
    
    This handles cases where the LLM returns malformed JSON.
    
    Args:
        idx: Patient index
        row: Patient data row
        embedded_db: Embedded RAG database
    
    If response parsing fails, retries the ENTIRE patient pipeline (DTW matching + LLM + parsing).
    This handles cases where the LLM returns malformed JSON.
    
    Args:
        idx: Patient index
        row: Patient data row
        text_db: Text-based patient database
        top_k: Number of similar patients to retrieve
    max_patient_retries: Maximum full patient retries (default: 3)
    
    Returns:
        Dictionary with results or None if all retries failed
    """
    for attempt in range(max_patient_retries):
        try:
            # Log start of attempt
            if attempt > 0:
                print(f"\n   🔄 PATIENT-LEVEL RETRY: Patient {idx} | Attempt {attempt + 1}/{max_patient_retries}")
                print(f"   ↻  Retrying full RAG pipeline: DTW retrieval → augmentation → generation")
            
            result = _process_patient_core(idx, row, text_db, top_k, weights=weights)
            
            if result is not None:
                # Success!
                if attempt > 0:
                    print(f"   ✅ Patient {idx}: SUCCESS on patient-level retry {attempt + 1}/{max_patient_retries}!")
                return result
            else:
                # Failed - determine why and decide if we should retry
                if attempt < max_patient_retries - 1:
                    print(f"   ⚠️  Patient {idx}: RAG pipeline failed on attempt {attempt + 1}/{max_patient_retries}")
                    print(f"   🔄 Initiating PATIENT-LEVEL RETRY (will re-run entire RAG pipeline)...")
                    time.sleep(3)  # Brief pause before full retry
                else:
                    print(f"   ❌ Patient {idx}: FAILED after {max_patient_retries} complete RAG pipeline attempts")
                    
        except Exception as e:
            if attempt < max_patient_retries - 1:
                print(f"   ⚠️  Patient {idx}: Exception on attempt {attempt + 1}/{max_patient_retries}: {e}")
                print(f"   🔄 Initiating PATIENT-LEVEL RETRY (will re-run entire RAG pipeline)...")
                time.sleep(3)
            else:
                print(f"   ❌ Patient {idx}: FAILED after {max_patient_retries} attempts with exception: {e}")
    
    return None


def forecast_with_dtw(test_data, text_db, top_k=5, max_patients=None, n_workers=4, weights=None):
    """
    Generate forecasts for test patients using DTW-based RAG with parallel processing.
    
    This function orchestrates the full DTW-based RAG pipeline:
      - RETRIEVAL: Find similar patients using DTW on depression time series
      - AUGMENTATION: Create prompts with retrieved similar patient trajectories  
      - GENERATION: LLM generates forecasts based on augmented context
    
    Args:
        test_data: DataFrame with test patients
        text_db: Text-based patient database
        top_k: Number of similar patients to retrieve via DTW
        max_patients: Maximum number of test patients to process (None = all)
        n_workers: Number of parallel workers (default: 4)
    
    Returns:
        DataFrame with predictions and ground truth
    """
    results = []
    
    # Get primary-condition day columns (e.g., 'depression_day1' or 'anxiety_day1')
    dep_days_1_10 = [f'{CONDITION}_day{i}' for i in range(1, 11)]
    dep_days_11_20 = [f'target_{CONDITION}_day{i}' for i in range(11, 21)]
    
    # Limit number of patients if specified
    test_subset = test_data.head(max_patients) if max_patients else test_data
    
    print(f"\n{'='*70}")
    print(f"FORECASTING WITH DTW-BASED RAG (Top-{top_k} similar patients)")
    print(f"{'='*70}")
    print(f"Processing {len(test_subset)} test patients with {n_workers} parallel workers...")
    emb_state = 'ENABLED' if USE_EMBEDDING_RAG else 'DISABLED'
    print(f"Matching on: {CONDITION.capitalize()} only (DTW on primary trajectory). Embedding-RAG (comorbid/contextual) = {emb_state}.\n")
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks (DTW-based RAG - no embedding API needed!)
        future_to_idx = {}
        for idx, row in test_subset.iterrows():
            fut = executor.submit(process_single_patient, idx, row, text_db, top_k, max_patient_retries=3, weights=weights)
            future_to_idx[fut] = idx

        # Process completed tasks with progress bar
        processed_count = 0
        periodic_interval = 250
        with tqdm(total=len(test_subset), desc="Forecasting") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                    else:
                        tqdm.write(f"   ⚠️  Failed to process patient {idx}")
                except Exception as e:
                    tqdm.write(f"   ❌ Error processing patient {idx}: {e}")

                # Count this completed patient (success or failure)
                processed_count += 1

                # Periodic concise summary every `periodic_interval` patients
                try:
                    if processed_count % periodic_interval == 0 or processed_count == len(test_subset):
                        # Count how many of the processed patients (successful results)
                        # had 3 or more embedding matches.
                        found_3 = sum(1 for r in results if int(r.get('embedding_matches_count', 0)) >= 3)
                        tqdm.write(f"\nPeriodic summary (processed {processed_count}/{len(test_subset)}): Embedding-RAG identified 3 matches for {found_3}/{processed_count} patients")
                        if found_3 == processed_count:
                            tqdm.write("Success!")
                        else:
                            tqdm.write("⚠️  Some processed patients did not receive 3 embedding matches. Inspect results for details.")
                except Exception:
                    # Non-fatal; continue the loop
                    pass

                pbar.update(1)
    
    if not results:
        raise ValueError("No successful predictions were generated!")
    
    results_df = pd.DataFrame(results)
    print(f"\n✅ Successfully generated {len(results_df)} predictions")

    # Compact Embedding-RAG health summary: count how many patients received the
    # expected top-3 embedding matches. Keep output minimal when everything is
    # healthy (all patients got 3 matches) to avoid huge logs for large runs.
    try:
        total_patients = len(test_subset)
        found_3 = int(results_df['embedding_matches_count'].ge(3).sum()) if 'embedding_matches_count' in results_df.columns else 0
        print(f"Embedding-RAG successfully identified 3 matches for {found_3}/{total_patients} patients")
        if found_3 == total_patients:
            # All good — short celebratory message and suppress further per-patient logs
            print("Success!")
        else:
            # Warn the user that some patients didn't get 3 matches; include a hint
            print(f"⚠️  Warning: Embedding-RAG did not find 3 matches for all patients ({found_3}/{total_patients}). Check 'embedding_error' or 'embedding_matches_count' in results for details.")
    except Exception:
        # Non-fatal: don't break pipeline if aggregation fails
        pass
    
    return results_df


def evaluate_predictions(results_df):
    """
    Evaluate LLM predictions and compare with Random Forest.
    
    Args:
        results_df: DataFrame with predicted and actual values
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*70}")
    print("EVALUATION METRICS")
    print(f"{'='*70}\n")
    
    # Extract predicted and actual values
    predicted_cols = [f'predicted_day_{i}' for i in range(11, 21)]
    actual_cols = [f'actual_day_{i}' for i in range(11, 21)]
    
    y_pred = results_df[predicted_cols].values.flatten()
    y_true = results_df[actual_cols].values.flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Per-day metrics
    day_metrics = []
    for i in range(11, 21):
        pred_col = f'predicted_day_{i}'
        actual_col = f'actual_day_{i}'
        
        day_mae = mean_absolute_error(results_df[actual_col], results_df[pred_col])
        day_rmse = np.sqrt(mean_squared_error(results_df[actual_col], results_df[pred_col]))
        
        day_metrics.append({
            'day': i,
            'mae': day_mae,
            'rmse': day_rmse
        })
    
    metrics = {
        'model': LLM_MODEL,
        'overall': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'n_predictions': len(results_df)
        },
        'per_day': day_metrics
    }
    
    # Print overall metrics
    print("Overall Metrics:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    # Print per-day metrics
    print("\nPer-Day Metrics:")
    for dm in day_metrics:
        print(f"  Day {dm['day']:2d}: MAE={dm['mae']:.4f}, RMSE={dm['rmse']:.4f}")
    
    return metrics


def load_random_forest_results():
    """Load Random Forest results for comparison.

    This function only searches under the project's `random_forest_output/` directory
    (recursively). It prefers files named like `evaluation_metrics*.csv`, then
    `predictions*.csv`. If multiple matches exist it returns the most recently
    modified file as a pandas DataFrame. Returns None if nothing usable is found.
    """
    base = Path('/Users/wilsonyeh/flaredown_project')
    rf_dir = base / 'random_forest_output'

    if not rf_dir.exists():
        print(f"⚠️  random_forest_output directory not found at expected location: {rf_dir}")
        return None

    # Gather candidate files (search recursively)
    eval_files = list(rf_dir.rglob('evaluation_metrics*.csv'))
    pred_files = list(rf_dir.rglob('predictions*.csv'))
    # Fallback: any csv under the directory
    other_csvs = list(rf_dir.rglob('*.csv'))

    chosen = None
    if eval_files:
        # Prefer the newest evaluation_metrics file
        chosen = max(eval_files, key=lambda p: p.stat().st_mtime)
    elif pred_files:
        # Otherwise prefer the newest predictions file
        chosen = max(pred_files, key=lambda p: p.stat().st_mtime)
    elif other_csvs:
        # As a last resort, pick the newest CSV in the folder
        chosen = max(other_csvs, key=lambda p: p.stat().st_mtime)
    else:
        print(f"⚠️  No CSV files found under {rf_dir}")
        return None

    try:
        print(f"\nLoading Random Forest baseline metrics from: {chosen}")
        rf_metrics = pd.read_csv(chosen)
        print("✅ Loaded Random Forest metrics")
        return rf_metrics
    except Exception as e:
        print(f"   ⚠️  Failed to read {chosen}: {e}")
        return None


def compare_with_random_forest(llm_metrics, rf_metrics_df):
    """
    Compare Multi-variate DTW-based RAG performance with Random Forest baseline.
    
    Args:
        llm_metrics: Dictionary with Multi-variate DTW-based RAG evaluation metrics
        rf_metrics_df: DataFrame with Random Forest metrics
    """
    print(f"\n{'='*70}")
    print("COMPARISON: Multi-variate DTW-RAG vs Random Forest")
    print(f"{'='*70}\n")
    
    if rf_metrics_df is None:
        print("⚠️  No Random Forest baseline available for comparison")
        return
    
    # Calculate overall RF metrics by averaging across all days (test set metrics)
    # RF CSV now has per-day metrics with columns: test_mae_mean, test_rmse_mean, test_r2_mean
    # Or it might have an 'Overall' row with aggregated statistics
    
    # Check if there's an 'Overall' row in the dataframe
    def _robust_overall_from_df(df):
        """
        Robust extraction of overall RF metrics from a potentially variable-format DataFrame.
        Tries several common column name patterns, then falls back to heuristic column matching.
        Returns a dict with keys 'mae','rmse','r2' (values may be None if unavailable).
        """
        # 1) If there's an 'Overall' row, prefer explicit aggregate columns if present
        if 'day' in df.columns:
            try:
                overall_row = df[df['day'].astype(str).str.lower() == 'overall']
            except Exception:
                overall_row = df[df['day'] == 'Overall'] if 'Overall' in df['day'].values else df[df['day'].astype(str).str.lower() == 'overall']

            if len(overall_row) > 0:
                row = overall_row.iloc[0]
                # Try several common column names
                mae = None
                rmse = None
                r2 = None
                for col in ['test_mae_mean', 'test_mae', 'mae']:
                    if col in row.index:
                        mae = float(row[col])
                        break
                for col in ['test_rmse_mean', 'test_rmse', 'rmse']:
                    if col in row.index:
                        rmse = float(row[col])
                        break
                for col in ['test_r2_mean', 'test_r2', 'r2']:
                    if col in row.index:
                        r2 = float(row[col])
                        break
                return {'mae': mae, 'rmse': rmse, 'r2': r2}

            # 2) No Overall row: attempt to derive from per-day columns
            per_day = df[df['day'].astype(str).str.lower() != 'overall'] if 'day' in df.columns else df
            # Common aggregated column names
            if all(c in per_day.columns for c in ['test_mae_mean', 'test_rmse_mean', 'test_r2_mean']):
                return {
                    'mae': float(per_day['test_mae_mean'].mean()),
                    'rmse': float(per_day['test_rmse_mean'].mean()),
                    'r2': float(per_day['test_r2_mean'].mean())
                }
            # Fallback heuristic: find any columns containing mae/rmse/mse/r2 and average
            def mean_of_matching(patterns, df_local, treat_mse_as_rmse=False):
                cols = [c for c in df_local.columns if any(p in c.lower() for p in patterns)]
                if not cols:
                    return None
                vals = df_local[cols].apply(pd.to_numeric, errors='coerce')
                # If treating mse as rmse, take sqrt after averaging MSEs
                if treat_mse_as_rmse:
                    mse_cols = [c for c in cols if 'mse' in c.lower() and 'rmse' not in c.lower()]
                    if mse_cols:
                        mean_mse = vals[mse_cols].stack().mean()
                        return float(np.sqrt(mean_mse)) if pd.notna(mean_mse) else None
                mean_val = vals.stack().mean()
                return float(mean_val) if pd.notna(mean_val) else None

            mae = mean_of_matching(['mae'], per_day)
            rmse = mean_of_matching(['rmse'], per_day)
            if rmse is None:
                # maybe only MSE columns exist
                rmse = mean_of_matching(['mse'], per_day, treat_mse_as_rmse=True)
            r2 = mean_of_matching(['r2'], per_day)
            return {'mae': mae, 'rmse': rmse, 'r2': r2}

        else:
            # No 'day' column: try several common layouts
            if 'test_mae_mean' in df.columns and 'test_r2_mean' in df.columns:
                return {
                    'mae': float(df['test_mae_mean'].mean()),
                    'rmse': float(df['test_rmse_mean'].mean()) if 'test_rmse_mean' in df.columns else None,
                    'r2': float(df['test_r2_mean'].mean())
                }
            if 'test_mae' in df.columns and 'test_r2' in df.columns:
                return {
                    'mae': float(df['test_mae'].mean()),
                    'rmse': float(np.sqrt(df['test_mse'].mean())) if 'test_mse' in df.columns else None,
                    'r2': float(df['test_r2'].mean())
                }
            # Generic fallback
            mae = None
            rmse = None
            r2 = None
            mae = mean_of_matching(['mae'], df)
            rmse = mean_of_matching(['rmse'], df)
            if rmse is None:
                rmse = mean_of_matching(['mse'], df, treat_mse_as_rmse=True)
            r2 = mean_of_matching(['r2'], df)
            return {'mae': mae, 'rmse': rmse, 'r2': r2}

    rf_overall = _robust_overall_from_df(rf_metrics_df)
    
    llm_overall = llm_metrics['overall']
    
    print(f"{'Metric':<10} {'MV-DTW-RAG':<12} {'Random Forest':<15} {'Difference':<12} {'Winner'}")
    print("-" * 70)
    
    for metric in ['mae', 'rmse', 'r2']:
        llm_val = llm_overall.get(metric)
        rf_val = rf_overall.get(metric)
        
        if llm_val is not None and rf_val is not None:
            diff = llm_val - rf_val
            
            # For MAE and RMSE, lower is better; for R², higher is better
            if metric in ['mae', 'rmse']:
                winner = 'MV-DTW-RAG ✅' if diff < 0 else 'Random Forest ✅'
                diff_str = f"{diff:+.4f}"
            else:  # r2
                winner = 'MV-DTW-RAG ✅' if diff > 0 else 'Random Forest ✅'
                diff_str = f"{diff:+.4f}"
            
            print(f"{metric.upper():<10} {llm_val:<12.4f} {rf_val:<15.4f} {diff_str:<12} {winner}")
        else:
            print(f"{metric.upper():<10} {llm_val or 'N/A':<12} {rf_val or 'N/A':<15} {'N/A':<12}")


def create_visualizations(results_df, llm_metrics):
    """Create visualization plots"""
    print(f"\n{'='*70}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*70}\n")
    
    fig_dir = RESULTS_DIR / 'figures'
    fig_dir.mkdir(exist_ok=True)
    
    # 1. Predicted vs Actual scatter plot
    plt.figure(figsize=(10, 8))
    
    predicted_cols = [f'predicted_day_{i}' for i in range(11, 21)]
    actual_cols = [f'actual_day_{i}' for i in range(11, 21)]
    
    y_pred = results_df[predicted_cols].values.flatten()
    y_true = results_df[actual_cols].values.flatten()
    
    plt.scatter(y_true, y_pred, alpha=0.5, s=20)
    plt.plot([0, 4], [0, 4], 'r--', label='Perfect Prediction')
    plt.xlabel(f'Actual {CONDITION.capitalize()} Level', fontsize=12)
    plt.ylabel(f'Predicted {CONDITION.capitalize()} Level', fontsize=12)
    plt.title(f'Multi-variate DTW-RAG: Predicted vs Actual {CONDITION.capitalize()} Levels', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / 'predicted_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: predicted_vs_actual.png")
    
    # 2. Per-day MAE
    plt.figure(figsize=(12, 6))
    day_metrics = llm_metrics['per_day']
    days = [dm['day'] for dm in day_metrics]
    maes = [dm['mae'] for dm in day_metrics]
    
    plt.bar(days, maes, color='steelblue', alpha=0.7)
    plt.xlabel('Day', fontsize=12)
    plt.ylabel('Mean Absolute Error', fontsize=12)
    plt.title(f'Multi-variate DTW-RAG: MAE by Forecast Day ({CONDITION.capitalize()})', fontsize=14, fontweight='bold')
    plt.xticks(days)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / 'mae_by_day.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: mae_by_day.png")
    
    # 3. Error distribution
    plt.figure(figsize=(10, 6))
    errors = y_pred - y_true
    plt.hist(errors, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    plt.xlabel('Prediction Error (Predicted - Actual)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Multi-variate DTW-RAG: Prediction Error Distribution ({CONDITION.capitalize()})', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: error_distribution.png")
    
    # 4. MAE comparison with Random Forest (per-day)
    try:
        rf_df = load_random_forest_results()

        def _extract_rf_maes(rf_df):
            """Robust extraction of RF per-day MAE for days 11-20 from various RF file formats."""
            if rf_df is None:
                return None

            # Case 1: DataFrame has 'day' column and per-day metric columns
            if 'day' in rf_df.columns:
                # Normalize day to numeric when possible
                tmp = rf_df.copy()
                tmp['day_num'] = pd.to_numeric(tmp['day'], errors='coerce')
                rows = tmp[tmp['day_num'].between(11, 20)]
                if not rows.empty:
                    if 'test_mae_mean' in rows.columns:
                        mapping = {int(r['day_num']): r['test_mae_mean'] for _, r in rows.iterrows()}
                    elif 'test_mae' in rows.columns:
                        mapping = {int(r['day_num']): r['test_mae'] for _, r in rows.iterrows()}
                    elif 'mae' in rows.columns:
                        mapping = {int(r['day_num']): r['mae'] for _, r in rows.iterrows()}
                    else:
                        mapping = {}
                    return [mapping.get(d, np.nan) for d in range(11, 21)]

            # Case 2: Columns named per-day (day_11_mae, test_mae_day_11, etc.)
            col_maes = []
            for d in range(11, 21):
                candidates = [
                    f'test_mae_day_{d}', f'day_{d}_mae', f'mae_day_{d}', f'day{d}_mae',
                    f'test_mae_{d}', f'mae_{d}', f'day_{d}_mae_mean'
                ]
                found = None
                for c in candidates:
                    if c in rf_df.columns:
                        found = c
                        break
                if found:
                    try:
                        col_maes.append(float(rf_df[found].iloc[0]))
                    except Exception:
                        col_maes.append(np.nan)
                else:
                    col_maes.append(np.nan)

            if not all(np.isnan(col_maes)):
                return col_maes

            # Case 3: Predictions file - compute per-day MAE from predictions.csv (if it has actual/pred columns)
            # Try to compute from predictions table if available
            try:
                # Look for columns like 'actual_day_11' / 'predicted_day_11' in rf_df
                per_day_maes = []
                found_any = False
                for d in range(11, 21):
                    act_col = f'actual_day_{d}'
                    pred_col_candidates = [f'predicted_day_{d}', f'prediction_day_{d}', f'pred_day_{d}']
                    if act_col in rf_df.columns:
                        found_any = True
                        pred_col = None
                        for pc in pred_col_candidates:
                            if pc in rf_df.columns:
                                pred_col = pc
                                break
                        if pred_col:
                            per_day_maes.append(mean_absolute_error(rf_df[act_col], rf_df[pred_col]))
                        else:
                            per_day_maes.append(np.nan)
                    else:
                        per_day_maes.append(np.nan)

                if found_any and not all(np.isnan(per_day_maes)):
                    return per_day_maes
            except Exception:
                pass

            return None

        rf_maes = _extract_rf_maes(rf_df)

        if rf_maes is not None:
            # Plot grouped bar chart comparing LLM MAE vs RF MAE per-day
            x = np.arange(11, 21)
            width = 0.35

            plt.figure(figsize=(12, 6))
            plt.bar(x - width/2, maes, width, label='MV-DTW-RAG', color='steelblue')
            plt.bar(x + width/2, rf_maes, width, label='Random Forest', color='orange', alpha=0.9)
            plt.xlabel('Day', fontsize=12)
            plt.ylabel('Mean Absolute Error', fontsize=12)
            plt.title(f'MAE by Day: MV-DTW-RAG vs Random Forest ({CONDITION.capitalize()})', fontsize=14, fontweight='bold')
            plt.xticks(x)
            plt.legend()
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(fig_dir / 'mae_comparison_rf.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Saved: mae_comparison_rf.png")
        else:
            print("⚠️  Could not extract per-day RF MAE values; skipping RF comparison plot.")

    except Exception as e:
        print(f"   ⚠️  Error when attempting RF comparison plot: {e}")
    
    print(f"\n📊 All visualizations saved to: {fig_dir}/")


def save_results(results_df, metrics, extra_info=None):
    """Save results and metrics to files"""
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}\n")
    
    # Save predictions
    predictions_path = RESULTS_DIR / 'predictions.csv'
    results_df.to_csv(predictions_path, index=False)
    print(f"✅ Saved predictions: {predictions_path}")
    
    # Save metrics
    metrics_path = RESULTS_DIR / 'evaluation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved metrics: {metrics_path}")
    
    # Save an expanded summary with run metadata
    summary_path = RESULTS_DIR / 'training_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"MULTI-VARIATE DTW-BASED RAG {CONDITION.upper()} FORECASTING SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {LLM_MODEL}\n")
        f.write(f"Method: Multi-variate DTW-based RAG (DTW retrieval + LLM augmentation + generation)\n")
        f.write("\n")

        # Input dataset info
        f.write("INPUTS:\n")
        # Prefer explicit paths from extra_info when available (these are set in __main__)
        text_db_path = extra_info.get('text_db') if extra_info and 'text_db' in extra_info else TEXT_DB_PATH
        test_data_path = extra_info.get('test_data') if extra_info and 'test_data' in extra_info else TEST_DATA_PATH

        f.write(f"  Text DB path: {text_db_path}\n")
        f.write(f"  Test data path: {test_data_path}\n")
        # Also write the directories explicitly so it's clear where the data came from
        try:
            f.write(f"  Text DB directory: {Path(text_db_path).parent}\n")
            f.write(f"  Test data directory: {Path(test_data_path).parent}\n")
        except Exception:
            # If paths are None or not parseable, skip directory lines
            pass
        f.write(f"\n")

        # Parameters used (capture from extra_info where available)
        f.write("RUN PARAMETERS:\n")
        f.write(f"  Condition: {CONDITION}\n")
        f.write(f"  Model: {LLM_MODEL}\n")
        if extra_info:
            for k, v in extra_info.items():
                f.write(f"  {k}: {v}\n")
        # Explicitly state requested number of testing windows (max_patients)
        # NOTE: 'max_patients' in this script refers to the number of testing windows
        # (i.e., separate forecast instances), not unique patients.
        requested = None
        if extra_info and 'max_patients' in extra_info:
            requested = extra_info.get('max_patients')
        else:
            # default used in script
            requested = 1500
        f.write(f"  Requested testing windows (max_patients): {requested}\n")
        f.write(f"  Results directory: {RESULTS_DIR}\n")
        f.write("\n")

        # Overall metrics
        f.write("EVALUATION METRICS:\n")
        f.write(f"  Number of prediction windows saved: {metrics['overall']['n_predictions']}\n")
        # Number of unique patients represented by those windows (based on patient_id column if available)
        try:
            if 'patient_id' in results_df.columns:
                unique_patients = int(results_df['patient_id'].nunique())
            else:
                unique_patients = int(len(results_df))
        except Exception:
            unique_patients = 'Unknown'
        f.write(f"  Unique patients tested: {unique_patients}\n")
        f.write(f"  MSE:  {metrics['overall']['mse']:.6f}\n")
        f.write(f"  RMSE: {metrics['overall']['rmse']:.6f}\n")
        f.write(f"  MAE:  {metrics['overall']['mae']:.6f}\n")
        f.write(f"  R²:   {metrics['overall']['r2']:.6f}\n\n")

        f.write("PER-DAY METRICS:\n")
        for dm in metrics['per_day']:
            f.write(f"  Day {dm['day']:2d}: MAE={dm['mae']:.6f}, RMSE={dm['rmse']:.6f}\n")

        f.write("\nFILES WRITTEN:\n")
        for p in ['predictions.csv', 'evaluation_metrics.json', 'training_summary.txt']:
            f.write(f"  - {RESULTS_DIR / p}\n")
        f.write(f"  - {RESULTS_DIR / 'figures'} (visualizations directory)\n")

        f.write("\nNOTES:\n")
        f.write("  - Retrieval used multi-variate DTW with primary symptom weighted 50% and comorbidities 50%.\n")
        f.write("  - LLM calls were made to the configured OpenRouter model with temperature=0 and retry/backoff.\n")
        f.write("  - Any patients that failed the full pipeline were excluded from the saved predictions; check logs for retry details.\n")

    print(f"✅ Saved summary: {summary_path}")


def print_startup_notice(args=None):
    """Print a concise startup notice describing which retrieval mode will be used and which precomputed data is available.

    Args:
        args: argparse Namespace from main (optional). If provided, the function will include flag-derived choices.
    """
    base = Path('/Users/wilsonyeh/flaredown_project')
    cond = CONDITION
    print('\n' + '='*70)
    print('STARTUP: Retrieval configuration')
    print('='*70)
    print(f"  Condition: {cond}")
    print(f"  Embedding-RAG enabled: {bool(USE_EMBEDDING_RAG)}")
    # Note: numeric/vector-RAG support has been removed. Embedding-RAG (text-based)
    # is the supported augmentation mechanism. A persisted numeric vector DB may
    # still exist on disk as a historical artifact but is not used by this script.
    print("  Numeric/vector-RAG support: removed (embedding-RAG only)")

    # Check presence of embedded DB (forecast embeddings)
    emb_db_path = base / 'llm_dataset' / cond / f'rag_{cond}_forecast_embedded.json'
    precomp_q_path = base / 'llm_dataset' / cond / f'rag_{cond}_test_embedded.json'
    vec_db_path = base / 'llm_dataset' / cond / f'vector_db_{cond}.pkl'

    if emb_db_path.exists():
        try:
            size = emb_db_path.stat().st_size
            print(f"  Embedded DB available: {emb_db_path} ({size//1024} KB)")
        except Exception:
            print(f"  Embedded DB available: {emb_db_path}")
    else:
        print(f"  Embedded DB not found at: {emb_db_path}")

    if precomp_q_path.exists() or PRECOMPUTED_QUERY_EMBS:
        if PRECOMPUTED_QUERY_EMBS:
            n = len(PRECOMPUTED_QUERY_EMBS)
            print(f"  Precomputed test-query embeddings loaded: {n} entries")
        else:
            try:
                size = precomp_q_path.stat().st_size
                print(f"  Precomputed test-query embeddings file present: {precomp_q_path} ({size//1024} KB)")
            except Exception:
                print(f"  Precomputed test-query embeddings file present: {precomp_q_path}")
    else:
        print(f"  No precomputed test-query embeddings found at: {precomp_q_path}")

    if vec_db_path.exists():
        try:
            vsize = vec_db_path.stat().st_size
            print(f"  Note: persisted numeric vector DB found (historical): {vec_db_path} ({vsize//1024} KB)")
        except Exception:
            print(f"  Note: persisted numeric vector DB found (historical): {vec_db_path}")
    else:
        print(f"  No persisted numeric vector DB found (expected): {vec_db_path}")

    print('\n  To change behavior: use --use-embedding-rag to enable embedding-RAG,')
    print('  --precompute-query-embeddings to precompute queries, or --use-precomputed-query-embeddings to force loading precomputed embeddings.')
    print('='*70 + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multi-variate DTW-based RAG forecasting')
    parser.add_argument('--condition', type=str, default=CONDITION,
                        help="Primary condition: 'depression' or 'anxiety'")
    # Vector-RAG related arguments removed - only embedding-RAG is supported)
    # New defaults: embedding-RAG disabled by default; enable explicitly with --use-embedding-rag
    parser.set_defaults(use_embedding_rag=False)
    parser.add_argument('--use-embedding-rag', dest='use_embedding_rag', action='store_true',
                        help='Use text-embedding RAG (precomputed embeddings in llm_dataset). Default: disabled')
    # Allow an explicit opt-out flag for clarity and scripting convenience
    parser.add_argument('--no-embedding-rag', dest='use_embedding_rag', action='store_false',
                        help='Explicitly disable text-embedding RAG (overrides any default)')
    parser.add_argument('--precompute-query-embeddings', dest='precompute_query_embeddings', action='store_true',
                        help='Precompute embeddings for test CSV and save them to llm_dataset/<condition>/rag_<condition>_test_embedded.json then exit')
    parser.add_argument('--use-precomputed-query-embeddings', dest='use_precomputed_query_embeddings', action='store_true',
                        help='Use precomputed test query embeddings (avoid per-patient embedding API calls)')
    args = parser.parse_args()

    # Validate and set CONDITION
    if args.condition.lower() not in VALID_CONDITIONS:
        raise ValueError(f"Unknown condition: {args.condition}. Valid: {VALID_CONDITIONS}")

    CONDITION = args.condition.lower()

    # Enable embedding-based RAG if requested (default: True)
    USE_EMBEDDING_RAG = bool(args.use_embedding_rag)

    # Auto-load precomputed test-query embeddings if they exist (convenience)
    base = Path('/Users/wilsonyeh/flaredown_project')
    precomp_path = base / 'llm_dataset' / CONDITION / f'rag_{CONDITION}_test_embedded.json'
    if precomp_path.exists():
        try:
            load_precomputed_query_embeddings(CONDITION)
            print(f"Auto-loaded precomputed test query embeddings from {precomp_path}")
        except Exception as e:
            print(f"   ⚠️  Failed to auto-load precomputed query embeddings at {precomp_path}: {e}")

    # Precompute on demand (explicit flag). If set, compute and exit.
    if args.precompute_query_embeddings:
        precompute_test_query_embeddings(CONDITION)
        print('Precompute complete; exiting as requested.')
        import sys
        sys.exit(0)

    # Load embedded DB once before parallel processing (if embedding-RAG is enabled)
    embedded_entries = None
    embedded_vecs = None
    if USE_EMBEDDING_RAG:
        try:
            print(f"🔁 Embedding-RAG enabled: loading embedded DB for '{CONDITION}'")
            embedded_entries, embedded_vecs = load_embedded_db(CONDITION)
            print(f"✅ Loaded embedded DB for '{CONDITION}': {len(embedded_entries)} entries, dim={embedded_vecs.shape[1]}")
        except Exception as e:
            print(f"   ⚠️  Failed to load embedded DB: {e}")
            USE_EMBEDDING_RAG = False  # Disable embedding-RAG if loading fails

    # Optionally explicitly request loading of precomputed embeddings via flag
    if args.use_precomputed_query_embeddings and not PRECOMPUTED_QUERY_EMBS:
        load_precomputed_query_embeddings(CONDITION)

    # Update dataset paths based on condition
    base = Path('/Users/wilsonyeh/flaredown_project')
    TEXT_DB_PATH = base / 'llm_dataset' / CONDITION / f'rag_{CONDITION}_forecast_text_database.json'
    TEST_DATA_PATH = base / 'llm_dataset' / CONDITION / f'llm_{CONDITION}_test.csv'

    print("="*70)
    print(f"DTW-BASED RAG {CONDITION.upper()} FORECASTING")
    print("="*70)

    # Print concise startup notice summarizing retrieval modes and available precomputed data
    try:
        print_startup_notice(args)
    except Exception as e:
        print(f"   ⚠️  Failed to print startup notice: {e}")

    # Load data
    text_db = load_text_database()
    test_data = load_test_data()
    # Validate inputs
    try:
        validate_text_db(text_db)
        validate_test_dataframe(test_data)
    except Exception as e:
        print(f"\nERROR: Input validation failed: {e}")
        raise
    
    # Create per-run output directory under llm_output/<condition>/<model>-MMDDYY-HHMM
    model_sanitized = _sanitize_model_name(LLM_MODEL)
    timestamp = datetime.now().strftime('%m%d%y-%H%M')
    run_name = f"{model_sanitized}-{timestamp}"
    run_dir = base / 'llm_output' / CONDITION / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Override RESULTS_DIR for this run
    RESULTS_DIR = run_dir
    print(f"Outputs will be written to: {RESULTS_DIR}")
    
    # Attempt to compute adaptive weights from available training data (prefer llm_dataset, fallback to random_forest_dataset)
    train_candidates = [
        base / 'llm_dataset' / CONDITION / f'llm_{CONDITION}_train.csv',
        base / 'random_forest_dataset' / CONDITION / f'random_forest_{CONDITION}_train.csv',
        base / 'random_forest_dataset' / f'rf_{CONDITION}_train.csv',
        base / 'random_forest_dataset' / f'{CONDITION}_train.csv'
    ]

    ADAPTIVE_WEIGHTS = None
    for tp in train_candidates:
        try:
            if tp.exists():
                print(f"Found training CSV for adaptive weights at: {tp}")
                train_df = pd.read_csv(tp)
                ADAPTIVE_WEIGHTS = compute_adaptive_weights_from_df(train_df, condition=CONDITION, min_primary_weight=0.35)
                print(f"Using adaptive weights: {ADAPTIVE_WEIGHTS}")
                break
        except Exception as e:
            print(f"   ⚠️  Could not compute adaptive weights from {tp}: {e}")

    if ADAPTIVE_WEIGHTS is None:
        print("No training CSV found or failed to compute adaptive weights; using default fixed weights in compute_multivariate_dtw_distance().")

    # Generate forecasts with parallel processing using DTW-based RAG
    # Primary-only DTW matches patients based on target condition trajectory:
    #   - Depression/anxiety trajectory (days 1-10) for temporal similarity
    #   - Comorbid/contextual features handled via embedding-RAG (default)
    # This focuses DTW on temporal patterns where it excels, while using other
    # retrieval methods for contextual similarity.
    # 
    # PERFORMANCE OPTIMIZATIONS ENABLED:
    #   ✅ Fast pre-filter: Euclidean distance → top 100 candidates (20x speedup)
    #   ✅ DTW radius constraint: radius=2 for speed/robustness (reduced from 4)
    #   ✅ Parallel processing: Increase n_workers to match your CPU cores
    #
    # Change max_patients=None to process all test patients (5,662 total)
    # Adjust n_workers based on your CPU cores (16+ recommended for speed)
    # Run parameters
    # Number of DTW-matched cases to retrieve per patient. Increased to 50
    # so the empirical distribution block can aggregate days 11-20 across
    # a larger set of similar patients (top-50 by DTW similarity).
    TOP_K = 50
    MAX_PATIENTS = 1500 #3 to test, 1500 for full run
    N_WORKERS = 16

    # Record start time for this run (used in summary/diagnostics)
    run_start_time = time.time()

    results_df = forecast_with_dtw(
        test_data=test_data,
        text_db=text_db,
        top_k=TOP_K,
        max_patients=MAX_PATIENTS,
        n_workers=N_WORKERS,
        weights=ADAPTIVE_WEIGHTS
    )
    
    # Evaluate predictions
    metrics = evaluate_predictions(results_df)
    
    # Load and compare with Random Forest
    rf_metrics = load_random_forest_results()
    compare_with_random_forest(metrics, rf_metrics)
    
    # Create visualizations
    create_visualizations(results_df, metrics)

    # Record end time and build an extensive extra_info dict for the summary
    run_end_time = time.time()
    run_duration = run_end_time - run_start_time

    # Count embedded DB entries if available
    embedded_db_count = 0
    try:
        embedded_db_count = len(embedded_entries) if embedded_entries is not None else 0
    except Exception:
        embedded_db_count = 0

    precomp_q_count = len(PRECOMPUTED_QUERY_EMBS) if PRECOMPUTED_QUERY_EMBS else 0

    # Compose extra_info with useful experiment knobs and runtime diagnostics
    extra_info = {
        'top_k': TOP_K,
        'max_patients': MAX_PATIENTS,
        'n_workers': N_WORKERS,
        'text_db': str(TEXT_DB_PATH),
        'test_data': str(TEST_DATA_PATH),
        'run_name': run_name,
        'run_start_timestamp': datetime.fromtimestamp(run_start_time).strftime('%Y-%m-%d %H:%M:%S'),
        'run_end_timestamp': datetime.fromtimestamp(run_end_time).strftime('%Y-%m-%d %H:%M:%S'),
        'run_duration_seconds': float(run_duration),
        'use_embedding_rag': bool(USE_EMBEDDING_RAG),
        'embedded_db_count': int(embedded_db_count),
        'precomputed_query_embedding_count': int(precomp_q_count),
        'adaptive_weights': ADAPTIVE_WEIGHTS,
        'api_call_count': int(_api_call_count) if '_api_call_count' in globals() else None,
        'api_examples_collected': int(_api_examples_collected) if '_api_examples_collected' in globals() else None,
        'python_executable': sys.executable if 'sys' in globals() else None,
        'python_version': sys.version.replace('\n', ' '),
        'platform': platform.platform(),
        'git_run_name': run_name,
    }

    # Attempt to collect key package versions (best-effort)
    pkg_versions = {}
    for mod in ['pandas', 'numpy', 'sklearn', 'fastdtw', 'scipy', 'requests', 'matplotlib', 'seaborn']:
        try:
            import importlib
            m = importlib.import_module(mod)
            v = getattr(m, '__version__', None)
            if v is None:
                # fallback for sklearn package name
                try:
                    from importlib.metadata import version as _ver
                    v = _ver(mod)
                except Exception:
                    v = None
            pkg_versions[mod] = v
        except Exception:
            pkg_versions[mod] = None

    extra_info['package_versions'] = pkg_versions

    save_results(results_df, metrics, extra_info=extra_info)
    
    print("\n" + "="*70)
    print("✅ DTW-BASED RAG FORECASTING COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {RESULTS_DIR}/")
