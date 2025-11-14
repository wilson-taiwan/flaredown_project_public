import json
import pickle
import random
from pathlib import Path
import numpy as np
import pandas as pd
import time

# Robust OPENAI key lookup
import os
try:
    from config import OPENAI_API_KEY
except Exception:
    OPENAI_API_KEY = None
    # Try to load llm_output/config.py by path if it exists
    cfg_path = Path('llm_output') / 'config.py'
    if cfg_path.exists():
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location('llm_output_config', str(cfg_path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            OPENAI_API_KEY = getattr(mod, 'OPENAI_API_KEY', None)
        except Exception:
            OPENAI_API_KEY = None

    if not OPENAI_API_KEY:
        OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

    if not OPENAI_API_KEY:
        raise ImportError("OPENAI_API_KEY not found. Set llm_output/config.py or env OPENAI_API_KEY")

from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)


def build_text_from_row(row, condition):
    """Serialize a DataFrame row into the same text format used by prepare_rag_database.create_text_database()."""
    # row is a pandas Series
    rid = row.get('user_seq_id') or row.name
    text = f"Patient Record {rid}:\n"
    text += f"Demographics: Age {row.get('age', '')}, Country: {row.get('country', '')}, Sex: {row.get('sex', '')}\n\n"
    text += f"{condition.capitalize()} levels for days 1-10:\n"
    for i in range(1, 11):
        col = f'{condition}_day{i}'
        v = row.get(col, '')
        text += f"  day_{i}: {v}\n"
    text += "\nOther symptoms and conditions (days 1-10):\n"
    # include other non-empty columns except a short list
    skip = set(['age', 'country', 'sex', 'user_seq_id', 'window_start_date', 'window_end_date'])
    for k, v in row.items():
        if k in skip:
            continue
        if k.startswith(condition):
            continue
        if pd.isna(v):
            continue
        if v == '' or v == 0:
            continue
        text += f"  {k}: {v}\n"
    return text


def load_embedded(condition):
    p = Path('llm_dataset') / condition / f'rag_{condition}_forecast_embedded.json'
    with open(p, 'r') as f:
        data = json.load(f)
    embeddings = np.array([r['embedding'] for r in data], dtype=float)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-12)
    return data, embeddings


def load_numeric_db(condition):
    pkl = Path('llm_dataset') / condition / f'vector_db_{condition}.pkl'
    if not pkl.exists():
        return None
    with open(pkl, 'rb') as f:
        d = pickle.load(f)
    entries = d.get('entries', [])
    feature_keys = d.get('feature_keys', [])
    # entry vectors are expected to be normalized already
    vecs = np.array([e['vector'] for e in entries], dtype=float)
    return {'entries': entries, 'feature_keys': feature_keys, 'vecs': vecs}


def embed_texts(texts):
    # batch call
    resp = client.embeddings.create(input=texts, model='text-embedding-3-small', encoding_format='float')
    embs = [np.array(r.embedding, dtype=float) for r in resp.data]
    embs = np.array(embs)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / (norms + 1e-12)


def numeric_query_vector(row, feature_keys):
    vals = []
    for k in feature_keys:
        v = row.get(k, 0.0)
        try:
            if pd.isna(v):
                vals.append(0.0)
            else:
                vals.append(float(v))
        except Exception:
            vals.append(1.0 if str(v).strip() != '' else 0.0)
    vec = np.array(vals, dtype=float)
    n = np.linalg.norm(vec)
    if n > 0:
        vec = vec / n
    return vec


def compare(condition='depression', sample_size=30, top_k=3, seed=42, exclude_duplicates=False):
    random.seed(seed)
    np.random.seed(seed)

    print(f"Loading embedded DB for {condition}...")
    embedded_data, embedded_vecs = load_embedded(condition)
    print(f"  loaded {len(embedded_data)} embedded records (dim={embedded_vecs.shape[1]})")

    numeric_db = load_numeric_db(condition)
    if numeric_db is not None:
        print(f"Loaded numeric vector DB with {len(numeric_db['entries'])} entries and {len(numeric_db['feature_keys'])} feature keys")
    else:
        print("No numeric vector DB found for this condition.")

    # load test CSV
    test_csv = Path('llm_dataset') / condition / f'llm_{condition}_test.csv'
    if not test_csv.exists():
        print(f"No test CSV found at {test_csv}; will sample from embedded metadata as fallback")
        # fallback use first N embedded metadata rows as pseudo-queries
        queries = []
        for r in embedded_data[:sample_size]:
            queries.append({'text': r['text'], 'row': r['metadata']})
        use_embedded_metadata = True
    else:
        df = pd.read_csv(test_csv)
        n = min(sample_size, len(df))
        sample_idx = random.sample(range(len(df)), n)
        queries = []
        for i in sample_idx:
            row = df.iloc[i]
            text = build_text_from_row(row, condition)
            queries.append({'text': text, 'row': row.to_dict()})
        use_embedded_metadata = False

    # embed queries in batches
    texts = [q['text'] for q in queries]
    print(f"Embedding {len(texts)} queries via OpenAI...")
    q_embs = embed_texts(texts)

    # Optionally exclude exact Day1-10 duplicates from the embedded DB
    if exclude_duplicates:
        print("Filtering embedded DB to exclude exact Day1-10 sequences present in test set...")
        # Build set of test sequences
        test_seq_set = set()
        if test_csv.exists():
            df_all = pd.read_csv(test_csv)
            for _, r in df_all.iterrows():
                tup = tuple((r.get(f'{condition}_day{i}', None) for i in range(1, 11)))
                test_seq_set.add(tup)
        else:
            # if no test csv, build set from queries (which are embedded metadata)
            for q in queries:
                # metadata may have depression_days_1_10 dict
                meta = q.get('row')
                if isinstance(meta, dict):
                    d = meta.get(f'{condition}_days_1_10') or {}
                    tup = tuple((d.get(f'day_{i}', None) for i in range(1, 11)))
                    test_seq_set.add(tup)

        # Filter embedded_data and embedded_vecs
        keep_mask = []
        for r in embedded_data:
            d = r.get('metadata', {}).get(f'{condition}_days_1_10', {})
            tup = tuple((d.get(f'day_{i}', None) for i in range(1, 11)))
            keep_mask.append(tup not in test_seq_set)

        keep_idxs = [i for i, k in enumerate(keep_mask) if k]
        print(f"  removed {len(embedded_data) - len(keep_idxs)} records; keeping {len(keep_idxs)}")
        if len(keep_idxs) == 0:
            raise RuntimeError("All embedded records were filtered out; cannot proceed")
        embedded_vecs = embedded_vecs[keep_idxs]
        embedded_data = [embedded_data[i] for i in keep_idxs]
        # Also filter numeric DB entries by the same Day1-10 sequence set (if numeric DB is present)
        if numeric_db is not None:
            n_entries = numeric_db['entries']
            n_vecs = numeric_db['vecs']
            keep_mask_num = []
            for e in n_entries:
                row = e.get('row', {})
                # try nested dict first (consistent with embedded metadata), else look for flat day_i keys
                d = row.get(f'{condition}_days_1_10') or {}
                if not d:
                    # build from flat keys like 'depression_day1' ...
                    tup = tuple((row.get(f'{condition}_day{i}', None) for i in range(1, 11)))
                else:
                    tup = tuple((d.get(f'day_{i}', None) for i in range(1, 11)))
                keep_mask_num.append(tup not in test_seq_set)

            keep_idxs_num = [i for i, k in enumerate(keep_mask_num) if k]
            print(f"  numeric DB: removed {len(n_entries) - len(keep_idxs_num)} records; keeping {len(keep_idxs_num)}")
            if len(keep_idxs_num) == 0:
                raise RuntimeError("All numeric DB records were filtered out; cannot proceed")
            numeric_db['vecs'] = numeric_db['vecs'][keep_idxs_num]
            numeric_db['entries'] = [n_entries[i] for i in keep_idxs_num]

    # For each query compute top-k in embedded_vecs
    emb_top1_sims = []
    num_top1_sims = []
    emb_better_count = 0

    for qi, qvec in enumerate(q_embs):
        sims = embedded_vecs.dot(qvec)
        top_idx = np.argsort(sims)[-top_k:][::-1]
        top_records = [(embedded_data[i], float(sims[i])) for i in top_idx]
        emb_top1_sims.append(top_records[0][1])

        # numeric retrieval
        if numeric_db is not None:
            fk = numeric_db['feature_keys']
            qrow = queries[qi]['row']
            qvec_num = numeric_query_vector(qrow, fk)
            sims_num = numeric_db['vecs'].dot(qvec_num)
            num_top1_sims.append(float(np.max(sims_num)))
            if top_records[0][1] > float(np.max(sims_num)):
                emb_better_count += 1
        else:
            num_top1_sims.append(None)

        # print one example of matches
        if qi < 3:
            print('\n--- Query example ---')
            print(texts[qi][:400].replace('\n', ' | '))
            print('\nTop embedding match (sim):', top_records[0][1])
            m = top_records[0][0]['metadata']
            print('  matched metadata snippet:', {k: m.get('patient_metadata', {}).get(k) for k in ['age','sex']})
            if numeric_db is not None:
                max_idx = int(np.argmax(sims_num))
                nmeta = numeric_db['entries'][max_idx]['row']
                print('Top numeric match (sim):', float(np.max(sims_num)))
                print('  numeric matched row snippet:', {k: nmeta.get(k) for k in ['age','sex'] if k in nmeta})

    emb_mean = float(np.mean(emb_top1_sims))
    num_mean = float(np.mean([v for v in num_top1_sims if v is not None])) if any(v is not None for v in num_top1_sims) else None

    print('\nSUMMARY')
    print(f'  Queries evaluated: {len(texts)}')
    print(f'  Embedding top-1 mean similarity: {emb_mean:.4f}')
    if num_mean is not None:
        print(f'  Numeric top-1 mean similarity:   {num_mean:.4f}')
        print(f'  Embedding > Numeric top1 count: {emb_better_count}/{len(texts)}')


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description='Compare embedding vs numeric retrieval for a condition')
    p.add_argument('--condition', type=str, default='depression')
    p.add_argument('--sample_size', type=int, default=30)
    p.add_argument('--top_k', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--exclude_duplicates', action='store_true', help='Exclude exact Day1-10 duplicates from DBs')
    args = p.parse_args()

    compare(condition=args.condition, sample_size=args.sample_size, top_k=args.top_k, seed=args.seed, exclude_duplicates=args.exclude_duplicates)
