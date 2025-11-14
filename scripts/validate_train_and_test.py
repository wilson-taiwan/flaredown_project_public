#!/usr/bin/env python3
"""Validate that random_forest_dataset train/test CSVs contain the same patient data as the RAG forecast JSON databases.

This script locates the train/test CSVs under random_forest_dataset/{anxiety,depression}
and the rag_*_forecast_database.json files under llm_dataset/{anxiety,depression}.

It will print a short summary per condition including unique patient counts and a small
diff (IDs present in one but not the other).

Usage: python scripts/validate_train_and_test.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, Iterable, List, Optional, Set

try:
    import pandas as pd
except Exception:
    pd = None


ROOT = os.path.dirname(os.path.dirname(__file__))
RF_DIR = os.path.join(ROOT, "random_forest_dataset")
LLM_DIR = os.path.join(ROOT, "llm_dataset")

ID_CANDIDATES = [
    "patient_id",
    "patientid",
    "patient",
    "user_id",
    "user",
    "userId",
    "flaredown_id",
    "flaredown_user_id",
    "participant_id",
    "id",
]


def choose_id_column(columns: Iterable[str]) -> Optional[str]:
    cols = list(columns)
    lower = {c.lower(): c for c in cols}
    # prefer exact candidate matches
    for cand in ID_CANDIDATES:
        if cand.lower() in lower:
            return lower[cand.lower()]
    # fallback: any column containing 'patient' or 'user' or '_id'
    for c in cols:
        lc = c.lower()
        if "patient" in lc or "user" in lc or lc.endswith("_id") or lc == "id":
            return c
    return None


def ids_from_csv(path: str) -> tuple[Set[str], Optional[str]]:
    """Return set of unique ids and the column used (or None)."""
    if pd is not None:
        df = pd.read_csv(path, dtype=str, low_memory=False)
        if df.shape[0] == 0:
            return set(), None
        col = choose_id_column(df.columns)
        if col is None:
            # fallback: use all columns concatenated as a provenance key
            vals = ("|".join(row.astype(str).fillna("")) for _, row in df.iterrows())
            return set(vals), None
        return set(df[col].dropna().astype(str).unique()), col
    # fallback without pandas
    import csv

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        col = choose_id_column(cols)
        ids = set()
        if col is None:
            for row in reader:
                vals = "|".join((row.get(c, "") or "") for c in cols)
                ids.add(vals)
            return ids, None
        for row in reader:
            v = row.get(col)
            if v:
                ids.add(str(v))
        return ids, col


def extract_id_from_json_record(rec: Dict) -> Optional[str]:
    # prefer keys that look like id
    for k in rec.keys():
        lk = k.lower()
        if lk in ("patient_id", "patientid", "user_id", "userid", "id", "patient"):
            return rec.get(k)
    # fallback: any key containing patient/user/id
    for k in rec.keys():
        lk = k.lower()
        if "patient" in lk or "user" in lk or "id" in lk:
            return rec.get(k)
    return None


def ids_from_rag_json(path: str) -> tuple[Set[str], Optional[str]]:
    """Return set of record_ids found in the RAG JSON (these are numeric indices).

    Note: the RAG JSON produced by prepare_rag_database uses `record_id` equal to the
    source DataFrame index. We return the set of those numeric ids so the caller can
    map them back to `user_seq_id` by loading the original CSV used to build the RAG DB.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # If the file is a dict envelope, try to unwrap
    if isinstance(data, dict):
        for k in ("documents", "data", "items", "rows"):
            if k in data and isinstance(data[k], list):
                data = data[k]
                break

    if not isinstance(data, list):
        return set(), None

    record_ids: Set[str] = set()
    for rec in data:
        if not isinstance(rec, dict):
            continue
        # prefer explicit 'record_id' key
        if "record_id" in rec:
            record_ids.add(str(rec["record_id"]))
            continue
        # if the RAG already contains user_seq_id directly, prefer that
        if "user_seq_id" in rec:
            # return these as user ids
            record_ids.add(str(rec["user_seq_id"]))
            continue
        # some text-based databases put the full record under 'metadata'
        if "metadata" in rec and isinstance(rec["metadata"], dict) and "record_id" in rec["metadata"]:
            record_ids.add(str(rec["metadata"]["record_id"]))
            continue
        if "metadata" in rec and isinstance(rec["metadata"], dict) and "user_seq_id" in rec["metadata"]:
            record_ids.add(str(rec["metadata"]["user_seq_id"]))
            continue
        # fallback: try top-level 'id'
        if "id" in rec:
            record_ids.add(str(rec["id"]))

    return record_ids, "record_id"


def short_sample(s: Iterable[str], n: int = 10) -> List[str]:
    out = []
    for i, v in enumerate(s):
        if i >= n:
            break
        out.append(v)
    return out


def validate_condition(condition: str):
    rf_anom = os.path.join(RF_DIR, condition)
    llm_anom = os.path.join(LLM_DIR, condition)

    # detect files
    train_csv = None
    test_csv = None
    rag_json = None

    for fname in os.listdir(rf_anom):
        if fname.endswith("_train.csv"):
            train_csv = os.path.join(rf_anom, fname)
        if fname.endswith("_test.csv"):
            test_csv = os.path.join(rf_anom, fname)

    for fname in os.listdir(llm_anom):
        if fname.endswith("forecast_database.json") or fname.endswith("_forecast_database.json") or "forecast_database" in fname:
            rag_json = os.path.join(llm_anom, fname)

    print(f"\n=== Condition: {condition} ===")
    if not train_csv or not test_csv:
        print(f"Missing train/test CSV under {rf_anom}: train={train_csv}, test={test_csv}")
        return
    if not rag_json:
        print(f"Missing forecast JSON under {llm_anom}")
        return

    train_ids, train_col = ids_from_csv(train_csv)
    test_ids, test_col = ids_from_csv(test_csv)
    combined_rf = set(train_ids) | set(test_ids)
    rag_ids, rag_key = ids_from_rag_json(rag_json)

    # Locate llm-provided test CSV (these live under llm_dataset/{condition}/llm_{condition}_test.csv)
    llm_test_path = os.path.join(LLM_DIR, condition, f"llm_{condition}_test.csv")
    llm_test_ids: Set[str] = set()
    llm_test_col = None
    if os.path.exists(llm_test_path):
        llm_test_ids, llm_test_col = ids_from_csv(llm_test_path)

    # If rag_ids are record indices, map them back to user_seq_id by re-loading
    # the cleaned_data RF train CSV that was used to produce the RAG DB.
    rag_mapped_user_ids: Set[str] = set()
    rag_mapping_note = None
    if rag_key == "record_id" and len(rag_ids) > 0:
        cleaned_path = os.path.join(ROOT, "cleaned_data", f"rf_{condition}_forecast_train.csv")
        if os.path.exists(cleaned_path):
            # load and map
            if pd is not None:
                df_map = pd.read_csv(cleaned_path, dtype=str, low_memory=False)
                # ensure index alignment: record_id was set to the iteration index
                for rid in rag_ids:
                    try:
                        rid_i = int(rid)
                        if 0 <= rid_i < len(df_map):
                            rag_mapped_user_ids.add(str(df_map['user_seq_id'].iat[rid_i]))
                    except Exception:
                        continue
                rag_mapping_note = f"mapped from {os.path.basename(cleaned_path)}"
            else:
                # fallback: manual CSV read
                import csv
                rows = []
                with open(cleaned_path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        rows.append(row)
                for rid in rag_ids:
                    try:
                        rid_i = int(rid)
                        if 0 <= rid_i < len(rows):
                            rag_mapped_user_ids.add(str(rows[rid_i].get('user_seq_id')))
                    except Exception:
                        continue
                rag_mapping_note = f"mapped from {os.path.basename(cleaned_path)}"
        else:
            rag_mapped_user_ids = set()
            rag_mapping_note = f"cleaned source not found: {cleaned_path}"
    else:
        # If record_id wasn't present or mapping not attempted, leave rag_ids as-is
        rag_mapped_user_ids = set(rag_ids)

    print(f"RF train: {len(train_ids)} unique patients (id column: {train_col})")
    print(f"RF test : {len(test_ids)} unique patients (id column: {test_col})")
    print(f"RF combined: {len(combined_rf)} unique patients")
    print(f"RAG json : {len(rag_mapped_user_ids)} unique patients (note: {rag_mapping_note}, key guessed: {rag_key})")

    # Check A: train vs RAG (RAG is expected to reflect the training set)
    train_vs_rag_same = set(train_ids) == rag_mapped_user_ids
    print(f"Match RF TRAIN == RAG JSON (by user_seq_id)? {'YES' if train_vs_rag_same else 'NO'}")
    if not train_vs_rag_same:
        only_in_train = sorted(list(set(train_ids) - rag_mapped_user_ids))
        only_in_rag = sorted(list(rag_mapped_user_ids - set(train_ids)))
        print(f"Only in RF TRAIN (count={len(only_in_train)}) sample: {short_sample(only_in_train, 10)}")
        print(f"Only in RAG (count={len(only_in_rag)}) sample: {short_sample(only_in_rag, 10)}")

    # Check B: RF TEST vs llm_dataset test (llm provides separate test CSVs)
    if os.path.exists(llm_test_path):
        # Print the llm test path and how many unique users were found there
        print(f"LLM test CSV used: {llm_test_path}")
        print(f"LLM test: {len(llm_test_ids)} unique patients (id column: {llm_test_col})")
        test_vs_llm_same = set(test_ids) == set(llm_test_ids)
        print(f"Match RF TEST == LLM_TEST CSV (by user_seq_id)? {'YES' if test_vs_llm_same else 'NO'}")
        if not test_vs_llm_same:
            only_in_rf_test = sorted(list(set(test_ids) - set(llm_test_ids)))
            only_in_llm_test = sorted(list(set(llm_test_ids) - set(test_ids)))
            print(f"Only in RF TEST (count={len(only_in_rf_test)}) sample: {short_sample(only_in_rf_test, 10)}")
            print(f"Only in LLM_TEST (count={len(only_in_llm_test)}) sample: {short_sample(only_in_llm_test, 10)}")
    else:
        print(f"LLM test CSV not found at {llm_test_path}; skipped RF TEST vs LLM_TEST check")


def main():
    for condition in ("anxiety", "depression"):
        validate_condition(condition)


if __name__ == "__main__":
    main()
