import pandas as pd
import json
import numpy as np
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
import time
import os

# Try multiple locations for OPENAI_API_KEY to be robust in different run contexts.
try:
    from config import OPENAI_API_KEY
except Exception:
    try:
        # fall back to llm_output/config.py used in some runs
        from llm_output.config import OPENAI_API_KEY
    except Exception:
        OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
        if not OPENAI_API_KEY:
            raise ImportError(
                "OPENAI_API_KEY not found. Please set it in config.py, llm_output/config.py, or the environment variable OPENAI_API_KEY"
            )

def compute_embeddings():
    """
    Pre-compute embeddings for the RAG text database.
    Uses OpenAI's text-embedding-3-small model (cheap and effective).
    
    API key is imported from config.py (excluded from Git).
    """
    
    print("Initializing OpenAI client...")
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # We process both 'depression' and 'anxiety' text DBs located under llm_dataset/<condition>/
    base = Path(__file__).parent
    conditions = ['depression', 'anxiety']
    batch_size = 100  # OpenAI allows batch requests

    all_outputs = {}
    print("Initializing embedding generation for conditions:", conditions)

    for condition in conditions:
        text_db_path = base / 'llm_dataset' / condition / f'rag_{condition}_forecast_text_database.json'
        if not text_db_path.exists():
            print(f"⚠️  Text DB not found for '{condition}' at {text_db_path}; skipping")
            continue

        print(f"\nLoading text database for '{condition}' from: {text_db_path}")
        with open(text_db_path, 'r') as f:
            text_records = json.load(f)

        print(f"Loaded {len(text_records)} records for '{condition}'")

        embedded_records = []
        print("\nComputing embeddings...")
        print("Model: text-embedding-3-small (1536 dimensions)")
        print(f"Estimated cost: ${(len(text_records) * 500 / 1_000_000) * 0.02:.4f}")

        for i in tqdm(range(0, len(text_records), batch_size)):
            batch = text_records[i:i + batch_size]
            texts = [record['text'] for record in batch]

            try:
                # Call OpenAI embedding API
                response = client.embeddings.create(
                    input=texts,
                    model="text-embedding-3-small",  # Fast, cheap, 1536 dimensions
                    encoding_format="float"
                )

                # Add embeddings to records
                for j, record in enumerate(batch):
                    embedding = response.data[j].embedding
                    embedded_records.append({
                        "id": record['id'],
                        "text": record['text'],
                        "embedding": embedding,
                        "metadata": record['metadata']
                    })

                # Rate limiting (OpenAI allows ~3000 RPM for embeddings)
                time.sleep(0.1)

            except Exception as e:
                print(f"\n❌ Error processing batch {i//batch_size} for '{condition}': {e}")
                # Save progress so far for this condition
                save_embeddings(embedded_records, partial=True, condition=condition, out_dir=base / 'llm_dataset' / condition)
                raise

        # Save embedded database into llm_dataset/<condition>/
        out_dir = base / 'llm_dataset' / condition
        out_dir.mkdir(parents=True, exist_ok=True)
        save_embeddings(embedded_records, partial=False, condition=condition, out_dir=out_dir)
        all_outputs[condition] = embedded_records

    return all_outputs


def save_embeddings(embedded_records, partial=False, condition=None, out_dir=None):
    """Save embedded records to file.

    If `out_dir` and `condition` are provided, save to
    llm_dataset/<condition>/rag_<condition>_forecast_embedded{suffix}.json
    otherwise falls back to project root rag_depression_forecast_embedded.json
    for backward compatibility.
    """
    suffix = "_partial" if partial else ""
    if out_dir is not None and condition:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f'rag_{condition}_forecast_embedded{suffix}.json'
    else:
        output_path = Path(__file__).parent / f'rag_depression_forecast_embedded{suffix}.json'

    with open(output_path, 'w') as f:
        json.dump(embedded_records, f, indent=2)

    print(f"\n✅ Saved {len(embedded_records)} embedded records to: {output_path}")

    # Calculate storage size
    try:
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"   File size: {file_size_mb:.2f} MB")
    except Exception:
        pass


def compute_embedding_stats(embedded_records):
    """Compute statistics about embeddings"""
    print("\n📊 Embedding Statistics:")
    print(f"   Total records: {len(embedded_records)}")
    print(f"   Embedding dimension: {len(embedded_records[0]['embedding'])}")
    
    # Calculate average embedding norm
    norms = [np.linalg.norm(record['embedding']) for record in embedded_records[:100]]
    print(f"   Average embedding norm (first 100): {np.mean(norms):.4f}")
    
    # Estimate cost
    # text-embedding-3-small: $0.02 per 1M tokens
    # Rough estimate: ~500 tokens per record
    total_tokens = len(embedded_records) * 500
    estimated_cost = (total_tokens / 1_000_000) * 0.02
    print(f"   Actual API cost: ~${estimated_cost:.4f}")


def test_similarity_search(embedded_records, query_id=0):
    """Test similarity search with a sample query"""
    print(f"\n🔍 Testing similarity search with record {query_id}...")
    
    query_record = embedded_records[query_id]
    query_embedding = np.array(query_record['embedding'])
    
    # Compute cosine similarity with all other records
    similarities = []
    for record in embedded_records:
        if record['id'] == query_id:
            continue
        
        embedding = np.array(record['embedding'])
        # Cosine similarity
        similarity = np.dot(query_embedding, embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
        )
        similarities.append((record['id'], similarity))
    
    # Get top 5 most similar
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_5 = similarities[:5]
    
    print(f"\nQuery patient (ID {query_id}):")
    print(f"  Age: {query_record['metadata']['patient_metadata']['age']}")
    print(f"  Sex: {query_record['metadata']['patient_metadata']['sex']}")
    
    print(f"\nTop 5 most similar patients:")
    for rank, (record_id, similarity) in enumerate(top_5, 1):
        similar_record = next(r for r in embedded_records if r['id'] == record_id)
        meta = similar_record['metadata']['patient_metadata']
        print(f"  {rank}. ID {record_id} (similarity: {similarity:.4f})")
        print(f"     Age: {meta['age']}, Sex: {meta['sex']}")


if __name__ == "__main__":
    print("=" * 70)
    print("COMPUTING EMBEDDINGS FOR RAG DATABASE")
    print("=" * 70)
    
    # Compute embeddings
    results = compute_embeddings()

    # Show statistics and run a quick similarity test per condition
    if isinstance(results, dict):
        for condition, embedded_records in results.items():
            if not embedded_records:
                print(f"\nNo embeddings for '{condition}' to analyze")
                continue
            print(f"\n--- Stats for: {condition} ---")
            compute_embedding_stats(embedded_records)
            # run a quick similarity test using the first record if present
            try:
                test_similarity_search(embedded_records, query_id=0)
            except Exception as e:
                print(f"   ⚠️  test_similarity_search failed for {condition}: {e}")
    else:
        # backward compatible single-list return
        compute_embedding_stats(results)
        test_similarity_search(results, query_id=0)
    
    print("\n" + "=" * 70)
    print("✅ EMBEDDINGS COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Use 'llm_dataset/<condition>/rag_<condition>_forecast_embedded.json' for RAG-based forecasting")
    print("2. For new patients, compute their embedding at runtime (use same text serialization) and find similar cases")
    print("3. Send top-k similar cases to LLM for context-aware predictions")
