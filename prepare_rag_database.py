import pandas as pd
import json
from pathlib import Path

def prepare_rag_database():
    """
    Prepare a RAG database for LLM-based depression severity forecasting.
    Creates a JSON database with patient records containing:
    - First 10 days of symptoms/conditions
    - Patient metadata (age, country)
    - Target: Next 10 days of depression severity (days 11-20)
    """
    
    # Load the RF training data
    base_path = Path(__file__).parent
    rf_data_path = base_path / 'cleaned_data' / 'rf_depression_forecast_train.csv'
    df = pd.read_csv(rf_data_path)
    
    # Create RAG database structure
    rag_database = []
    
    for idx, row in df.iterrows():
        # Extract patient metadata
        patient_record = {
            "record_id": idx,
            "user_seq_id": row.get('user_seq_id', None),
            "patient_metadata": {
                "age": row.get('age', None),
                "country": row.get('country', None),
                "sex": row.get('sex', None)
            },
            
            # First 10 days of depression (input features)
            "depression_days_1_10": {
                f"day_{i}": row.get(f'depression_day{i}', None) 
                for i in range(1, 11)
            },
            
            # Other symptoms/conditions for days 1-10
            "contextual_features": {},
            
            # Target: Days 11-20 depression severity
            "target_depression_days_11_20": {
                f"day_{i}": row.get(f'target_depression_day{i}', None) 
                for i in range(11, 21)
            }
        }
        
        # Extract all other symptom/condition features from days 1-10
        for col in df.columns:
            if col not in ['age', 'country', 'sex', 'user_seq_id', 'window_start_date', 'window_end_date'] and 'depression' not in col and 'target_' not in col:
                # Include mean features and other aggregated features from days 1-10
                patient_record["contextual_features"][col] = row[col]
        
        rag_database.append(patient_record)
    
    # Save as JSON for RAG
    base_path = Path(__file__).parent
    output_path = base_path / 'rag_depression_forecast_database.json'
    with open(output_path, 'w') as f:
        json.dump(rag_database, f, indent=2)
    
    print(f"RAG database created with {len(rag_database)} patient records")
    print(f"Saved to: {output_path}")
    
    # Also create a text-based version for embedding
    create_text_database(rag_database)
    
    return rag_database


def create_text_database(rag_database):
    """
    Create a text-based version of the database for semantic search/RAG.
    
    IMPORTANT: The text used for EMBEDDING includes ONLY days 1-10 (input features).
    This ensures similarity search finds patients with similar INPUT patterns,
    not similar OUTCOMES.
    
    However, the full metadata (including days 11-20) is preserved so the LLM
    can see the complete trajectory when making predictions.
    """
    text_records = []
    
    for record in rag_database:
        # TEXT FOR EMBEDDING: Only days 1-10 (no data leakage)
        text = f"Patient Record {record['record_id']}:\n"
        text += f"Demographics: Age {record['patient_metadata']['age']}, "
        text += f"Country: {record['patient_metadata']['country']}, "
        text += f"Sex: {record['patient_metadata']['sex']}\n\n"
        
        text += "Depression levels for days 1-10:\n"
        for day, value in record['depression_days_1_10'].items():
            text += f"  {day}: {value}\n"
        
        text += "\nOther symptoms and conditions (days 1-10):\n"
        for feature, value in record['contextual_features'].items():
            if pd.notna(value):
                text += f"  {feature}: {value}\n"
        
        # NOTE: Days 11-20 are NOT included in the embedding text
        # They are only in metadata for the LLM to use after retrieval
        
        # Ensure metadata includes user_seq_id for reliable mapping
        metadata = dict(record)
        metadata["user_seq_id"] = record.get("user_seq_id")

        text_records.append({
            "id": record['record_id'],
            "text": text,  # Only days 1-10
            "metadata": metadata  # Full record including days 11-20 and user_seq_id
        })
    
    # Save text database
    base_path = Path(__file__).parent
    output_path = base_path / 'rag_depression_forecast_text_database.json'
    with open(output_path, 'w') as f:
        json.dump(text_records, f, indent=2)
    
    print(f"Text-based RAG database saved to: {output_path}")


if __name__ == "__main__":
    rag_database = prepare_rag_database()
    print(f"\nSample record:\n{json.dumps(rag_database[0], indent=2)}")