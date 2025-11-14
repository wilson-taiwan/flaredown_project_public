import importlib.util
from pathlib import Path

# Load module by path so we don't need package imports
mod_path = Path(__file__).resolve().parent / 'train_llm_rag_with_dtw.py'
spec = importlib.util.spec_from_file_location('train_llm_rag_with_dtw', str(mod_path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

create_dtw_prompt = mod.create_dtw_prompt

# Minimal synthetic inputs
test_patient = {
    'age': 30,
    'sex': 'F',
    'country': 'USA',
    'depression_days_1_10': {f'day_{i}': float(i%5) for i in range(1,11)},
    'contextual_features': {
        'anxiety_mean_10d': 1.5,
        'fatigue_mean_10d': 0.5
    }
}

# Similar patients: simple mock records matching expected structure
similar_patients = [
    ({'metadata': {
        'patient_metadata': {'id': 'p1'},
        'depression_days_1_10': {f'day_{i}': float((i+1)%5) for i in range(1,11)},
        'target_depression_days_11_20': {f'day_{i}': float((i+2)%5) for i in range(11,21)},
        'contextual_features': {'anxiety_mean_10d': 1.5}
    }}, 0.9)
]

prompt = create_dtw_prompt(test_patient, similar_patients, top_k=1)
print(prompt[:1000])
