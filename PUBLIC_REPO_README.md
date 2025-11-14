# Flaredown Project - Depression & Anxiety Forecasting

This repository contains code and analysis for forecasting depression and anxiety severity using machine learning (Random Forest) and large language models (LLMs) with Retrieval-Augmented Generation (RAG).

## Project Overview

This research compares different forecasting approaches for mental health symptom trajectories:
- **Random Forest (RF)**: Traditional ML baseline using temporal and clinical features
- **LLM with DTW-based RAG**: Large language models augmented with dynamically retrieved similar patient cases using Dynamic Time Warping

## Repository Structure

```
├── scripts/                          # Data preparation and validation scripts
├── random_forest_output/             # RF model training results
│   ├── train_random_forest.py       # RF training script
│   └── test_results/                # Per-condition evaluation metrics
├── llm_output/                      # LLM training and results
│   ├── train_llm_rag_with_dtw.py   # Main LLM+RAG training script
│   └── [condition]/[model]/         # Results by condition and model
├── publication_ready_tables_and_figures/  # Publication outputs
├── compute_embeddings.py            # Precompute embeddings for RAG
├── prepare_rag_database.py          # Create RAG database from training data
├── generate_per_day_summaries.py   # Generate per-horizon performance summaries
└── config.py                        # API keys (NOT included - see setup)
```

## Setup

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/flaredown_project_public.git
cd flaredown_project_public
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up API keys (for LLM experiments):
Create a `config.py` file in the project root:
```python
OPENROUTER_API_KEY = "your_openrouter_key"
OPENAI_API_KEY = "your_openai_key"
```

## Usage

### Random Forest Baseline

Train Random Forest models:
```bash
python random_forest_output/train_random_forest.py --condition depression
python random_forest_output/train_random_forest.py --condition anxiety
```

Generate per-day summaries:
```bash
python generate_per_day_summaries.py
```

### LLM with DTW-RAG

1. Prepare RAG database:
```bash
python prepare_rag_database.py
```

2. Compute embeddings (optional, for embedding-based RAG):
```bash
python compute_embeddings.py
```

3. Train LLM forecasting models:
```bash
python llm_output/train_llm_rag_with_dtw.py \
    --condition depression \
    --model meta-llama/llama-3.3-70b-instruct \
    --top-k 3
```

## Data

**Note**: The raw patient data is not included in this public repository for privacy reasons. The repository structure assumes data is organized as follows:

- `cleaned_data/`: Preprocessed datasets
- `llm_dataset/[condition]/`: LLM-ready datasets with RAG databases
- `random_forest_dataset/`: Feature-engineered datasets for RF

If you have your own symptom tracking data in a similar format, you can adapt the preprocessing scripts.

## Key Features

### Random Forest
- Per-horizon (day 11-20) forecasting
- Feature engineering from days 1-10
- Cross-validated hyperparameter tuning
- Detailed per-day performance metrics

### LLM with DTW-RAG
- Dynamic Time Warping for temporal similarity matching
- Retrieval-Augmented Generation with historical cases
- Empirical distribution statistics from matched cases
- Structured reasoning + JSON output format
- Support for multiple LLM backends (OpenRouter, OpenAI)

## Results

Performance metrics and visualizations are available in:
- `publication_ready_tables_and_figures/`: Comparison tables and figures
- `random_forest_output/test_results/[condition]/`: RF evaluation metrics
- `llm_output/[condition]/[model]/`: LLM evaluation results

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{your_paper_here,
  title={Forecasting Mental Health Symptoms with LLMs and DTW-based RAG},
  author={Your Name},
  journal={Journal Name},
  year={2025}
}
```

## License

[Choose an appropriate license - MIT, Apache 2.0, etc.]

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For questions or collaboration inquiries, please contact: [your email]

## Acknowledgments

Data sourced from the Flaredown symptom tracking platform.
