"""
Random Forest Forecasting Model (Depression or Anxiety)
=======================================================

This script trains a Random Forest model to predict severity for days 11-20
based on the first 10 days of input data. The script is configurable to run
for either `depression` or `anxiety` using the `--condition` CLI flag.

Usage examples:
    # Train/test for depression (default)
    python train_random_forest.py

    # Train/test for anxiety
    python train_random_forest.py --condition anxiety

    # Optionally limit test set after loading (not recommended for final eval):
    python train_random_forest.py --limit-test 1000

Requirements:
    - pandas
    - numpy
    - scikit-learn
    - matplotlib (optional, for visualizations)

Input (per condition):
    - random_forest_dataset/depression/random_forest_depression_train.csv
    - random_forest_dataset/depression/random_forest_depression_test.csv
    - random_forest_dataset/anxiety/random_forest_anxiety_train.csv
    - random_forest_dataset/anxiety/random_forest_anxiety_test.csv

Output (per condition, configurable):
    - Model: random_forest_output/models/{condition}/{condition}_rf_model.pkl
    - Results (metrics, predictions, summary): random_forest_output/test_results/{condition}/
    - Figures: random_forest_output/test_results/{condition}/figures/

Notes:
    - The script expects the train/test CSVs to be produced upstream (e.g. by
      `scripts/create_rf_datasets.py`) and will by default use those deterministic
      stratified test sets. Avoid re-truncating the test set unless you understand
      the impact on evaluation reproducibility.

"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import argparse
from datetime import datetime

# Scikit-learn imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    mean_absolute_percentage_error
)

# Optional: matplotlib for visualizations
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️  matplotlib/seaborn not available. Visualizations will be skipped.")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Use the sampled test/train datasets produced by scripts/create_rf_datasets.py
    'train_file': 'random_forest_dataset/depression/random_forest_depression_train.csv',
    'test_file': 'random_forest_dataset/depression/random_forest_depression_test.csv',
    # these will be overridden at runtime based on chosen condition (depression|anxiety)
    'model_output_dir': 'models/',
    'results_output_dir': 'results_rf/',
    'figures_output_dir': 'results_rf/figures/',
    'condition': 'depression',
    'results_root': 'random_forest_output/test_results',
    'models_root': 'random_forest_output/models',
    
    # Model hyperparameters
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1,  # Use all CPU cores
    
    # Multiple runs for robustness
    'n_runs': 5,  # Number of times to train the model with different random states
    
    # Data processing
    'imputation_strategy': 'mean',  # 'mean', 'median', or 'most_frequent'
    'scale_features': True,  # Whether to standardize features
    # If set to an integer, limit the loaded test set to this many rows. If None, use the test CSV as-is.
    'limit_test_to': None,
    
    # Evaluation
    'save_predictions': True,
    'generate_plots': True,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_directories():
    """Create output directories if they don't exist."""
    for dir_path in [CONFIG['model_output_dir'], 
                     CONFIG['results_output_dir'], 
                     CONFIG['figures_output_dir']]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directories created")


def load_data():
    """Load training and test datasets."""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Load train
    train_file = CONFIG['train_file']
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}\n"
                              f"Please run the data cleaning pipeline first to generate RF datasets.")
    
    df_train = pd.read_csv(train_file)
    print(f"✓ Loaded training data: {df_train.shape}")
    
    # Load test
    test_file = CONFIG['test_file']
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}\n"
                              f"Please run the data cleaning pipeline first to generate RF datasets.")
    
    df_test = pd.read_csv(test_file)
    print(f"✓ Loaded test data: {df_test.shape}")
    
    # Optionally limit test set size (default: use test CSV as produced by the data pipeline).
    limit = CONFIG.get('limit_test_to')
    if limit and len(df_test) > limit:
        # Do a reproducible random sample when limiting (stratification is handled upstream).
        df_test = df_test.sample(n=limit, random_state=CONFIG['random_state']).reset_index(drop=True)
        print(f"✓ Limited test set to {limit} instances (random sample)")
    
    return df_train, df_test


def prepare_features_and_targets(df_train, df_test):
    """Separate features and targets, handle missing values, and scale."""
    print("\n" + "="*80)
    print("PREPARING FEATURES AND TARGETS")
    print("="*80)
    
    # Identify feature and target columns
    exclude_cols = ['user_seq_id', 'window_start_date', 'window_end_date']
    feature_cols = [col for col in df_train.columns 
                   if not col.startswith('target_') and col not in exclude_cols]
    target_cols = [col for col in df_train.columns if col.startswith('target_')]
    
    print(f"✓ Feature columns: {len(feature_cols)}")
    print(f"✓ Target columns: {len(target_cols)}")
    
    # Extract features and targets
    X_train = df_train[feature_cols].copy()
    y_train = df_train[target_cols].copy()
    X_test = df_test[feature_cols].copy()
    y_test = df_test[target_cols].copy()
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Handle categorical variables (sex, country)
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"\n✓ Encoding categorical variables: {list(categorical_cols)}")
        X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
        X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
        
        # Align columns (in case test set has different categories)
        X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    
    # Handle missing values
    print(f"\n✓ Imputing missing values (strategy: {CONFIG['imputation_strategy']})")

    # Some features may be entirely missing in the training set (all-NaN). sklearn's
    # SimpleImputer will warn and may skip such features. To keep shapes stable,
    # fill all-NaN columns with a neutral value (0) before running the imputer.
    all_nan_cols = [c for c in X_train.columns if X_train[c].isna().all()]
    if all_nan_cols:
        print(f"  ⚠️  Found all-NaN feature columns in training set: {all_nan_cols}")
        print("  → Filling those columns with 0 to preserve feature shape for imputation")
        X_train[all_nan_cols] = X_train[all_nan_cols].fillna(0)
        # Also fill in test set columns to keep transforms consistent
        X_test[all_nan_cols] = X_test[all_nan_cols].fillna(0)

    imputer = SimpleImputer(strategy=CONFIG['imputation_strategy'])
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Convert back to DataFrame to preserve column names
    X_train = pd.DataFrame(X_train_imputed, columns=X_train.columns)
    X_test = pd.DataFrame(X_test_imputed, columns=X_test.columns)
    
    # Scale features (optional but recommended)
    scaler = None
    if CONFIG['scale_features']:
        print(f"✓ Scaling features (StandardScaler)")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    print(f"\n✓ Final feature shape: {X_train.shape}")
    print(f"✓ Final target shape: {y_train.shape}")
    
    return X_train, X_test, y_train, y_test, feature_cols, target_cols, imputer, scaler


def train_model(X_train, y_train, run_number=None):
    """Train Random Forest model."""
    if run_number is None:
        print("\n" + "="*80)
        print("TRAINING RANDOM FOREST MODEL")
        print("="*80)
    else:
        print(f"\n{'='*80}")
        print(f"TRAINING RUN {run_number}/{CONFIG['n_runs']}")
        print(f"{'='*80}")
    
    print(f"\nHyperparameters:")
    print(f"  • n_estimators: {CONFIG['n_estimators']}")
    print(f"  • max_depth: {CONFIG['max_depth']}")
    print(f"  • min_samples_split: {CONFIG['min_samples_split']}")
    print(f"  • min_samples_leaf: {CONFIG['min_samples_leaf']}")
    print(f"  • max_features: {CONFIG['max_features']}")
    if run_number is not None:
        random_state = CONFIG['random_state'] + run_number - 1
        print(f"  • random_state: {random_state}")
    else:
        print(f"  • random_state: {CONFIG['random_state']}")
    
    # Determine random state
    if run_number is not None:
        random_state = CONFIG['random_state'] + run_number - 1
    else:
        random_state = CONFIG['random_state']
    
    # Create base Random Forest
    rf_base = RandomForestRegressor(
        n_estimators=CONFIG['n_estimators'],
        max_depth=CONFIG['max_depth'],
        min_samples_split=CONFIG['min_samples_split'],
        min_samples_leaf=CONFIG['min_samples_leaf'],
        max_features=CONFIG['max_features'],
        random_state=random_state,
        n_jobs=CONFIG['n_jobs'],
        verbose=0  # Set to 0 to reduce output clutter with multiple runs
    )
    
    # Wrap in MultiOutputRegressor for multiple target prediction
    model = MultiOutputRegressor(rf_base)
    
    print(f"\n⏳ Training model on {X_train.shape[0]} samples...")
    start_time = datetime.now()
    
    model.fit(X_train, y_train)
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    print(f"✓ Training complete! Time: {training_time:.2f} seconds")
    
    return model, training_time


def evaluate_model(model, X_train, y_train, X_test, y_test, target_cols, run_number=None):
    """Evaluate model performance on train and test sets."""
    if run_number is None:
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)
    
    # Make predictions
    if run_number is None:
        print(f"\n⏳ Making predictions...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics for each day (target)
    results = []
    
    if run_number is None:
        print(f"\n{'Day':<15} {'Train MSE':<12} {'Test MSE':<12} {'Train MAE':<12} {'Test MAE':<12} {'Test R²':<12}")
        print("-" * 80)
    
    for i, target_col in enumerate(target_cols):
        # Extract day number from format: target_depression_day11 -> 11
        day_num = int(target_col.split('day')[-1])
        
        # Training metrics
        train_mse = mean_squared_error(y_train.iloc[:, i], y_train_pred[:, i])
        train_mae = mean_absolute_error(y_train.iloc[:, i], y_train_pred[:, i])
        train_r2 = r2_score(y_train.iloc[:, i], y_train_pred[:, i])
        
        # Test metrics
        test_mse = mean_squared_error(y_test.iloc[:, i], y_test_pred[:, i])
        test_mae = mean_absolute_error(y_test.iloc[:, i], y_test_pred[:, i])
        test_r2 = r2_score(y_test.iloc[:, i], y_test_pred[:, i])
        
        results.append({
            'day': day_num,
            'target_column': target_col,
            'train_mse': train_mse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_r2': test_r2
        })
        
        if run_number is None:
            print(f"Day {day_num:<11} {train_mse:<12.4f} {test_mse:<12.4f} {train_mae:<12.4f} {test_mae:<12.4f} {test_r2:<12.4f}")
    
    # Overall metrics
    if run_number is None:
        print("\n" + "-" * 80)
    overall_train_mse = mean_squared_error(y_train, y_train_pred)
    overall_test_mse = mean_squared_error(y_test, y_test_pred)
    overall_train_mae = mean_absolute_error(y_train, y_train_pred)
    overall_test_mae = mean_absolute_error(y_test, y_test_pred)
    overall_test_r2 = r2_score(y_test, y_test_pred)
    
    if run_number is None:
        print(f"{'Overall':<15} {overall_train_mse:<12.4f} {overall_test_mse:<12.4f} {overall_train_mae:<12.4f} {overall_test_mae:<12.4f} {overall_test_r2:<12.4f}")
        
        # Summary
        print(f"\n📊 Model Performance Summary:")
        print(f"  • Overall Test RMSE: {np.sqrt(overall_test_mse):.4f}")
        print(f"  • Overall Test MAE: {overall_test_mae:.4f}")
        print(f"  • Overall Test R²: {overall_test_r2:.4f}")
    else:
        print(f"  Run {run_number} - Test RMSE: {np.sqrt(overall_test_mse):.4f}, MAE: {overall_test_mae:.4f}, R²: {overall_test_r2:.4f}")
    
    results_df = pd.DataFrame(results)
    
    return results_df, y_test_pred


def save_model_and_results(model, imputer, scaler, results_df, y_test_pred, 
                           df_test, target_cols, training_time, all_run_metrics=None):
    """Save trained model, preprocessing objects, and results."""
    print("\n" + "="*80)
    print("SAVING MODEL AND RESULTS")
    print("="*80)
    
    # Save model (save the last model or best performing model)
    condition = CONFIG.get('condition', 'depression')
    model_filename = f"{condition}_rf_model.pkl"
    model_path = os.path.join(CONFIG['model_output_dir'], model_filename)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved: {model_path}")
    
    # Save preprocessing objects
    preprocessing_path = os.path.join(CONFIG['model_output_dir'], 'preprocessing.pkl')
    with open(preprocessing_path, 'wb') as f:
        pickle.dump({'imputer': imputer, 'scaler': scaler}, f)
    print(f"✓ Preprocessing objects saved: {preprocessing_path}")
    
    # Save evaluation metrics
    metrics_path = os.path.join(CONFIG['results_output_dir'], 'evaluation_metrics.csv')
    results_df.to_csv(metrics_path, index=False)
    print(f"✓ Evaluation metrics saved: {metrics_path}")
    
    # Save per-run metrics if available
    if all_run_metrics is not None:
        run_metrics_path = os.path.join(CONFIG['results_output_dir'], 'evaluation_metrics_per_run.csv')
        all_run_metrics.to_csv(run_metrics_path, index=False)
        print(f"✓ Per-run metrics saved: {run_metrics_path}")
    
    # Save predictions
    if CONFIG['save_predictions']:
        predictions_df = df_test[['user_seq_id', 'window_start_date', 'window_end_date']].copy()
        
        # Add actual values
        for col in target_cols:
            predictions_df[f'actual_{col}'] = df_test[col].values
        
        # Add predictions
        for i, col in enumerate(target_cols):
            predictions_df[f'predicted_{col}'] = y_test_pred[:, i]
        
        # Add residuals
        for i, col in enumerate(target_cols):
            predictions_df[f'residual_{col}'] = (
                predictions_df[f'actual_{col}'] - predictions_df[f'predicted_{col}']
            )
        
        pred_path = os.path.join(CONFIG['results_output_dir'], 'predictions.csv')
        predictions_df.to_csv(pred_path, index=False)
        print(f"✓ Predictions saved: {pred_path}")
    
    # Save summary report
    summary_path = os.path.join(CONFIG['results_output_dir'], 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RANDOM FOREST DEPRESSION FORECASTING - TRAINING SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Configuration:\n")
        for key, value in CONFIG.items():
            if not key.endswith('_dir'):
                f.write(f"  • {key}: {value}\n")
        
        f.write(f"\nTotal Training Time: {training_time:.2f} seconds\n")
        
        if all_run_metrics is not None:
            f.write(f"\nMulti-Run Performance (averaged over {CONFIG['n_runs']} runs):\n")
            f.write(f"  • Test RMSE: {results_df['test_rmse_mean'].iloc[0]:.4f} ± {results_df['test_rmse_std'].iloc[0]:.4f}\n")
            f.write(f"  • Test MAE: {results_df['test_mae_mean'].iloc[0]:.4f} ± {results_df['test_mae_std'].iloc[0]:.4f}\n")
            f.write(f"  • Test R²: {results_df['test_r2_mean'].iloc[0]:.4f} ± {results_df['test_r2_std'].iloc[0]:.4f}\n")
        else:
            f.write("\nOverall Performance:\n")
            f.write(f"  • Test RMSE: {np.sqrt(results_df['test_mse'].mean()):.4f}\n")
            f.write(f"  • Test MAE: {results_df['test_mae'].mean():.4f}\n")
            f.write(f"  • Test R²: {results_df['test_r2'].mean():.4f}\n")
    
    print(f"✓ Training summary saved: {summary_path}")


def generate_visualizations(results_df, y_test, y_test_pred, target_cols):
    """Generate evaluation visualizations."""
    if not PLOTTING_AVAILABLE or not CONFIG['generate_plots']:
        print("\n⚠️  Skipping visualizations (matplotlib not available or disabled)")
        return
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Metrics by day
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # MSE or RMSE
    if 'test_mse' in results_df.columns:
        metric_col = 'test_mse'
        metric_label = 'Mean Squared Error (MSE)'
    elif 'test_rmse_mean' in results_df.columns:
        metric_col = 'test_rmse_mean'
        metric_label = 'Root Mean Squared Error (RMSE)'
    else:
        metric_col = 'test_rmse'
        metric_label = 'Root Mean Squared Error (RMSE)'
    
    # Filter out 'Overall' row if present
    plot_df = results_df[results_df['day'] != 'Overall'].copy()
    plot_df['day'] = plot_df['day'].astype(int)
    
    axes[0, 0].plot(plot_df['day'], plot_df[metric_col], 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Prediction Day', fontweight='bold')
    axes[0, 0].set_ylabel(metric_label, fontweight='bold')
    axes[0, 0].set_title(f'Test {metric_label} by Prediction Day', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE
    if 'test_mae' in results_df.columns:
        mae_col = 'test_mae'
    else:
        mae_col = 'test_mae_mean'
    
    axes[0, 1].plot(plot_df['day'], plot_df[mae_col], 'o-', 
                   linewidth=2, markersize=8, color='coral')
    axes[0, 1].set_xlabel('Prediction Day', fontweight='bold')
    axes[0, 1].set_ylabel('Mean Absolute Error (MAE)', fontweight='bold')
    axes[0, 1].set_title('Test MAE by Prediction Day', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # R²
    if 'test_r2' in results_df.columns:
        r2_col = 'test_r2'
    else:
        r2_col = 'test_r2_mean'
    
    axes[1, 0].plot(plot_df['day'], plot_df[r2_col], 'o-', 
                   linewidth=2, markersize=8, color='green')
    axes[1, 0].set_xlabel('Prediction Day', fontweight='bold')
    axes[1, 0].set_ylabel('R² Score', fontweight='bold')
    axes[1, 0].set_title('Test R² by Prediction Day', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    # Error bars or metric progression (instead of train vs test comparison)
    if 'test_rmse_std' in results_df.columns:
        # Show RMSE with error bars
        axes[1, 1].errorbar(plot_df['day'], plot_df['test_rmse_mean'], 
                          yerr=plot_df['test_rmse_std'],
                          fmt='o-', linewidth=2, markersize=8, capsize=5,
                          label='RMSE (mean ± std)', color='blue')
        axes[1, 1].set_xlabel('Prediction Day', fontweight='bold')
        axes[1, 1].set_ylabel('Root Mean Squared Error (RMSE)', fontweight='bold')
        axes[1, 1].set_title('Test RMSE with Standard Deviation', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Show RMSE progression
        axes[1, 1].plot(plot_df['day'], np.sqrt(plot_df['test_mse']), 'o-', 
                       linewidth=2, markersize=8, label='RMSE', color='blue')
        axes[1, 1].set_xlabel('Prediction Day', fontweight='bold')
        axes[1, 1].set_ylabel('Root Mean Squared Error (RMSE)', fontweight='bold')
        axes[1, 1].set_title('Test RMSE by Prediction Day', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['figures_output_dir'], '01_metrics_by_day.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 01_metrics_by_day.png")
    
    # 2. Actual vs Predicted (scatter plot for each day)
    n_days = len(target_cols)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, target_col in enumerate(target_cols):
        # Extract day number from format: target_depression_day11 -> 11
        day_num = int(target_col.split('day')[-1])
        ax = axes[i]
        
        ax.scatter(y_test.iloc[:, i], y_test_pred[:, i], alpha=0.5, s=30)
        
        # Add diagonal line (perfect prediction)
        min_val = min(y_test.iloc[:, i].min(), y_test_pred[:, i].min())
        max_val = max(y_test.iloc[:, i].max(), y_test_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.5)
        
        ax.set_xlabel('Actual', fontweight='bold')
        ax.set_ylabel('Predicted', fontweight='bold')
        ax.set_title(f'Day {day_num}', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Actual vs Predicted Depression Severity (Days 11-20)', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['figures_output_dir'], '02_actual_vs_predicted.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 02_actual_vs_predicted.png")
    
    # 3. Residual plots
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, target_col in enumerate(target_cols):
        # Extract day number from format: target_depression_day11 -> 11
        day_num = int(target_col.split('day')[-1])
        ax = axes[i]
        
        residuals = y_test.iloc[:, i] - y_test_pred[:, i]
        
        ax.scatter(y_test_pred[:, i], residuals, alpha=0.5, s=30)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2, alpha=0.5)
        
        ax.set_xlabel('Predicted', fontweight='bold')
        ax.set_ylabel('Residual', fontweight='bold')
        ax.set_title(f'Day {day_num}', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Residual Plots (Days 11-20)', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['figures_output_dir'], '03_residuals.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 03_residuals.png")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    # Parse command-line args to select condition and optional test limit
    parser = argparse.ArgumentParser(description='Train Random Forest for depression or anxiety forecasting')
    parser.add_argument('--condition', choices=['depression', 'anxiety'], default=CONFIG.get('condition','depression'),
                        help='Condition to train/test (depression or anxiety)')
    parser.add_argument('--limit-test', type=int, default=CONFIG.get('limit_test_to'),
                        help='Optionally limit number of test instances after loading')
    args = parser.parse_args()

    # Update CONFIG based on chosen condition
    CONFIG['condition'] = args.condition
    CONFIG['limit_test_to'] = args.limit_test

    # Set dataset paths and output folders per condition
    CONFIG['train_file'] = f"random_forest_dataset/{CONFIG['condition']}/random_forest_{CONFIG['condition']}_train.csv"
    CONFIG['test_file'] = f"random_forest_dataset/{CONFIG['condition']}/random_forest_{CONFIG['condition']}_test.csv"
    CONFIG['results_output_dir'] = os.path.join(CONFIG['results_root'], CONFIG['condition'])
    CONFIG['figures_output_dir'] = os.path.join(CONFIG['results_output_dir'], 'figures')
    CONFIG['model_output_dir'] = os.path.join(CONFIG['models_root'], CONFIG['condition'])

    print("\n" + "="*80)
    print(f"RANDOM FOREST {CONFIG['condition'].upper()} FORECASTING")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Number of runs: {CONFIG['n_runs']}")
    
    # Create output directories
    create_directories()
    
    # Load data
    df_train, df_test = load_data()
    
    # Prepare features and targets
    X_train, X_test, y_train, y_test, feature_cols, target_cols, imputer, scaler = \
        prepare_features_and_targets(df_train, df_test)
    
    # Multiple training runs
    all_predictions = []
    all_metrics = []
    all_models = []
    total_training_time = 0
    
    print("\n" + "="*80)
    print(f"RUNNING {CONFIG['n_runs']} TRAINING ITERATIONS")
    print("="*80)
    
    for run in range(1, CONFIG['n_runs'] + 1):
        # Train model
        model, training_time = train_model(X_train, y_train, run_number=run)
        total_training_time += training_time
        
        # Evaluate model
        results_df, y_test_pred = evaluate_model(
            model, X_train, y_train, X_test, y_test, target_cols, run_number=run
        )
        
        # Store results
        all_predictions.append(y_test_pred)
        results_df['run'] = run
        all_metrics.append(results_df)
        all_models.append(model)
    
    # Average predictions across all runs
    print("\n" + "="*80)
    print("AVERAGING PREDICTIONS ACROSS RUNS")
    print("="*80)
    
    y_test_pred_avg = np.mean(all_predictions, axis=0)
    print(f"✓ Averaged predictions from {CONFIG['n_runs']} runs")
    
    # Evaluate averaged predictions
    print("\n" + "="*80)
    print("FINAL AVERAGED MODEL EVALUATION")
    print("="*80)
    
    final_results = []
    print(f"\n{'Day':<15} {'Test MSE':<12} {'Test MAE':<12} {'Test R²':<12} {'Test RMSE':<12}")
    print("-" * 80)
    
    for i, target_col in enumerate(target_cols):
        day_num = int(target_col.split('day')[-1])
        
        test_mse = mean_squared_error(y_test.iloc[:, i], y_test_pred_avg[:, i])
        test_mae = mean_absolute_error(y_test.iloc[:, i], y_test_pred_avg[:, i])
        test_r2 = r2_score(y_test.iloc[:, i], y_test_pred_avg[:, i])
        test_rmse = np.sqrt(test_mse)
        
        final_results.append({
            'day': day_num,
            'target_column': target_col,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'test_rmse': test_rmse
        })
        
        print(f"Day {day_num:<11} {test_mse:<12.4f} {test_mae:<12.4f} {test_r2:<12.4f} {test_rmse:<12.4f}")
    
    # Calculate overall metrics
    overall_test_mse = mean_squared_error(y_test, y_test_pred_avg)
    overall_test_mae = mean_absolute_error(y_test, y_test_pred_avg)
    overall_test_r2 = r2_score(y_test, y_test_pred_avg)
    overall_test_rmse = np.sqrt(overall_test_mse)
    
    print("\n" + "-" * 80)
    print(f"{'Overall':<15} {overall_test_mse:<12.4f} {overall_test_mae:<12.4f} {overall_test_r2:<12.4f} {overall_test_rmse:<12.4f}")
    
    # Calculate statistics across runs
    print("\n" + "="*80)
    print("MULTI-RUN STATISTICS")
    print("="*80)
    
    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    
    # Group by day and calculate mean and std
    stats_by_day = []
    for day_num in range(11, 21):
        day_metrics = all_metrics_df[all_metrics_df['day'] == day_num]
        stats_by_day.append({
            'day': day_num,
            'test_mse_mean': day_metrics['test_mse'].mean(),
            'test_mse_std': day_metrics['test_mse'].std(),
            'test_mae_mean': day_metrics['test_mae'].mean(),
            'test_mae_std': day_metrics['test_mae'].std(),
            'test_r2_mean': day_metrics['test_r2'].mean(),
            'test_r2_std': day_metrics['test_r2'].std(),
            'test_rmse_mean': np.sqrt(day_metrics['test_mse'].mean()),
            'test_rmse_std': np.sqrt(day_metrics['test_mse'].std())
        })
    
    stats_df = pd.DataFrame(stats_by_day)
    
    # Overall statistics
    overall_stats = {
        'day': 'Overall',
        'test_mse_mean': all_metrics_df.groupby('run')['test_mse'].mean().mean(),
        'test_mse_std': all_metrics_df.groupby('run')['test_mse'].mean().std(),
        'test_mae_mean': all_metrics_df.groupby('run')['test_mae'].mean().mean(),
        'test_mae_std': all_metrics_df.groupby('run')['test_mae'].mean().std(),
        'test_r2_mean': all_metrics_df.groupby('run')['test_r2'].mean().mean(),
        'test_r2_std': all_metrics_df.groupby('run')['test_r2'].mean().std(),
        'test_rmse_mean': np.sqrt(all_metrics_df.groupby('run')['test_mse'].mean().mean()),
        'test_rmse_std': np.sqrt(all_metrics_df.groupby('run')['test_mse'].mean().std())
    }
    
    stats_df = pd.concat([stats_df, pd.DataFrame([overall_stats])], ignore_index=True)
    
    print(f"\n{'Day':<15} {'RMSE (mean±std)':<25} {'MAE (mean±std)':<25} {'R² (mean±std)':<25}")
    print("-" * 90)
    
    for _, row in stats_df.iterrows():
        day_label = f"Day {int(row['day'])}" if row['day'] != 'Overall' else 'Overall'
        rmse_str = f"{row['test_rmse_mean']:.4f} ± {row['test_rmse_std']:.4f}"
        mae_str = f"{row['test_mae_mean']:.4f} ± {row['test_mae_std']:.4f}"
        r2_str = f"{row['test_r2_mean']:.4f} ± {row['test_r2_std']:.4f}"
        print(f"{day_label:<15} {rmse_str:<25} {mae_str:<25} {r2_str:<25}")
    
    print(f"\n📊 Final Performance Summary (Averaged over {CONFIG['n_runs']} runs):")
    print(f"  • Overall Test RMSE: {overall_test_rmse:.4f}")
    print(f"  • Overall Test MAE: {overall_test_mae:.4f}")
    print(f"  • Overall Test R²: {overall_test_r2:.4f}")
    print(f"\n  • Mean RMSE across runs: {overall_stats['test_rmse_mean']:.4f} ± {overall_stats['test_rmse_std']:.4f}")
    print(f"  • Mean MAE across runs: {overall_stats['test_mae_mean']:.4f} ± {overall_stats['test_mae_std']:.4f}")
    print(f"  • Mean R² across runs: {overall_stats['test_r2_mean']:.4f} ± {overall_stats['test_r2_std']:.4f}")
    
    # Save results (use the last model, but save averaged predictions)
    final_results_df = pd.DataFrame(final_results)
    save_model_and_results(
        all_models[-1], imputer, scaler, stats_df, y_test_pred_avg, 
        df_test, target_cols, total_training_time, all_run_metrics=all_metrics_df
    )
    
    # Generate visualizations (using averaged predictions)
    generate_visualizations(final_results_df, y_test, y_test_pred_avg, target_cols)
    
    # Final summary
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\n📁 Output Locations:")
    print(f"  • Model: {CONFIG['model_output_dir']}")
    print(f"  • Results: {CONFIG['results_output_dir']}")
    print(f"  • Figures: {CONFIG['figures_output_dir']}")
    print(f"\n⏰ End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Total Training Time: {total_training_time:.2f} seconds")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
