#!/usr/bin/env python3
"""
Main training script for Turbofan Engine Predictive Maintenance System
Implements the complete training pipeline with business-aligned evaluation
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import argparse
import logging

# Add src to path
sys.path.append('src')

from data_processing.data_loader import TurbofanDataLoader
from models.dual_lstm_models import DualLSTMPredictor
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories for the project"""
    directories = [
        'models', 'logs', 'results', 'plots', 
        'data/processed', 'config'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logger.info("Project directories created successfully")


def train_dual_models(dataset_name='FD001', temp_epochs=100, vib_epochs=50):
    """
    Train both temperature trend and vibration alert models
    
    Args:
        dataset_name: Dataset to use ('FD001', 'FD002', 'FD003', 'FD004')
        temp_epochs: Epochs for temperature model training
        vib_epochs: Epochs for vibration model training
    
    Returns:
        Trained predictor and evaluation results
    """
    logger.info(f"Starting training pipeline for dataset {dataset_name}")
    
    # Initialize components
    loader = TurbofanDataLoader(data_path="data/raw")
    predictor = DualLSTMPredictor()
    
    # Prepare datasets
    logger.info("Preparing dual datasets with 25% baseline normalization...")
    temp_data, vib_data = loader.prepare_dual_datasets(dataset_name)
    
    logger.info(f"Temperature model training data shape: {temp_data['X_train'].shape}")
    logger.info(f"Vibration model training data shape: {vib_data['X_train'].shape}")
    
    # Save processed data
    np.save(f'data/processed/temp_X_train_{dataset_name}.npy', temp_data['X_train'])
    np.save(f'data/processed/temp_y_train_{dataset_name}.npy', temp_data['y_train'])
    np.save(f'data/processed/vib_X_train_{dataset_name}.npy', vib_data['X_train'])
    np.save(f'data/processed/vib_y_train_{dataset_name}.npy', vib_data['y_train'])
    
    # Train models
    logger.info("Training dual LSTM models...")
    training_history = predictor.train_models(
        temp_data, vib_data, 
        temp_epochs=temp_epochs, 
        vib_epochs=vib_epochs
    )
    
    # Save trained models
    logger.info("Saving trained models...")
    predictor.save_models(f"models/{dataset_name}_")
    
    # Save training history
    with open(f'results/training_history_{dataset_name}.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Evaluate model performance using standard ML metrics
    logger.info("Evaluating model performance...")
    
    # Get predictions on training data for evaluation
    temp_predictions = predictor.temperature_model.predict(temp_data['X_train'])
    vib_probs, vib_preds = predictor.vibration_model.predict(vib_data['X_train'])
    
    # Temperature model evaluation (regression)
    temp_rmse = np.sqrt(mean_squared_error(temp_data['y_train'], temp_predictions))
    temp_mae = mean_absolute_error(temp_data['y_train'], temp_predictions)
    
    # Vibration model evaluation (classification)
    vib_accuracy = accuracy_score(vib_data['y_train'], vib_preds)
    vib_report = classification_report(vib_data['y_train'], vib_preds, output_dict=True)
    
    # Create evaluation results
    evaluation_results = {
        'temperature_model': {
            'rmse': float(temp_rmse),
            'mae': float(temp_mae),
            'data_shape': temp_data['X_train'].shape
        },
        'vibration_model': {
            'accuracy': float(vib_accuracy),
            'precision': float(vib_report['1']['precision']) if '1' in vib_report else 0.0,
            'recall': float(vib_report['1']['recall']) if '1' in vib_report else 0.0,
            'f1_score': float(vib_report['1']['f1-score']) if '1' in vib_report else 0.0,
            'data_shape': vib_data['X_train'].shape
        }
    }
    
    # Generate and save evaluation report
    report = f"""=== TURBOFAN ENGINE PREDICTIVE MAINTENANCE EVALUATION REPORT ===

1. TEMPERATURE TREND MODEL (RUL PREDICTION) PERFORMANCE:
   - RMSE: {temp_rmse:.2f} cycles
   - MAE: {temp_mae:.2f} cycles
   - Training data shape: {temp_data['X_train'].shape}

2. VIBRATION ALERT MODEL (CHANGE DETECTION) PERFORMANCE:
   - Accuracy: {vib_accuracy:.3f}
   - Precision: {evaluation_results['vibration_model']['precision']:.3f}
   - Recall: {evaluation_results['vibration_model']['recall']:.3f}
   - F1-Score: {evaluation_results['vibration_model']['f1_score']:.3f}
   - Training data shape: {vib_data['X_train'].shape}

=== MODEL PERFORMANCE SUMMARY ===
✅ Temperature model predicts RUL within ±{temp_mae:.1f} cycles on average
✅ Vibration model detects changes with {vib_accuracy:.1%} accuracy
"""
    
    with open(f'results/evaluation_report_{dataset_name}.txt', 'w') as f:
        f.write(report)
    
    logger.info("Evaluation report generated")
    print(report)
    
    # Create visualizations
    logger.info("Creating performance visualizations...")
    create_training_plots(training_history, dataset_name)
    
    # Save evaluation results
    with open(f'results/evaluation_results_{dataset_name}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    logger.info(f"Training pipeline completed for {dataset_name}")
    
    return predictor, evaluation_results


def create_training_plots(training_history, dataset_name):
    """Create plots for training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Temperature model loss
    temp_history = training_history['temperature_history']
    axes[0, 0].plot(temp_history['loss'], label='Training Loss')
    if 'val_loss' in temp_history:
        axes[0, 0].plot(temp_history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Temperature Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Temperature model MAE
    axes[0, 1].plot(temp_history['mae'], label='Training MAE')
    if 'val_mae' in temp_history:
        axes[0, 1].plot(temp_history['val_mae'], label='Validation MAE')
    axes[0, 1].set_title('Temperature Model MAE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    
    # Vibration model loss
    vib_history = training_history['vibration_history']
    axes[1, 0].plot(vib_history['loss'], label='Training Loss')
    if 'val_loss' in vib_history:
        axes[1, 0].plot(vib_history['val_loss'], label='Validation Loss')
    axes[1, 0].set_title('Vibration Model Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    
    # Vibration model accuracy
    axes[1, 1].plot(vib_history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in vib_history:
        axes[1, 1].plot(vib_history['val_accuracy'], label='Validation Accuracy')
    axes[1, 1].set_title('Vibration Model Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f'plots/training_history_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def compare_datasets():
    """Compare performance across all datasets"""
    logger.info("Comparing performance across all datasets...")
    
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    comparison_results = {}
    
    for dataset in datasets:
        if os.path.exists(f'results/evaluation_results_{dataset}.json'):
            with open(f'results/evaluation_results_{dataset}.json', 'r') as f:
                results = json.load(f)
                comparison_results[dataset] = results
    
    # Create comparison visualization
    if comparison_results:
        create_comparison_plots(comparison_results)
        
        # Create comparison table
        create_comparison_table(comparison_results)


def create_comparison_plots(comparison_results):
    """Create plots comparing performance across datasets"""
    datasets = list(comparison_results.keys())
    
    # Extract metrics for comparison
    roi_values = []
    catastrophic_rates = []
    alert_accuracy = []
    
    for dataset in datasets:
        results = comparison_results[dataset]
        roi_values.append(results['combined_evaluation']['roi_percentage'])
        catastrophic_rates.append(results['rul_evaluation']['business_penalties']['catastrophic_rate'])
        alert_accuracy.append(results['alert_evaluation']['accuracy'])
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # ROI comparison
    bars1 = axes[0].bar(datasets, roi_values, color='green', alpha=0.7)
    axes[0].set_title('System ROI by Dataset')
    axes[0].set_ylabel('ROI (%)')
    axes[0].set_xlabel('Dataset')
    for i, v in enumerate(roi_values):
        axes[0].text(i, v + 5, f'{v:.1f}%', ha='center', va='bottom')
    
    # Catastrophic rate comparison
    bars2 = axes[1].bar(datasets, [r*100 for r in catastrophic_rates], color='red', alpha=0.7)
    axes[1].set_title('Catastrophic Prediction Rate by Dataset')
    axes[1].set_ylabel('Catastrophic Rate (%)')
    axes[1].set_xlabel('Dataset')
    for i, v in enumerate(catastrophic_rates):
        axes[1].text(i, v*100 + 0.5, f'{v*100:.1f}%', ha='center', va='bottom')
    
    # Alert accuracy comparison
    bars3 = axes[2].bar(datasets, [a*100 for a in alert_accuracy], color='blue', alpha=0.7)
    axes[2].set_title('Alert System Accuracy by Dataset')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_xlabel('Dataset')
    for i, v in enumerate(alert_accuracy):
        axes[2].text(i, v*100 + 1, f'{v*100:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('plots/dataset_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_comparison_table(comparison_results):
    """Create a comparison table of key metrics"""
    datasets = list(comparison_results.keys())
    
    table_data = []
    for dataset in datasets:
        results = comparison_results[dataset]
        
        row = {
            'Dataset': dataset,
            'ROI (%)': f"{results['combined_evaluation']['roi_percentage']:.1f}",
            'Total Savings ($)': f"{results['combined_evaluation']['total_savings']:,.0f}",
            'Catastrophic Rate (%)': f"{results['rul_evaluation']['business_penalties']['catastrophic_rate']*100:.1f}",
            'Optimal Rate (%)': f"{results['rul_evaluation']['business_penalties']['optimal_rate']*100:.1f}",
            'Alert Accuracy': f"{results['alert_evaluation']['accuracy']:.3f}",
            'Alert Precision': f"{results['alert_evaluation']['precision']:.3f}",
            'Alert Recall': f"{results['alert_evaluation']['recall']:.3f}",
            'Business Penalty': f"{results['rul_evaluation']['business_penalties']['total_penalty']:.2f}"
        }
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    
    # Save as CSV
    df.to_csv('results/dataset_comparison.csv', index=False)
    
    # Print formatted table
    print("\n" + "="*100)
    print("DATASET COMPARISON SUMMARY")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Turbofan Engine Predictive Maintenance Models')
    parser.add_argument('--dataset', type=str, default='FD001', 
                       choices=['FD001', 'FD002', 'FD003', 'FD004', 'all'],
                       help='Dataset to train on')
    parser.add_argument('--temp-epochs', type=int, default=100, 
                       help='Epochs for temperature model')
    parser.add_argument('--vib-epochs', type=int, default=50, 
                       help='Epochs for vibration model')
    parser.add_argument('--compare', action='store_true', 
                       help='Compare results across datasets')
    
    args = parser.parse_args()
    
    # Setup project structure
    setup_directories()
    
    # Train models
    if args.dataset == 'all':
        datasets = ['FD001', 'FD002', 'FD003', 'FD004']
        for dataset in datasets:
            logger.info(f"Training models for {dataset}")
            train_dual_models(dataset, args.temp_epochs, args.vib_epochs)
    else:
        train_dual_models(args.dataset, args.temp_epochs, args.vib_epochs)
    
    # Compare datasets if requested
    if args.compare or args.dataset == 'all':
        compare_datasets()
    
    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()