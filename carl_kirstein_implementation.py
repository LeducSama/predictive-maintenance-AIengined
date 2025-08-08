#!/usr/bin/env python3
"""
NASA Turbofan Engine RUL Prediction - Carl Kirstein's Exact Implementation
Following the methodology from: https://www.kaggle.com/code/carlkirstein/predictive-maintenance-nasa-turbofan-regression/notebook
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import statsmodels.api as sm
import time
import warnings

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
warnings.filterwarnings('ignore')

# Configure pandas display
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', 500) 
pd.set_option('display.expand_frame_repr', False)

def load_data():
    """Load data with exact column names from Carl Kirstein's notebook"""
    print("Loading NASA Turbofan dataset...")
    
    # Define column names exactly as in the notebook
    index_names = ['engine', 'cycle']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = [
        "(Fan inlet temperature) (◦R)",
        "(LPC outlet temperature) (◦R)",
        "(HPC outlet temperature) (◦R)",
        "(LPT outlet temperature) (◦R)",
        "(Fan inlet Pressure) (psia)",
        "(bypass-duct pressure) (psia)",
        "(HPC outlet pressure) (psia)",
        "(Physical fan speed) (rpm)",
        "(Physical core speed) (rpm)",
        "(Engine pressure ratio(P50/P2)",
        "(HPC outlet Static pressure) (psia)",
        "(Ratio of fuel flow to Ps30) (pps/psia)",
        "(Corrected fan speed) (rpm)",
        "(Corrected core speed) (rpm)",
        "(Bypass Ratio) ",
        "(Burner fuel-air ratio)",
        "(Bleed Enthalpy)",
        "(Required fan speed)",
        "(Required fan conversion speed)",
        "(High-pressure turbines Cool air flow)",
        "(Low-pressure turbines Cool air flow)"
    ]
    col_names = index_names + setting_names + sensor_names
    
    # Load data files
    df_train = pd.read_csv('data/raw/train_FD001.txt', sep=r'\s+', header=None, names=col_names)
    df_test = pd.read_csv('data/raw/test_FD001.txt', sep=r'\s+', header=None, names=col_names)
    df_test_RUL = pd.read_csv('data/raw/RUL_FD001.txt', sep=r'\s+', header=None, names=['RUL'])
    
    print(f"Train shape: {df_train.shape}")
    print(f"Test shape: {df_test.shape}")
    print(f"RUL shape: {df_test_RUL.shape}")
    
    return df_train, df_test, df_test_RUL

def remove_constant_features(df_train, df_test):
    """Remove sensors with constant values - exact implementation from notebook"""
    print("Removing constant features...")
    
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = [
        "(Fan inlet temperature) (◦R)",
        "(LPC outlet temperature) (◦R)",
        "(HPC outlet temperature) (◦R)",
        "(LPT outlet temperature) (◦R)",
        "(Fan inlet Pressure) (psia)",
        "(bypass-duct pressure) (psia)",
        "(HPC outlet pressure) (psia)",
        "(Physical fan speed) (rpm)",
        "(Physical core speed) (rpm)",
        "(Engine pressure ratio(P50/P2)",
        "(HPC outlet Static pressure) (psia)",
        "(Ratio of fuel flow to Ps30) (pps/psia)",
        "(Corrected fan speed) (rpm)",
        "(Corrected core speed) (rpm)",
        "(Bypass Ratio) ",
        "(Burner fuel-air ratio)",
        "(Bleed Enthalpy)",
        "(Required fan speed)",
        "(Required fan conversion speed)",
        "(High-pressure turbines Cool air flow)",
        "(Low-pressure turbines Cool air flow)"
    ]
    
    # Find constant features
    sens_const_values = []
    for feature in setting_names + sensor_names:
        try:
            if df_train[feature].min() == df_train[feature].max():
                sens_const_values.append(feature)
        except:
            pass
    
    print(f"Constant features to remove: {sens_const_values}")
    
    # Drop constant features
    df_train.drop(sens_const_values, axis=1, inplace=True)
    df_test.drop(sens_const_values, axis=1, inplace=True)
    
    return df_train, df_test

def remove_highly_correlated_features(df_train, df_test, threshold=0.95):
    """Remove highly correlated features - exact implementation from notebook"""
    print("Removing highly correlated features...")
    
    # Calculate correlation matrix
    cor_matrix = df_train.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
    print(f"Highly correlated features to remove: {corr_features}")
    
    # Drop correlated features
    df_train.drop(corr_features, axis=1, inplace=True)
    df_test.drop(corr_features, axis=1, inplace=True)
    
    return df_train, df_test

def add_rul_to_training_data(df_train):
    """Add RUL calculation to training data - exact implementation from notebook"""
    print("Adding RUL to training data...")
    
    # Calculate maximum life of each engine
    df_train_RUL = df_train.groupby(['engine']).agg({'cycle': 'max'})
    df_train_RUL.rename(columns={'cycle': 'life'}, inplace=True)
    
    # Merge and calculate RUL
    df_train = df_train.merge(df_train_RUL, how='left', on=['engine'])
    df_train['RUL'] = df_train['life'] - df_train['cycle']
    df_train.drop(['life'], axis=1, inplace=True)
    
    # Apply upper limit of 125 - this is the "sneaky" part mentioned in the notebook
    df_train.loc[df_train['RUL'] > 125, 'RUL'] = 125
    
    return df_train

def backward_regression(X, y, threshold_out=0.05, verbose=True):
    """
    Backward Stepwise Regression for feature selection - exact implementation from notebook
    Code from: https://www.kaggle.com/code/adibouayjan/house-price-step-by-step-modeling
    """
    print("Performing backward regression for feature selection...")
    
    included = list(X.columns)
    while True:
        changed = False
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # null if pvalues is empty
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print(f"Removing feature: {worst_feature}, p-value: {worst_pval}")
        if not changed:
            break
    
    print(f"\nSelected Features: {included}")
    return included

def create_sequences(data, feature_names, sequence_length=30):
    """Create proper sequences for LSTM training"""
    print(f"Creating sequences with length {sequence_length}...")
    
    sequences = []
    targets = []
    
    for engine_id in data['engine'].unique():
        engine_data = data[data['engine'] == engine_id].sort_values('cycle')
        
        if len(engine_data) < sequence_length:
            continue
            
        engine_features = engine_data[feature_names].values
        engine_rul = engine_data['RUL'].values
        
        # Create sliding window sequences
        for i in range(sequence_length, len(engine_features)):
            sequences.append(engine_features[i-sequence_length:i])
            targets.append(engine_rul[i])
    
    return np.array(sequences), np.array(targets)

def prepare_test_data(df_test, df_test_RUL, feature_names, scaler, sequence_length=30):
    """Prepare test data with proper sequences"""
    print("Preparing test data with sequences...")
    
    test_sequences = []
    test_targets = []
    
    for i, engine_id in enumerate(df_test['engine'].unique()):
        engine_data = df_test[df_test['engine'] == engine_id].sort_values('cycle')
        
        if len(engine_data) >= sequence_length:
            # Take the last sequence_length cycles for prediction
            engine_features = engine_data[feature_names].values[-sequence_length:]
            engine_features_scaled = scaler.transform(engine_features)
            
            test_sequences.append(engine_features_scaled)
            test_targets.append(df_test_RUL.iloc[i, 0])
    
    return np.array(test_sequences), np.array(test_targets)

def build_lstm_model(input_shape):
    """Build LSTM model with corrected architecture"""
    print("Building corrected LSTM model...")
    
    model = Sequential()
    model.add(LSTM(100, 
                   return_sequences=True,
                   input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LSTM(50,
                   return_sequences=True,
                   activation='tanh'))
    model.add(Dropout(0.3))
    model.add(LSTM(25,
                   return_sequences=False,  # Final LSTM should not return sequences
                   activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(50,
                    activation='relu'))
    model.add(Dense(1))
    
    # Compile model
    model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['mae'])
    
    return model

def time_based_split(sequences, targets, train_ratio=0.7, val_ratio=0.15):
    """Time-based split to avoid data leakage"""
    print("Performing time-based train/validation/test split...")
    
    total_samples = len(sequences)
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))
    
    X_train = sequences[:train_end]
    y_train = targets[:train_end]
    
    X_val = sequences[train_end:val_end]
    y_val = targets[train_end:val_end]
    
    X_test_seq = sequences[val_end:]
    y_test_seq = targets[val_end:]
    
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test_seq)}")
    
    return X_train, X_val, X_test_seq, y_train, y_val, y_test_seq

def comprehensive_evaluation(y_true, y_pred, model_name="LSTM"):
    """Comprehensive evaluation with realistic metrics"""
    print(f"\n{model_name} Model Evaluation:")
    print("=" * 40)
    
    # Basic regression metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # RUL-specific metrics
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    
    # Prognostic Accuracy (predictions within ±20% of actual)
    relative_error = np.abs(y_pred - y_true) / np.maximum(y_true, 1)
    prognostic_accuracy_20 = np.mean(relative_error <= 0.20) * 100
    prognostic_accuracy_10 = np.mean(relative_error <= 0.10) * 100
    
    # Early/Late prediction analysis
    early_predictions = np.sum(y_pred > y_true)
    late_predictions = np.sum(y_pred < y_true)
    
    print(f'RMSE: {rmse:.2f} cycles')
    print(f'MAE: {mae:.2f} cycles')
    print(f'R²: {r2:.4f}')
    print(f'MAPE: {mape:.2f}%')
    print(f'Prognostic Accuracy (±20%): {prognostic_accuracy_20:.1f}%')
    print(f'Prognostic Accuracy (±10%): {prognostic_accuracy_10:.1f}%')
    print(f'Early predictions: {early_predictions}/{len(y_true)} ({early_predictions/len(y_true)*100:.1f}%)')
    print(f'Late predictions: {late_predictions}/{len(y_true)} ({late_predictions/len(y_true)*100:.1f}%)')
    
    return {
        'rmse': rmse,
        'mae': mae, 
        'r2': r2,
        'mape': mape,
        'prog_acc_20': prognostic_accuracy_20,
        'prog_acc_10': prognostic_accuracy_10,
        'y_pred': y_pred
    }

def plot_results(y_test, y_predictions, r2, rmse):
    """Plot results exactly as in the notebook"""
    plt.style.use('seaborn-v0_8-white')
    plt.rcParams['figure.figsize'] = 15, 5
    plt.rcParams['font.family'] = 'Calibri'
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 16
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Actual vs Predicted
    ax1.set_title('Actual vs Predicted')
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    sns.scatterplot(x=y_test, y=y_predictions, s=100, alpha=0.6, 
                   linewidth=1, edgecolor='black', ax=ax1)
    sns.lineplot(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)], 
                linewidth=4, color='gray', ax=ax1)
    
    ax1.annotate(text=(f'R-squared: {r2:.2%}\n' +
                      f'RMSE: {rmse:.2f}'),
                xy=(0, 150), size='medium')
    
    # Predictions by engine
    ax2.set_ylabel('RUL')
    ax2.set_xlabel('Engine nr')
    engine_range = np.arange(0, len(y_test))
    sns.lineplot(x=engine_range, y=y_test, color='gray', label='actual', ax=ax2)
    sns.lineplot(x=engine_range, y=y_predictions, color='steelblue', label='predictions', ax=ax2)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('carl_kirstein_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main execution with corrected methodology"""
    print("NASA Turbofan Engine RUL Prediction - CORRECTED Implementation")
    print("=" * 70)
    
    # Load data
    df_train, df_test, df_test_RUL = load_data()
    
    # Remove constant features
    df_train, df_test = remove_constant_features(df_train, df_test)
    
    # Remove highly correlated features
    df_train, df_test = remove_highly_correlated_features(df_train, df_test)
    
    # Add RUL to training data
    df_train = add_rul_to_training_data(df_train)
    
    print(f"Features after preprocessing: {list(df_train.columns)}")
    
    # Feature selection using backward regression
    X = df_train.iloc[:, 1:-1]  # All features except engine and RUL
    y = df_train.iloc[:, -1]    # RUL
    selected_features = backward_regression(X, y)
    
    # Scale features using MinMaxScaler
    print("Scaling features...")
    sc = MinMaxScaler()
    df_train[selected_features] = sc.fit_transform(df_train[selected_features])
    df_test[selected_features] = sc.transform(df_test[selected_features])
    
    # Create proper sequences for LSTM
    sequence_length = 30
    X_sequences, y_sequences = create_sequences(df_train, selected_features, sequence_length)
    
    # Time-based split to avoid data leakage
    X_train, X_val, X_test_seq, y_train, y_val, y_test_seq = time_based_split(X_sequences, y_sequences)
    
    # Also prepare original test set
    X_test_original, y_test_original = prepare_test_data(df_test, df_test_RUL, selected_features, sc, sequence_length)
    
    print(f"Sequence training data shape: {X_train.shape}")
    print(f"Original test data shape: {X_test_original.shape}")
    
    # Build and train LSTM model
    model = build_lstm_model((sequence_length, len(selected_features)))
    model.summary()
    
    # Setup callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    
    print("Training corrected LSTM model...")
    start_time = time.time()
    history = model.fit(x=X_train, y=y_train,
                       validation_data=(X_val, y_val),
                       epochs=50,
                       batch_size=64,
                       callbacks=[early_stop, reduce_lr],
                       verbose=1)
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on both test sets
    print("\n" + "="*50)
    print("EVALUATION ON TIME-BASED TEST SPLIT:")
    y_pred_seq = model.predict(X_test_seq).flatten()
    metrics_seq = comprehensive_evaluation(y_test_seq, y_pred_seq, "LSTM (Time-based)")
    
    print("\n" + "="*50)
    print("EVALUATION ON ORIGINAL TEST SET:")
    y_pred_orig = model.predict(X_test_original).flatten()
    metrics_orig = comprehensive_evaluation(y_test_original, y_pred_orig, "LSTM (Original)")
    
    # Create comprehensive plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Time-based test results
    axes[0, 0].scatter(y_test_seq, y_pred_seq, alpha=0.6, s=50)
    axes[0, 0].plot([y_test_seq.min(), y_test_seq.max()], [y_test_seq.min(), y_test_seq.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual RUL')
    axes[0, 0].set_ylabel('Predicted RUL')
    axes[0, 0].set_title(f'Time-based Split (R² = {metrics_seq["r2"]:.3f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Original test results
    axes[0, 1].scatter(y_test_original, y_pred_orig, alpha=0.6, s=50)
    axes[0, 1].plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual RUL')
    axes[0, 1].set_ylabel('Predicted RUL')
    axes[0, 1].set_title(f'Original Test Set (R² = {metrics_orig["r2"]:.3f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training history - Loss
    axes[1, 0].plot(history.history['loss'], label='Training Loss')
    axes[1, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[1, 0].set_title('Model Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Training history - MAE
    axes[1, 1].plot(history.history['mae'], label='Training MAE')
    axes[1, 1].plot(history.history['val_mae'], label='Validation MAE')
    axes[1, 1].set_title('Model MAE')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('corrected_lstm_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save model
    model.save('corrected_lstm_model.h5')
    print("Model saved as 'corrected_lstm_model.h5'")
    
    return model, history, metrics_seq, metrics_orig

if __name__ == "__main__":
    model, history, metrics_seq, metrics_orig = main()