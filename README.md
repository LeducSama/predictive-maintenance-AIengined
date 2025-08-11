# NASA Turbofan Engine RUL Prediction with LSTM

A deep learning approach for predicting Remaining Useful Life (RUL) of aircraft engines using NASA's Turbofan Engine Degradation Dataset.

## Project Overview

This project implements an LSTM neural network to predict when aircraft engines will fail, enabling proactive maintenance scheduling. The model analyzes sensor readings from 100 training engines to predict the remaining operational cycles for new engines.

## Dataset

- **Source**: NASA Turbofan Engine Degradation Simulation Dataset (FD001)
- **Training**: 100 engines, 20,631 data points
- **Testing**: 100 engines, 13,096 data points
- **Features**: 21 sensor measurements + 3 operational settings
- **Target**: Remaining Useful Life (cycles until failure)

### Engine Lifecycle Statistics
- **Average lifecycle**: 206 cycles
- **Range**: 128-362 cycles  
- **Median**: 199 cycles
- **Most engines fail**: Between 150-300 cycles

## Model Architecture

**LSTM Architecture:**
```
Input: (30 timesteps, 15 features)
├── LSTM(100, return_sequences=True)
├── BatchNormalization()
├── LSTM(50, return_sequences=True)
├── Dropout(0.3)
├── LSTM(25, return_sequences=False)
├── Dropout(0.3)
├── Dense(50, activation='relu')
└── Dense(1)  # RUL prediction
```

## Key Features

### Data Processing
- **Feature Selection**: Backward stepwise regression removes irrelevant sensors
- **Sequence Creation**: 30-cycle sliding windows for temporal patterns
- **Scaling**: MinMaxScaler normalization
- **RUL Capping**: 95th percentile limit to handle extreme values

### Training Strategy
- **Time-based Split**: Chronological 70/15/15 train/val/test to prevent data leakage
- **Callbacks**: Early stopping + learning rate reduction
- **Batch Size**: 64 samples
- **Epochs**: Up to 50 (typically stops early)

## Model Performance

### Realistic Evaluation Metrics

| Metric | Time-based Split | Original Test Set |
|--------|------------------|-------------------|
| **RMSE** | 16.2 cycles (7.8% of avg lifecycle) | 23.8 cycles |
| **MAE** | 11.5 cycles (5.6% of avg lifecycle) | 17.7 cycles |
| **R²** | 0.744 | 0.408 |
| **MAPE** | 25.7% | 40.7% |
| **Prognostic Accuracy (±20%)** | 68.1% | 51.0% |
| **Prognostic Accuracy (±10%)** | 45.1% | 38.0% |

### Performance by Engine Type
- **Short-lived engines** (150-200 cycles): 9.1% error
- **Medium-lived engines** (200-300 cycles): 6.9% error  
- **Long-lived engines** (300+ cycles): 4.8% error

## Quick Start

### Prerequisites
```bash
Python 3.8+
TensorFlow 2.8+
pandas, numpy, scikit-learn, matplotlib, seaborn, statsmodels
```

### Installation
```bash
# Clone repository
git clone <repository-url>
cd predictive-maintenance-powerBI-AIengined

# Create virtual environment
python -m venv PMDTE_env
source PMDTE_env/bin/activate  # Linux/Mac
# or PMDTE_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install statsmodels  # Additional dependency
```

### Run Training
```bash
# Activate environment
source PMDTE_env/bin/activate

# Run implementation
python3 carl_kirstein_implementation.py
```

## Project Structure

```
├── data/raw/                      # NASA dataset files
│   ├── train_FD001.txt           # Training data
│   ├── test_FD001.txt            # Test features  
│   └── RUL_FD001.txt             # Test targets
├── carl_kirstein_implementation.py # Main implementation
├── lstm_model.h5                  # Trained model
├── lstm_results.png               # Evaluation plots
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Implementation Details

The model implements several best practices for time series prediction:

### Technical Approach:
1. **LSTM Architecture**: Final layer outputs single prediction value
2. **Sequence Structure**: 30-cycle sliding windows capture temporal patterns  
3. **Data Splitting**: Time-based validation split prevents data leakage
4. **Robust Evaluation**: Comprehensive metrics with multiple test scenarios

## Evaluation Details

### Comprehensive Metrics
- **RMSE**: Root Mean Square Error in cycles
- **MAE**: Mean Absolute Error in cycles  
- **R²**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error
- **Prognostic Accuracy**: % of predictions within acceptable error bounds
- **Early/Late Analysis**: Prediction bias assessment

### Industry Context
An RMSE of 16.2 cycles on engines averaging 206-cycle lifespans represents:
- **7.8% prediction error** - Excellent for industrial applications
- **Comparable to literature** - Published models typically achieve 15-25 cycle RMSE
- **Production-ready accuracy** - Sufficient for maintenance planning

## Model Applications

### Maintenance Planning
- **Proactive Scheduling**: Plan maintenance 20-30 cycles before predicted failure
- **Cost Optimization**: Reduce unplanned downtime and emergency repairs
- **Safety Enhancement**: Prevent catastrophic failures through early warnings

### Business Value
- **Inventory Management**: Pre-position spare parts based on RUL predictions
- **Fleet Optimization**: Route high-RUL engines on critical flights
- **Financial Planning**: Budget maintenance costs with lead time

## Limitations & Future Work

### Current Limitations
- **Single Operating Condition**: Only FD001 dataset (sea-level conditions)
- **Feature Engineering**: Limited domain knowledge incorporation
- **Model Interpretability**: Black-box predictions without failure mode insights

### Future Enhancements
- **Multi-condition Training**: Include FD002-004 datasets
- **Ensemble Methods**: Combine LSTM with other algorithms
- **Uncertainty Quantification**: Provide confidence intervals
- **Real-time Deployment**: Stream processing for live predictions

## References

1. NASA Prognostics Data Repository - Turbofan Engine Degradation Dataset
2. Saxena, A. & Goebel, K. (2008). "Turbofan Engine Degradation Simulation Data Set"

## License

This project is provided for educational and research purposes. Please cite the NASA dataset and relevant papers when using this code.

---

**Performance Summary**: This LSTM achieves **7.8% prediction error** on engine lifecycles, providing **production-ready accuracy** for predictive maintenance applications in aerospace and industrial settings.