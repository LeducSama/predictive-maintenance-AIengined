# Turbofan Engine Predictive Maintenance System

A comprehensive machine learning solution for turbofan engine predictive maintenance with integrated Power BI dashboards. This system implements a multi-signal approach with business-aligned evaluation metrics to optimize maintenance costs and prevent catastrophic failures.

## Business Objectives

- **Minimize Total Cost**: Optimize planned maintenance costs + emergency failure costs
- **Prevent Catastrophic Failures**: Early detection of engine degradation patterns
- **Maximize Fleet Availability**: Strategic maintenance scheduling based on RUL predictions
- **Enable Data-Driven Decisions**: Real-time insights for different stakeholder groups

## System Architecture

### Multi-Signal Approach
- **Signal A: Temperature Trend Analysis** - Long-term RUL prediction (100+ cycle sequences)
- **Signal B: Vibration Change Detection** - Immediate failure warning (10-cycle sequences)
- **Combined Risk Assessment** - Integrated decision support system

### Key Technical Innovations

#### 1. Business-First Feature Engineering
- **25% Baseline Normalization**: Account for different operating environments
- **Engine-Specific Baselines**: Healthy state established from early operational cycles
- **Formula**: `Normalized_reading = Current_reading - Healthy_baseline_for_this_engine`

#### 2. Custom Business Penalty System
- **Catastrophic (10x penalty)**: Late predictions (failure before predicted)
- **Moderate (3x penalty)**: Overly pessimistic predictions (resource waste)
- **Minimal (1x penalty)**: Predictions within ±5 cycles (optimal)

#### 3. Dual LSTM Architecture
- **Temperature Model**: Deep LSTM for long-term patterns
- **Vibration Model**: Lightweight LSTM for change detection
- **Combined Predictor**: Multi-signal risk assessment

## Power BI Integration

### Four Specialized Dashboards

1. **Executive Overview**
   - Fleet health summary and KPIs
   - Cost impact metrics and ROI
   - Strategic maintenance insights

2. **Operations Dashboard**
   - Real-time engine status grid
   - Immediate alert management
   - Priority-based action lists

3. **Maintenance Planning**
   - Maintenance timeline and scheduling
   - Parts inventory coordination
   - Resource allocation optimization

4. **Technical Performance**
   - Model accuracy monitoring
   - Prediction vs actual analysis
   - System performance metrics

## Quick Start

### Prerequisites
```bash
Python 3.8+
TensorFlow 2.13+
Power BI Desktop
8GB+ RAM recommended
```

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd "Predictive Maintenance Dashboard for Turbofan Engines"

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
python -c "import os; [os.makedirs(d, exist_ok=True) for d in ['models', 'logs', 'results', 'plots']]"
```

### Training the Models
```bash
# Train on single dataset
python train_models.py --dataset FD001

# Train on all datasets with comparison
python train_models.py --dataset all --compare

# Custom training parameters
python train_models.py --dataset FD002 --temp-epochs 150 --vib-epochs 75
```

### Starting the API Server
```bash
# Start the Power BI integration API
python src/api/powerbi_api.py

# API will be available at http://localhost:5000
```

### Power BI Setup
1. Follow the [Power BI Integration Guide](POWER_BI_INTEGRATION_GUIDE.md)
2. Configure data sources pointing to `http://localhost:5000/api`
3. Import dashboard templates
4. Set up real-time data refresh

## Project Structure

```
Predictive Maintenance Dashboard for Turbofan Engines/
├── data/
│   ├── raw/                    # NASA turbofan dataset files
│   └── processed/              # Preprocessed training data
├── src/
│   ├── data_processing/
│   │   └── data_loader.py      # 25% baseline normalization
│   ├── models/
│   │   └── dual_lstm_models.py # Temperature + Vibration models
│   ├── evaluation/
│   │   └── business_metrics.py # Custom penalty system
│   └── api/
│       └── powerbi_api.py      # Power BI integration endpoints
├── config/
│   └── powerbi_config.json     # Dashboard configuration
├── models/                     # Trained model files
├── logs/                       # Training and API logs
├── results/                    # Evaluation reports and metrics
├── plots/                      # Visualization outputs
├── notebooks/                  # Jupyter analysis notebooks
├── train_models.py             # Main training script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── POWER_BI_INTEGRATION_GUIDE.md  # Power BI setup guide
```

## API Endpoints

### Core Endpoints
- `GET /api/health` - System health check
- `GET /api/engines/status` - Fleet status summary
- `POST /api/engines/{id}/predict` - Engine-specific predictions
- `GET /api/alerts` - Current system alerts
- `GET /api/maintenance/schedule` - Maintenance planning data
- `GET /api/analytics/performance` - Model performance metrics

### Real-time Features
- **Streaming Data**: Real-time engine status updates
- **Alert Management**: Immediate notification system
- **Mobile Support**: Field technician access

## Model Performance & Key Design Decisions

### Why Two Separate Models?
Rather than building one complex model, I chose a **dual-LSTM architecture** because:
- **Different time horizons**: Temperature trends emerge over 100+ cycles, while vibration changes are immediate
- **Different objectives**: RUL prediction needs accuracy, while fault detection needs high recall
- **Computational efficiency**: Specialized models are faster and more interpretable

### Temperature RUL Model Results (FD001 Dataset)
```
RMSE: 4.28 cycles
MAE: 3.34 cycles
Training sequences: 10,731 samples of 100 cycles each
Features: 6 normalized temperature sensors
```

**Why these results matter**: Predicting RUL within ±3.3 cycles allows maintenance teams to schedule work 2-3 weeks in advance, optimizing parts inventory and labor allocation.

### Vibration Change Detection Results (FD001 Dataset)
```
Accuracy: 97.6%
Precision: 75.3%
Recall: 98.7%
F1-Score: 85.4%
Training sequences: 19,731 samples of 10 cycles each
```

**Design choice explained**: I optimized for **high recall (98.7%)** rather than precision. Missing a real failure (false negative) costs 10x more than a false alarm, so I accept more false positives to catch every real issue.

### Critical Engineering Decisions

#### 1. 25% Baseline Normalization
```python
normalized_value = current_reading - healthy_baseline_25th_percentile
```
**Why this works**: Each engine has different "normal" operating ranges. Using the 25th percentile of early cycles as baseline removes engine-specific bias while preserving degradation signals.

#### 2. Sequence Length Selection
- **Temperature**: 100 cycles (why: degradation patterns need long context)
- **Vibration**: 10 cycles (why: sudden changes are immediate, longer sequences add noise)

#### 3. Business-Aligned Loss Function
Instead of pure RMSE, I minimize:
```
Cost = Σ(prediction_error × business_penalty_multiplier)
```
Where catastrophic failures (late predictions) have 10x penalty vs. conservative predictions (3x penalty).

## Success Criteria

✅ **Models predict RUL trends with business-optimized penalties**  
✅ **Alert system catches vibration changes within 10 cycles**  
✅ **Power BI dashboards provide actionable insights for all stakeholders**  
✅ **Real-time integration delivers alerts within 15 minutes**  
✅ **Executive dashboard clearly shows ROI and cost savings**  
✅ **Mobile access enables immediate field technician response**  
✅ **System demonstrates cost optimization over pure accuracy**  

## Development

### Adding New Features
1. **New Sensor Types**: Extend `data_loader.py` sensor categories
2. **Additional Models**: Add models to `dual_lstm_models.py`
3. **Custom Metrics**: Extend `business_metrics.py` evaluation
4. **API Endpoints**: Add endpoints to `powerbi_api.py`

### Testing
```bash
# Run unit tests
python -m pytest tests/

# Validate API endpoints
python -m pytest tests/test_api.py

# Check model performance
python -m pytest tests/test_models.py
```

### Deployment
```bash
# Production API server
gunicorn --bind 0.0.0.0:5000 src.api.powerbi_api:app

# Docker deployment
docker build -t turbofan-maintenance .
docker run -p 5000:5000 turbofan-maintenance
```

## Data Sources

### NASA Turbofan Engine Degradation Dataset
- **4 Sub-datasets**: FD001, FD002, FD003, FD004
- **Operating Conditions**: 1-6 different conditions
- **Fault Modes**: HPC degradation, Fan degradation
- **Sensors**: 21 sensor measurements + 3 operational settings
- **Format**: Multivariate time series, run-to-failure

### Data Characteristics
- **Training Engines**: 100-260 per dataset
- **Test Engines**: 100-259 per dataset  
- **Sensor Measurements**: Temperature, pressure, vibration, flow rates
- **Operational Settings**: Altitude, throttle, bypass ratio

## Monitoring and Maintenance

### System Monitoring
- **API Performance**: Response time, error rates
- **Model Drift**: Prediction accuracy over time
- **Data Quality**: Missing values, sensor anomalies
- **Business Metrics**: ROI, cost savings, maintenance efficiency

### Regular Maintenance Tasks
- **Model Retraining**: Monthly with new operational data
- **Threshold Tuning**: Quarterly review of alert thresholds
- **Dashboard Updates**: Bi-annual dashboard optimization
- **Performance Review**: Annual system performance evaluation

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation for API changes
- Test Power BI integration before submitting

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **NASA**: For providing the turbofan engine degradation dataset
- **Business Domain Experts**: For guidance on cost optimization strategies
- **Power BI Team**: For integration support and best practices
- **Open Source Community**: For the excellent ML and visualization libraries

## Support

For technical support:
- **API Issues**: Check logs in `logs/` directory
- **Model Performance**: Review evaluation reports in `results/`
- **Power BI Integration**: See [Power BI Integration Guide](POWER_BI_INTEGRATION_GUIDE.md)
- **General Questions**: Open an issue in the repository

---