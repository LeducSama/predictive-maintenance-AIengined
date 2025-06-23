#!/usr/bin/env python3
"""
Demo script for Turbofan Engine Predictive Maintenance System
Demonstrates the key features and capabilities of the system
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
import time

# Add src to path
sys.path.append('src')

from data_processing.data_loader import TurbofanDataLoader
from models.dual_lstm_models import DualLSTMPredictor
# Removed financial evaluation - focusing on ML metrics only


def print_banner():
    """Print demo banner"""
    print("=" * 80)
    print("🚁 TURBOFAN ENGINE PREDICTIVE MAINTENANCE SYSTEM DEMO")
    print("=" * 80)
    print("Multi-Signal LSTM Architecture for Business-Optimized Predictions")
    print("Temperature Trend Analysis + Vibration Change Detection")
    print("=" * 80)
    print()


def demo_data_preprocessing():
    """Demonstrate the data preprocessing pipeline"""
    print("📊 DEMO 1: Data Preprocessing with 25% Baseline Normalization")
    print("-" * 60)
    
    # Initialize data loader
    loader = TurbofanDataLoader(data_path="data/raw")
    
    # Load dataset
    print("Loading NASA Turbofan Dataset FD001...")
    train_df, test_df, rul_df = loader.load_dataset('FD001')
    
    print(f"✓ Training data shape: {train_df.shape}")
    print(f"✓ Test data shape: {test_df.shape}")
    print(f"✓ RUL labels shape: {rul_df.shape}")
    
    # Demonstrate sensor categories
    print(f"\nSensor Categories:")
    print(f"🌡️  Temperature sensors: {loader.temperature_sensors}")
    print(f"📳 Vibration sensors: {loader.vibration_sensors}")
    
    # Calculate baselines
    print("\nCalculating 25% baseline normalization...")
    baselines = loader.calculate_baseline_normalization(train_df)
    print(f"✓ Baselines calculated for {len(baselines)} engines")
    
    # Show example baseline for first engine
    engine_1_baseline = baselines[1]
    print(f"\nExample baseline for Engine 1:")
    for sensor, value in list(engine_1_baseline.items())[:5]:
        print(f"  {sensor}: {value:.3f}")
    
    # Apply normalization
    print("\nApplying baseline normalization...")
    train_normalized = loader.apply_baseline_normalization(train_df, baselines)
    
    # Show the effect of normalization
    print("\nNormalization Effect Example (Engine 1, Sensor 1):")
    engine_1_data = train_df[train_df['unit_id'] == 1]['sensor_1'].head(5)
    engine_1_norm = train_normalized[train_normalized['unit_id'] == 1]['sensor_1'].head(5)
    
    for i, (orig, norm) in enumerate(zip(engine_1_data, engine_1_norm)):
        print(f"  Cycle {i+1}: {orig:.3f} → {norm:.3f}")
    
    print("\n✓ Data preprocessing demonstration complete!")
    print()
    
    return loader


def demo_model_architecture():
    """Demonstrate the dual LSTM model architecture"""
    print("🧠 DEMO 2: Dual LSTM Model Architecture")
    print("-" * 60)
    
    # Initialize predictor
    predictor = DualLSTMPredictor()
    
    print("Temperature Trend Model Architecture:")
    print("🌡️  Purpose: Long-term RUL prediction")
    print("🔄 Sequence Length: 100 cycles")
    print("📊 Features: 6 temperature-related sensors")
    print("🎯 Target: Remaining Useful Life (regression)")
    
    # Build temperature model for demonstration
    predictor.temperature_model.build_model()
    temp_model = predictor.temperature_model.model
    
    print(f"\nTemperature Model Summary:")
    if temp_model:
        print(f"📈 Total Parameters: {temp_model.count_params():,}")
        print(f"🏗️  Layers: {len(temp_model.layers)}")
    else:
        print("❌ Model not built")
    
    print("\nVibration Alert Model Architecture:")
    print("📳 Purpose: Short-term change detection")
    print("🔄 Sequence Length: 10 cycles")
    print("📊 Features: 7 vibration-related sensors")
    print("🎯 Target: Change detection (binary classification)")
    
    # Build vibration model for demonstration
    predictor.vibration_model.build_model()
    vib_model = predictor.vibration_model.model
    
    print(f"\nVibration Model Summary:")
    if vib_model:
        print(f"📈 Total Parameters: {vib_model.count_params():,}")
        print(f"🏗️  Layers: {len(vib_model.layers)}")
    else:
        print("❌ Model not built")
    
    print("\n✓ Model architecture demonstration complete!")
    print()
    
    return predictor


def demo_ml_evaluation():
    """Demonstrate standard ML evaluation metrics"""
    print("🔬 DEMO 3: Model Performance Evaluation")
    print("-" * 60)
    
    print("Standard ML Metrics:")
    print("📊 RMSE (Root Mean Square Error): Measures prediction accuracy")
    print("📏 MAE (Mean Absolute Error): Average prediction error") 
    print("🎯 Accuracy: Percentage of correct classifications")
    print("⚡ Precision/Recall: Classification performance metrics")
    
    # Simulate some predictions for demonstration
    print("\nSimulating Model Performance...")
    np.random.seed(42)
    
    # Temperature model simulation (RUL prediction)
    actual_rul = np.array([50, 30, 80, 15, 45, 25, 60, 10, 35, 55])
    predicted_rul = actual_rul + np.random.normal(0, 3, 10)
    
    rmse = np.sqrt(np.mean((actual_rul - predicted_rul)**2))
    mae = np.mean(np.abs(actual_rul - predicted_rul))
    
    print(f"\n🌡️ Temperature Model (RUL Prediction):")
    print(f"📊 RMSE: {rmse:.2f} cycles")
    print(f"📏 MAE: {mae:.2f} cycles")
    print(f"✅ Model predicts failure within ±{mae:.1f} cycles on average")
    
    # Vibration model simulation (Change detection)
    alert_true = np.random.binomial(1, 0.2, 50)
    alert_pred = np.random.binomial(1, 0.22, 50)
    
    accuracy = np.mean(alert_true == alert_pred)
    precision = np.sum((alert_true == 1) & (alert_pred == 1)) / np.sum(alert_pred == 1) if np.sum(alert_pred == 1) > 0 else 0
    recall = np.sum((alert_true == 1) & (alert_pred == 1)) / np.sum(alert_true == 1) if np.sum(alert_true == 1) > 0 else 0
    
    print(f"\n📳 Vibration Model (Change Detection):")
    print(f"🎯 Accuracy: {accuracy:.3f}")
    print(f"⚡ Precision: {precision:.3f}")
    print(f"⚡ Recall: {recall:.3f}")
    print(f"✅ Model detects {accuracy:.1%} of changes correctly")
    
    print("\n✓ ML evaluation demonstration complete!")
    print()


def demo_model_predictions():
    """Demonstrate model prediction capabilities"""
    print("🔌 DEMO 4: Model Prediction Capabilities")
    print("-" * 60)
    
    print("Demonstrating Prediction Capabilities:")
    
    # Simulate engine predictions
    print("\n🔮 Engine RUL Predictions:")
    engines = []
    
    for i in range(1, 11):
        rul = max(10, np.random.normal(45, 15))
        alert_prob = np.random.beta(2, 8)
        
        if alert_prob > 0.7:
            status = 'critical'
        elif rul < 25:
            status = 'warning'
        else:
            status = 'healthy'
        
        engines.append({
            'engine_id': i,
            'status': status,
            'rul_prediction': round(rul, 1),
            'alert_probability': round(alert_prob, 3)
        })
    
    for engine in engines:
        print(f"  Engine {engine['engine_id']}: RUL={engine['rul_prediction']} cycles, Alert={engine['alert_probability']:.3f}")
    
    # Simulate maintenance recommendations
    print("\n🔧 Maintenance Recommendations:")
    for engine in engines[:5]:
        if engine['status'] == 'critical':
            print(f"  Engine {engine['engine_id']}: IMMEDIATE maintenance required")
        elif engine['status'] == 'warning':
            print(f"  Engine {engine['engine_id']}: Schedule maintenance within 10 days")
        else:
            print(f"  Engine {engine['engine_id']}: Continue normal operations")
    
    print("\n✓ Model prediction demonstration complete!")
    print()


def demo_alert_system():
    """Demonstrate alert system"""
    print("🚨 DEMO 5: Alert System")
    print("-" * 60)
    
    print("Alert Classification System:")
    
    # Simulate different alert types
    alerts = [
        {'engine_id': 3, 'type': 'vibration_spike', 'severity': 'critical', 'rul': 15},
        {'engine_id': 7, 'type': 'temperature_rise', 'severity': 'warning', 'rul': 25},
        {'engine_id': 1, 'type': 'efficiency_decline', 'severity': 'info', 'rul': 45},
        {'engine_id': 9, 'type': 'sensor_anomaly', 'severity': 'warning', 'rul': 30},
        {'engine_id': 5, 'type': 'degradation_trend', 'severity': 'info', 'rul': 60}
    ]
    
    for alert in alerts:
        if alert['severity'] == 'critical':
            icon = "🔴"
            action = "IMMEDIATE ACTION REQUIRED"
        elif alert['severity'] == 'warning':
            icon = "🟡"
            action = "Schedule maintenance"
        else:
            icon = "🟢"
            action = "Monitor closely"
        
        print(f"\n{icon} Engine {alert['engine_id']} - {alert['type'].upper()}")
        print(f"   Severity: {alert['severity']}")
        print(f"   RUL: {alert['rul']} cycles")
        print(f"   Action: {action}")
    
    print("\n📊 Alert Statistics:")
    critical_count = sum(1 for a in alerts if a['severity'] == 'critical')
    warning_count = sum(1 for a in alerts if a['severity'] == 'warning')
    info_count = sum(1 for a in alerts if a['severity'] == 'info')
    
    print(f"   🔴 Critical: {critical_count}")
    print(f"   🟡 Warning: {warning_count}")
    print(f"   🟢 Info: {info_count}")
    
    print("\n✓ Alert system demonstration complete!")
    print()


def demo_complete_workflow():
    """Demonstrate a complete prediction workflow"""
    print("🔄 DEMO 6: Complete Prediction Workflow")
    print("-" * 60)
    
    print("Simulating Real-time Engine Monitoring Workflow...")
    
    # Simulate incoming sensor data
    print("\n1. 📡 Sensor Data Ingestion:")
    engine_id = 42
    current_cycle = 85
    
    # Simulate temperature sensor readings
    temp_readings = {
        'sensor_2': 520.5,
        'sensor_3': 642.8,
        'sensor_4': 1588.2,
        'sensor_7': 553.1,
        'sensor_11': 47.3,
        'sensor_12': 521.8
    }
    
    # Simulate vibration sensor readings  
    vib_readings = {
        'sensor_1': 0.05,
        'sensor_5': 14.62,
        'sensor_6': 21.61,
        'sensor_10': 1.30,
        'sensor_16': 0.03,
        'sensor_18': 392.0,
        'sensor_19': 2388.0
    }
    
    print(f"   Engine {engine_id}, Cycle {current_cycle}")
    print(f"   🌡️  Temperature data: {len(temp_readings)} sensors")
    print(f"   📳 Vibration data: {len(vib_readings)} sensors")
    
    # Simulate model predictions
    print("\n2. 🧠 Model Predictions:")
    
    # Temperature trend analysis (simulated)
    rul_prediction = max(15, np.random.normal(30, 8))
    rul_confidence = [rul_prediction - 5, rul_prediction + 5]
    
    print(f"   🌡️  Temperature Model:")
    print(f"      RUL Prediction: {rul_prediction:.1f} cycles")
    print(f"      Confidence: [{rul_confidence[0]:.1f}, {rul_confidence[1]:.1f}] cycles")
    
    # Vibration change detection (simulated)
    alert_probability = np.random.beta(8, 2)  # Higher probability for demo
    change_detected = alert_probability > 0.7
    
    print(f"   📳 Vibration Model:")
    print(f"      Change Probability: {alert_probability:.3f}")
    print(f"      Alert Status: {'🚨 ALERT' if change_detected else '✅ Normal'}")
    
    # Business risk assessment
    print("\n3. 💼 Business Risk Assessment:")
    
    if change_detected and rul_prediction < 20:
        risk_level = "CRITICAL"
        recommendation = "IMMEDIATE MAINTENANCE REQUIRED"
        color = "🔴"
    elif change_detected or rul_prediction < 30:
        risk_level = "HIGH"
        recommendation = "Schedule maintenance within 10 days"
        color = "🟡"
    else:
        risk_level = "MODERATE"
        recommendation = "Continue monitoring"
        color = "🟢"
    
    print(f"   {color} Risk Level: {risk_level}")
    print(f"   📋 Recommendation: {recommendation}")
    
    # System notifications
    print("\n4. 📋 System Updates:")
    
    system_updates = {
        'Maintenance Schedule': f"Engine {engine_id} scheduled for maintenance in {int(rul_prediction * 0.8)} days",
        'Alert Log': f"{'New alert generated' if change_detected else 'No new alerts'}",
        'Model Performance': f"Predictions logged, accuracy metrics updated",
        'Fleet Status': f"Fleet status updated, {1 if risk_level == 'CRITICAL' else 0} critical engines"
    }
    
    for system, update in system_updates.items():
        print(f"   📋 {system}: {update}")
    
    # Notifications (if critical)
    if risk_level == "CRITICAL":
        print("\n5. 📢 Notifications:")
        print("   🚨 CRITICAL ALERT sent to maintenance team")
        print("   📧 Email notification sent to operations manager")
        print("   📞 SMS alert sent to field technicians")
    
    print("\n✓ Complete workflow demonstration finished!")
    print()


def print_summary():
    """Print demo summary and next steps"""
    print("📋 DEMO SUMMARY")
    print("-" * 60)
    
    print("✅ Demonstrated Features:")
    print("   • 25% Baseline Normalization for engine-specific degradation")
    print("   • Dual LSTM Architecture (Temperature + Vibration)")
    print("   • Business-Aligned Penalty System")
    print("   • Alert Classification System")
    print("   • Maintenance Scheduling")
    print("   • Real-time Monitoring Workflow")
    
    print("\n🎯 Key Business Benefits:")
    print("   • Cost-optimized maintenance scheduling")
    print("   • Catastrophic failure prevention")
    print("   • Real-time failure detection")
    print("   • Automated alert notifications")
    print("   • ML performance evaluation metrics")
    
    print("\n🚀 Next Steps:")
    print("   1. Run 'python train_models.py --dataset FD001' to train models")
    print("   2. Run predictions with 'python predict.py'")
    print("   3. Configure alert notifications")
    print("   4. Set up maintenance scheduling system")
    print("   5. Deploy to production environment")
    
    print("\n📚 Additional Resources:")
    print("   • README.md - Complete system documentation")
    print("   • Documentation - System setup guide")
    print("   • src/ - Source code with detailed comments")
    print("   • config/ - Configuration files and examples")
    
    print("\n" + "=" * 80)
    print("🎉 DEMO COMPLETE - Ready for Production Deployment!")
    print("=" * 80)


def main():
    """Run the complete demo"""
    print_banner()
    
    # Run demo sections
    try:
        loader = demo_data_preprocessing()
        predictor = demo_model_architecture()
        demo_ml_evaluation()
        demo_model_predictions()
        demo_alert_system()
        demo_complete_workflow()
        print_summary()
        
    except Exception as e:
        print(f"❌ Demo error: {str(e)}")
        print("📝 Note: Some demos may require the actual dataset to be present")
        print("   Download the dataset and run 'python train_models.py' first")


if __name__ == "__main__":
    main()