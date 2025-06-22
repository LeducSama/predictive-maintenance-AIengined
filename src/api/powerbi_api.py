from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import os
import sys
import logging
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.dual_lstm_models import DualLSTMPredictor
from data_processing.data_loader import TurbofanDataLoader
from evaluation.business_metrics import BusinessPenaltyEvaluator

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models and data
predictor = None
data_loader = None
evaluator = None
current_engine_data = {}
alert_history = []


class PowerBIAPIServer:
    """
    Flask API server for Power BI integration
    Provides real-time endpoints for predictive maintenance data
    """
    
    def __init__(self):
        self.predictor = DualLSTMPredictor()
        self.data_loader = TurbofanDataLoader()
        self.evaluator = BusinessPenaltyEvaluator()
        self.engine_status = {}
        self.alert_queue = []
        
    def initialize_models(self, model_path: str = "models/"):
        """Load trained models"""
        try:
            self.predictor.load_models(model_path)
            logger.info("Models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            return False
    
    def simulate_real_time_data(self, engine_id: int, num_cycles: int = 10) -> Dict[str, Any]:
        """
        Simulate real-time sensor data for demonstration
        In production, this would connect to actual sensor feeds
        """
        # Generate simulated sensor data
        np.random.seed(engine_id)
        
        temperature_data = np.random.normal(500, 50, (num_cycles, 6))
        vibration_data = np.random.normal(0, 1, (num_cycles, 7))
        
        # Add some degradation trend
        degradation_factor = np.linspace(1.0, 1.2, num_cycles).reshape(-1, 1)
        temperature_data = temperature_data * degradation_factor
        
        # Add occasional vibration spikes
        if np.random.random() < 0.1:  # 10% chance of vibration spike
            spike_idx = np.random.randint(0, num_cycles)
            vibration_data[spike_idx] *= 3
        
        return {
            'engine_id': engine_id,
            'timestamp': datetime.now().isoformat(),
            'temperature_sensors': temperature_data.tolist(),
            'vibration_sensors': vibration_data.tolist(),
            'operating_conditions': [100.0, 0.0, 0.0]  # Simulated operating settings
        }

# Initialize the API server
api_server = PowerBIAPIServer()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': api_server.predictor.temperature_model.is_trained
    })

@app.route('/api/engines/status', methods=['GET'])
def get_engine_status():
    """
    Get current status of all engines
    Returns fleet-wide health summary for executive dashboard
    """
    try:
        # Simulate data for demonstration fleet
        engine_count = int(request.args.get('count', 10))
        engines = []
        
        for engine_id in range(1, engine_count + 1):
            # Simulate real-time data
            sensor_data = api_server.simulate_real_time_data(engine_id)
            
            # Mock predictions (in production, use actual model predictions)
            rul_prediction = max(10, np.random.normal(50, 20))
            alert_probability = np.random.beta(2, 8)  # Most engines should be low risk
            
            # Determine status based on predictions
            if alert_probability > 0.8:
                status = 'critical'
                priority = 1
            elif rul_prediction < 20:
                status = 'warning'
                priority = 2
            else:
                status = 'healthy'
                priority = 3
            
            engines.append({
                'engine_id': engine_id,
                'status': status,
                'priority': priority,
                'rul_prediction': round(rul_prediction, 1),
                'alert_probability': round(alert_probability, 3),
                'last_maintenance': (datetime.now() - timedelta(days=np.random.randint(30, 200))).isoformat(),
                'next_maintenance': (datetime.now() + timedelta(days=int(rul_prediction))).isoformat(),
                'location': f"Site_{engine_id % 5 + 1}",
                'model': f"CFM56-{engine_id % 3 + 1}"
            })
        
        # Calculate fleet summary
        critical_count = sum(1 for e in engines if e['status'] == 'critical')
        warning_count = sum(1 for e in engines if e['status'] == 'warning')
        healthy_count = sum(1 for e in engines if e['status'] == 'healthy')
        
        avg_rul = np.mean([e['rul_prediction'] for e in engines])
        
        return jsonify({
            'fleet_summary': {
                'total_engines': engine_count,
                'critical_engines': critical_count,
                'warning_engines': warning_count,
                'healthy_engines': healthy_count,
                'average_rul': round(avg_rul, 1),
                'last_updated': datetime.now().isoformat()
            },
            'engines': engines
        })
        
    except Exception as e:
        logger.error(f"Error in get_engine_status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/engines/<int:engine_id>/predict', methods=['POST'])
def predict_engine_health(engine_id: int):
    """
    Get predictions for a specific engine
    Used for detailed engine analysis
    """
    try:
        # Get sensor data from request
        sensor_data = request.json
        
        if not sensor_data:
            # Use simulated data for demonstration
            sensor_data = api_server.simulate_real_time_data(engine_id)
        
        # Prepare data for model prediction
        temp_sequences = np.array([sensor_data['temperature_sensors']])
        vib_sequences = np.array([sensor_data['vibration_sensors']])
        
        # Make predictions (mock for demonstration)
        rul_prediction = max(10, np.random.normal(50, 20))
        alert_probability = np.random.beta(2, 8)
        
        # Calculate confidence intervals
        rul_confidence_low = max(0, rul_prediction - 10)
        rul_confidence_high = rul_prediction + 10
        
        # Generate recommendations
        recommendations = []
        if alert_probability > 0.7:
            recommendations.append("Immediate inspection recommended")
        if rul_prediction < 30:
            recommendations.append("Schedule maintenance within 30 days")
        if rul_prediction < 15:
            recommendations.append("URGENT: Schedule immediate maintenance")
        
        return jsonify({
            'engine_id': engine_id,
            'timestamp': datetime.now().isoformat(),
            'predictions': {
                'rul_prediction': round(rul_prediction, 1),
                'rul_confidence_interval': [round(rul_confidence_low, 1), round(rul_confidence_high, 1)],
                'alert_probability': round(alert_probability, 3),
                'risk_level': 'high' if alert_probability > 0.7 else 'medium' if alert_probability > 0.4 else 'low'
            },
            'recommendations': recommendations,
            'sensor_readings': {
                'temperature_avg': round(np.mean(sensor_data['temperature_sensors']), 2),
                'vibration_rms': round(np.sqrt(np.mean(np.array(sensor_data['vibration_sensors'])**2)), 3),
                'operating_normal': all(abs(x) < 2 for x in sensor_data['operating_conditions'])
            }
        })
        
    except Exception as e:
        logger.error(f"Error in predict_engine_health: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """
    Get current alerts for operations dashboard
    Real-time alerts for immediate attention
    """
    try:
        # Simulate alerts for demonstration
        alerts = []
        current_time = datetime.now()
        
        # Generate some sample alerts
        for i in range(5):
            alert_time = current_time - timedelta(minutes=np.random.randint(1, 60))
            engine_id = np.random.randint(1, 11)
            
            alert_types = ['vibration_spike', 'temperature_anomaly', 'pressure_drop', 'efficiency_decline']
            alert_type = np.random.choice(alert_types)
            
            severity = np.random.choice(['critical', 'warning', 'info'], p=[0.2, 0.5, 0.3])
            
            alerts.append({
                'alert_id': f'ALT_{i+1:03d}',
                'engine_id': engine_id,
                'alert_type': alert_type,
                'severity': severity,
                'timestamp': alert_time.isoformat(),
                'message': f"Engine {engine_id}: {alert_type.replace('_', ' ').title()} detected",
                'acknowledged': bool(np.random.choice([True, False], p=[0.7, 0.3])),
                'estimated_response_time': np.random.randint(5, 30)
            })
        
        # Sort by severity and time
        severity_order = {'critical': 0, 'warning': 1, 'info': 2}
        alerts.sort(key=lambda x: (severity_order[x['severity']], x['timestamp']), reverse=True)
        
        return jsonify({
            'alerts': alerts,
            'summary': {
                'total_alerts': len(alerts),
                'critical_alerts': sum(1 for a in alerts if a['severity'] == 'critical'),
                'unacknowledged_alerts': sum(1 for a in alerts if not a['acknowledged']),
                'last_updated': current_time.isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_alerts: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/maintenance/schedule', methods=['GET'])
def get_maintenance_schedule():
    """
    Get maintenance schedule based on RUL predictions
    For maintenance planning dashboard
    """
    try:
        # Simulate maintenance schedule
        schedule = []
        current_date = datetime.now()
        
        for i in range(1, 11):  # 10 engines
            engine_id = i
            rul_prediction = max(10, np.random.normal(50, 20))
            
            # Calculate maintenance dates
            next_maintenance = current_date + timedelta(days=int(rul_prediction * 0.8))  # Schedule at 80% of RUL
            
            maintenance_type = np.random.choice(['routine', 'major', 'overhaul'], p=[0.6, 0.3, 0.1])
            
            schedule.append({
                'engine_id': engine_id,
                'maintenance_type': maintenance_type,
                'scheduled_date': next_maintenance.isoformat(),
                'estimated_duration': np.random.randint(4, 24),  # hours
                'rul_prediction': round(rul_prediction, 1),
                'maintenance_window': f"{next_maintenance.strftime('%Y-%m-%d')} to {(next_maintenance + timedelta(days=7)).strftime('%Y-%m-%d')}",
                'parts_required': np.random.choice([True, False], p=[0.8, 0.2]),
                'technician_assigned': f"Tech_{np.random.randint(1, 6)}"
            })
        
        # Sort by date
        schedule.sort(key=lambda x: x['scheduled_date'])
        
        return jsonify({
            'maintenance_schedule': schedule,
            'summary': {
                'total_scheduled': len(schedule),
                'this_week': sum(1 for s in schedule if datetime.fromisoformat(s['scheduled_date'].replace('Z', '+00:00')) < current_date + timedelta(days=7)),
                'this_month': sum(1 for s in schedule if datetime.fromisoformat(s['scheduled_date'].replace('Z', '+00:00')) < current_date + timedelta(days=30)),
                'parts_needed': sum(1 for s in schedule if s['parts_required'])
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_maintenance_schedule: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/performance', methods=['GET'])
def get_model_performance():
    """
    Get model performance metrics for technical dashboard
    """
    try:
        # Simulate model performance data
        performance_data = {
            'temperature_model': {
                'accuracy_metrics': {
                    'mae': round(np.random.uniform(8, 12), 2),
                    'rmse': round(np.random.uniform(10, 15), 2),
                    'business_penalty': round(np.random.uniform(150, 250), 2)
                },
                'prediction_distribution': {
                    'catastrophic_rate': round(np.random.uniform(0.05, 0.15), 3),
                    'optimal_rate': round(np.random.uniform(0.6, 0.8), 3),
                    'moderate_rate': round(np.random.uniform(0.15, 0.25), 3)
                },
                'last_updated': datetime.now().isoformat()
            },
            'vibration_model': {
                'classification_metrics': {
                    'accuracy': round(np.random.uniform(0.85, 0.95), 3),
                    'precision': round(np.random.uniform(0.80, 0.90), 3),
                    'recall': round(np.random.uniform(0.75, 0.85), 3),
                    'f1_score': round(np.random.uniform(0.78, 0.88), 3)
                },
                'business_impact': {
                    'alerts_generated': np.random.randint(50, 100),
                    'true_positives': np.random.randint(35, 60),
                    'false_positives': np.random.randint(5, 15),
                    'predicted_maintenance_hours': round(np.random.uniform(20, 80), 1)
                },
                'last_updated': datetime.now().isoformat()
            },
            'combined_system': {
                'performance_score': round(np.random.uniform(0.75, 0.95), 3),
                'total_maintenance_hours_saved': round(np.random.uniform(1000, 2000), 1),
                'maintenance_efficiency': round(np.random.uniform(0.85, 0.95), 3),
                'uptime_improvement': round(np.random.uniform(0.02, 0.08), 3)
            }
        }
        
        return jsonify(performance_data)
        
    except Exception as e:
        logger.error(f"Error in get_model_performance: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/engines/<int:engine_id>/history', methods=['GET'])
def get_engine_history(engine_id: int):
    """
    Get historical data for a specific engine
    Used for trend analysis and detailed diagnostics
    """
    try:
        days = int(request.args.get('days', 30))
        
        # Generate historical data
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        history = []
        for date in dates:
            # Simulate degradation over time
            degradation_factor = 1 + (days - len(history)) * 0.01
            
            rul = max(5, 60 - len(history) * 2 + np.random.normal(0, 5))
            alert_prob = min(0.9, len(history) * 0.02 + np.random.beta(1, 9))
            
            history.append({
                'date': date.isoformat(),
                'rul_prediction': round(rul, 1),
                'alert_probability': round(alert_prob, 3),
                'temperature_avg': round(500 * degradation_factor + np.random.normal(0, 10), 2),
                'vibration_rms': round(0.5 * degradation_factor + np.random.normal(0, 0.1), 3),
                'efficiency': round(0.95 / degradation_factor + np.random.normal(0, 0.02), 3)
            })
        
        return jsonify({
            'engine_id': engine_id,
            'history': history,
            'trends': {
                'rul_trend': 'declining',
                'alert_trend': 'increasing',
                'temperature_trend': 'increasing',
                'efficiency_trend': 'declining'
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_engine_history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/<alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id: str):
    """
    Acknowledge an alert
    Used by operations dashboard for alert management
    """
    try:
        user_id = (request.json or {}).get('user_id', 'unknown')
        
        # In production, this would update the alert in the database
        response = {
            'alert_id': alert_id,
            'acknowledged': True,
            'acknowledged_by': user_id,
            'acknowledged_at': datetime.now().isoformat(),
            'status': 'success'
        }
        
        logger.info(f"Alert {alert_id} acknowledged by {user_id}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error acknowledging alert {alert_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/maintenance/schedule', methods=['GET'])
def get_maintenance_schedule():
    """
    Get maintenance schedule optimized for Power BI
    Includes upcoming maintenance, parts requirements, and technician assignments
    """
    try:
        # Simulate maintenance schedule data
        schedule = []
        current_time = datetime.now()
        
        for i in range(15):  # 15 maintenance items
            scheduled_date = current_time + timedelta(days=np.random.randint(1, 90))
            engine_id = np.random.randint(1, 11)
            
            maintenance_types = ['routine', 'preventive', 'corrective', 'overhaul']
            maintenance_type = np.random.choice(maintenance_types)
            
            duration_hours = {
                'routine': np.random.randint(2, 6),
                'preventive': np.random.randint(8, 16), 
                'corrective': np.random.randint(4, 12),
                'overhaul': np.random.randint(24, 48)
            }[maintenance_type]
            
            schedule.append({
                'maintenance_id': f'MNT_{i+1:03d}',
                'engine_id': engine_id,
                'maintenance_type': maintenance_type,
                'scheduled_date': scheduled_date.isoformat(),
                'estimated_duration': duration_hours,
                'parts_required': bool(np.random.choice([True, False], p=[0.6, 0.4])),
                'technician_assigned': f'Tech_{np.random.randint(1, 8)}',
                'priority': np.random.choice(['high', 'medium', 'low'], p=[0.2, 0.5, 0.3]),
                'estimated_hours': np.random.randint(8, 48),
                'status': np.random.choice(['scheduled', 'in_progress', 'completed'], p=[0.7, 0.2, 0.1])
            })
        
        # Sort by scheduled date
        schedule.sort(key=lambda x: x['scheduled_date'])
        
        return jsonify({
            'maintenance_schedule': schedule,
            'summary': {
                'total_scheduled': len(schedule),
                'this_week': len([m for m in schedule if 
                    datetime.fromisoformat(m['scheduled_date'].replace('Z', '+00:00')) <= current_time + timedelta(days=7)]),
                'parts_required': len([m for m in schedule if m['parts_required']]),
                'high_priority': len([m for m in schedule if m['priority'] == 'high']),
                'last_updated': current_time.isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_maintenance_schedule: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance/metrics', methods=['GET'])
def get_performance_metrics():
    """
    Get historical performance metrics for technical dashboard
    Model accuracy and performance metrics data
    """
    try:
        # Generate historical performance data
        metrics = []
        current_date = datetime.now()
        
        for i in range(30):  # 30 days of data
            date = current_date - timedelta(days=i)
            
            # Temperature model metrics
            temp_accuracy = 0.85 + np.random.normal(0, 0.05)
            temp_precision = 0.82 + np.random.normal(0, 0.04)
            temp_recall = 0.78 + np.random.normal(0, 0.06)
            
            # Vibration model metrics  
            vib_accuracy = 0.79 + np.random.normal(0, 0.07)
            vib_precision = 0.76 + np.random.normal(0, 0.05)
            vib_recall = 0.81 + np.random.normal(0, 0.04)
            
            # Performance metrics
            prediction_confidence = np.random.uniform(0.75, 0.95)
            response_time_ms = np.random.randint(50, 200)
            data_quality_score = np.random.uniform(0.85, 0.98)
            
            metrics.append({
                'date': date.date().isoformat(),
                'temperature_model': {
                    'accuracy': round(max(0, min(1, temp_accuracy)), 3),
                    'precision': round(max(0, min(1, temp_precision)), 3),
                    'recall': round(max(0, min(1, temp_recall)), 3)
                },
                'vibration_model': {
                    'accuracy': round(max(0, min(1, vib_accuracy)), 3),
                    'precision': round(max(0, min(1, vib_precision)), 3),
                    'recall': round(max(0, min(1, vib_recall)), 3)
                },
                'performance_metrics': {
                    'prediction_confidence': round(prediction_confidence, 3),
                    'response_time_ms': response_time_ms,
                    'data_quality_score': round(data_quality_score, 3),
                    'maintenance_efficiency': round(np.random.uniform(0.8, 0.95), 3)
                }
            })
        
        return jsonify({
            'performance_metrics': metrics,
            'summary': {
                'avg_temperature_accuracy': round(np.mean([m['temperature_model']['accuracy'] for m in metrics]), 3),
                'avg_vibration_accuracy': round(np.mean([m['vibration_model']['accuracy'] for m in metrics]), 3),
                'avg_confidence': round(np.mean([m['performance_metrics']['prediction_confidence'] for m in metrics]), 3),
                'avg_response_time': round(np.mean([m['performance_metrics']['response_time_ms'] for m in metrics]), 1),
                'last_updated': current_date.isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_performance_metrics: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/streaming/engines', methods=['GET'])
def get_streaming_data():
    """
    Optimized endpoint for Power BI streaming datasets
    Returns simplified, frequently-updated engine status
    """
    try:
        streaming_data = []
        
        for engine_id in range(1, 11):
            # Simulate real-time variations
            base_rul = np.random.uniform(10, 100)
            alert_prob = np.random.uniform(0.05, 0.5)
            
            # Status based on RUL and alert probability
            if base_rul < 15 or alert_prob > 0.4:
                status = "critical"
            elif base_rul < 30 or alert_prob > 0.2:
                status = "warning"
            else:
                status = "healthy"
            
            streaming_data.append({
                'timestamp': datetime.now().isoformat(),
                'engine_id': engine_id,
                'rul_prediction': round(base_rul, 1),
                'alert_probability': round(alert_prob, 3),
                'status': status,
                'temperature_avg': round(np.random.uniform(400, 600), 1),
                'vibration_rms': round(np.random.uniform(0.1, 2.0), 3),
                'efficiency': round(np.random.uniform(0.75, 0.95), 3)
            })
        
        return jsonify(streaming_data)
        
    except Exception as e:
        logger.error(f"Error in get_streaming_data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/webhook', methods=['POST'])
def handle_powerbi_webhook():
    """
    Handle webhooks from Power BI alerts
    Process alert notifications and trigger external systems
    """
    try:
        webhook_data = request.json or {}
        
        # Log the webhook event
        logger.info(f"Power BI webhook received: {webhook_data}")
        
        # Process different types of alerts
        alert_type = webhook_data.get('alertType', 'unknown')
        
        if alert_type == 'critical_engines':
            # Handle critical engines alert
            response = handle_critical_engines_webhook(webhook_data)
        elif alert_type == 'low_rul':
            # Handle low RUL alert
            response = handle_low_rul_webhook(webhook_data)
        else:
            # Generic alert handling
            response = handle_generic_webhook(webhook_data)
        
        return jsonify({
            'status': 'processed',
            'webhook_id': webhook_data.get('id', 'unknown'),
            'processed_at': datetime.now().isoformat(),
            'response': response
        })
        
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        return jsonify({'error': str(e)}), 500

def handle_critical_engines_webhook(data):
    """Handle critical engines webhook"""
    # In production, this would:
    # 1. Send notifications to operations team
    # 2. Create service tickets
    # 3. Trigger automated responses
    return {
        'action': 'critical_alert_processed',
        'notifications_sent': 3,
        'tickets_created': 1
    }

def handle_low_rul_webhook(data):
    """Handle low RUL webhook"""
    # In production, this would:
    # 1. Schedule maintenance
    # 2. Order parts
    # 3. Notify maintenance team
    return {
        'action': 'maintenance_scheduled',
        'parts_ordered': True,
        'team_notified': True
    }

def handle_generic_webhook(data):
    """Handle generic webhook"""
    return {
        'action': 'generic_alert_logged',
        'logged_at': datetime.now().isoformat()
    }

if __name__ == '__main__':
    # Initialize models if available
    api_server.initialize_models()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)