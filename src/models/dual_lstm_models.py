import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import joblib
import os
from typing import Tuple, Dict, Any


class TemperatureTrendModel:
    """
    LSTM model for long-term temperature trend analysis and RUL prediction
    Uses long sequences (100+ cycles) to predict remaining useful life
    """
    
    def __init__(self, sequence_length: int = 100, n_features: int = 6):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def build_model(self) -> None:
        """Build the LSTM architecture for temperature trend prediction"""
        self.model = Sequential([
            Input(shape=(self.sequence_length, self.n_features)),
            
            # First LSTM layer with return sequences
            LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            BatchNormalization(),
            
            # Second LSTM layer with return sequences
            LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            BatchNormalization(),
            
            # Third LSTM layer without return sequences
            LSTM(32, dropout=0.2, recurrent_dropout=0.2),
            BatchNormalization(),
            
            # Dense layers for regression
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='linear')  # Linear activation for regression
        ])
        
        # Compile with appropriate loss for regression
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray = None, fit_scaler: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the input data with scaling
        
        Args:
            X: Input sequences
            y: Target values (optional)
            fit_scaler: Whether to fit the scaler on this data
        
        Returns:
            Tuple of (scaled_X, y)
        """
        # Reshape for scaling
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X_reshaped)
        else:
            X_scaled = self.scaler.transform(X_reshaped)
        
        # Reshape back to original sequence format
        X_scaled = X_scaled.reshape(original_shape)
        
        return X_scaled, y
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 100, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train the temperature trend model
        
        Args:
            X_train: Training sequences
            y_train: Training RUL values
            X_val: Validation sequences (optional)
            y_val: Validation RUL values (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
        
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Preprocess training data
        X_train_scaled, _ = self.preprocess_data(X_train, y_train, fit_scaler=True)
        
        # Preprocess validation data if provided
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled, _ = self.preprocess_data(X_val, y_val, fit_scaler=False)
            validation_data = (X_val_scaled, y_val)
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss' if validation_data else 'loss', 
                         patience=10, restore_best_weights=True),
            ModelCheckpoint('models/temp_model_best.h5', 
                          monitor='val_loss' if validation_data else 'loss',
                          save_best_only=True)
        ]
        
        # Train the model
        history = self.model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Input sequences
        
        Returns:
            Predicted RUL values
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled, _ = self.preprocess_data(X, fit_scaler=False)
        predictions = self.model.predict(X_scaled)
        return predictions.flatten()
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model and scaler"""
        if self.model is not None:
            self.model.save(f"{filepath}_model.h5")
            joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model and scaler"""
        self.model = tf.keras.models.load_model(f"{filepath}_model.h5")
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        self.is_trained = True


class VibrationAlertModel:
    """
    LSTM model for short-term vibration change detection
    Uses short sequences (10 cycles) to detect sudden pattern changes
    """
    
    def __init__(self, sequence_length: int = 10, n_features: int = 7):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def build_model(self) -> None:
        """Build the LSTM architecture for vibration change detection"""
        self.model = Sequential([
            Input(shape=(self.sequence_length, self.n_features)),
            
            # First LSTM layer
            LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
            BatchNormalization(),
            
            # Second LSTM layer
            LSTM(32, dropout=0.3, recurrent_dropout=0.3),
            BatchNormalization(),
            
            # Dense layers for binary classification
            Dense(32, activation='relu'),
            Dropout(0.4),
            Dense(16, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')  # Sigmoid for binary classification
        ])
        
        # Compile with binary crossentropy for classification
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray = None, fit_scaler: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the input data with scaling
        
        Args:
            X: Input sequences
            y: Target values (optional)
            fit_scaler: Whether to fit the scaler on this data
        
        Returns:
            Tuple of (scaled_X, y)
        """
        # Reshape for scaling
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X_reshaped)
        else:
            X_scaled = self.scaler.transform(X_reshaped)
        
        # Reshape back to original sequence format
        X_scaled = X_scaled.reshape(original_shape)
        
        return X_scaled, y
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 50, batch_size: int = 64) -> Dict[str, Any]:
        """
        Train the vibration alert model
        
        Args:
            X_train: Training sequences
            y_train: Training change detection labels
            X_val: Validation sequences (optional)
            y_val: Validation change detection labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
        
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Preprocess training data
        X_train_scaled, _ = self.preprocess_data(X_train, y_train, fit_scaler=True)
        
        # Preprocess validation data if provided
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled, _ = self.preprocess_data(X_val, y_val, fit_scaler=False)
            validation_data = (X_val_scaled, y_val)
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss' if validation_data else 'loss',
                         patience=8, restore_best_weights=True),
            ModelCheckpoint('models/vib_model_best.h5',
                          monitor='val_loss' if validation_data else 'loss',
                          save_best_only=True)
        ]
        
        # Train the model
        history = self.model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1,
            class_weight={0: 1, 1: 3}  # Weight positive class more heavily
        )
        
        self.is_trained = True
        return history.history
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the trained model
        
        Args:
            X: Input sequences
            threshold: Classification threshold
        
        Returns:
            Tuple of (probabilities, binary_predictions)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled, _ = self.preprocess_data(X, fit_scaler=False)
        probabilities = self.model.predict(X_scaled).flatten()
        binary_predictions = (probabilities > threshold).astype(int)
        
        return probabilities, binary_predictions
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model and scaler"""
        if self.model is not None:
            self.model.save(f"{filepath}_model.h5")
            joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model and scaler"""
        self.model = tf.keras.models.load_model(f"{filepath}_model.h5")
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        self.is_trained = True


class DualLSTMPredictor:
    """
    Combined predictor using both temperature trend and vibration alert models
    Implements the multi-signal system architecture
    """
    
    def __init__(self):
        self.temperature_model = TemperatureTrendModel()
        self.vibration_model = VibrationAlertModel()
    
    def train_models(self, temp_data: Dict, vib_data: Dict, 
                    temp_epochs: int = 100, vib_epochs: int = 50) -> Dict[str, Any]:
        """
        Train both models with their respective datasets
        
        Args:
            temp_data: Temperature model training data
            vib_data: Vibration model training data
            temp_epochs: Epochs for temperature model
            vib_epochs: Epochs for vibration model
        
        Returns:
            Combined training history
        """
        print("Training Temperature Trend Model...")
        temp_history = self.temperature_model.train(
            temp_data['X_train'], temp_data['y_train'],
            epochs=temp_epochs
        )
        
        print("\nTraining Vibration Alert Model...")
        vib_history = self.vibration_model.train(
            vib_data['X_train'], vib_data['y_train'],
            epochs=vib_epochs
        )
        
        return {
            'temperature_history': temp_history,
            'vibration_history': vib_history
        }
    
    def predict_combined(self, temp_sequences: np.ndarray, vib_sequences: np.ndarray) -> Dict[str, Any]:
        """
        Make predictions using both models
        
        Args:
            temp_sequences: Temperature sensor sequences
            vib_sequences: Vibration sensor sequences
        
        Returns:
            Combined predictions dictionary
        """
        # Temperature trend predictions
        rul_predictions = self.temperature_model.predict(temp_sequences)
        
        # Vibration alert predictions
        alert_probs, alert_binary = self.vibration_model.predict(vib_sequences)
        
        return {
            'rul_predictions': rul_predictions,
            'alert_probabilities': alert_probs,
            'alert_binary': alert_binary,
            'combined_risk': self._calculate_combined_risk(rul_predictions, alert_probs)
        }
    
    def _calculate_combined_risk(self, rul_predictions: np.ndarray, alert_probs: np.ndarray) -> np.ndarray:
        """
        Calculate combined risk score from both models
        
        Args:
            rul_predictions: RUL predictions from temperature model
            alert_probs: Alert probabilities from vibration model
        
        Returns:
            Combined risk scores
        """
        # Normalize RUL to 0-1 scale (higher risk = lower RUL)
        rul_normalized = 1 / (1 + rul_predictions / 100)  # Assume max RUL ~100
        
        # Combine risks with weighted average
        # Vibration alerts get higher weight for immediate risk
        combined_risk = 0.3 * rul_normalized + 0.7 * alert_probs
        
        return combined_risk
    
    def save_models(self, base_path: str = "models/") -> None:
        """Save both trained models"""
        os.makedirs(base_path, exist_ok=True)
        self.temperature_model.save_model(f"{base_path}temperature")
        self.vibration_model.save_model(f"{base_path}vibration")
    
    def load_models(self, base_path: str = "models/") -> None:
        """Load both trained models"""
        self.temperature_model.load_model(f"{base_path}temperature")
        self.vibration_model.load_model(f"{base_path}vibration")


if __name__ == "__main__":
    # Example usage
    print("Dual LSTM Models for Turbofan Engine Predictive Maintenance")
    print("Temperature Model: Long-term RUL prediction")
    print("Vibration Model: Short-term change detection")
    print("Combined System: Multi-signal risk assessment")