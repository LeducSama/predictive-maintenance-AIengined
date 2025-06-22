import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import os


class TurbofanDataLoader:
    """
    Data loader for NASA Turbofan Engine Degradation Dataset
    Implements the 25% baseline normalization strategy for business-aligned feature engineering
    """
    
    def __init__(self, data_path: str = "data/raw"):
        self.data_path = data_path
        self.sensor_columns = [f'sensor_{i}' for i in range(1, 22)]
        self.setting_columns = ['setting_1', 'setting_2', 'setting_3']
        self.columns = ['unit_id', 'cycle'] + self.setting_columns + self.sensor_columns
        
        # Temperature-related sensors (based on domain knowledge)
        self.temperature_sensors = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_11', 'sensor_12']
        
        # Vibration-related sensors (based on domain knowledge)  
        self.vibration_sensors = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
    
    def load_dataset(self, dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load training data, test data, and RUL values for a specific dataset
        
        Args:
            dataset_name: One of 'FD001', 'FD002', 'FD003', 'FD004'
        
        Returns:
            Tuple of (train_df, test_df, rul_df)
        """
        train_file = f"{self.data_path}/train_{dataset_name}.txt"
        test_file = f"{self.data_path}/test_{dataset_name}.txt"
        rul_file = f"{self.data_path}/RUL_{dataset_name}.txt"
        
        # Load training data
        train_df = pd.read_csv(train_file, sep=' ', header=None)
        train_df = train_df.dropna(axis=1)
        train_df.columns = self.columns[:len(train_df.columns)]
        
        # Load test data
        test_df = pd.read_csv(test_file, sep=' ', header=None)
        test_df = test_df.dropna(axis=1)
        test_df.columns = self.columns[:len(test_df.columns)]
        
        # Load RUL values
        rul_df = pd.read_csv(rul_file, header=None)
        rul_df.columns = ['rul']
        
        return train_df, test_df, rul_df
    
    def calculate_baseline_normalization(self, df: pd.DataFrame, early_age_percent: float = 0.25) -> Dict[int, Dict[str, float]]:
        """
        Calculate baseline values for each engine using early age (25%) normalization approach
        
        Args:
            df: Training dataframe
            early_age_percent: Percentage of early life to use for baseline (default 0.25)
        
        Returns:
            Dictionary mapping unit_id to sensor baseline values
        """
        baselines = {}
        
        for unit_id in df['unit_id'].unique():
            unit_data = df[df['unit_id'] == unit_id].sort_values('cycle')
            max_cycle = unit_data['cycle'].max()
            early_age_cycles = int(max_cycle * early_age_percent)
            
            # Get early age data (first 25% of cycles)
            early_age_data = unit_data[unit_data['cycle'] <= early_age_cycles]
            
            # Calculate mean values for each sensor during early age
            baseline_values = {}
            for sensor in self.sensor_columns:
                if sensor in early_age_data.columns:
                    baseline_values[sensor] = early_age_data[sensor].mean()
            
            baselines[unit_id] = baseline_values
        
        return baselines
    
    def apply_baseline_normalization(self, df: pd.DataFrame, baselines: Dict[int, Dict[str, float]]) -> pd.DataFrame:
        """
        Apply baseline normalization to the dataframe
        
        Args:
            df: Dataframe to normalize
            baselines: Baseline values from training data
        
        Returns:
            Normalized dataframe
        """
        df_normalized = df.copy()
        
        for unit_id in df['unit_id'].unique():
            if unit_id in baselines:
                unit_mask = df_normalized['unit_id'] == unit_id
                for sensor in self.sensor_columns:
                    if sensor in baselines[unit_id] and sensor in df_normalized.columns:
                        baseline_value = baselines[unit_id][sensor]
                        if sensor in df_normalized.columns and pd.api.types.is_numeric_dtype(df_normalized[sensor]):
                            df_normalized.loc[unit_mask, sensor] = pd.to_numeric(df_normalized.loc[unit_mask, sensor]) - baseline_value
        
        return df_normalized
    
    def add_rul_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Remaining Useful Life (RUL) labels to training data
        
        Args:
            df: Training dataframe
        
        Returns:
            Dataframe with RUL column added
        """
        df_with_rul = df.copy()
        df_with_rul['rul'] = 0
        
        for unit_id in df['unit_id'].unique():
            unit_data = df[df['unit_id'] == unit_id]
            max_cycle = unit_data['cycle'].max()
            
            unit_mask = df_with_rul['unit_id'] == unit_id
            df_with_rul.loc[unit_mask, 'rul'] = max_cycle - df_with_rul.loc[unit_mask, 'cycle']
        
        return df_with_rul
    
    def create_sequences(self, df: pd.DataFrame, sequence_length: int, sensor_subset: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        
        Args:
            df: Normalized dataframe with RUL labels
            sequence_length: Length of sequences to create
            sensor_subset: List of sensors to include in sequences
        
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_sequences = []
        y_sequences = []
        
        for unit_id in df['unit_id'].unique():
            unit_data = df[df['unit_id'] == unit_id].sort_values('cycle')
            
            # Select only the specified sensors
            sensor_data = unit_data[sensor_subset].values
            rul_data = unit_data['rul'].values
            
            # Create sequences
            for i in range(len(sensor_data) - sequence_length + 1):
                X_sequences.append(sensor_data[i:i+sequence_length])
                y_sequences.append(rul_data[i+sequence_length-1])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def create_change_detection_labels(self, df: pd.DataFrame, window_size: int = 10) -> pd.DataFrame:
        """
        Create binary labels for vibration change detection
        
        Args:
            df: Normalized dataframe
            window_size: Window size for change detection
        
        Returns:
            Dataframe with change_detected column
        """
        df_with_labels = df.copy()
        df_with_labels['change_detected'] = 0
        
        for unit_id in df['unit_id'].unique():
            unit_data = df[df['unit_id'] == unit_id].sort_values('cycle')
            unit_mask = df_with_labels['unit_id'] == unit_id
            
            # Calculate rolling standard deviation for vibration sensors
            vibration_data = unit_data[self.vibration_sensors]
            rolling_std = vibration_data.rolling(window=window_size).std().mean(axis=1)
            
            # Detect significant changes (simple threshold-based approach)
            threshold = rolling_std.quantile(0.8)
            change_points = rolling_std > threshold
            
            df_with_labels.loc[unit_mask, 'change_detected'] = change_points.astype(int)
        
        return df_with_labels
    
    def prepare_dual_datasets(self, dataset_name: str) -> Tuple[Dict, Dict]:
        """
        Prepare both temperature trend and vibration alert datasets
        
        Args:
            dataset_name: Dataset to process ('FD001', 'FD002', 'FD003', 'FD004')
        
        Returns:
            Tuple of (temperature_data, vibration_data) dictionaries
        """
        # Load raw data
        train_df, test_df, rul_df = self.load_dataset(dataset_name)
        
        # Calculate baselines from training data
        baselines = self.calculate_baseline_normalization(train_df)
        
        # Apply normalization
        train_normalized = self.apply_baseline_normalization(train_df, baselines)
        test_normalized = self.apply_baseline_normalization(test_df, baselines)
        
        # Add RUL labels to training data
        train_with_rul = self.add_rul_labels(train_normalized)
        
        # Create change detection labels
        train_with_changes = self.create_change_detection_labels(train_with_rul)
        
        # Prepare temperature trend data (long sequences)
        temp_X_train, temp_y_train = self.create_sequences(
            train_with_rul, 
            sequence_length=100, 
            sensor_subset=self.temperature_sensors
        )
        
        # Prepare vibration alert data (short sequences)
        vib_X_train, vib_y_train = self.create_sequences(
            train_with_changes[train_with_changes.columns[:-1]], 
            sequence_length=10, 
            sensor_subset=self.vibration_sensors
        )
        
        # Get change detection labels for vibration model
        vib_change_labels = []
        for unit_id in train_with_changes['unit_id'].unique():
            unit_data = train_with_changes[train_with_changes['unit_id'] == unit_id].sort_values('cycle')
            change_data = unit_data['change_detected'].values
            for i in range(len(change_data) - 10 + 1):
                vib_change_labels.append(change_data[i+10-1])
        
        temperature_data = {
            'X_train': temp_X_train,
            'y_train': temp_y_train,
            'test_df': test_normalized,
            'baselines': baselines,
            'sensor_subset': self.temperature_sensors
        }
        
        vibration_data = {
            'X_train': vib_X_train,
            'y_train': np.array(vib_change_labels),
            'test_df': test_normalized,
            'baselines': baselines,
            'sensor_subset': self.vibration_sensors
        }
        
        return temperature_data, vibration_data


if __name__ == "__main__":
    # Example usage
    loader = TurbofanDataLoader()
    temp_data, vib_data = loader.prepare_dual_datasets('FD001')
    
    print(f"Temperature model data shapes:")
    print(f"X_train: {temp_data['X_train'].shape}")
    print(f"y_train: {temp_data['y_train'].shape}")
    
    print(f"\nVibration model data shapes:")
    print(f"X_train: {vib_data['X_train'].shape}")
    print(f"y_train: {vib_data['y_train'].shape}")