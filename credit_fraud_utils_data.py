import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import RobustScaler

def apply_feature_engineering(df):
    """
    Applies cyclic encoding for time and custom feature creation.
    Maintains the circular connection as per the project requirements.
    """
    data = df.copy()
    
    # Calculate Hour if not present
    if 'Hour' not in data.columns:
        data['Hour'] = (data['Time'] // 3600) % 24
    
    # Circular encoding (Sin/Cos)
    data['Hour_Sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
    data['Hour_Cos'] = np.cos(2 * np.pi * data['Hour'] / 24)
    
    # Custom binary feature
    data['Is_Small'] = (data['Amount'] < 5.0).astype(int)
    
    return data

def process_train_set(df, apply_smote=True):
    """
    Processes the training data: Engineering, Scaling, and optional SMOTE.
    """
    data = apply_feature_engineering(df)
    
    # Scaling Amount and Hour using RobustScaler as per your notebook
    scaler = RobustScaler()
    data[['Scaled_Amount', 'Scaled_Hour']] = scaler.fit_transform(data[['Amount', 'Hour']])
    
    # Drop raw columns
    data.drop(columns=['Time', 'Hour', 'Amount'], inplace=True)
    
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    # Handling Imbalance
    if apply_smote:
        sm = SMOTE(random_state=42)
        X, y = sm.fit_resample(X, y)
        print(f"Data is now balanced via SMOTE: New length {len(y)}") 
    
    return X, y, scaler

def process_inference_set(df, scaler_instance):
    """
    Processes validation or test sets using a pre-fitted scaler to avoid leakage.
    """
    data = apply_feature_engineering(df)
    
    # Transform using the training set scaler
    data[['Scaled_Amount', 'Scaled_Hour']] = scaler_instance.transform(data[['Amount', 'Hour']])
    
    data.drop(columns=['Time', 'Hour', 'Amount'], inplace=True)
    
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    return X, y