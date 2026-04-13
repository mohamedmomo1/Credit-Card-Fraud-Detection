import argparse
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Importing your custom utilities
from credit_fraud_utils_data import process_train_set, process_inference_set
from credit_fraud_utils_eval import evaluate_predictions, print_results

def main():
    # CLI arguments setup as requested
    parser = argparse.ArgumentParser(description='Credit Fraud Detection Training Pipeline')
    parser.add_argument('--train_path', type=str, required=True, help='Path to training csv')
    parser.add_argument('--val_path', type=str, required=True, help='Path to validation csv')
    parser.add_argument('--model_save_path', type=str, default='model.pkl', help='Path to save .pkl file')
    
    args = parser.parse_args()

    # 1. Loading datasets
    print("[*] Loading data...")
    df_train = pd.read_csv(args.train_path)
    df_val = pd.read_csv(args.val_path)

    # 2. Preprocessing (Using your notebook logic)
    print("[*] Preprocessing training data...")
    X_train, y_train, scaler = process_train_set(df_train, apply_smote=True)
    X_val, y_val = process_inference_set(df_val, scaler)

    # 3. Model Definitions (Your Ensemble C components)
    print("[*] Initializing Ensemble components...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

    # Training models
    lr_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    nn_model.fit(X_train, y_train)

    # 4. Simple Weighted Ensemble Prediction (Logic from notebook)
    print("[*] Evaluating Ensemble...")
    p1 = lr_model.predict_proba(X_val)[:, 1]
    p2 = rf_model.predict_proba(X_val)[:, 1]
    p3 = nn_model.predict_proba(X_val)[:, 1]
    
    # Average probabilities for the ensemble
    y_pred_proba = (p1 + p2 + p3) / 3.0
    
    # Evaluation
    f1, cm, report = evaluate_predictions(y_val, y_pred_proba, threshold=0.55)
    print_results(f1, cm)

    # 5. Saving the final Model Package (Dictionary)
    print(f"[*] Saving model to {args.model_save_path}...")
    model_package = {
        'lr': lr_model,
        'rf': rf_model,
        'nn': nn_model,
        'scaler': scaler,
        'threshold': 0.55,
        'features': X_train.columns.tolist()
    }
    
    with open(args.model_save_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    print("[+] Done.")

if __name__ == "__main__":
    main()