from sklearn.metrics import f1_score, confusion_matrix, classification_report

def evaluate_predictions(y_true, y_pred_proba, threshold=0.55):
    """
    Evaluates model performance using a specific decision threshold.
    """
    # Convert probabilities to binary labels based on threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate key metrics
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    
    return f1, cm, report

def print_results(f1, cm):
    """
    Utility for clean console output of metrics.
    """
    print(f"\n[+] Final F1-Score: {f1:.4f}")
    print("[+] Confusion Matrix:")
    print(cm)