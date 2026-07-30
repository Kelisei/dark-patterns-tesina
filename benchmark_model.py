import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

def main():
    artifact = joblib.load('models/dark_pattern_model.joblib')
    pipeline = artifact["pipeline"]
    labels = artifact["labels"]
    
    df = pd.read_csv('research/datasets/unified_dataset.csv')
    X = df['content'].values
    
    y_true = np.zeros((len(df), len(labels)), dtype=int)
    for i, row in df.iterrows():
        text_labels = str(row['type']).split(',')
        for l in text_labels:
            if l in labels:
                y_true[i, labels.index(l)] = 1
                
    y_prob = pipeline.predict_proba(X)
    dynamic_thresholds = artifact["thresholds"]

    print("=== MÉTRICAS CLÁSICAS (UMBRALES ÓPTIMOS) ===")
    dyn_preds = (y_prob >= dynamic_thresholds).astype(int)
    
    for i, label in enumerate(labels):
        tn, fp, fn, tp = confusion_matrix(y_true[:, i], dyn_preds[:, i], labels=[0, 1]).ravel()
        
        acc = accuracy_score(y_true[:, i], dyn_preds[:, i])
        prec = precision_score(y_true[:, i], dyn_preds[:, i], zero_division=0)
        rec = recall_score(y_true[:, i], dyn_preds[:, i], zero_division=0)
        f1 = f1_score(y_true[:, i], dyn_preds[:, i], zero_division=0)
        
        print(f"[{label.upper()}]")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1-Score : {f1:.4f}")
        print(f"  TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
        print("-" * 40)

if __name__ == "__main__":
    main()
