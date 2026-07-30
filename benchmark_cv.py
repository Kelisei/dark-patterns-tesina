import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, fbeta_score
from sklearn.model_selection import GroupKFold
import joblib

def main():
    # Load dataset
    df = pd.read_csv('research/datasets/unified_dataset.csv')
    X = df['content'].values
    
    artifact = joblib.load('models/dark_pattern_model.joblib')
    labels = artifact["labels"]
    thresholds = artifact["thresholds"]
    
    y_true = np.zeros((len(df), len(labels)), dtype=int)
    for i, row in df.iterrows():
        text_labels = str(row['type']).split(',')
        for l in text_labels:
            if l in labels:
                y_true[i, labels.index(l)] = 1
                
    groups = df['source'].values

    # Lists to store metrics per fold
    metrics_dyn = {label: {'acc': [], 'prec': [], 'rec': [], 'f1': []} for label in labels}
    metrics_stat = {label: {'acc': [], 'prec': [], 'rec': [], 'f1': []} for label in labels}
    
    # Define static thresholds
    static_thresholds = np.full(len(labels), 0.5)
    if "shaming" in labels:
        static_thresholds[labels.index("shaming")] = 0.6
        
    gkf = GroupKFold(n_splits=5)
    
    print("Ejecutando 5-Fold Cross Validation (Out-of-Sample)...")
    fold = 1
    for tr, te in gkf.split(X, y_true, groups):
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', MultiOutputClassifier(LogisticRegression(class_weight='balanced', max_iter=1000)))
        ])
        
        pipeline.fit(X[tr], y_true[tr])
        y_prob = pipeline.predict_proba(X[te])
        
        y_prob_formatted = np.zeros((len(te), len(labels)))
        for j in range(len(labels)):
            y_prob_formatted[:, j] = y_prob[j][:, 1]
            
        dyn_preds = (y_prob_formatted >= thresholds).astype(int)
        stat_preds = (y_prob_formatted >= static_thresholds).astype(int)
        
        for j, label in enumerate(labels):
            # Dynamic Metrics
            tn, fp, fn, tp = confusion_matrix(y_true[te, j], dyn_preds[:, j], labels=[0, 1]).ravel()
            fpr_d = fp / (fp + tn) if (fp + tn) > 0 else 0
            spec_d = tn / (fp + tn) if (fp + tn) > 0 else 0
            
            prec_d = precision_score(y_true[te, j], dyn_preds[:, j], zero_division=0)
            rec_d = recall_score(y_true[te, j], dyn_preds[:, j], zero_division=0)
            f05_d = fbeta_score(y_true[te, j], dyn_preds[:, j], beta=0.5, zero_division=0)
            
            metrics_dyn[label]['acc'].append(fpr_d) # Using acc list for FPR temporarily
            metrics_dyn[label]['prec'].append(prec_d)
            metrics_dyn[label]['rec'].append(rec_d)
            metrics_dyn[label]['f1'].append(f05_d) # Using f1 list for F0.5 temporarily
            
        fold += 1

    print("\n=== MÉTRICAS OUT-OF-SAMPLE (F0.5 y FPR) ===")
    for label in labels:
        fpr_d = np.mean(metrics_dyn[label]['acc'])
        prec_d = np.mean(metrics_dyn[label]['prec'])
        rec_d = np.mean(metrics_dyn[label]['rec'])
        f05_d = np.mean(metrics_dyn[label]['f1'])
        
        print(f"[{label.upper()}] (Umbral: {thresholds[labels.index(label)]})")
        print(f"  FPR      : {fpr_d:.4f}")
        print(f"  Precision: {prec_d:.4f}")
        print(f"  Recall   : {rec_d:.4f}")
        print(f"  F0.5     : {f05_d:.4f}")
        print("-" * 60)

if __name__ == "__main__":
    main()
