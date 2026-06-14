import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score,
)
import joblib

from features import build_preprocessor, ALL_FEATURES, TARGET

DATA_PATH = Path(__file__).parent / "data" / "anomaly_samples.csv"
MODEL_DIR = Path(__file__).parent / "model"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df[ALL_FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Train: {len(X_train)}  Test: {len(X_test)}")
print(f"Train anomaly rate: {y_train.mean():.1%}  Test anomaly rate: {y_test.mean():.1%}")

preprocessor = build_preprocessor()

lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
])

gb_pipeline = Pipeline([
    ("preprocessor", build_preprocessor()),
    ("classifier", GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nRunning 5-fold CV...")
lr_scores = cross_val_score(lr_pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
gb_scores = cross_val_score(gb_pipeline, X_train, y_train, cv=cv, scoring="roc_auc")

print(f"Logistic Regression CV ROC-AUC: {lr_scores.mean():.4f} ± {lr_scores.std():.4f}")
print(f"Gradient Boosting  CV ROC-AUC: {gb_scores.mean():.4f} ± {gb_scores.std():.4f}")

if gb_scores.mean() >= lr_scores.mean():
    winner_pipeline = gb_pipeline
    winner_name = "GradientBoosting"
else:
    winner_pipeline = lr_pipeline
    winner_name = "LogisticRegression"

print(f"\nWinner: {winner_name}")

winner_pipeline.fit(X_train, y_train)
y_proba = winner_pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

test_roc_auc = roc_auc_score(y_test, y_proba)
test_f1 = f1_score(y_test, y_pred)
test_ap = average_precision_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

print(f"\nTest ROC-AUC:  {test_roc_auc:.4f}")
print(f"Test F1:       {test_f1:.4f}")
print(f"Test Avg Prec: {test_ap:.4f}")
print(f"Confusion matrix:\n{cm}")

joblib.dump(winner_pipeline, MODEL_DIR / "model.joblib")
print(f"\nModel saved to {MODEL_DIR / 'model.joblib'}")

if winner_name == "GradientBoosting":
    clf = winner_pipeline.named_steps["classifier"]
    pre = winner_pipeline.named_steps["preprocessor"]
    num_names = list(pre.transformers_[0][2])
    cat_names = list(pre.transformers_[1][1].get_feature_names_out(pre.transformers_[1][2]))
    feature_names = num_names + cat_names
    importances = clf.feature_importances_
else:
    clf = winner_pipeline.named_steps["classifier"]
    pre = winner_pipeline.named_steps["preprocessor"]
    num_names = list(pre.transformers_[0][2])
    cat_names = list(pre.transformers_[1][1].get_feature_names_out(pre.transformers_[1][2]))
    feature_names = num_names + cat_names
    importances = np.abs(clf.coef_[0])

feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
top5 = feat_imp.head(5)

metrics = {
    "algorithm": winner_name,
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "cv_roc_auc_mean": float(round(max(lr_scores.mean(), gb_scores.mean()), 4)),
    "test_roc_auc": float(round(test_roc_auc, 4)),
    "test_f1": float(round(test_f1, 4)),
    "test_avg_precision": float(round(test_ap, 4)),
    "confusion_matrix": cm.tolist(),
    "top5_features": {k: float(round(v, 6)) for k, v in top5.items()},
    "threshold": 0.5,
}

with open(MODEL_DIR / "metrics.json", "w") as fh:
    json.dump(metrics, fh, indent=2)
print(f"Metrics saved to {MODEL_DIR / 'metrics.json'}")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(f"Model Evaluation — {winner_name}", fontsize=14, fontweight="bold")

fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[0, 0].plot(fpr, tpr, color="#00d4ff", lw=2, label=f"AUC = {test_roc_auc:.4f}")
axes[0, 0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0, 0].set_xlabel("False Positive Rate")
axes[0, 0].set_ylabel("True Positive Rate")
axes[0, 0].set_title("ROC Curve")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

precision, recall, _ = precision_recall_curve(y_test, y_proba)
axes[0, 1].plot(recall, precision, color="#ff6b35", lw=2, label=f"AP = {test_ap:.4f}")
axes[0, 1].set_xlabel("Recall")
axes[0, 1].set_ylabel("Precision")
axes[0, 1].set_title("Precision-Recall Curve")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

im = axes[1, 0].imshow(cm, interpolation="nearest", cmap="Blues")
axes[1, 0].set_title("Confusion Matrix")
axes[1, 0].set_xlabel("Predicted")
axes[1, 0].set_ylabel("Actual")
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_yticks([0, 1])
axes[1, 0].set_xticklabels(["Normal", "Anomaly"])
axes[1, 0].set_yticklabels(["Normal", "Anomaly"])
for i in range(2):
    for j in range(2):
        axes[1, 0].text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)

top10 = feat_imp.head(10)
axes[1, 1].barh(range(len(top10)), top10.values[::-1], color="#00d4ff", alpha=0.8)
axes[1, 1].set_yticks(range(len(top10)))
axes[1, 1].set_yticklabels([n.replace("cat__", "").replace("num__", "") for n in top10.index[::-1]], fontsize=8)
axes[1, 1].set_title("Feature Importance (Top 10)")
axes[1, 1].set_xlabel("Importance")
axes[1, 1].grid(True, axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(MODEL_DIR / "evaluation.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"Evaluation plot saved to {MODEL_DIR / 'evaluation.png'}")
print("\nTraining complete.")
