import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------
df = pd.read_csv("email_dataset.csv")
print("Dataset Loaded:", df.shape)
print("Columns:", df.columns.tolist())

# Validate
if "Category" not in df.columns or "Message" not in df.columns:
    raise Exception("Dataset must contain 'Category' and 'Message'")

# --------------------------------------------------
# 2. Preprocess
# --------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

df["clean_text"] = df["Message"].apply(clean_text)
df["label"] = df["Category"].map({"ham": 0, "spam": 1})

# --------------------------------------------------
# 3. TF-IDF Vectorizer
# --------------------------------------------------
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"].values

# --------------------------------------------------
# 4. Train/Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# --------------------------------------------------
# 5. Train Model
# --------------------------------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# --------------------------------------------------
# 6. Evaluate
# --------------------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n===== MODEL PERFORMANCE =====\n")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --------------------------------------------------
# 7. Confusion Matrix
# --------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --------------------------------------------------
# 8. ROC Curve
# --------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# --------------------------------------------------
# 9. Save Model + Vectorizer
# --------------------------------------------------
joblib.dump({"model": model, "vectorizer": vectorizer}, "sms_spam_model.pkl")
print("\n🎉 Model saved as sms_spam_model.pkl")
