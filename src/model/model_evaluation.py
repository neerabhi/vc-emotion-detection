import pickle, json
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score
)

clf = pickle.load(open("./models/model.pkl", "rb"))
test_data = pd.read_csv("./data/processed/test_bow.csv")

X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)

n_classes = len(set(y_test))

if n_classes == 2:
    # Binary case – same as before
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_pred_proba[:, 1])
else:
    # Multiclass case
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="macro", zero_division=0)
    roc_auc   = roc_auc_score(
        y_test,
        y_pred_proba,
        multi_class="ovo",      # or "ovr"
        average="macro"
    )

metrics_dict = {
    "accuracy":  accuracy_score(y_test, y_pred),
    "precision": precision,
    "recall":    recall,
    "roc_auc":   roc_auc
}

with open("metrics_.json", "w") as f:
    json.dump(metrics_dict, f, indent=4)
