import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from graphviz import Source
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


data_path = 'data/test_data/w_features.tsv'
feature_names = ["emdash",
                 "emoji",
                 "bold",
                 "title_case",
                 "list",
                 "curly_quotation",
                 "table",
                 "template",
                 "grammar",
                 "vocab",
                 "neg_parallelism",
                 "rule_of_three",
                 "basic_copulative",
                 "eleg_variation",
                 "subject_line",
                 "summary"
                 ]


df = pd.read_csv(data_path, sep='\t', index_col=False)
df = df.dropna()

y = df['label']
X = df[feature_names].values

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    shuffle=True)


model = RandomForestClassifier(
    n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# # Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
# Precision
precision = precision_score(y_test, y_pred, pos_label="AI")
# Recall
recall = recall_score(y_test, y_pred, pos_label="AI")
# F1-Score
f1 = f1_score(y_test, y_pred, pos_label="AI")
# # ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label="AI")
roc_auc = roc_auc_score(y_test, y_pred_prob)

# # Plot the ROC curve
# plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
# # roc curve for tpr = fpr
# plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend(loc="lower right")
# plt.show()

# # Plot the predicted class probabilities
# plt.hist(y_pred_prob, bins=10)
# plt.xlim(0, 1)
# plt.title('Histogram of predicted probabilities')
# plt.xlabel('Predicted probability of AI')
# plt.ylabel('Frequency')
# plt.show()


print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 Score:  {f1:.2f}")
print(f"AUROC:  {roc_auc:.2f}")
