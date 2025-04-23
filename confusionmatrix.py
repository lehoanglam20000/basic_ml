import numpy as np
from sklearn.metrics import confusion_matrix

# Example true labels and predicted labels
y_true = [0, 1, 0, 1, 0, 1, 0, 1]
y_pred = [0, 0, 0, 1, 0, 1, 1, 1]

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Extract FP and TP
FP = cm.sum(axis=0) - np.diag(cm)
TP = np.diag(cm)

print("False Positives (FP):", FP)
print("True Positives (TP):", TP)
