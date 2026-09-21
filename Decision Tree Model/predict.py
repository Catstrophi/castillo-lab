from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

from dataset import create_datasets
from model import BreastCancerClassifier


DATA_PATH = "data/wdbc.data"
MODEL_PATH = "saved_models/best_model.json"

THRESHOLD = 0.500


# Get same dataset split
training_features, validation_features, test_features, training_labels, validation_labels, test_labels = create_datasets(DATA_PATH)


# Create and load model
model = BreastCancerClassifier()
model.load_model(MODEL_PATH)


# Get the preditions
probabilities = model.predict_proba(test_features)
probabilities = probabilities[:, 1]
predictions = probabilities >= THRESHOLD
predictions = predictions.astype(int)

# All metrics
loss = log_loss(test_labels, probabilities)
accuracy = accuracy_score(test_labels, predictions)

# zero_division = 0 reports zero if a metric cannot be calculated.
precision = precision_score(test_labels, predictions, zero_division = 0)
recall = recall_score(test_labels, predictions, zero_division = 0)
f1 = f1_score(test_labels, predictions, zero_division = 0)
matrix = confusion_matrix(test_labels, predictions, labels = [0, 1])

# Prints the final results
print(f"Loss: {loss:.3f}")
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1: {f1:.3f}")

print("predicted benign, predicted malignant")
print("actual benign\nactual malignant")
print(matrix)