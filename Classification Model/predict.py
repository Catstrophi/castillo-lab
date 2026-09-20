import math
import torch

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from dataset import create_dataloaders
from model import BreastCancerClassifier



DATA_PATH = "data/wdbc.data"
MODEL_PATH = "saved_models/best_model.pth"

BATCH_SIZE = 32
THRESHOLD = 0.500


# Use the same split of data
training_loader, validation_loader, test_loader, scaler = create_dataloaders(DATA_PATH, batch_size = BATCH_SIZE)


# Loading the best saved mdoel
model = BreastCancerClassifier(number_of_features=30)

saved_weights = torch.load(MODEL_PATH, map_location = "cpu", weights_only = True)

model.load_state_dict(saved_weights)
model.eval()

loss_function = torch.nn.BCEWithLogitsLoss()

all_predictions = []
all_labels = []

total_loss = 0.0
sample_count = 0


# Makes predictions with no change in gradient
with torch.no_grad():

    for feature_batch, label_batch in test_loader:

        raw_scores = model(feature_batch)

        batch_loss = loss_function(raw_scores, label_batch)

        batch_sample_count = label_batch.shape[0]
        batch_total_loss = batch_loss.item() * batch_sample_count
        total_loss += batch_total_loss
        sample_count += batch_sample_count

        # Convert raw scores into estimated probabilities.
        probabilities = torch.sigmoid(raw_scores)

        # True becomes 1 (malignant). False becomes 0 (benign).
        predictions = probabilities >= THRESHOLD
        predictions = predictions.int()

        # Since prediction is a column, turn to row, then to list, then add it to all predictions since it was in batchs
        prediction_list = predictions.flatten()
        prediction_list = prediction_list.tolist()
        all_predictions += prediction_list

        # Do the same for the labs
        label_list = label_batch.int()
        label_list = label_list.flatten()
        label_list = label_list.tolist()
        all_labels += label_list


# All metrics
average_loss = total_loss / sample_count

accuracy = accuracy_score(all_labels, all_predictions)

# zero_division = 0 reports zero if a metric cannot be calculated.
precision = precision_score(all_labels, all_predictions, zero_division = 0)

recall = recall_score(all_labels, all_predictions, zero_division = 0)

f1 = f1_score(all_labels, all_predictions, zero_division = 0)

matrix = confusion_matrix(all_labels, all_predictions, labels = [0, 1])

# Prints the final results
print(f"Loss: {average_loss:.3f}")
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1: {f1:.3f}")

print("predicted benign, predicted malignant")
print("actual benign\nactual malignant")
print(matrix)
