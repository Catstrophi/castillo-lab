import os

from dataset import create_datasets
from model import BreastCancerClassifier

DATA_PATH = "data/wdbc.data"
MODEL_PATH = "saved_models/best_model.json"

# Create folder if it does not exist
os.makedirs("saved_models", exist_ok = True)

# Get datasets
training_features, validation_features, test_features, training_labels, validation_labels, test_labels = create_datasets(DATA_PATH)

# Create model
model = BreastCancerClassifier()


# Train model
model.fit(training_features, training_labels)

# Save model
model.save_model(MODEL_PATH)

print("The end is near...")