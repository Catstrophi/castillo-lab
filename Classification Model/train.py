import os
import math
import torch

from dataset import create_dataloaders
from model import BreastCancerClassifier


# Allows the model to find where the data is and where to save the best model
DATA_PATH = "data/wdbc.data"
MODEL_PATH = "saved_models/best_model.pth"

# I want to give 32 samples each time
BATCH_SIZE = 32

# Common learning rate for Adamm
LEARNING_RATE = 0.001

# Run all the data 100 times for better fitting
EPOCHS = 100

# Start with the same random weights when the script runs again.
torch.manual_seed(42)

# Creates the folder for the saved model
os.makedirs("saved_models", exist_ok = True)


# Get the data from the dataset.py
training_loader, validation_loader, test_loader, scaler = create_dataloaders(DATA_PATH, batch_size = BATCH_SIZE)

# get the model, data set has 30 features
model = BreastCancerClassifier(number_of_features = 30)

# This loss function accepts raw scores and handles sigmoid internally
loss_function = torch.nn.BCEWithLogitsLoss()

# Use torch built in optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

# As a starting point we set the loss to inf, saves best epoch so we know
best_validation_loss = float("inf")
best_epoch = 0


# Goes through the number of epochs and trains the model
for epoch in range(EPOCHS):

    # Sets model into training mode
    model.train()

    total_training_loss = 0.0
    training_sample_count = 0

    for feature_batch, label_batch in training_loader:

        # Clear gradients left over from the previous batch.
        optimizer.zero_grad()

        # Make predictions using the current weights.
        raw_scores = model(feature_batch)

        # Compare the predictions with the correct labels.
        training_loss = loss_function(raw_scores, label_batch)

        # Calculate gradients, then update the weights.
        training_loss.backward()
        optimizer.step()

        # Count each sample equally, including the smaller final batch.
        batch_sample_count = label_batch.shape[0]
        batch_total_loss = training_loss.item() * batch_sample_count
        total_training_loss += batch_total_loss
        training_sample_count += batch_sample_count

    average_training_loss = total_training_loss / training_sample_count


    # Validation step
    model.eval()

    total_validation_loss = 0.0
    validation_sample_count = 0

    # No gradients and no weight updates during validation.
    with torch.no_grad():

        for feature_batch, label_batch in validation_loader:

            raw_scores = model(feature_batch)

            validation_loss = loss_function(raw_scores, label_batch)

            batch_sample_count = label_batch.shape[0]
            batch_total_loss = validation_loss.item() * batch_sample_count
            total_validation_loss += batch_total_loss
            validation_sample_count += batch_sample_count

    average_validation_loss = total_validation_loss / validation_sample_count


    # Saves the best model
    if average_validation_loss < best_validation_loss:

        best_validation_loss = average_validation_loss
        best_epoch = epoch + 1

        torch.save(model.state_dict(), MODEL_PATH)

# Let me know it done
print("The end is near...")
