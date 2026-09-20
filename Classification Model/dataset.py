import pandas
import torch

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


COLUMN_NAMES = [
    "id",
    "diagnosis",

    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",

    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",

    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst"
]


def create_dataloaders(file_path, batch_size = 32, random_state = 42):

    # Use pandas to open the csv file 
    data = pandas.read_csv(file_path, header = None)

    data.columns = COLUMN_NAMES

    # Remove repeated rows to make sure we dont train our model on the same data
    data = data.drop_duplicates()

    
    # Chainging M to 1 and B to 0
    data["diagnosis"] = data["diagnosis"].str.upper()
    data["diagnosis"] = data["diagnosis"].map({"B": 0, "M": 1})

    # Convert each measurement column into numbers since it is a text right now
    for column_name in COLUMN_NAMES:
        if column_name != "id" and column_name != "diagnosis":
            data[column_name] = pandas.to_numeric(data[column_name])

    # We dont care about patient ID and and the diagnosis is not a feature but label
    features = data.drop(columns = ["id", "diagnosis"]).values

    # Turn the diagnosis to label to check later
    labels = data["diagnosis"].values

    # Split:
    # 70% train
    # 15% validation
    # 15% test

    # First step, we seperate 70 30. 70 for training and 30 for others
    training_features, remaining_features, training_labels, remaining_labels = train_test_split(
        features,
        labels,
        test_size = 0.30,
        random_state = random_state,
        stratify = labels
    )

    # Take that 30 and seperate it for 15 and 15 for validation and then 15 for test
    validation_features, test_features, validation_labels, test_labels = train_test_split(
        remaining_features,
        remaining_labels,
        test_size = 0.50,
        random_state = random_state,
        stratify = remaining_labels
    )

    
    # Since all the values may be on different scales we normalize the data
    scaler = StandardScaler()

    # Get the mean and sd from training data only
    training_features = scaler.fit_transform(training_features)

    # Use same scaler on validation/test
    validation_features = scaler.transform(validation_features)
    test_features = scaler.transform(test_features)

    # Now store data into tensors
    training_features = torch.tensor(training_features, dtype = torch.float32)
    training_labels = torch.tensor(training_labels, dtype = torch.float32)
    training_labels = training_labels.reshape(-1, 1)

    validation_features = torch.tensor(validation_features, dtype = torch.float32)
    validation_labels = torch.tensor(validation_labels, dtype = torch.float32)
    validation_labels = validation_labels.reshape(-1, 1)
    
    test_features = torch.tensor(test_features, dtype = torch.float32)
    test_labels = torch.tensor(test_labels, dtype = torch.float32)
    test_labels = test_labels.reshape(-1, 1)


    # Create datasets
    training_dataset = TensorDataset(training_features, training_labels)

    validation_dataset = TensorDataset(validation_features, validation_labels)

    test_dataset = TensorDataset(test_features, test_labels)

    
    # Create batch loaders
    training_loader = DataLoader(training_dataset, batch_size = batch_size, shuffle = True)

    validation_loader = DataLoader(validation_dataset, batch_size = batch_size,shuffle = False)

    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    return (training_loader, validation_loader, test_loader, scaler)
