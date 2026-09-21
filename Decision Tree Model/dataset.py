import pandas

from sklearn.model_selection import train_test_split


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


def create_datasets(file_path, random_state = 42):

    # Open the dataset
    data = pandas.read_csv(file_path, header = None)

    # Give every column its correct name
    data.columns = COLUMN_NAMES

    # Remove duplicate rows
    data = data.drop_duplicates()

    # Change diagnosis:
    # B = benign = 0
    # M = malignant = 1
    data["diagnosis"] = data["diagnosis"].str.upper()
    data["diagnosis"] = data["diagnosis"].map({"B": 0, "M": 1})

    # Convert measurement columns to numeric values
    for column_name in COLUMN_NAMES:
        if column_name != "id" and column_name != "diagnosis":
            data[column_name] = pandas.to_numeric(data[column_name])


    # Remove ID and diagnosis from the features
    features = data.drop(columns = ["id", "diagnosis"]).values

    # Diagnosis is what the model is trying to predict
    labels = data["diagnosis"].values


    # First split:
    # 70% training
    # 30% remaining

    training_features, remaining_features, training_labels, remaining_labels = train_test_split(
        features,
        labels,
        test_size = 0.30,
        random_state = random_state,
        stratify = labels
    )


    # Split remaining 30% into:
    # 15% validation
    # 15% test

    validation_features, test_features, validation_labels, test_labels = train_test_split(
        remaining_features,
        remaining_labels,
        test_size = 0.50,
        random_state = random_state,
        stratify = remaining_labels
    )


    return (training_features, validation_features, test_features, training_labels, validation_labels, test_labels)