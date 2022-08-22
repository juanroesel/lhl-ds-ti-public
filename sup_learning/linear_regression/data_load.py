import numpy as np


def extract_data(filepath):
    """Helper function to extract the dataset from the data file."""
    with open(filepath, "r") as f:
        data = f.readlines()[1:]
    return data


def transform_data(data):
    """Helper function to transform regression data into numpy arrays."""
    x_values = []
    y_values = []
    for d in data:
        arr = d.split("\t")
        x_values.append(float(arr[1]))
        y_values.append(float(arr[2]))
    assert len(x_values) == len(y_values)
    X = np.array(x_values).reshape(-1, 1)
    y = np.array(y_values)
    return X, y
