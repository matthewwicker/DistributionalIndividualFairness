import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


DATA_ADULT_TRAIN = '../Datasets/adult.data.csv'
DATA_ADULT_TEST = '../Datasets/adult.test.csv'
DATA_CRIME_FILENAME = '../Datasets/crime.csv'
DATA_GERMAN_FILENAME = '../Datasets/german.csv'


def get_adult_data(sensitive_features, drop_columns=[], test_size=0.2, random_state=42):
    """
    train_path: path to training data
    test_path: path to test data
    returns: tuple of training features, training labels, test features and test labels
    """
    train_df = pd.read_csv(DATA_ADULT_TRAIN, na_values='?').dropna()
    test_df = pd.read_csv(DATA_ADULT_TEST, na_values='?').dropna()
    target = 'target'

    X_train = train_df.drop(columns=[target])
    y_train = train_df[[target]]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[[target]]

    train_ds = TabularDataset(
        X_train, y_train, sensitive_features=sensitive_features, drop_columns=drop_columns)
    test_ds = TabularDataset(
        X_test, y_test, sensitive_features=sensitive_features, drop_columns=drop_columns)

    return train_ds, test_ds