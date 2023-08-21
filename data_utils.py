import gc
import numpy as np
import pandas as pd
import re

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl

# When true this will generate a random covariate shifted version of the data
# We use this to generate a dataset, and then save it under cache/CovShift
# in order that covariate shift results are reproducible :-) 
COV_SHIFT = False  # This should always be false, only set to true via data-generating scripts
global COV_SHIFT_RATE
COV_SHIFT_RATE = 0.1

DATASET_DIR = "cache"
DATA_ADULT_TRAIN = f'{DATASET_DIR}/adult.data.csv'
DATA_ADULT_TEST = f'{DATASET_DIR}/adult.test.csv'
DATA_CRIME_FILENAME = f'{DATASET_DIR}/crime.csv'
DATA_GERMAN_FILENAME = f'{DATASET_DIR}/german.csv'
DATA_STUDENT_FILENAME = f'{DATASET_DIR}/studentInfo.csv'
DATA_AMEX_FILENAME = f'{DATASET_DIR}/amex_train_data.ftr'


def get_student_data(sensitive_features, drop_columns=[], test_size=0.2, random_state=42, country='ENG'):
    """
    sensitive_features: features that should be considered sensitive when building the
        BiasedDataset object
    drop_columns: columns we can ignore and drop
    random_state: to pass to train_test_split
    return: two BiasedDataset objects, for training and test data respectively
    """

    # Force categorical data do be dtype: category
    data = pd.read_csv(DATA_STUDENT_FILENAME, na_values='?').dropna()
    data = data[data.final_result != "Withdrawn"]
    if (country == "ENG"):
        data = data[data.region != 'Scotland']
    elif (country == 'SCO'):
        data = data[data.region == 'Scotland']
    categorical_features = ['code_module', 'region', 'highest_education', 'imd_band', 'disability']
    for cf in categorical_features:
        data[cf] = data[cf].astype(str).astype('category')

    # Encode output
    target = 'final_result'
    features = data.drop(columns=[target])
    target = data[[target]]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size,
                                                        random_state=random_state)

    drop_columns.append('region')
    drop_columns = list(set(drop_columns))
    train_ds = TabularDataset(
        X_train, y_train, sensitive_features=sensitive_features, drop_columns=drop_columns)
    test_ds = TabularDataset(
        X_test, y_test, sensitive_features=sensitive_features, drop_columns=drop_columns)

    return train_ds, test_ds


def get_crime_data(sensitive_features, drop_columns=[], test_size=0.2, random_state=42):
    data_df = pd.read_csv(DATA_CRIME_FILENAME, na_values='?').dropna()
    train_df, test_df = train_test_split(data_df, test_size=test_size, random_state=random_state)
    target = 'ViolentCrimesPerPop'

    X_train = train_df.drop(columns=[target])
    y_train = train_df[[target]]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[[target]]

    train_ds = TabularDataset(
        X_train, y_train, sensitive_features=sensitive_features, drop_columns=drop_columns)
    test_ds = TabularDataset(
        X_test, y_test, sensitive_features=sensitive_features, drop_columns=drop_columns)

    return train_ds, test_ds


def get_AMEX_data(sensitive_features, drop_columns=[], test_size=0.2, random_state=42):
    train_dataset_ = pd.read_feather(DATA_AMEX_FILENAME)
    # Keep the latest statement features for each customer
    train_dataset = train_dataset_.groupby('customer_ID').tail(1).set_index('customer_ID', drop=True).sort_index()
    del train_dataset_
    gc.collect()

    train_dataset = train_dataset.drop(['S_2', 'D_66', 'D_42', 'D_49', 'D_73', 'D_76', 'R_9', 'B_29',
                                        'D_87', 'D_88', 'D_106', 'R_26', 'D_108', 'D_110', 'D_111', 'B_39',
                                        'B_42', 'D_132', 'D_134', 'D_135', 'D_136', 'D_137', 'D_138',
                                        'D_142', 'B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120',
                                        'D_126', 'D_63', 'D_64', 'D_66', 'D_68'], axis=1)

    selected_col = np.array(['P_2', 'S_3', 'B_2', 'D_41', 'D_43', 'B_3', 'D_44', 'D_45', 'D_46', 'D_48',
                             'D_50', 'D_53', 'S_7', 'D_56', 'S_9', 'B_6', 'B_8', 'D_52', 'P_3', 'D_54',
                             'D_55', 'B_13', 'D_59', 'D_61', 'B_15', 'D_62', 'B_16', 'B_17', 'D_77', 'B_19',
                             'B_20', 'D_69', 'B_22', 'D_70', 'D_72', 'D_74', 'R_7', 'B_25', 'B_26', 'D_78',
                             'D_79', 'D_80', 'B_27', 'D_81', 'R_12', 'D_82', 'D_105', 'S_27', 'D_83', 'R_14',
                             'D_84', 'D_86', 'R_20', 'B_33', 'D_89', 'D_91', 'S_22', 'S_23', 'S_24', 'S_25',
                             'S_26', 'D_102', 'D_103', 'D_104', 'D_107', 'B_37', 'R_27', 'D_109', 'D_112',
                             'B_40', 'D_113', 'D_115', 'D_118', 'D_119', 'D_121', 'D_122', 'D_123', 'D_124',
                             'D_125', 'D_128', 'D_129', 'B_41', 'D_130', 'D_131', 'D_133', 'D_139', 'D_140',
                             'D_141', 'D_143', 'D_144', 'D_145'])
    for col in selected_col:
        train_dataset[col] = train_dataset[col].fillna(train_dataset[col].median())

    data_df = train_dataset
    train_df, test_df = train_test_split(data_df, test_size=test_size, random_state=random_state)
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


class CustomDataModule(pl.LightningDataModule):
    """
    Make pl-Lightning data module from torch cache
    """

    def __init__(self, train, val, test, batch_size=128):
        super().__init__()
        self.train_data = train
        self.val_data = val
        self.test_data = test
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=8)
