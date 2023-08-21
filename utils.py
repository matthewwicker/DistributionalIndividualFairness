import os
import random
import re

import numpy as np
import torch

import data_utils
from models import FairnessCertifier


def split(dataset, frac):
    """
    Splits the provided dataset in to frac and 1-frac
    :param dataset: Must be a torch.utils.data.Dataset
    returns two cache
    """
    all_idxs = np.random.permutation(np.arange(len(dataset)))
    s1_ln = int(0.8*len(all_idxs))
    s1_idxs, s2_idxs = all_idxs[:s1_ln], all_idxs[s1_ln:]
    return torch.utils.data.Subset(dataset, s1_idxs), torch.utils.data.Subset(dataset, s2_idxs)


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_default_model(dataset, num_redacted_features=0):
    """
    Returns the default model architecture for the dataset
    :param dataset: Name of the dataset
    :param num_redacted_features: Number of redacted features that may have been removed later (eg. sensitive)
    """
    dataset = dataset.upper()
    # assert dataset in ["ADULT", "CREDIT", "GERMAN", "CRIME", "STUDENT", "AMEX"]
    # @todo to fix later: should not expose shifted logic here.
    if dataset.startswith('SHIFTED_'):
        dataset, _ = data_utils.parse_special_dataset_names(dataset.lower())
        dataset = dataset.upper()

    if(dataset=="ADULT"):
        # class_weights = torch.Tensor([1, 3.017])
        in_dim = 63 
        num_cls = 2
    elif(dataset=="CREDIT"):
        # class_weights = torch.Tensor([1, 3.508])
        in_dim = 146
        num_cls = 2
    elif(dataset=="GERMAN"):
        # class_weights = torch.Tensor([2.39, 1])
        in_dim = 62
        num_cls = 2
    elif(dataset=="CRIME"):
        # class_weights = torch.Tensor([1, 42.081])
        in_dim = 54
        num_cls = 2
    elif(dataset=="STUDENT"):
        # class_weights = torch.Tensor([1, 1.78])
        in_dim = 25
        num_cls = 2
    elif(dataset=="AMEX"):
        #self.class_weights = torch.Tensor([1,2*42.081])
        in_dim = 155
        num_cls = 2
        
    (x_train, y_train), (_, _), (_, _) = data_utils.get_data(dataset)
    y_counts = np.unique(y_train, return_counts=True)[1]
    # set to balance loss and such that class_weights sum to one.
    class_weights = torch.Tensor((y_counts.sum()-y_counts)/y_counts.sum())*(num_cls/(num_cls-1))
    print(class_weights)
    # class_weights = torch.Tensor(np.ones(num_cls))
    in_dim -= num_redacted_features
    model = FairnessCertifier(input_dimension=in_dim, num_cls=num_cls, hidden_dim=128, hidden_lay=1)
    model.class_weights = class_weights
    return model


def get_best_ckpt(fldr):
    """
    From a folder with many ckpts, returns the best ckpt with least val_loss
    expects the checkpoint files contain the pattern 'val_loss={val_loss}'
    """
    if not os.path.exists(fldr):
        print(f"Warning!! Folder {fldr} does not exist.")
        return None
    return os.path.join(fldr, 'last.ckpt')
    # patt = re.compile('val_quality=([0-9]+.[0-9]+)')
    # lst = []
    # for fname in os.listdir(fldr):
    #     if fname.endswith('.ckpt'):
    #         m = patt.search(fname)
    #         if m:
    #             val_loss = float(m.group(1))
    #             lst.append((fname, val_loss))
    # if len(lst) > 0:
    #     best_fname = max(lst, key=lambda _: _[1])[0]
    #     return os.path.join(fldr, best_fname)
    # else:
    #     return None
