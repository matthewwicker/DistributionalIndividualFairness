import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fair")
parser.add_argument("--meth") # DONE
parser.add_argument("--width") # DONE
parser.add_argument("--year") # DONE
parser.add_argument("--state") # DONE

args = parser.parse_args()
fair = str(args.fair)
meth = str(args.meth)
width = int(args.width)
year = str(args.year)
state = str(args.state)

import numpy as np
import folktables
from folktables import ACSDataSource, ACSIncome

data_source = ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
CustomIncome = folktables.BasicProblem(
    features=[
        'AGEP',
        'COW',
        'SCHL',
        'MAR',
        'OCCP',
        'POBP',
        #'RELP',
        'WKHP',
        'SEX',
        #'RAC1P',
    ],
    target='PINCP',
    target_transform=lambda x: x > 25000,    
    group='RAC1P',
    preprocess=folktables.adult_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

print("FETCHING DATA FOR STATE: ", state)
ca_data = data_source.get_data(states=[state], download=True)
ca_features, ca_labels, _ = CustomIncome.df_to_numpy(ca_data)

import os
import sys
import copy
sys.path.append('..')
import models
import FairCertModule
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import models, transforms
from FullyConnected import FullyConnected
import pytorch_lightning as pl



X_train, X_test, y_train, y_test = train_test_split(ca_features, ca_labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)



y_train = y_train.astype(int)
y_val   = y_val.astype(int)
y_test  = y_test.astype(int)


if(fair == "LP"):
    f_epsilon = FairCertModule.get_fairness_intervals(X_train, [-1],  metric="LP", use_sens=False, eps=1)
    f_epsilon /= sum(f_epsilon)
elif(fair == "SENSR"):
    f_epsilon = FairCertModule.get_fairness_intervals(X_train, [-1],  metric="SENSR", use_sens=False, eps=1)
    f_epsilon /= sum(f_epsilon)
elif(fair == "JOHN"):
    f_epsilon = np.asarray([1,1,1,1,1,1,1])/7.0
else:
    f_epsilon = np.asarray([1,1,1,1,1,1,1])/7.0
    f_epsilon *= 0
    
X_train = X_train[:,:-1]
X_test = X_test[:,:-1]
X_val = X_val[:,:-1]


class custDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.Tensor(X).float()
        self.y = y
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.X.shape[0]
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    

CustTrain = custDataset(X_train, y_train)  
CustVal = custDataset(X_val, y_val) 
CustTest = custDataset(X_test, y_test)

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, train, val, test, batch_size=32):
        super().__init__()
        self.train_data = train
        self.val_data = val
        self.test_data = test
        self.batch_size = batch_size
        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)
    
dm = CustomDataModule(CustTrain, CustVal, CustTest)




model = FullyConnected(hidden_lay=2, hidden_dim=width, learning_rate = 0.001, mode=meth)
model.set_fair_interval(f_epsilon)
model.MAX_EPOCHS = 10
trainer = pl.Trainer(max_epochs=15, accelerator="cpu", devices=1)
trainer.fit(model, datamodule=dm)
result = trainer.test(model, datamodule=dm)


directory = "FolkModels"
if not os.path.exists(directory):
    os.makedirs(directory)
MODEL_ID = "FCN_s=%s_y=%s_w=%s_f=%s_m=%s"%(state, year, width, fair, meth)
trainer.save_checkpoint("FolkModels/%s.ckpt"%(MODEL_ID))
torch.save(model.state_dict(), "FolkModels/%s.pt"%(MODEL_ID))





