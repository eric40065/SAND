import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from Model import CNPModel
from DataUtils import DataLoader

# implememnt data generating process

folder = "train1/"
Path(folder).mkdir(parents=True, exist_ok=True)
X = pd.read_csv("../data/dense_X.csv", header=None)
T = pd.read_csv("../data/dense_T.csv", header=None)

if True:
    device = torch.device("cpu")
    batch_size = 16
else:
    # torch.autograd.set_detect_anomaly(True)
    # set up GPU
    device = torch.device("cuda:0")

checkpoint = torch.load(folder + 'best_checkpoint.pth')
model = checkpoint["model"]
model.load_state_dict(checkpoint["state_dict"])
for parameter in model.parameters():
    parameter.requires_grad = False
# send model to GPU
model.to(device)
model.eval()

X_obs = pd.read_csv("../data/X.csv", header=None)
T_obs = pd.read_csv("../data/T.csv", header=None)

X_den = pd.read_csv("../data/dense_X.csv", header=None)
_, L = X_den.shape
T_den = pd.read_csv("../data/dense_T.csv", header=None)