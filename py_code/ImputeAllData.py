import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from ModelRelated import evaluation

## Get the data. 
# Options are: "/HighDim_E", "/LowDim_G", "/HighDim_G", "/LowDim_E", "/LowDim_T", "/HighDim_T", "/UK"
data_name = "/HighDim_E" 
iidt = True

## Define the device
cuda_device = "cpu" # use "cuda:0" if gpu is avaliable

## Define the model
output_structure = "SAND" # "Vanilla" "SelfAtt" "SAND"

## d and split must be the same as in the dataloader defined from train.py
d, split = 60, (90, 5, 5)

result = evaluation(data_name, iidt, output_structure, d = d, split = split, cuda_device = cuda_device)
