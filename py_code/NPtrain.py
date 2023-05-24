# implement CNP in PyTorch
import sys, os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.optim import Adam
from NPModel import CNPModel
from NPDataUtils import DataLoader

## Get the data. 
# Options are: "HighDim_E", "LowDim_G", "HighDim_G", "LowDim_E", "LowDim_T", "HighDim_T", "UK"
data_name = "/HighDim_E" 
# Pick a kind of data to analyze.
# Options for (data_is_dense, error) are: case 1 (True, True), case 2 (False, True), case 3 (True, False), case 4 (False, False).
# Options for iidt are: True, False
data_is_dense, error = True, True
iidt = True

## Define the device
cuda_device = "cpu" # use "cuda:0" if gpu is avaliable

## Define the dataloader. 
# d: number of embedding layers. batch_size: batch size. split: proportions of training/validation/testing
dataloader_setting = {"batch_size": 512, "split": (90, 5, 5), "context": (0.8, 0.9)}

## Define the models.
output_structure_folder = "/CNP"
in_dim = 2
epoch = int(5e4)
save_model_every = 1000
batch_size = 512
e_hidden, e_out_dim = [128, 128, 128], 128
d_hidden = [128, 128]

## Get the data.
denoise_method = "None" # weight decay is applied in Adam so l2 penalty is used.
device = torch.device(cuda_device)
real = (data_name == "/UK") # A useful indicator that records if the data is simulated
data_type = "/dense" if data_is_dense else "/sparse"
sparsity_error_folder = (data_type + "/w_error" if error else data_type + "/wo_error") if not real else data_type
sparsity_error_folder = "" if data_name == "/Framingham" else sparsity_error_folder

# implememnt data generating process
X = pd.read_csv("../Data" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + sparsity_error_folder + "/X_obs.csv", header = None)
T = pd.read_csv("../Data" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + sparsity_error_folder + "/T_obs.csv", header = None)

model = CNPModel(in_dim, e_hidden, e_out_dim, d_hidden)
model = model.to(device)

loss_train_history = []
loss_valid_history = []
optimizer = Adam(model.parameters(), lr = 2e-4, weight_decay = 1e-8)
dataLoader = DataLoader(dataloader_setting["batch_size"], X, T, context = dataloader_setting["context"], split = dataloader_setting["split"])
min_valid_loss = sys.maxsize

for k in range(epoch):
    if k and k % save_model_every == 0:
        checkpoint = {"model": CNPModel(in_dim, e_hidden, e_out_dim, d_hidden), "state_dict": model.state_dict(), "optimizer" : optimizer.state_dict()}
        torch.save(checkpoint, "../Checkpoints/Tmp" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + output_structure_folder + sparsity_error_folder + "/ckpts_" + str(denoise_method) + "_" + str(k) + ".pth")
                   
    loss_train = []
    loss_valid = []
    model = model.train()
    # set model training state
    for i, (context_t, context_x, context_mask, trg_t, trg_x, trg_mask) in enumerate(dataLoader.get_train_batch()):
        # stop
        context_t, context_x, context_mask, trg_t, trg_x, trg_mask = context_t.to(device), context_x.to(device), context_mask.to(device), trg_t.to(device), trg_x.to(device), trg_mask.to(device)
        log_prob, _, _ = model(context_t, context_x, context_mask, trg_t, trg_x, trg_mask)
        loss = -torch.mean(log_prob)
        # record training loss history
        loss_train.append(loss.item())
        
        # update parameters using backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    loss_train_history.append(np.mean(loss_train))

    # model evaluation mode
    with torch.no_grad():
        model = model.eval()
        for context_t, context_x, context_mask, trg_t, trg_x, trg_mask in dataLoader.get_valid_batch():
            context_t, context_x, context_mask, trg_t, trg_x, trg_mask = context_t.to(device), context_x.to(device), context_mask.to(device), trg_t.to(device), trg_x.to(device), trg_mask.to(device)
            log_prob, _, _ = model(context_t, context_x, context_mask, trg_t, trg_x, trg_mask)
            valid_loss = -torch.mean(log_prob)
            loss_valid.append(valid_loss.item())
    
    loss_valid_history.append(np.mean(loss_valid))
    if np.mean(loss_valid) < min_valid_loss:
        checkpoint_best = {"model": CNPModel(in_dim, e_hidden, e_out_dim, d_hidden), "state_dict": model.state_dict()}
        torch.save(checkpoint_best, "../Checkpoints" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + output_structure_folder + sparsity_error_folder + "/best_ckpts_" + str(denoise_method) + ".pth")
        min_valid_loss = np.mean(loss_valid)
        print("epoch:", k, "training loss = ", loss_train_history[-1], "validation loss = ", loss_valid_history[-1], flush = True)

plt.clf()
plt.subplots_adjust(left=0.2, right=0.9, top=0.95, bottom=0.15)
epoch_range = list(range(1, len(loss_train_history) + 1))
plt.plot(epoch_range, loss_train_history, label = 'train', lw = 1)
plt.plot(epoch_range, loss_valid_history, label = 'valid', lw = 1)
plt.xlabel("epoch vs. log(loss)", fontsize = 32)
plt.ylabel("CNP", fontsize = 32)
plt.legend(fontsize = 15)
plt.savefig("../Plots" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + "/Loss" + output_structure_folder + "/" + str(denoise_method) + sparsity_error_folder.replace("/", "_") + "_loss.png")
plt.show()
plt.close()




