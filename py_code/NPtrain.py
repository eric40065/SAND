# implement CNP in PyTorch
for name in dir():
    if not name.startswith("_"):
        del globals()[name]

import sys, os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.optim import Adam

## Find data
output_structure_folder = "/CNP"
iidt = True

my_computer = True
if my_computer:
    data_name = "/Framingham" # "UK" "Framingham" "LowDim_G" "HighDim_G" "LowDim_NG" "HighDim_NG"
    # data_is_dense, error = False, False
    data_is_dense, error = False, True
    # data_is_dense, error = True, False
    # data_is_dense, error = True, True
    server_specified_folder = "/Users/eric/Desktop/UCD"
    device = torch.device("cpu")
else:
    server_specified_folder = "/home/jshong96"
    data_name = "/" + sys.argv[1]
    data_is_dense = (sys.argv[2] == "True")
    real = True if data_name == "/Framingham" or data_name == "/UK" else False
    if real:
        cuda_device = "cuda:" + str(0 + data_is_dense + iidt * 2)
    else:
        error = (sys.argv[3] == "True")
        cuda_device = "cuda:" + str(error * 2 + data_is_dense)
    cuda_device = "cuda:" + (str(error + 0) if iidt else str(1 - error))
    device = torch.device(cuda_device)

denoise_method = "None"
real = True if data_name == "/Framingham" or data_name == "/UK" else False
folder = server_specified_folder + "/TransFD/py_code"
os.chdir(folder)
from NPModel import CNPModel
from NPDataUtils import DataLoader

data_type = "/dense" if data_is_dense else "/sparse"
sparsity_error_folder = (data_type + "/w_error" if error else data_type + "/wo_error") if not real else data_type
sparsity_error_folder = "" if data_name == "/Framingham" else sparsity_error_folder

# implememnt data generating process
X = pd.read_csv("../Data" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + sparsity_error_folder + "/X_obs.csv", header = None)
T = pd.read_csv("../Data" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + sparsity_error_folder + "/T_obs.csv", header = None)

if my_computer:
    epoch = int(4)
    batch_size = 32
    save_model_every = 2
    e_hidden, e_out_dim = [32, 32], 32
    d_hidden = [32, 32]
else:
    # torch.autograd.set_detect_anomaly(True)
    # set up GPU
    epoch = int(3e4) if data_is_dense else int(1e4)
    save_model_every = 1000
    batch_size = 512
    batch_size = 32 if data_name == "/Framingham" else batch_size
    e_hidden, e_out_dim = [128, 128, 128], 128
    d_hidden = [128, 128]

in_dim = 2
model = CNPModel(in_dim, e_hidden, e_out_dim, d_hidden)
model = model.to(device)

loss_train_history = []
loss_valid_history = []
optimizer = Adam(model.parameters(), lr = 2e-4, weight_decay = 1e-8)
dataLoader = DataLoader(batch_size, X, T, context = (0.8, 0.9), split = (90, 5, 5))
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
if my_computer:
    plt.show()
plt.close()




