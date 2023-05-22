import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

split = (90, 5, 5)
data_name_list = list(("/LowDim_E", "/HighDim_E", "/LowDim_T", "/HighDim_T", "/LowDim_G", "/HighDim_G", "/UK")) # "UK" "LowDim_G" "HighDim_G" "LowDim_NG" "HighDim_NG"
iidt = True

## Define model
denoise_method = "None"
output_structure = "CNP"
output_structure_folder = "/" + output_structure

device = torch.device("cpu")
error_mat_train_noi = np.zeros((4, len(data_name_list))) + np.nan
error_mat_test_noi = np.zeros((4, len(data_name_list))) + np.nan
error_mat_train = np.zeros((4, len(data_name_list))) + np.nan
error_mat_test = np.zeros((4, len(data_name_list))) + np.nan

for counter_col, data_name in enumerate(data_name_list):
    # stop
    real = (data_name == "/UK")
    
    if not real:
        dense_error_list = list(((True, True), (True, False), (False, True), (False, False)))
        X_den = np.array(pd.read_csv("../Data/IID/Simulation" + data_name + "/X_full_true.csv", header = None)) ## true but without noise
        X_den_noi = np.array(pd.read_csv("../Data/IID/Simulation" + data_name + "/X_full_noise.csv", header = None)) ## true but with noise
    else:
        dense_error_list = list(((True, True), (True, False)))
        X_den = np.array(pd.read_csv("../Data/IID/RealData/UK/X_full_noise.csv", header = None)) ## true but with noise
    
    for counter_row, (error, data_is_dense) in enumerate(dense_error_list):
        # stop
        # load data
        data_type = "/dense" if data_is_dense else "/sparse"
        sparsity_error_folder = (data_type + "/w_error" if error else data_type + "/wo_error") if not real else data_type

        X_obs = pd.read_csv("../Data" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + sparsity_error_folder + "/X_obs.csv", header = None)
        T_obs = pd.read_csv("../Data" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + sparsity_error_folder + "/T_obs.csv", header = None)
        
        # process data
        M_obs = np.array(X_obs.iloc[:, 0], dtype = int)
        X_obs = X_obs.iloc[:, 1:]
        
        X_obs.replace(np.nan, 0, inplace=True)
        T_obs.replace(np.nan, 0, inplace=True)
        n, m = X_obs.shape
        n_test = round(n * split[2] / sum(split))
        
        # index_training = np.array(range(0, int(n - n_test), 1)) # run all
        index_training = np.array(range(0, int(n - n_test), int((n - n_test)/500)))
        index_testing = np.array(range(n - n_test, n))
        index_all = np.unique(np.append(index_training, index_testing))
        
        T_den = np.array(pd.read_csv("../Data/IID" + ("/RealData" if real else "/Simulation") + data_name + "/T_full.csv", header=None)) ## true
        t_true = T_den[0, :]
        L = len(t_true)
        
        # load model
        checkpoint = torch.load("../Checkpoints" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + output_structure_folder + sparsity_error_folder + "/best_ckpts_" + str(denoise_method) + ".pth", map_location = torch.device('cpu'))
        model = checkpoint["model"]
        model.load_state_dict(checkpoint["state_dict"])
        for parameter in model.parameters():
            parameter.requires_grad = False
        
        # send model to CPU
        model = model.to(device)
        model = model.eval()
        
        pred = np.zeros((len(index_all), L))
        context_t = torch.Tensor(np.array(T_obs)[index_all, ]).unsqueeze(-1)
        context_x = torch.Tensor(np.array(X_obs)[index_all, ]).unsqueeze(-1)
        context_mask = torch.zeros(context_t.size())
        context_mask[np.repeat(np.arange(len(index_all)), M_obs[index_all]), np.concatenate([np.arange(M_obs[i]) for i in index_all])] = 1
        trg_t = torch.repeat_interleave(torch.Tensor(t_true).reshape(-1, L), len(index_all), 0).unsqueeze(-1)
        trg_mask = torch.zeros(trg_t.size()) + 1
        pred = model(context_t, context_x, context_mask, trg_t, trg_mask = trg_mask)
        pred_mean = np.array(pred[1].squeeze(-1))
        pred_sd = np.array(pred[2].squeeze(-1))
        
        error_mat_train[counter_row, counter_col] = np.mean(np.power(pred_mean[:len(index_training)] - X_den[index_training, ], 2))
        error_mat_test[counter_row, counter_col] = np.mean(np.power(pred_mean[len(index_training):] - X_den[index_testing, ], 2))
        if not real:
            error_mat_train_noi[counter_row, counter_col] = np.mean(np.power(pred_mean[:len(index_training)] - X_den_noi[index_training, ], 2))
            error_mat_test_noi[counter_row, counter_col] = np.mean(np.power(pred_mean[len(index_training):] - X_den_noi[index_testing, ], 2))
        
        plt.clf()
        n_train = round(n * split[0] / sum(split))
        tmp_X = np.array(X_den)[index_all, ]
        ylim = [np.min(tmp_X), np.max(tmp_X) * 2]
        for i, index in enumerate(index_all):
            if (i + 1) % 100 == 0:
                Tnow = T_obs.iloc[index, :M_obs[index]]
                Xnow = X_obs.iloc[index, :M_obs[index]]
                
                plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.1)
                plt.gca().set_ylim(ylim);
                plt.scatter(Tnow, Xnow, label = "obs.");
                plt.plot(t_true, pred_mean[i, :], label = "CNP");
                plt.plot(t_true, X_den[index, :], label = "True");
                
                plt.ylabel("CNP", fontsize = 32)
                plt.legend(fontsize = 20)
                plt.savefig("../Plots" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + "/ImputedCurves" + sparsity_error_folder + "/" + str(i + 1) + output_structure + denoise_method + ".png")
                plt.show()
                plt.close()

error_mat_test = pd.DataFrame(error_mat_test, list(("DwE", "SwE", "DwoE", "SwoE")), data_name_list) * 1000
error_mat_test.transpose()



