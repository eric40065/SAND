for name in dir():
    if not name.startswith("_"):
        del globals()[name]

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

d, split = 20, (90, 5, 5)
add_random = False
denoise_method_list = list((None, "l2w", "l2o", "TVo", "MLE", "DAE"))
# denoise_method_list = list((None, "MLE", "DAE"))
# denoise_method_list = list((None, None))
my_computer = True

if my_computer:
    server_specified_folder = "/Users/eric/Desktop/UCD/"
else:
    server_specified_folder = "/home/jshong96/"

device = torch.device("cpu")
folder = server_specified_folder + "TransFD/py_code"
os.chdir(folder)

counter_row = 0
error_mat = np.zeros((4, len(denoise_method_list)))

for error in list((True, False)):
    for data_is_dense in list((True, False)):
        # data_is_dense, error = True, False
        # load data
        data_type = "dense" if data_is_dense else "sparse"
        sparsity_error_folder = "/" + data_type + "/w_error" if error else "/" + data_type + "/wo_error"
        
        X_obs = pd.read_csv("../data" + sparsity_error_folder + "/X_sparse.csv", header=None)
        T_obs = pd.read_csv("../data" + sparsity_error_folder + "/T_sparse.csv", header=None)
        X_den = pd.read_csv("../data" + sparsity_error_folder + "/X_sparse_true.csv", header=None)
        T_den = pd.read_csv("../data" + sparsity_error_folder + "/T_sparse_true.csv", header=None)
        
        # slice out testing data
        n, _ = X_obs.shape
        n_test = round(n * split[2] / sum(split))
        # X_obs = X_obs.iloc[(n - n_test):n, ]
        # T_obs = T_obs.iloc[(n - n_test):n, ]
        # T_den = T_den.iloc[(n - n_test):n, ]
        # X_den = X_den.iloc[(n - n_test):n, ]
        
        X_obs = X_obs.iloc[:n_test, ]
        T_obs = T_obs.iloc[:n_test, ]
        T_den = T_den.iloc[:n_test, ]
        X_den = X_den.iloc[:n_test, ]
        
        # process data
        M_obs = X_obs.iloc[:, 0].to_list()
        X_obs = X_obs.iloc[:, 1:]
        
        X_obs.replace(np.nan, 0, inplace=True)
        T_obs.replace(np.nan, 0, inplace=True)
        n, m = X_obs.shape
        _, L = T_den.shape
        t_true = T_den.iloc[0, :].to_list()
        
        counter_col = 0
        for denoise_method in denoise_method_list:
            # denoise_method = None 
            # load model
            if add_random:
                checkpoint = torch.load("../ckpts" + sparsity_error_folder + "/best_ckpts_addW_" + str(denoise_method) + ".pth", map_location = torch.device('cpu'))
            else:
                checkpoint = torch.load("../ckpts" + sparsity_error_folder + "/best_ckpts_" + str(denoise_method) + ".pth", map_location = torch.device('cpu'))
            
            model = checkpoint["model"]
            model.load_state_dict(checkpoint["state_dict"])
            for parameter in model.parameters():
                parameter.requires_grad = False
            
            # send model to CPU
            model = model.to(device)
            model = model.eval()
            num_heads = 2 * (1 + data_is_dense)
            
            X_pred = []
            sd_pred = []
            
            if d == 0:
                d_m_mh = torch.Tensor([[1] for _ in range(L * num_heads)])
            else:
                full_T = np.array(T_den.iloc[0, :]) * 100 # after multiply by 100
                d_T = np.zeros([1, d + 1, len(full_T)])
                d_T[:, 0, :] = T_den.iloc[0, :] # before multiply by 100
                for i in range(int(d/2)):
                    d_T[0, 2*i+1, :] = np.sin(10 ** (-4*(i + 1)/d) * full_T)
                    d_T[0, 2*i+2, :] = np.cos(10 ** (-4*(i + 1)/d) * full_T)
                d_T = torch.Tensor(d_T)
                d_m_mh = torch.Tensor([[1] * L] * num_heads)
                
            # i = 0
            for i in range(n):
                if i % 10 == 0 and my_computer:
                    print(i, flush = True)
                src_mask = [1] * m
                src_mask[M_obs[i]:] = [0] * (m - M_obs[i])
                x = X_obs.iloc[i, :]
                t = T_obs.iloc[i, :]
                
                if d == 0:
                    e_X = torch.Tensor([[x.to_list(), t.to_list()] for _ in range(L)]).to(device)
                    d_T = torch.Tensor([[[t]] for t in T_den.iloc[i, :].to_list()]).to(device)
                    src_M = torch.Tensor([src_mask for _ in range(L)])
                    e_m_mh = torch.repeat_interleave(src_M, torch.tensor([num_heads] * src_M.shape[0]), dim = 0).to(device)
                else:
                    full_T = np.array(t) * 100 # after multiply by 100
                    e_XT = np.zeros([1, d + 1, len(full_T)])
                    e_XT[:, 0, :] = t # before multiply by 100
                    for i in range(int(d/2)):
                        e_XT[0, 2*i+1, :] = np.sin(10 ** (-4*(i + 1)/d) * full_T)
                        e_XT[0, 2*i+2, :] = np.cos(10 ** (-4*(i + 1)/d) * full_T)
                    
                    e_X = torch.Tensor(np.concatenate((np.array([[x]]), e_XT), axis = 1))
                    e_m_mh = torch.Tensor(src_mask).reshape(1, -1)
                    # e_m_mh = torch.repeat_interleave(e_m_mh, num_heads, dim = 0).to(device)
                
                if denoise_method == "MLE":
                    out = model.forward(e_X, d_T, e_m_mh, d_m_mh).squeeze(1).reshape(1, -1, 2)
                    out_sd = np.sqrt(np.exp(out[:, :, 1])).reshape(-1)
                    sd_pred.append(out_sd.cpu().numpy())
                    out = out[:, :, 0].reshape(-1)
                else:
                    out = model.forward(e_X, d_T, e_m_mh, d_m_mh).squeeze(1).reshape(-1)
                    
                # if add_random:
                #     e_X = torch.Tensor([[x.to_list(), t.to_list()]]).to(device)
                #     d_T = torch.Tensor([T_den.iloc[i, :]]).reshape(1, -1, 1)
                #     e_m_mh = torch.Tensor([src_mask])
                #     d_m_mh = torch.Tensor([[1 for _ in range(L * num_heads)]])
                #     out = model.forward(e_X, d_T, e_m_mh, d_m_mh).squeeze(1)
                # else:
                #     e_X = torch.Tensor([[x.to_list(), t.to_list()] for _ in range(L)]).to(device)
                #     d_T = torch.Tensor([[[t]] for t in T_den.iloc[i, :].to_list()]).to(device)
                #     src_M = torch.Tensor([src_mask for _ in range(L)])
                #     e_m_mh = torch.repeat_interleave(src_M, torch.tensor([num_heads] * src_M.shape[0]), dim = 0).to(device)
                #     
                #     if denoise_method == "MLE":
                #         out = model.forward(e_X, d_T, e_m_mh, d_m_mh).squeeze(1)
                #         out_sd = np.sqrt(np.exp(out[:, 1]))
                #         sd_pred.append(out_sd.cpu().numpy())
                #         out = out[:, 0]
                #     else:
                #         out = model.forward(e_X, d_T, e_m_mh, d_m_mh).squeeze(1)
                X_pred.append(out.cpu().numpy())
            
            err = []
            plt.clf()
            for i in range(n):
                err.append(np.mean(np.power(np.array(X_pred[i]) - np.array(X_den.iloc[i, :].to_list()), 2)))
                if i % 5 == 0:
                    Tnow = T_obs.iloc[i, :M_obs[i]]
                    Xnow = X_obs.iloc[i, :M_obs[i]]
                    
                    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)
                    plt.gca().set_ylim([-7, 7]);
                    plt.scatter(Tnow, Xnow, label = "obs");
                    plt.plot(t_true, X_pred[i] , label = "predicted");
                    plt.plot(t_true, X_den.iloc[i, :].to_list(), label = "true");
                    if denoise_method == "MLE":
                        plt.plot(t_true, X_pred[i] + 1.96 * sd_pred[i], color = "c");
                        plt.plot(t_true, X_pred[i] - 1.96 * sd_pred[i], color = "c");
                    
                    plt.ylabel("denoise: " + str(denoise_method), fontsize = 32);
                    plt.legend(fontsize = 20);
                    if add_random:
                        plt.savefig("../plots/investigate" + sparsity_error_folder + "/addW_" + str(denoise_method) + "_testing_" + str(i) + ".png");
                    else:
                        plt.savefig("../plots/investigate" + sparsity_error_folder + "/" + str(denoise_method) + "_testing_" + str(i) + ".png");
                    if my_computer:
                        plt.show();
                    plt.close();
            
            error_mat[counter_row, counter_col] = np.mean(err)
            counter_col += 1
        counter_row +=1

print(pd.DataFrame(error_mat, list(("DwE", "SwE", "DwoE", "SwoE")), denoise_method_list))
