for name in dir():
    if not name.startswith("_"):
        del globals()[name]

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import torch.nn.functional as F
from scipy.stats import norm

d, split = 60, (90, 5, 5)
data_name = "/LowDim_T" # "UK" "Framingham" "LowDim_G" "HighDim_G" "LowDim_E" "HighDim_E" "LowDim_T" "HighDim_T"
iidt = True
# iidt = False
iidt = True if data_name == "/Framingham" else iidt

## Define model
output_structure = "SAND" # "Vanilla" "SelfAtt" "SAND"
output_structure_folder = "/" + output_structure
real = True if data_name == "/Framingham" or data_name == "/UK" else False

if output_structure == "Vanilla":
    denoise_method_list = ["None", "l1w", "l2w", "l2o", "TVo"]
    denoise_method_list = ["l2w", "l2o"]
    denoise_method_list = ["None"]
elif output_structure == "SelfAtt":
    denoise_method_list = ["l2w"]
elif output_structure == "SAND":
    denoise_method_list = ["l2w"]

my_computer = True
server_specified_folder = "/Users/eric/Desktop/UCD/TransFD/py_code/" if my_computer else "/home/jshong96/TransFD/py_code/"
os.chdir(server_specified_folder)

device = torch.device("cpu")
counter_row = 0
error_mat_train_org = np.zeros((4, len(denoise_method_list)))
error_mat_test_org = np.zeros((4, len(denoise_method_list)))
error_mat_train_smooth = np.zeros((4, len(denoise_method_list)))
error_mat_test_smooth = np.zeros((4, len(denoise_method_list)))

if not real:
    error_mat_train_org_noi = np.zeros((4, len(denoise_method_list)))
    error_mat_test_org_noi = np.zeros((4, len(denoise_method_list)))
    error_mat_train_smooth_noi = np.zeros((4, len(denoise_method_list)))
    error_mat_test_smooth_noi = np.zeros((4, len(denoise_method_list)))

if not real:
    X_den = np.array(pd.read_csv("../Data/IID/Simulation" + data_name + "/X_full_true.csv", header = None)) ## true but with noise
    X_den_noi = np.array(pd.read_csv("../Data/IID/Simulation" + data_name + "/X_full_noise.csv", header = None)) ## true but with noise
elif data_name == "/Framingham":
    X_den = np.array(pd.read_csv("../Data/IID/RealData/Framingham/X_obs.csv", header = None).iloc[:, 1:]) ## true but with noise
else:
    X_den = np.array(pd.read_csv("../Data/IID/RealData/UK/X_full_noise.csv", header = None)) ## true but with noise

for error in list((True, False)):
    for data_is_dense in list((True, False)):
        # data_is_dense, error = True, True
        # load data
        data_type = "/dense" if data_is_dense else "/sparse"
        sparsity_error_folder = (data_type + "/w_error" if error else data_type + "/wo_error") if not real else data_type
        sparsity_error_folder = "" if data_name == "/Framingham" else sparsity_error_folder
        
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
        
        if data_name != "/Framingham":
            T_den = np.array(pd.read_csv("../Data/IID" + ("/RealData" if real else "/Simulation") + data_name + "/T_full.csv", header=None)) ## true
            t_true = T_den[0, :]
            L = len(t_true)
        else:
            # data_name = "/Framingham"
            t_true = np.unique(T_obs)
            L = len(t_true)
            index_obs = np.array(T_obs * (L - 1), dtype = int)
        counter_col = 0
        for denoise_method in denoise_method_list:
            # denoise_method = "l2w"
            # load model
            checkpoint = torch.load("../Checkpoints" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + output_structure_folder + sparsity_error_folder + "/best_ckpts_" + str(denoise_method) + ".pth", map_location = torch.device('cpu'))
            # checkpoint = torch.load("../Checkpoints" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + output_structure_folder + sparsity_error_folder + "/ckpts_l2w_3000.pth", map_location = torch.device('cpu'))
            model = checkpoint["model"]
            model.load_state_dict(checkpoint["state_dict"])
            model.model_settings["device"] = torch.device("cpu")
            for parameter in model.parameters():
                parameter.requires_grad = False
            
            # send model to CPU
            model = model.to(device)
            model = model.eval()
            num_heads = model.model_settings["num_heads"]
            
            org_pred = []
            smooth_pred = []
            
            d_T = np.zeros([1, d + 1, L])
            d_T[:, 0, :] = t_true # before multiply by 100
            for i in range(int(d/4)):
                d_T[0, 4*i+1, :] = np.sin(10 ** (- 4*(i + 1)/d) * t_true * (L - 1))
                d_T[0, 4*i+2, :] = np.cos(10 ** (- 4*(i + 1)/d) * t_true * (L - 1))
                d_T[0, 4*i+3, :] = np.sin(2 * np.pi * (i + 1) * t_true)
                d_T[0, 4*i+4, :] = np.cos(2 * np.pi * (i + 1) * t_true)
            
            d_T = torch.Tensor(d_T)
            d_m_mh = torch.Tensor([[1] * L] * num_heads)
            d_m = torch.Tensor([[1] * L])
            prob_list = np.exp((1 - torch.abs(torch.Tensor(range(-L, L + 1))/L)) * 20)
            prob_list = prob_list/prob_list.sum() + (2e-16)
            
            # i = 0
            for counter, i in enumerate(index_all):
                if counter % 200 == 0 and my_computer:
                    print(counter, flush = True)
                src_mask = [1] * m
                src_mask[M_obs[i]:] = [0] * (m - M_obs[i])
                x = X_obs.iloc[i, :]
                t = np.array(T_obs.iloc[i, :])
                
                e_XT = np.zeros([1, d + 1, len(t)])
                e_XT[:, 0, :] = t # before multiply by 100
                for j in range(int(d/4)):
                    e_XT[0, 4*j+1, :] = np.sin(10 ** (2 - 4*(j + 1)/d) * t)
                    e_XT[0, 4*j+2, :] = np.cos(10 ** (2 - 4*(j + 1)/d) * t)
                    e_XT[0, 4*j+3, :] = np.sin(2 * np.pi * (j + 1) * t)
                    e_XT[0, 4*j+4, :] = np.cos(2 * np.pi * (j + 1) * t)
                e_X = torch.Tensor(np.concatenate((np.array([[x]]), e_XT), axis = 1))
                e_m = torch.Tensor(src_mask).reshape(1, -1)
                e_m_mh = torch.repeat_interleave(e_m, num_heads, dim = 0)
                
                if output_structure == "Vanilla":
                    org = model.forward(e_X, d_T, e_m_mh, e_m_mh, d_m_mh, d_m)
                elif output_structure == "SelfAtt":
                    [smooth, org] = model.forward(e_X, d_T, e_m_mh, e_m_mh, d_m_mh, d_m, TAs_position = 0)
                else:
                    TAs_position = np.array(e_X[0, 1, :] * (L - 1), dtype = int)
                    TAs_position = torch.Tensor(TAs_position[np.diff(np.concatenate(([-1], TAs_position))) > 0]).unsqueeze(-1)
                    # d_m = torch.zeros(1, L)
                    # d_m[0, TAs_position.squeeze(-1).long()] = 1
                    # d_m_mh = torch.repeat_interleave(d_m, 2, dim = 0)
                    [smooth, org] = model.forward(e_X, d_T, e_m_mh, e_m_mh, d_m_mh, d_m, TAs_position = TAs_position)
                    weight = smooth[:, :, 0] * 0
                    for j, pos in enumerate(TAs_position.squeeze(-1).long()):
                        weight[j, :] = prob_list[(L - pos + 1):(2 * L - pos + 1)]
                        
                    weight = weight/torch.sum(weight, dim = 0)
                    smooth = torch.sum(smooth * weight.unsqueeze(-1), dim = 0)
                    # org = torch.sum(org * weight.unsqueeze(-1), dim = 0)
                # plt.clf()
                # plt.gca().set_ylim(ylim);
                # plt.scatter(t, x, label = "obs.");
                # plt.plot(t_true, org[0, :, 0], label = "Original");
                # plt.fill_between(t_true, org[0, :, 1], org[0, :, 2], alpha=0.2, color = "blue");
                # plt.plot(t_true, smooth[0, :, 0], label = "Original");
                # plt.fill_between(t_true, smooth[0, :, 1], smooth[0, :, 2], alpha=0.2, color = "orange");
                # plt.show()
                
                org_pred.append(org.squeeze(0).cpu().numpy())
                if output_structure != "Vanilla":
                    smooth_pred.append(smooth.squeeze(0).cpu().numpy())
            
            org_pred = np.array(org_pred)
            smooth_pred = np.array(smooth_pred)
            mat_filename = "../ImputedData" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + output_structure_folder + sparsity_error_folder + "/X_imputed_" + denoise_method + ".csv"
            np.savetxt(mat_filename, smooth_pred[:, :, 0], delimiter=",") if output_structure != "Vanilla" else np.savetxt(mat_filename, org_pred[:, :, 0], delimiter=",")
            mat_filename10 = "../ImputedData" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + output_structure_folder + sparsity_error_folder + "/X_imputed_10" + denoise_method + ".csv"
            mat_filename90 = "../ImputedData" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + output_structure_folder + sparsity_error_folder + "/X_imputed_90" + denoise_method + ".csv"
            np.savetxt(mat_filename10, org_pred[:, :, 1], delimiter=",")
            np.savetxt(mat_filename90, org_pred[:, :, 2], delimiter=",")
            
            err_org = []
            err_smooth = []
            err_org_noi = []
            err_smooth_noi = []
            plt.clf()
            tmp_X = np.array(X_den)[index_all, ]
            ylim = [np.min(tmp_X), np.max(tmp_X)]
            for i, index in enumerate(index_all):
                # stop
                if data_name == "/Framingham":
                    err_org.append(np.mean(np.power(org_pred[i, index_obs[index, :M_obs[index]], 0] - X_den[index, :M_obs[index]], 2)))
                    if output_structure != "Vanilla":
                        err_smooth.append(np.mean(np.power(smooth_pred[i, index_obs[index, :M_obs[index]], 0] - X_den[index, :M_obs[index]], 2)))
                else:
                    err_org.append(np.mean(np.power(org_pred[i, :, 0] - X_den[index, :], 2)))
                    if output_structure != "Vanilla":
                        err_smooth.append(np.mean(np.power(smooth_pred[i, :, 0] - X_den[index, :], 2)))
                if not real:
                    err_org_noi.append(np.mean(np.power(org_pred[i, :, 0] - X_den_noi[index, :], 2)))
                    if output_structure != "Vanilla":
                        err_smooth_noi.append(np.mean(np.power(smooth_pred[i, :, 0] - X_den_noi[index, :], 2)))
                if (i + 1) % 100 == 0 and index > (3000 if real else 9450):
                    Tnow = T_obs.iloc[index, :M_obs[index]]
                    Xnow = X_obs.iloc[index, :M_obs[index]]
                    
                    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.1)
                    plt.gca().set_ylim(ylim);
                    plt.gca().set_xlim([-0.03, 1.03]);
                    plt.scatter(Tnow, Xnow, label = "obs.");
                    if output_structure == "Vanilla":
                        # plt.plot(Tnow, org_pred[i, :len(Tnow), 0], label = "VT");
                        # plt.fill_between(Tnow, org_pred[i, :len(Tnow), 1], org_pred[i, :len(Tnow), 2], alpha=0.12, color = "blue");
                        plt.plot(t_true, org_pred[i, :, 0], label = "VT");
                        plt.fill_between(t_true, org_pred[i, :, 1], org_pred[i, :, 2], alpha=0.12, color = "blue");
                    if output_structure != "Vanilla":
                        plt.plot(t_true, org_pred[i, :, 0], label = "Org");
                        plt.plot(t_true, smooth_pred[i, :, 0], label = "Self-Att" if output_structure == "SelfAtt" else "SAND");
                        gaps = (org_pred[i, :, 2] - org_pred[i, :, 1])/2
                        plt.fill_between(t_true, smooth_pred[i, :, 0] - gaps, smooth_pred[i, :, 0] + gaps, alpha=0.12, color = "blue");
                    if data_name != "/Framingham":
                        plt.plot(t_true, X_den[index, :], label = "True", color = "orange");
                    
                    if output_structure == "Vanilla":
                        plt.ylabel(output_structure + "/" + denoise_method, fontsize = 32)
                    else:
                        plt.ylabel(("SAND" if output_structure == "SAND" else output_structure), fontsize = 32)
                    plt.legend(fontsize = 20)
                    plt.savefig("../Plots" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + "/ImputedCurves" + sparsity_error_folder + "/" + str(i + 1) + output_structure + denoise_method + ".png")
                    if my_computer:
                        plt.show()
                    plt.close()
            
            error_mat_train_org[counter_row, counter_col] = np.mean(err_org[:len(index_training)])
            error_mat_test_org[counter_row, counter_col] = np.mean(err_org[len(index_training):])
            if not real:
                error_mat_train_org_noi[counter_row, counter_col] = np.mean(err_org_noi[:len(index_training)])
                error_mat_test_org_noi[counter_row, counter_col] = np.mean(err_org_noi[len(index_training):])
            if output_structure != "Vanilla":
                error_mat_train_smooth[counter_row, counter_col] = np.mean(err_smooth[:len(index_training)])
                error_mat_test_smooth[counter_row, counter_col] = np.mean(err_smooth[len(index_training):])
                if not real:
                    error_mat_train_smooth_noi[counter_row, counter_col] = np.mean(err_smooth_noi[:len(index_training)])
                    error_mat_test_smooth_noi[counter_row, counter_col] = np.mean(err_smooth_noi[len(index_training):])
            
            counter_col += 1
        counter_row +=1
        if data_name == "/Framingham":
            break
    if real:
        break

if not real:
    print(pd.DataFrame(error_mat_train_org_noi, list(("DwE", "SwE", "DwoE", "SwoE")), denoise_method_list) * 1000)
    print(pd.DataFrame(error_mat_test_org_noi, list(("DwE", "SwE", "DwoE", "SwoE")), denoise_method_list) * 1000)
    if output_structure != "Vanilla":
        print(pd.DataFrame(error_mat_train_smooth_noi, list(("DwE", "SwE", "DwoE", "SwoE")), denoise_method_list) * 1000)
        print(pd.DataFrame(error_mat_test_smooth_noi, list(("DwE", "SwE", "DwoE", "SwoE")), denoise_method_list) * 1000)
print("compare to ture")
print(pd.DataFrame(error_mat_train_org, list(("DwE", "SwE", "DwoE", "SwoE")), denoise_method_list) * 1000)
print(pd.DataFrame(error_mat_test_org, list(("DwE", "SwE", "DwoE", "SwoE")), denoise_method_list) * 1000)
if output_structure != "Vanilla":
    print(pd.DataFrame(error_mat_train_smooth, list(("DwE", "SwE", "DwoE", "SwoE")), denoise_method_list) * 1000)
    print(pd.DataFrame(error_mat_test_smooth, list(("DwE", "SwE", "DwoE", "SwoE")), denoise_method_list) * 1000)

np.round(error_mat_train_smooth.transpose() * 1000, 1)
np.round(error_mat_test_org.transpose() * 1000, 3)
np.round(error_mat_test_smooth.transpose() * 1000, 3)



