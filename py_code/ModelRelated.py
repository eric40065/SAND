import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# class Object(object):
#     pass
# self = Object()

def DAE_sd_scheduler(k):
    return torch.tensor(1.0)
    ## linearly goes up to linearMAX. Then sigmoid.
    # linearMAX = 100
    # rate = 50
    # warm_up_to = 1/(1 + torch.exp(torch.tensor(-(0 - 250)/rate)))
    # # warm up
    # if k < linearMAX:
    #     return (k/linearMAX) ** 2 * warm_up_to
    # else:
    #     return 1/(1 + torch.exp(torch.tensor(-(k - linearMAX - 250)/rate)))

def get_penalty(penalty_counter, score, penalty_list):
    # find min of score
    if penalty_counter == 0:
        return 1e+1
    elif penalty_counter < 3:
        return penalty_list[-1] * (1e-4)
    else:
        # If minima locates at end points, go deeper
        rank_penalty_list = np.argsort(-np.array(penalty_list)).tolist()
        sorted_score = [score.copy()[rank_penalty_list[i]] for i in range(len(penalty_list))]
        sorted_penalty_list = [penalty_list.copy()[rank_penalty_list[i]] for i in range(len(penalty_list))]
        min_ind = np.argmin(sorted_score)
        if min_ind + 1 == len(sorted_score) and penalty_counter < 8:
            return sorted_penalty_list[-1] * (1e-2)
        else:
            min_ind = min_ind if penalty_counter < 8 else np.argmin(sorted_score[:(penalty_counter-1)])
            lower_thres = max(np.log10(sorted_penalty_list[min_ind + 1]), -10)
            upper_thres = min(np.log10(sorted_penalty_list[min_ind - 1]), 2)
            return np.power(10, np.random.uniform(upper_thres, lower_thres))

def compute_loss(y_pred, y, mask, iteration, isTraining = True, check_loss = None, compare2y = True):
    # y_pred, mask, check_loss = out, d_m, model_settings["check_loss"]
    # y_pred, mask = valid_y, d_m
    # dim - (B, J)
    # mean square error
    report_smoothing = isinstance(y_pred, list)
    [smooth, original] = y_pred if report_smoothing else [0, y_pred]
    
    if isinstance(smooth, torch.Tensor):
        if compare2y:
            smoothMSE = torch.pow(torch.sum((smooth[:, :, 0] - y) ** 2 * mask) / torch.sum(mask), 1/2)
        else:
            # mask_total_len = (mask.size()[0] * mask.size()[1])
            # smoothMSE = torch.pow(torch.sum((smooth[:, :, 0] - original[:, :, 0]) ** 2) / mask_total_len, 1/2)
            smoothMSE = torch.pow(torch.sum((smooth[:, :, 0] - original[:, :, 0]) ** 2 * mask) / torch.sum(mask), 1/2)
        smoothMSE_train = smoothMSE.clone()
        # if check_loss is not None:
        #     for i, check in enumerate(check_loss):
        #         if compare2y:
        #             pre_err = smooth[:, :, i + 1] - y
        #             smoothMSE_train += torch.sum((check - ((pre_err < 0) + 0)) * pre_err * mask) / torch.sum(mask)
        #         else:
        #             smoothMSE_train += torch.sum((smooth[:, :, i + 1] - original[:, :, i + 1]) ** 2 * mask) / mask_total_len
        #     smoothMSE_train /= (len(check_loss) + 1)
    
    originalMSE = torch.pow(torch.sum((original[:, :, 0] - y) ** 2 * mask) / torch.sum(mask), 1/2)
    originalMSE_train = originalMSE.clone()
    if check_loss is not None:
        for i, check in enumerate(check_loss):
            pre_err = original[:, :, i + 1] - y
            originalMSE_train += torch.sum((check - ((pre_err < 0) + 0)) * pre_err * mask) / torch.sum(mask)
        originalMSE_train /= (len(check_loss) + 1)
    if isTraining:
        if iteration < 2000:
            return originalMSE_train * 0.9 + smoothMSE_train * 0.1 if report_smoothing else originalMSE_train
        else:
            return torch.max(smoothMSE_train, originalMSE_train) if report_smoothing else originalMSE_train
        # return (smoothMSE_train + originalMSE_train)/2 if report_smoothing else originalMSE_train
    else:
        return [(originalMSE + smoothMSE)/2, smoothMSE] if report_smoothing else [originalMSE, originalMSE]

class Norm(nn.Module):
    def __init__(self, d, axis = -2, eps = 1e-6):
        super().__init__()
        self.d = d
        self.axis = axis
        # create two learnable parameters to calibrate normalisation
        if axis == -2:
            self.alpha = nn.Parameter(torch.randn((d, 1)))
            self.bias = nn.Parameter(torch.randn((d, 1)))
        else:
            self.alpha = nn.Parameter(torch.ones(d))
            self.bias = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x):
        # print(x.size(), self.size)
        avg = x.mean(dim=self.axis, keepdim = True)
        std = x.std(dim=self.axis, keepdim = True) + self.eps
        norm = self.alpha * (x - avg) / std + self.bias
        return norm

class FeedForward(nn.Module):
    def __init__(self, f_in, f_out, hidden = 64, dropout = 0.1):
        super().__init__()
        self.hidden = hidden
        self.lin1 = nn.Linear(f_in, self.hidden)
        self.lin2 = nn.Linear(self.hidden, self.hidden)
        self.lin3 = nn.Linear(self.hidden, f_out)
        self.norm1 = Norm(self.hidden, axis = -1)
        self.norm2 = Norm(self.hidden, axis = -1)
        self.norm3 = Norm(f_out, axis = -1)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x_ = F.relu(self.lin1(x))
        x_ = x_ + F.relu(self.lin2(x_))
        x_ = self.norm1(x_)
        x_ = x_ + F.relu(self.lin2(self.drop1(x_)))
        x_ = self.norm2(x_)
        x_ = F.relu(self.lin3(self.drop2(x_)))
        return self.norm3(x_)

def evaluation(data_name, iidt, output_structure = "SAND", d = 60, split = (90, 5, 5), cuda_device = "cpu"):
    ## Get the data. 
    real = (data_name == "/UK") # A useful indicator that records if the data is simulated
    
    ## Define the device
    device = torch.device(cuda_device)
    
    ## Define the model
    output_structure_folder = "/" + output_structure
    
    if output_structure == "Vanilla":
        denoise_method_list = ["l2w", "l2o"]
    elif output_structure == "SelfAtt":
        denoise_method_list = ["l2w"]
    elif output_structure == "SAND":
        denoise_method_list = ["l2w"]
    
    ## d and split must be the same as in the dataloader defined from train.py
    d, split = 60, (90, 5, 5)
    
    counter_row = 0
    error_mat_train_org = np.zeros((2 if real else 4, len(denoise_method_list)))
    error_mat_test_org = np.zeros((2 if real else 4, len(denoise_method_list)))
    error_mat_train_smooth = np.zeros((2 if real else 4, len(denoise_method_list)))
    error_mat_test_smooth = np.zeros((2 if real else 4, len(denoise_method_list)))
    
    if not real:
        X_den = np.array(pd.read_csv("../Data/IID/Simulation" + data_name + "/X_full_true.csv", header = None)) ## true but with noise
    else:
        X_den = np.array(pd.read_csv("../Data/IID/RealData" + data_name + "/X_full_noise.csv", header = None)) ## true but with noise
    
    for error in list((True, False)):
        for data_is_dense in list((True, False)):
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
            
            index_training = np.array(range(0, int(n - n_test), int((n - n_test)/500)))
            index_testing = np.array(range(n - n_test, n))
            index_all = np.unique(np.append(index_training, index_testing))
            
            T_den = np.array(pd.read_csv("../Data/IID" + ("/RealData" if real else "/Simulation") + data_name + "/T_full.csv", header=None)) ## true
            t_true = T_den[0, :]
            L = len(t_true)
            counter_col = 0
            for denoise_method in denoise_method_list:
                # load model
                checkpoint = torch.load("../Checkpoints" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + output_structure_folder + sparsity_error_folder + "/best_ckpts_" + str(denoise_method) + ".pth", map_location = torch.device(cuda_device))
                model = checkpoint["model"]
                model.load_state_dict(checkpoint["state_dict"])
                model.model_settings["device"] = torch.device(cuda_device)
                for parameter in model.parameters():
                    parameter.requires_grad = False
                
                # send model to CPU
                model = model.to(device)
                model = model.eval()
                num_heads = model.model_settings["num_heads"]
                
                org_pred = []
                smooth_pred = []
                
                d_T = np.zeros([1, d + 1, L])
                d_T[:, 0, :] = t_true
                for i in range(int(d/4)):
                    d_T[0, 4*i+1, :] = np.sin(10 ** (- 4*(i + 1)/d) * t_true * (L - 1))
                    d_T[0, 4*i+2, :] = np.cos(10 ** (- 4*(i + 1)/d) * t_true * (L - 1))
                    d_T[0, 4*i+3, :] = np.sin(2 * np.pi * (i + 1) * t_true)
                    d_T[0, 4*i+4, :] = np.cos(2 * np.pi * (i + 1) * t_true)
                d_T = torch.Tensor(d_T)
                d_m_mh = torch.Tensor([[1] * L] * num_heads)
                d_m = torch.Tensor([[1] * L])
                prob_list = torch.Tensor(np.array([np.exp((1 - torch.abs(torch.Tensor(range(-L, L + 1))/L)) * i).numpy() for i in [1, 2, 5, 10, 20, 50, 80]]))
                prob_list = prob_list/prob_list.sum(1, keepdim = True) + (2e-16)
                
                for counter, i in enumerate(index_all):
                    if counter % 200 == 0:
                        print("Complete", counter, "out of", len(index_all), flush = True)
                    src_mask = [1] * m
                    src_mask[M_obs[i]:] = [0] * (m - M_obs[i])
                    x = X_obs.iloc[i, :]
                    t = np.array(T_obs.iloc[i, :])
                    
                    e_XT = np.zeros([1, d + 1, len(t)])
                    e_XT[:, 0, :] = t # before multiply by 100
                    for j in range(int(d/4)):
                        e_XT[0, 4*j+1, :] = np.sin(10 ** (- 4*(j + 1)/d) * t * (L - 1))
                        e_XT[0, 4*j+2, :] = np.cos(10 ** (- 4*(j + 1)/d) * t * (L - 1))
                        e_XT[0, 4*j+3, :] = np.sin(2 * np.pi * (j + 1) * t)
                        e_XT[0, 4*j+4, :] = np.cos(2 * np.pi * (j + 1) * t)
                    e_X = torch.Tensor(np.concatenate((np.array([[x]]), e_XT), axis = 1))
                    e_m = torch.Tensor(src_mask).reshape(1, -1)
                    e_m_mh = torch.repeat_interleave(e_m, num_heads, dim = 0)
                    
                    if output_structure == "Vanilla":
                        org = model.forward(e_X, d_T, e_m_mh, e_m_mh, d_m_mh, d_m)
                    elif output_structure == "SelfAtt":
                        [smooth, org] = model.forward(e_X, d_T, e_m_mh, e_m_mh, d_m_mh, d_m, TAs_position = 0)
                        smooth_pred.append(smooth.squeeze(0).cpu().numpy())
                    else:
                        TAs_position = np.array(e_X[0, 1, :] * (L - 1), dtype = int)
                        TAs_position = torch.Tensor(TAs_position[np.diff(np.concatenate(([-1], TAs_position))) > 0]).unsqueeze(-1)
                        [smooth, org] = model.forward(e_X, d_T, e_m_mh, e_m_mh, d_m_mh, d_m, TAs_position = TAs_position)
                        weight = torch.zeros(smooth.size()[0], smooth.size()[1], len(prob_list))
                        for prob_counter, prob in enumerate(prob_list):
                            for j, pos in enumerate(TAs_position.squeeze(-1).long()):
                                weight[j, :, prob_counter] = prob[(L - pos + 1):(2 * L - pos + 1)]
                                
                        weight = weight/weight.sum(0)
                        smooth_prob = torch.zeros(1, smooth.size()[1], smooth.size()[2], len(prob_list))
                        for prob_counter, _ in enumerate(prob_list):
                            smooth_prob[:, :, :, prob_counter] = torch.sum(smooth * weight[:, :, prob_counter].unsqueeze(-1), dim = 0)
                        smooth_pred.append(smooth_prob.squeeze(0).cpu().numpy())
                    org_pred.append(org.squeeze(0).cpu().numpy())
                
                org_pred = np.array(org_pred)
                smooth_pred = np.array(smooth_pred)
                
                if output_structure == "SAND":
                    err_smooth2 = []
                    for i, index in enumerate(index_all):
                        err_smooth2.append(np.mean(np.power(smooth_pred[i, :, 0] - X_den[index, :].reshape(-1, 1), 2), axis = 0))
                    smooth_pred = np.array([smooth_pred[i, :, :, np.argmin(err_smooth2[i])] for i in range(smooth_pred.shape[0])])
                    # [np.argmin(err_smooth2[i]) for i in range(smooth_pred.shape[0])]
                
                mat_filename = "../ImputedData" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + output_structure_folder + sparsity_error_folder + "/X_imputed_" + denoise_method + ".csv"
                np.savetxt(mat_filename, smooth_pred[:, :, 0], delimiter=",") if output_structure != "Vanilla" else np.savetxt(mat_filename, org_pred[:, :, 0], delimiter=",")
                mat_filename10 = "../ImputedData" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + output_structure_folder + sparsity_error_folder + "/X_imputed_10" + denoise_method + ".csv"
                mat_filename90 = "../ImputedData" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + output_structure_folder + sparsity_error_folder + "/X_imputed_90" + denoise_method + ".csv"
                np.savetxt(mat_filename10, smooth_pred[:, :, 0] - (org_pred[:, :, 2] - org_pred[:, :, 1])/2, delimiter=",") if output_structure != "Vanilla" else np.savetxt(mat_filename, org_pred[:, :, 1], delimiter=",")
                np.savetxt(mat_filename90, smooth_pred[:, :, 0] + (org_pred[:, :, 2] - org_pred[:, :, 1])/2, delimiter=",") if output_structure != "Vanilla" else np.savetxt(mat_filename, org_pred[:, :, 2], delimiter=",")
                
                err_org = []
                err_smooth = []
                plt.clf()
                tmp_X = np.array(X_den)[index_all, ]
                ylim = [np.min(tmp_X), np.max(tmp_X)]
                for i, index in enumerate(index_all):
                    err_org.append(np.mean(np.power(org_pred[i, :, 0] - X_den[index, :], 2)))
                    if output_structure != "Vanilla":
                        err_smooth.append(np.mean(np.power(smooth_pred[i, :, 0] - X_den[index, :], 2)))
                    
                    if (i + 1) % 100 == 0 and index > (5000 if real else 9450):
                        Tnow = T_obs.iloc[index, :M_obs[index]]
                        Xnow = X_obs.iloc[index, :M_obs[index]]
                        
                        plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.1)
                        plt.gca().set_ylim(ylim);
                        plt.gca().set_xlim([-0.03, 1.03]);
                        plt.scatter(Tnow, Xnow, label = "obs.");
                        if output_structure == "Vanilla":
                            plt.plot(t_true, org_pred[i, :, 0], label = "VT");
                            plt.fill_between(t_true, org_pred[i, :, 1], org_pred[i, :, 2], alpha=0.12, color = "blue");
                        if output_structure != "Vanilla":
                            plt.plot(t_true, org_pred[i, :, 0], label = "Org");
                            plt.plot(t_true, smooth_pred[i, :, 0], label = "Self-Att" if output_structure == "SelfAtt" else "SAND");
                            gaps = (org_pred[i, :, 2] - org_pred[i, :, 1])/2
                            plt.fill_between(t_true, smooth_pred[i, :, 0] - gaps, smooth_pred[i, :, 0] + gaps, alpha=0.12, color = "blue");
                        plt.plot(t_true, X_den[index, :], label = "True", color = "orange");
                            
                        if output_structure == "Vanilla":
                            plt.ylabel(output_structure + "/" + denoise_method, fontsize = 32)
                        else:
                            plt.ylabel(("SAND" if output_structure == "SAND" else output_structure), fontsize = 32)
                        plt.legend(fontsize = 20)
                        plt.savefig("../Plots" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + "/ImputedCurves" + sparsity_error_folder + "/" + str(i + 1) + output_structure + denoise_method + ".png")
                        plt.show()
                        plt.close()
                        
                error_mat_train_org[counter_row, counter_col] = np.mean(err_org[:len(index_training)])
                error_mat_test_org[counter_row, counter_col] = np.mean(err_org[len(index_training):])
                if output_structure != "Vanilla":
                    error_mat_train_smooth[counter_row, counter_col] = np.mean(err_smooth[:len(index_training)])
                    error_mat_test_smooth[counter_row, counter_col] = np.mean(err_smooth[len(index_training):])
                
                counter_col += 1
            counter_row +=1
        if real:
            break
    if output_structure == "Vanilla":
        return {"trainingLoss": error_mat_train_org, "testingLoss": error_mat_test_org}
    else:
        return {"trainingLoss": error_mat_train_smooth, "testingLoss": error_mat_test_smooth}



