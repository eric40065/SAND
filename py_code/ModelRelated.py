import numpy as np
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


