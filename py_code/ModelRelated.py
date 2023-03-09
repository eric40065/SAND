import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# class Object(object):
#     pass
# self = Object()
# def cost_in_penalty(theta, x_now, y_now):
#     return sum(np.power(y_now - theta[0] - theta[1] * np.log10(x_now) - theta[2] * np.power(np.log10(x_now), 2), 2))
#     # return sum(np.power(y_now - theta[0] - theta[1] * x_now - theta[2] * np.power(x_now, 2), 2))

# score_full = [4.503307860793038, 4.572633649426255, 4.427903253544626, 4.501810760761321, 4.456755257213316, 4.417535571380106, 4.5727015610538855, 4.587370601089161, 4.491998198613332, 4.428160567675093]
# penalty_list_full = [10.0, 0.1, 0.001, 1e-05, 0.0025694462547773043, 0.0009647135748793082, 8.259455921628662e-05, 0.00022331817688784735, 0.00039343108106525544, 0.0008698593479279202]
# score = score_full[:10]
# penalty_list = penalty_list_full[:10]
def get_penalty(penalty_counter, score, penalty_list):
    # find min of score
    if penalty_counter == 0:
        return 1e+2
    elif penalty_counter < 4:
        return penalty_list[-1] * (1e-3)
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
        
        # penalty_list = np.array([5e-3, 5e-5, 5e-7, 5e-9, 7e-10])
        # score = np.array([41.12, 23.67, 2.413, 0.2382, 2.465])
        # x_now, y_now = penalty_list[:penalty_counter].flatten(), score[:penalty_counter].flatten()
        # # delete outlier
        # q3, q1 = np.percentile(y_now, [75, 25])
        # uf = np.median(y_now) + 1.5 * (q3 - q1)
        # outliers_ind = (y_now > uf)
        # x_now, y_now = np.delete(x_now, outliers_ind), np.delete(y_now, outliers_ind)
        # penalty_counter = len(x_now)
        # 
        # # testing code
        # # penalty_counter = 5
        # # x_now = np.array([5e-3, 5e-5, 5e-7, 5e-9, 5e-11])
        # # y_now = np.array([41.12, 23.67, 24.13, 23.82, 24.65])
        # theta = np.array((0, 0, 0)).reshape(3, 1)
        # theta_best = minimize(cost_in_penalty, theta, (x_now, y_now)).x
        # if theta_best[2] < 0:
        #     try_times = int(5e2)
        #     theta_try = np.zeros((3, try_times))
        #     for try_iter in range(try_times):
        #         try_index = np.random.choice(range(penalty_counter), max(penalty_counter - 2, 3), replace = False)
        #         x_try, y_try = x_now[try_index], y_now[try_index]
        #         theta_try[:, try_iter] = minimize(cost_in_penalty, theta, (x_try, y_try)).x
        #     theta_try = theta_try[:, theta_try[2, ] > 0]
        #     if theta_try.shape[1] == 0:
        #         return np.array([np.power(10, np.random.uniform(-11, -3))])
        #     else:
        #         theta_best = theta_try[:, theta_try[2, ].argsort()][:, int(theta_try.shape[1]/2)]
        # candidate = np.array([np.power(10, -theta_best[1]/(2 * theta_best[2]))])
        # if candidate < 1e-12 or candidate > 1e-2:
        #     candidate = np.array([np.power(10, np.random.uniform(-12, -3))])
        # return candidate
    # x_full = np.power(10, np.linspace(-15, -3, 100))
    # plt.clf()
    # plt.plot(np.log10(x_full), theta_best[0] + theta_best[1] * np.log10(x_full) + theta_best[2] * np.power(np.log10(x_full), 2))
    # plt.scatter(np.log10(x_now), y_now)
    # plt.show()
def compute_loss(y_pred, y, mask, isTraining = True, check_loss = None):
    # y_pred, mask, check_loss = out, d_m, model_settings["check_loss"]
    # y_pred, mask = valid_y, d_m
    # dim - (B, J)
    # mean square error
    report_res = True if isinstance(y_pred, list) else False
    [y_pred, org] = y_pred if isinstance(y_pred, list) else [0, y_pred]
    
    if isinstance(y_pred, torch.Tensor):
        res = torch.sum((y_pred[:, :, 0] - y) ** 2 * mask) / torch.sum(mask)
        res_train = res.clone()
        if check_loss is not None:
            for i, check in enumerate(check_loss):
                pre_err = y_pred[:, :, i + 1] - y
                res_train += torch.sum((check - ((pre_err < 0) + 0)) * pre_err * mask) / torch.sum(mask)
            res_train /= (i + 1)
    result = torch.sum((org[:, :, 0] - y) ** 2 * mask) / torch.sum(mask)
    result_train = result.clone()
    if check_loss is not None:
        for i, check in enumerate(check_loss):
            pre_err = org[:, :, i + 1] - y
            result_train += torch.sum((check - ((pre_err < 0) + 0)) * pre_err * mask) / torch.sum(mask)
        result_train /= (i + 1)
    if isTraining:
        return (res_train + result_train)/2 if report_res else result_train
    else:
        return res if report_res else result

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

