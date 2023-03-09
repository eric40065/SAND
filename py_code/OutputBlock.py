import math
import pandas as pd
import numpy as np
import torch
from torch import nn
from Attention import MultiHeadAttention
from ModelRelated import Norm, FeedForward
# org_detach = org.detach().clone()

class TransformerOutput(nn.Module):
    """Transformer output"""
    def __init__(self, model_settings, output_structure):
        super(TransformerOutput, self).__init__()
        # model_settings_tmp = model_settings
        # model_settings = model_settings2
        self.model_settings = model_settings
        self.output_structure = output_structure
        self.outblks = nn.Sequential()
        self.sigmoid = nn.Sigmoid()
        self.alpha_smo, self.beta_smo = 1 * (model_settings["max_X"] - model_settings["min_X"]), model_settings["min_X"]
        
        for i in range(len(model_settings["check_loss"]) + 1):
            self.outblks.add_module("block" + str(i), MultiHeadAttention(model_settings, output_structure))
        if output_structure == "DiffSelfAtt":
            n_timepts = model_settings["tptsTraining"]
            self.timediff = 1/(n_timepts - 1)
            self.index_subs = np.repeat(range(model_settings["batch_size"]), n_timepts)
            self.range_pts = torch.arange(n_timepts, device = model_settings["device"]).unsqueeze(0)
            weight_rl = np.array(range(n_timepts))/(n_timepts - 1)
            weight_lr = 1 - weight_rl
            self.weight_lr_SL = torch.Tensor(weight_lr).to(model_settings["device"])
            self.weight_rl_SL = torch.Tensor(weight_rl).to(model_settings["device"])
            weight_lr_tmp = torch.Tensor(np.exp(weight_lr * 20))
            self.weight_lr = (weight_lr_tmp/weight_lr_tmp.sum() + (2e-16)).to(model_settings["device"])
        # model_settings = model_settings_tmp

    def forward(self, org_detach, y_t, d_m, iteration = 0, TAs_position = None, isTraining = True):
        # outblock_input - (B, seq_len, num_hiddens)
        # d_m            - (B, seq_len)
        device = self.model_settings["device"] if TAs_position is None else torch.device("cpu")
        # device = torch.device("cpu")
        smooth = torch.zeros(org_detach.size(), device = device)
        for i, blk in enumerate(self.outblks):
            # stop
            outblock_input = torch.cat([org_detach[:, :, i].unsqueeze(-1), y_t.transpose(-1, -2)], axis = -1)
            smooth[:, :, i] = blk(outblock_input, outblock_input, outblock_input, mask = d_m).squeeze(-1)
        
        if self.output_structure == "SelfAtt":
            return self.sigmoid(smooth) * self.alpha_smo + self.beta_smo
        else:
            [n_subs, n_timepts, _] = org_detach.size()
            if TAs_position == None:
                # iteration = 0; isTraining = True
                TAs_num = np.random.randint(low = 2, high = 15) if isTraining else 2
                TAs_position, _ = torch.sort(torch.multinomial(torch.ones(n_subs, int(n_timepts - 2)), int(TAs_num)) + 1)
                TAs_position = TAs_position.to(device)
                index_subs = self.index_subs if n_subs == self.model_settings["batch_size"] else np.repeat(range(n_subs), n_timepts)
            else:
                assert n_subs == 1
                n_subs = len(TAs_position)
                index_subs = np.repeat(0, n_timepts * n_subs)
            
            increments_lr = torch.cumsum(self.timediff * smooth, dim = 1) # increment compared to the first point
            TAs_position_lr = torch.cat((torch.zeros(n_subs, 1, device = device), TAs_position), dim = 1)
            repeats = torch.diff(torch.cat((TAs_position_lr, torch.full((n_subs, 1), n_timepts, device = device)), dim = 1), dim = 1).reshape(-1).long()
            index_timepts_lr = torch.repeat_interleave(TAs_position_lr.reshape(-1), repeats).long()
            smooth_lr = (org_detach[index_subs, index_timepts_lr, :] - increments_lr[index_subs, index_timepts_lr, :]).reshape(n_subs, n_timepts, -1) + increments_lr
            
            increments_rl = torch.cumsum(self.timediff * smooth[:, range(-1, -(n_timepts+1), -1), :], dim = 1)[:, range(-1, -(n_timepts+1), -1), :] # increment compared to the last point
            TAs_position_rl = torch.cat((TAs_position, torch.full((n_subs, 1), n_timepts - 1, device = device)), dim = 1)
            repeats = torch.diff(torch.cat((torch.full((n_subs, 1), -1, device = device), TAs_position_rl), dim = 1), dim = 1).reshape(-1).long()
            index_timepts_rl = torch.repeat_interleave(TAs_position_rl.reshape(-1), repeats).long()
            smooth_rl = (org_detach[index_subs, index_timepts_rl, :] + increments_rl[index_subs, index_timepts_rl, :]).reshape(n_subs, n_timepts, -1) - increments_rl
            
            TAs_position_lr = index_timepts_lr.reshape(n_subs, -1)
            TAs_position_rl = index_timepts_rl.reshape(n_subs, -1)
            weight_lr = self.weight_lr[self.range_pts - TAs_position_lr]
            weight_rl = self.weight_lr[TAs_position_rl - self.range_pts]
            weight_sum = weight_lr + weight_rl
            
            smooth_out = self.sigmoid(smooth_lr * (weight_lr/weight_sum).unsqueeze(-1) + smooth_rl * (weight_rl/weight_sum).unsqueeze(-1))
            return smooth_out * self.alpha_smo + self.beta_smo

class TransformerOutput_ver2(nn.Module):
    """Transformer output"""
    def __init__(self, model_settings, output_structure):
        super(TransformerOutput_ver2, self).__init__()
        # model_settings_tmp = model_settings
        # model_settings = model_settings2
        self.model_settings = model_settings
        self.output_structure = output_structure
        self.outblks = nn.Sequential()
        self.sigmoid = nn.Sigmoid()
        self.alpha_smo, self.beta_smo = 1 * (model_settings["max_X"] - model_settings["min_X"]), model_settings["min_X"]
        
        for i in range(len(model_settings["check_loss"]) + 1):
            self.outblks.add_module("block" + str(i), MultiHeadAttention(model_settings, output_structure))
        if output_structure == "DiffSelfAtt":
            n_timepts = model_settings["tptsTraining"]
            self.index_subs = np.repeat(range(model_settings["batch_size"]), n_timepts)
            self.range_pts = torch.arange(n_timepts, device = model_settings["device"]).unsqueeze(0)
            
            n_tpts_testing = model_settings["tptsTesting"]
            weight_rl = np.array(range(n_tpts_testing))/(n_tpts_testing - 1)
            weight_lr = 1 - weight_rl
            self.weight_lr_SL = torch.Tensor(weight_lr).to(model_settings["device"])
            self.weight_rl_SL = torch.Tensor(weight_rl).to(model_settings["device"])
            weight_lr_tmp = torch.Tensor(np.exp(weight_lr * 20))
            self.weight_lr = (weight_lr_tmp/weight_lr_tmp.sum() + (2e-16)).to(model_settings["device"])
        # model_settings = model_settings_tmp
        
    def forward(self, org_detach, y_t, d_m, iteration = 0, TAs_position = None, isTraining = True):
        # outblock_input - (B, seq_len, num_hiddens)
        # d_m            - (B, seq_len)
        device = self.model_settings["device"] if TAs_position is None else torch.device("cpu")
        # device = torch.device("cpu")
        smooth = torch.zeros(org_detach.size(), device = device)
        for i, blk in enumerate(self.outblks):
            # stop
            outblock_input = torch.cat([org_detach[:, :, i].unsqueeze(-1), y_t.transpose(-1, -2)], axis = -1)
            smooth[:, :, i] = blk(outblock_input, outblock_input, outblock_input, mask = d_m).squeeze(-1)
        
        # for i, blk in enumerate(self.outblock.outblks):
        #     # stop
        #     outblock_input = torch.cat([org_detach[:, :, i].unsqueeze(-1), y_t.transpose(-1, -2)], axis = -1)
        #     smooth[:, :, i] = blk(outblock_input, outblock_input, outblock_input, mask = d_m).squeeze(-1)
        
        if self.output_structure == "SelfAtt":
            return self.sigmoid(smooth) * self.alpha_smo + self.beta_smo
        else:
            [n_subs, n_timepts, _] = org_detach.size()
            M_obs = (torch.sum(d_m, dim = 1) - 1).long().reshape(n_subs, -1) # Since we want index, we minus 1.
            if TAs_position == None:
                # iteration = 0; isTraining = True
                TAs_num = np.random.randint(low = 2, high = n_timepts/2) if isTraining else 2
                TAs_position, _ = torch.sort(torch.multinomial(torch.ones(n_subs, int(n_timepts - 2)), int(TAs_num)) + 1)
                TAs_position = TAs_position.to(device)
                TAs_position[TAs_position > M_obs] = torch.repeat_interleave(M_obs, torch.sum(TAs_position > M_obs, dim = 1))
                index_subs = self.index_subs if n_subs == self.model_settings["batch_size"] else np.repeat(range(n_subs), n_timepts)
                zeros = torch.zeros(n_subs, 1, 3, device = device)
            else:
                assert n_subs == 1
                n_subs = len(TAs_position)
                index_subs = np.repeat(0, n_timepts * n_subs)
                zeros = torch.zeros(1, 1, 3, device = device)
                M_obs = torch.repeat_interleave(M_obs, len(TAs_position), dim = 0)
            
            smooth = smooth[:, 1:, :]
            time_diff = torch.diff(y_t[:, 0, :], dim = 1)
            changes = (time_diff * d_m[:, 1:]).unsqueeze(2) * smooth
            
            increments_lr = torch.cat((zeros, torch.cumsum(changes, dim = 1)), dim = 1) # increment compared to the first point
            TAs_position_lr = torch.cat((torch.zeros(n_subs, 1, device = device), TAs_position), dim = 1)
            repeats = torch.diff(torch.cat((TAs_position_lr, torch.full((n_subs, 1), n_timepts, device = device)), dim = 1), dim = 1).reshape(-1).long()
            index_timepts_lr = torch.repeat_interleave(TAs_position_lr.reshape(-1), repeats).long()
            smooth_lr = (org_detach[index_subs, index_timepts_lr, :] - increments_lr[index_subs, index_timepts_lr, :]).reshape(n_subs, n_timepts, -1) + increments_lr
            
            increments_rl = torch.cumsum(torch.cat((changes, zeros), dim = 1)[:, range(-1, -(n_timepts+1), -1), :], dim = 1)[:, range(-1, -(n_timepts+1), -1), :] # increment compared to the last point
            repeated_by = torch.cat((TAs_position, torch.full((n_subs, 1), n_timepts - 1, device = device)), dim = 1)
            repeats = torch.diff(torch.cat((torch.full((n_subs, 1), -1, device = device), repeated_by), dim = 1), dim = 1).reshape(-1).long()
            TAs_position_rl = torch.cat((TAs_position, M_obs), dim = 1)
            index_timepts_rl = torch.repeat_interleave(TAs_position_rl.reshape(-1), repeats).long()
            smooth_rl = (org_detach[index_subs, index_timepts_rl, :] + increments_rl[index_subs, index_timepts_rl, :]).reshape(n_subs, n_timepts, -1) - increments_rl
            
            weight_lr_index = ((self.model_settings["tptsTesting"] - 1) * (y_t[:, 0, :] - y_t[index_subs, 0, index_timepts_lr].reshape(n_subs, -1)) * d_m).long()
            weight_rl_index = ((self.model_settings["tptsTesting"] - 1) * (y_t[index_subs, 0, index_timepts_rl].reshape(n_subs, -1) - y_t[:, 0, :]) * d_m).long()
            weight_lr = self.weight_lr[weight_lr_index]
            weight_rl = self.weight_lr[weight_rl_index]
            weight_sum = weight_lr + weight_rl
            
            smooth_out = self.sigmoid(smooth_lr * (weight_lr/weight_sum).unsqueeze(-1) + smooth_rl * (weight_rl/weight_sum).unsqueeze(-1))
            return smooth_out * self.alpha_smo + self.beta_smo
            
            # if teaching_method == "Teacher_Forcing":
            #     smooth[:, 0] = org_detach[:, 0, 0] # hard encode the boundary
            #     smooth[:, 1:] = org_detach[:, range(n_timepts - 1), 0] + self.timediff * smooth[:, 1:]
            # elif teaching_method == "Self_Learning":
            #     smooth_lr = smooth.clone()
            #     smooth_lr[:, 0] = org_detach[:, 0, 0] # hard encode the boundary
            #     smooth_lr[:, 1:] = self.timediff * smooth_lr[:, 1:]
            #     smooth_lr = torch.cumsum(smooth_lr, dim = 1)
            #     
            #     smooth_rl = smooth.clone()
            #     smooth_rl[:, n_timepts - 1] = org_detach[:, n_timepts - 1, 0] # hard encode the boundary
            #     smooth_rl[:, :(n_timepts-1)] = -self.timediff * smooth_rl[:, 1:]
            #     smooth_rl = torch.cumsum(smooth_rl[:, range(-1, -(n_timepts+1), -1)], dim = 1)[:, range(-1, -(n_timepts+1), -1)] # increment compared to the last point
            #     
            #     org = org.squeeze(-1)
            #     return [smooth_lr * self.weight_lr_SL + smooth_rl * self.weight_rl_SL, org]
            # org = org.squeeze(-1)
            # return [smooth, org]


