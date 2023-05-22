import torch
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm
import matplotlib.pyplot as plt

# class Object(object):
#     pass
# self = Object()
# random_seed = 321

class DataLoader:
    def __init__(self, X, T, dataloader_setting, output_structure = "SAND", random_seed = 321):
        # train/valid/test split
        np.random.seed(random_seed)
        self.output_structure, self.device = output_structure, dataloader_setting["device"]
        d, split = dataloader_setting["d"], dataloader_setting["split"]
        num_obs = np.array(X.iloc[:, 0], dtype = int)
        X = X.iloc[:, 1:]
        self.n, self.m = X.shape
        
        # preprocess on X
        X.replace(np.nan, 0, inplace = True)
        
        # preprocess on T
        T.replace(np.nan, 0, inplace = True)
        self.tptsTesting = len(np.unique(T))
        
        full_prob_list = np.zeros((self.n, max(num_obs) * 2 + 1))
        for i in range(self.n):
            full_prob_list[i, :num_obs[i] * 2 + 1] = norm.pdf(x = np.array(range(-num_obs[i], num_obs[i] + 1)), loc = 0, scale = (T.iloc[i, num_obs[i] - 1] - T.iloc[i, 0]) * 15)
        
        encT = np.zeros([self.n, d, self.m])
        for i in range(int(d/4)):
            encT[:, 4 * i, :] = np.sin(10 ** (- 4*(i + 1)/d) * T * (self.tptsTesting - 1))
            encT[:, 4 * i + 1, :] = np.cos(10 ** (- 4*(i + 1)/d) * T * (self.tptsTesting - 1))
            encT[:, 4 * i + 2, :] = np.sin(2 * np.pi * (i + 1) * T)
            encT[:, 4 * i + 3, :] = np.cos(2 * np.pi * (i + 1) * T)
        
        self.batch_size = dataloader_setting["batch_size"]
        self.trai_n = round(self.n * split[0] / sum(split))
        self.vali_n = round(self.n * split[1] / sum(split))
        self.test_n = max(self.n - self.trai_n - self.vali_n, 0)
        
        self.trai_X = torch.Tensor(np.array(X.iloc[:self.trai_n, :])).to(self.device).unsqueeze(1) # (batchsize, 1, numpts)
        self.trai_T = torch.Tensor(np.array(T.iloc[:self.trai_n, :])).to(self.device).unsqueeze(1) # (batchsize, 1, numpts)
        self.trai_encT = torch.Tensor(encT[:self.trai_n, :, :]).to(self.device) # (batchsize, d, numpts)
        self.trai_O = num_obs[:self.trai_n]
        self.trai_prob_list = torch.Tensor(full_prob_list[:self.trai_n, :]).to(self.device) # (batchsize, numpts*2+1)
        
        self.vali_X = torch.Tensor(np.array(X.iloc[self.trai_n:(self.trai_n + self.vali_n), :])).to(self.device).unsqueeze(1)
        self.vali_T = torch.Tensor(np.array(T.iloc[self.trai_n:(self.trai_n + self.vali_n), :])).to(self.device).unsqueeze(1)
        self.vali_encT = torch.Tensor(encT[self.trai_n:(self.trai_n + self.vali_n), :, :]).to(self.device)
        self.vali_O = num_obs[self.trai_n:(self.trai_n + self.vali_n)]
        self.vali_prob_list = torch.Tensor(full_prob_list[self.trai_n:(self.trai_n + self.vali_n), :]).to(self.device)
        
        self.test_X = torch.Tensor(np.array(X.iloc[(self.trai_n + self.vali_n):, :])).to(self.device).unsqueeze(1)
        self.test_T = torch.Tensor(np.array(T.iloc[(self.trai_n + self.vali_n):, :])).to(self.device).unsqueeze(1)
        self.test_encT = torch.Tensor(encT[(self.trai_n + self.vali_n):, :, :]).to(self.device)
        self.test_O = num_obs[(self.trai_n + self.vali_n):]
        self.test_prob_list = torch.Tensor(full_prob_list[(self.trai_n + self.vali_n):]).to(self.device)
        
        self.max_X, self.min_X = torch.max(self.trai_X).to(self.device), torch.min(self.trai_X).to(self.device)
        
        # preprocess on full T
        if output_structure == "SAND":
            full_T = np.linspace(0, 1, len(np.unique(T)))
            enc_full_T = np.zeros([1, d + 1, len(full_T)])
            enc_full_T[:, 0, :] = full_T
            for i in range(int(d/4)):
                enc_full_T[:, 4 * i + 1, :] = np.sin(10 ** (- 4*(i + 1)/d) * full_T * (self.tptsTesting - 1))
                enc_full_T[:, 4 * i + 2, :] = np.cos(10 ** (- 4*(i + 1)/d) * full_T * (self.tptsTesting - 1))
                enc_full_T[:, 4 * i + 3, :] = np.sin(2 * np.pi * (i + 1) * full_T)
                enc_full_T[:, 4 * i + 4, :] = np.cos(2 * np.pi * (i + 1) * full_T)
            self.tptsTraining = full_T.shape[-1]
            self.full_T = torch.Tensor(enc_full_T).to(self.device) # (1, d + 1, allpts)
        else:
            self.tptsTraining = -1e9

    def shuffle(self): # For each epoch, we shuffle the order of inputs.
        # training dataset
        new_order = np.arange(self.trai_X.shape[0])
        np.random.shuffle(new_order)
        self.trai_X = self.trai_X[new_order]
        self.trai_T = self.trai_T[new_order]
        self.trai_encT = self.trai_encT[new_order]
        self.trai_O = self.trai_O[new_order]
        self.trai_prob_list = self.trai_prob_list[new_order]
        
    def _batch_generator(self, X, T, encT, O, n, prob_list): # n: sample size, B: batch size
        # i, X, T, encT, O, n, prob_list = 0, self.trai_X, self.trai_T, self.trai_encT, self.trai_O, self.trai_n, self.trai_prob_list
        # i, X, T, encT, O, n, prob_list = 0, self.vali_X, self.vali_T, self.vali_encT, self.vali_O, self.vali_n, self.vali_prob_list
        def generator_func():
            N = round(np.ceil(n/self.batch_size))
            treated_as_noniid = (np.random.randint(low = 0, high = 2, size = N) == 1)
            if self.output_structure == "SAND":
                y_t = torch.repeat_interleave(self.full_T, self.batch_size, axis = 0)
                
            for i in range(N):
                x = X[(i * self.batch_size):((i + 1)*self.batch_size)]
                t = T[(i * self.batch_size):((i + 1)*self.batch_size)]
                enct = encT[(i * self.batch_size):((i + 1)*self.batch_size)]
                obs = O[(i * self.batch_size):((i + 1)*self.batch_size)]
                prob_list_now = prob_list[(i * self.batch_size):((i + 1)*self.batch_size)]
                src = torch.cat([x, t, enct], dim = 1)
                batch_size_now = len(obs)
                
                if self.output_structure == "SAND":
                    y = torch.zeros(batch_size_now, self.full_T.shape[-1], device = self.device)
                    d_mask = torch.zeros(batch_size_now, self.full_T.shape[-1], device = self.device, dtype = int)
                    if batch_size_now != self.batch_size:
                        y_t = torch.repeat_interleave(self.full_T, batch_size_now, axis = 0)
                    index_mat = torch.round(t * (self.tptsTraining - 1)).squeeze(1).long()
                else:
                    y_t = src[:, 1:, :]
                    y = x.clone().squeeze(1)
                
                # mask related: 1 = seen; 0 = unseen
                enc_mask_size = np.random.randint(low = 1, high = obs/2 + 1, size = batch_size_now)
                e_mask = torch.tensor((t > 0).int(), device = self.device).squeeze(1)
                e_mask[:, 0] = 1
                if self.output_structure != "SAND":
                    d_mask = e_mask.clone()
                    
                emb_mask = e_mask.clone()
                if treated_as_noniid[i]:
                    center_vec = np.random.randint(low = 0, high = obs, size = batch_size_now)
                # j = 0
                
                for j in range(batch_size_now):
                    if self.output_structure == "SAND":
                        index_col = index_mat[j, range(obs[j])]
                        y[j, index_col] = x[j, 0, range(obs[j])]
                        d_mask[j, index_col] = 1
                    
                    if treated_as_noniid[i]:
                        center = center_vec[j]
                        prob = prob_list_now[j, (obs[j] - center + 1):(2 * obs[j] - center + 1)]
                        unmasked_idx = torch.multinomial(prob, obs[j] - enc_mask_size[j])
                        e_mask[j, unmasked_idx] = 2
                    else:
                        masked_idx = np.random.randint(low = 0, high = obs[j], size = enc_mask_size[j])
                        e_mask[j, masked_idx] = 0
                if treated_as_noniid[i]:
                    e_mask = (e_mask == 2).int()
                
                yield emb_mask, e_mask, d_mask, src, y_t, y
                # src: [x, t, encoded t]; y_t: [t, encoded t], for decoder, y: only x, for prediction
                # [d_mask[j, index_mat[j, range(obs[j])]] for j in range(self.batch_size)]
        return generator_func()

    def get_train_batch(self):
        return self._batch_generator(self.trai_X, self.trai_T, self.trai_encT, self.trai_O, self.trai_n, self.trai_prob_list)

    def get_valid_batch(self):
        return self._batch_generator(self.vali_X, self.vali_T, self.vali_encT, self.vali_O, self.vali_n, self.vali_prob_list)

    def get_test_batch(self):
        return self._batch_generator(self.test_X, self.test_T, self.test_encT, self.test_O, self.test_n, self.test_prob_list)

