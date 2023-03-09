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

class DataLoader_allpts:
    def __init__(self, X, T, dataloader_setting, output_structure = "DiffSelfAtt", random_seed = 321):
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
        
        full_prob_list = np.zeros((self.n, max(num_obs) * 2 + 1))
        for i in range(self.n):
            full_prob_list[i, :num_obs[i] * 2 + 1] = norm.pdf(x = np.array(range(-num_obs[i], num_obs[i] + 1)), loc = 0, scale = (T.iloc[i, num_obs[i] - 1] - T.iloc[i, 0]) * 10)
        
        encT = np.zeros([self.n, d, self.m])
        for i in range(int(d/4)):
            encT[:, 4 * i, :] = np.sin(10 ** (2 - 4*(i + 1)/d) * T)
            encT[:, 4 * i + 1, :] = np.cos(10 ** (2 - 4*(i + 1)/d) * T)
            encT[:, 4 * i + 2, :] = np.sin(2 * np.pi * (i + 1) * T)
            encT[:, 4 * i + 3, :] = np.cos(2 * np.pi * (i + 1) * T)
        
        self.batch_size = dataloader_setting["batch_size"]
        self.trai_n = round(self.n * split[0] / sum(split))
        self.vali_n = round(self.n * split[1] / sum(split))
        self.test_n = max(self.n - self.trai_n - self.vali_n, 0)
        
        self.trai_X = torch.Tensor(np.array(X.iloc[:self.trai_n, :])).to(dataloader_setting["device"]).unsqueeze(1) # (batchsize, 1, numpts)
        self.trai_T = torch.Tensor(np.array(T.iloc[:self.trai_n, :])).to(dataloader_setting["device"]).unsqueeze(1) # (batchsize, 1, numpts)
        self.trai_encT = torch.Tensor(encT[:self.trai_n, :, :]).to(dataloader_setting["device"]) # (batchsize, d, numpts)
        self.trai_O = num_obs[:self.trai_n]
        self.trai_prob_list = torch.Tensor(full_prob_list[:self.trai_n, :]).to(dataloader_setting["device"]) # (batchsize, numpts*2+1)
        
        self.vali_X = torch.Tensor(np.array(X.iloc[self.trai_n:(self.trai_n + self.vali_n), :])).to(dataloader_setting["device"]).unsqueeze(1)
        self.vali_T = torch.Tensor(np.array(T.iloc[self.trai_n:(self.trai_n + self.vali_n), :])).to(dataloader_setting["device"]).unsqueeze(1)
        self.vali_encT = torch.Tensor(encT[self.trai_n:(self.trai_n + self.vali_n), :, :]).to(dataloader_setting["device"])
        self.vali_O = num_obs[self.trai_n:(self.trai_n + self.vali_n)]
        self.vali_prob_list = torch.Tensor(full_prob_list[self.trai_n:(self.trai_n + self.vali_n), :]).to(dataloader_setting["device"])
        
        self.test_X = torch.Tensor(np.array(X.iloc[(self.trai_n + self.vali_n):, :])).to(dataloader_setting["device"]).unsqueeze(1)
        self.test_T = torch.Tensor(np.array(T.iloc[(self.trai_n + self.vali_n):, :])).to(dataloader_setting["device"]).unsqueeze(1)
        self.test_encT = torch.Tensor(encT[(self.trai_n + self.vali_n):, :, :]).to(dataloader_setting["device"])
        self.test_O = num_obs[(self.trai_n + self.vali_n):]
        self.test_prob_list = torch.Tensor(full_prob_list[(self.trai_n + self.vali_n):]).to(dataloader_setting["device"])
        
        self.max_X, self.min_X = torch.max(self.trai_X).to(dataloader_setting["device"]), torch.min(self.trai_X).to(dataloader_setting["device"])
        
        # preprocess on full T
        self.tptsTesting = len(np.unique(T))
        if output_structure == "DiffSelfAtt":
            full_T = np.linspace(0, 1, len(np.unique(T)))
            enc_full_T = np.zeros([1, d + 1, len(full_T)])
            enc_full_T[:, 0, :] = full_T
            for i in range(int(d/4)):
                enc_full_T[:, 4 * i + 1, :] = np.sin(10 ** (2 - 4*(i + 1)/d) * full_T)
                enc_full_T[:, 4 * i + 2, :] = np.cos(10 ** (2 - 4*(i + 1)/d) * full_T)
                enc_full_T[:, 4 * i + 3, :] = np.sin(2 * np.pi * (i + 1) * full_T)
                enc_full_T[:, 4 * i + 4, :] = np.cos(2 * np.pi * (i + 1) * full_T)
            self.tptsTraining = full_T.shape[-1]
            self.full_T = torch.Tensor(enc_full_T).to(dataloader_setting["device"]) # (1, d + 1, allpts)
        else:
            self.tptsTraining = self.m # we don't need it in this case but let's assign it a value.

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
            if self.output_structure == "DiffSelfAtt":
                y_t = torch.repeat_interleave(self.full_T, self.batch_size, axis = 0)
                y = torch.zeros(self.batch_size, self.full_T.shape[-1], device = self.device)
                d_mask = torch.zeros(self.batch_size, self.full_T.shape[-1], device = self.device)
            
            treated_as_noniid = (np.random.randint(low = 0, high = 2, size = N) == 1)
            for i in range(N):
                # data related
                x = X[(i * self.batch_size):((i + 1)*self.batch_size)]
                t = T[(i * self.batch_size):((i + 1)*self.batch_size)]
                enct = encT[(i * self.batch_size):((i + 1)*self.batch_size)]
                obs = O[(i * self.batch_size):((i + 1)*self.batch_size)]
                prob_list_now = prob_list[(i * self.batch_size):((i + 1)*self.batch_size)]
                
                batch_size_now = x.shape[0]
                src = torch.cat([x, t, enct], dim = 1)
                
                if self.output_structure == "DiffSelfAtt" and batch_size_now != self.batch_size:
                    y_t = torch.repeat_interleave(self.full_T, batch_size_now, axis = 0)
                    y = torch.zeros(batch_size_now, self.full_T.shape[-1], device = self.device)
                    d_mask = torch.zeros(batch_size_now, self.full_T.shape[-1], device = self.device)
                
                if self.output_structure == "DiffSelfAtt":
                    index_mat = torch.round(t * (self.tptsTraining - 1)).squeeze(1).long()
                else:
                    y_t = src[:, 1:, :]
                    y = x.clone().squeeze(1)
                    d_mask = torch.zeros(batch_size_now, y_t.shape[-1], device = self.device)
                
                # mask related: 1 = seen; 0 = unseen
                enc_mask_size = np.random.randint(low = 1, high = obs/2 + 1, size = batch_size_now)
                e_mask = torch.as_tensor((t > 0) + 0).squeeze(1)
                e_mask[:, 0] = 1
                emb_mask = e_mask.clone()
                
                if treated_as_noniid[i]:
                    center_vec = np.random.randint(low = 0, high = obs, size = len(obs))
                # j = 0
                
                for j in range(batch_size_now):
                    if self.output_structure == "DiffSelfAtt":
                        index_col = index_mat[j, range(obs[j])]
                        y[j, index_col] = x[j, 0, range(obs[j])]
                        d_mask[j, index_col] = 1
                    else:
                        d_mask[j, :obs[j]] = 1
                    
                    if treated_as_noniid[i]:
                        center = center_vec[j]
                        prob = prob_list_now[j, (obs[j] - center + 1):(2 * obs[j] - center + 1)]
                        unmasked_idx = torch.multinomial(prob, obs[j] - enc_mask_size[j])
                        e_mask[j, unmasked_idx] = 2
                    else:
                        masked_idx = np.random.randint(low = 0, high = obs[j], size = enc_mask_size[j])
                        e_mask[j, masked_idx] = 0
                if treated_as_noniid[i]:
                    e_mask = (e_mask == 2) + 0

                yield emb_mask, e_mask, d_mask, src, y_t, y
                    
        return generator_func()

    def get_train_batch(self):
        return self._batch_generator(self.trai_X, self.trai_T, self.trai_encT, self.trai_O, self.trai_n, self.trai_prob_list)

    def get_valid_batch(self):
        return self._batch_generator(self.vali_X, self.vali_T, self.vali_encT, self.vali_O, self.vali_n, self.vali_prob_list)

    def get_test_batch(self):
        return self._batch_generator(self.test_X, self.test_T, self.test_encT, self.test_O, self.test_n, self.test_prob_list)

class DataLoader:
    def __init__(self, X, T, dataloader_setting, random_seed = 321, iidt = True):
        # random_seed = 321
        np.random.seed(random_seed)
        print("not modified yet")
        self.iidt = iidt
        d, split = dataloader_setting["d"], dataloader_setting["split"]
        num_obs = X.iloc[:, 0].tolist()
        X = X.iloc[:, 1:]
        self.n, self.m = X.shape
        
        # preprocess on X
        X.replace(np.nan, 0, inplace = True)
        
        # preprocess on T
        T.replace(np.nan, 0, inplace=True)
        encT = np.zeros([self.n, d, self.m])
        self.tptsTraining = T.shape[-1]
        
        full_prob_list = []
        for i in range(self.n):
            full_prob_list.append(norm.pdf(x = np.array(range(-num_obs[i], num_obs[i] + 1)), loc = 0, scale = (T.iloc[i, num_obs[i] - 1] - T.iloc[i, 0]) * 20))

        for i in range(int(d/4)):
            encT[:, 4 * i, :] = np.sin(10 ** (2 - 4*(i + 1)/d) * T)
            encT[:, 4 * i + 1, :] = np.cos(10 ** (2 - 4*(i + 1)/d) * T)
            encT[:, 4 * i + 2, :] = np.sin(2 * np.pi * (i + 1) * T)
            encT[:, 4 * i + 3, :] = np.cos(2 * np.pi * (i + 1) * T)
        
        # train/valid/test split
        self.batch_size = dataloader_setting["batch_size"]
        self.trai_n = round(self.n * split[0] / sum(split))
        self.vali_n = round(self.n * split[1] / sum(split))
        self.test_n = max(self.n - self.trai_n - self.vali_n, 0)
        
        self.trai_X = X.iloc[:self.trai_n, :]
        self.trai_T = T.iloc[:self.trai_n, :]
        self.trai_encT = encT[:self.trai_n, :, :]
        self.trai_O = num_obs[:self.trai_n]
        self.trai_prob_list = full_prob_list[:self.trai_n]
        
        self.vali_X = X.iloc[self.trai_n:(self.trai_n + self.vali_n), :]
        self.vali_T = T.iloc[self.trai_n:(self.trai_n + self.vali_n), :]
        self.vali_encT = encT[self.trai_n:(self.trai_n + self.vali_n), :, :]
        self.vali_O = num_obs[self.trai_n:(self.trai_n + self.vali_n)]
        self.vali_prob_list = full_prob_list[self.trai_n:(self.trai_n + self.vali_n)]
        
        self.test_X = X.iloc[(self.trai_n + self.vali_n):, :]
        self.test_T = T.iloc[(self.trai_n + self.vali_n):, :]
        self.test_encT = encT[(self.trai_n + self.vali_n):, :, :]
        self.test_O = num_obs[(self.trai_n + self.vali_n):]
        self.test_prob_list = full_prob_list[(self.trai_n + self.vali_n):]
        self.max_X, self.min_X = np.max(np.array(self.trai_X)), np.min(np.array(self.trai_X))
        
    def shuffle(self): # For each epoch, you want to shuffle the order of inputs.
        # training dataset
        trai_size = self.trai_X.shape[0]
        new_order = list(range(trai_size))
        np.random.shuffle(new_order)
        self.trai_X = self.trai_X.iloc[new_order, :]
        self.trai_T = self.trai_T.iloc[new_order, :]
        self.trai_encT = self.trai_encT[new_order, :, :]
        self.trai_O = [self.trai_O[p] for p in new_order]
        self.trai_prob_list = [self.trai_prob_list[p] for p in new_order]
        
    def _batch_generator(self, X, T, encT, O, n, prob_list): # n: sample size
        # i, X, T, encT, O, n, prob_list = 0, self.vali_X, self.vali_T, self.vali_encT, self.vali_O, self.vali_n, self.vali_prob_list
        N = round(np.ceil(n/self.batch_size))
        def generator_func():
            for i in range(N):
                # data related
                x = np.array(X.iloc[(i * self.batch_size):((i + 1)*self.batch_size), :])
                t = np.array(T.iloc[(i * self.batch_size):((i + 1)*self.batch_size), :])
                enct = encT[(i * self.batch_size):((i + 1)*self.batch_size), :, :]
                obs = O[(i * self.batch_size):((i + 1)*self.batch_size)]
                prob_list_now = prob_list[(i * self.batch_size):((i + 1)*self.batch_size)]
                
                src = np.array([[x[j, :], t[j, :]] for j in range(x.shape[0])]) # source
                src = np.concatenate((src, enct), axis = 1)
                y = x
                y_t = np.concatenate((t.reshape(x.shape[0], 1, -1), enct), axis = 1)
                    
                # mask related: 1 = seen; 0 = unseen
                enc_mask_size = np.random.randint(low = 1, high = np.array(obs)/2 + 1, size = x.shape[0])
                e_mask = (t > 0) + 0
                e_mask[:, 0] = 1
                d_mask = torch.zeros(x.shape[0], y_t.shape[-1])
                
                if not self.iidt:
                    center_vec = np.random.randint(low = 0, high = obs, size = len(obs))
                
                for j in range(x.shape[0]):
                    d_mask[j, :obs[j]] = 1

                    if self.iidt:
                        masked_idx = np.random.randint(low = 0, high = obs[j], size = enc_mask_size[j])
                    else:
                        center = center_vec[j]
                        prob = prob_list_now[j][(obs[j] - center + 1):(2 * obs[j] - center + 1)]
                        masked_idx = np.setdiff1d(np.arange(obs[j]), np.random.choice(range(obs[j]), size = obs[j] - enc_mask_size[j], replace = False, p = prob/sum(prob)))
                    e_mask[j, masked_idx] = 0
                yield torch.Tensor(e_mask), d_mask, torch.Tensor(src), torch.Tensor(y_t), torch.Tensor(y)
        return generator_func()

    def get_train_batch(self):
        return self._batch_generator(self.trai_X, self.trai_T, self.trai_encT, self.trai_O, self.trai_n, self.trai_prob_list)

    def get_valid_batch(self):
        return self._batch_generator(self.vali_X, self.vali_T, self.vali_encT, self.vali_O, self.vali_n, self.vali_prob_list)

    def get_test_batch(self):
        return self._batch_generator(self.test_X, self.test_T, self.test_encT, self.test_O, self.test_n, self.test_prob_list)

