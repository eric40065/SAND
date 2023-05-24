import torch
import numpy as np

# class Object(object):
#     pass
# self = Object()

class DataLoader:
    def __init__(self, batch_size, X, T, context = (0.8, 0.9), split = (8, 1, 1), random_seed = 321):
        # context = (0.8, 0.9); split = (90, 5, 5); random_seed = 321
        np.random.seed(random_seed)
        
        X.replace(np.nan, 0, inplace=True)
        T.replace(np.nan, 0, inplace=True)
        self._context = context
        self.n, m = X.shape
        self._m = m - 1
        # train/valid/test split
        self.batch_size = batch_size
        
        train_n = self.n // sum(split) * split[0]
        valid_n = self.n // sum(split) * split[1]
        test_n = self.n - train_n - valid_n
        
        self.train_B = int(np.ceil(train_n / batch_size))
        self.valid_B = int(np.ceil(valid_n / batch_size))
        self.test_B = int(np.ceil(test_n / batch_size))
        
        mask_pos = X.iloc[:, 0].to_list()
        X = X.iloc[:, 1:]
        # train/valid/test split
        self.train_X = X.iloc[:train_n, :]
        self.train_T = T.iloc[:train_n, :]
        self.train_M = mask_pos[:train_n]

        self.valid_X = X.iloc[train_n:(train_n + valid_n), :]
        self.valid_T = T.iloc[train_n:(train_n + valid_n), :]
        self.valid_M = mask_pos[train_n:(train_n + valid_n)]

        self.test_X = X.iloc[(train_n + valid_n):, :]
        self.test_T = T.iloc[(train_n + valid_n):, :]
        self.test_M = mask_pos[(train_n + valid_n):]

    def shuffle(self):
        # training dataset
        train_size = self.train_X.shape[0]
        new_order = list(range(train_size))
        np.random.shuffle(new_order)
        self.train_X = self.train_X.iloc[new_order, :]
        self.train_T = self.train_T.iloc[new_order, :]
        self.train_M = [self.train_M[p] for p in new_order]

    def _batch_generator(self, X, T, M, N):
        # X, T, M, N, i, j = self.train_X, self.train_T, self.train_M, self.train_B, 1, 0
        # X, T, M, N, i, j = self.valid_X, self.valid_T, self.valid_M, self.valid_B, 1, 0
        def generator_func():
            for i in range(N):
                m = np.array(M[(i * self.batch_size):((i + 1) * self.batch_size)], dtype = int)
                max_obs = max(m)
                
                x = np.array(X.iloc[(i * self.batch_size):((i + 1) * self.batch_size), :max_obs])
                t = np.array(T.iloc[(i * self.batch_size):((i + 1) * self.batch_size), :max_obs])
                batch_size_now = x.shape[0]
                
                low, high = np.floor(self._context[0] * m), np.ceil(self._context[1] * m)
                context_n = np.random.randint(low = low, high = high)
                trg_n = m - context_n
                context_t = np.zeros((batch_size_now, max_obs))
                context_x = np.zeros((batch_size_now, max_obs))
                context_mask = np.zeros((batch_size_now, max_obs), dtype = int)
                
                max_trg_len = max(trg_n)
                trg_t = np.zeros((batch_size_now, max_trg_len))
                trg_x = np.zeros((batch_size_now, max_trg_len))
                trg_mask = np.zeros((batch_size_now, max_trg_len), dtype = int)
                
                for j in range(batch_size_now):
                    trg_idx = np.unique(np.sort(np.random.randint(low = 0, high = m[j], size = trg_n[j])))
                    trg_idx_bool = np.isin(np.arange(max_obs), trg_idx)
                    context_idx_bool = ~trg_idx_bool
                    context_idx_bool[m[j]:max_obs] = False
                    context_t[j, ] = np.concatenate((t[j, context_idx_bool], np.zeros(max_obs - m[j] + len(trg_idx))))
                    context_x[j, ] = np.concatenate((x[j, context_idx_bool], np.zeros(max_obs - m[j] + len(trg_idx))))
                    context_mask[j, :sum(context_idx_bool)] = 1
                    trg_t[j, ] = np.concatenate((t[j, trg_idx_bool], np.zeros(max_trg_len - len(trg_idx))))
                    trg_x[j, ] = np.concatenate((x[j, trg_idx_bool], np.zeros(max_trg_len - len(trg_idx))))
                    trg_mask[j, :sum(trg_idx_bool)] = 1

                yield torch.Tensor(context_t).unsqueeze(-1), \
                      torch.Tensor(context_x).unsqueeze(-1), \
                      torch.Tensor(context_mask).unsqueeze(-1), \
                      torch.Tensor(trg_t).unsqueeze(-1), \
                      torch.Tensor(trg_x), \
                      torch.Tensor(trg_mask).unsqueeze(-1)

        return generator_func()

    def get_train_batch(self):
        return self._batch_generator(self.train_X, self.train_T, self.train_M, self.train_B)

    def get_valid_batch(self):
        return self._batch_generator(self.valid_X, self.valid_T, self.valid_M, self.valid_B)

    def get_test_batch(self):
        return self._batch_generator(self.test_X, self.test_T, self.test_M, self.test_B)
