for name in dir():
    if not name.startswith("_"):
        del globals()[name]

import pandas as pd
import numpy as np
from pathlib import Path
class FraminghamDataset:
    def __init__(self, data_path, X, Y, x_rescale_range=(25, 90), y_rescale_range=(25, 90), sample_size=pow(10, 9) + 7, random_seed=123):
        self._data_path = data_path
        self._X, self._Y = X, Y
        self._data_dict = {}
        self._x_rescale_range = x_rescale_range
        self._y_rescale_range = y_rescale_range
        self._sample_size = sample_size
        self._random_seed = random_seed
        self._raw_data = {}
        self._read_data()
        self._prepare_data(self._death, self._age, *self._raw_data[X], *self._raw_data[Y])

    @property
    def sample_size(self):
        self._sample_size = min(
            self._sample_size,
            len(self._data_dict[self._X][0]),
            len(self._data_dict[self._Y][0])
        )
        return self._sample_size

    @property
    def sparseX(self):
        X, Xt = self._data_dict[self._X]
        return X[:self.sample_size], Xt[:self.sample_size]

    @property
    def sparseY(self):
        Y, Yt = self._data_dict[self._Y]
        return Y[:self.sample_size], Yt[:self.sample_size]

    def _read_data(self):
        df = pd.read_csv(self._data_path)
        self._age, _ = self._get_np_data(df, 1, 22, "age")
        self._death = df.iloc[:, 67].to_numpy()  # ncol=1

        self._raw_data = {
            "sbp": self._get_np_data(df, 145, 166, "sbp"),
            "gluc": self._get_np_data(df, 68, 86, "gluc"),
            "chol": self._get_np_data(df, 22, 38, "chol"),
            "wgt": self._get_np_data(df, 169, 190, "wgt"),
            "tg": self._get_np_data(df, 166, 169, "tg"),
        }

        hgt, hgt_index = self._get_np_data(df, 112, 124, "hgt")
        hgt[783, 7] = 65

        self._raw_data["hgt"] = (hgt, hgt_index)

    def _prepare_data(self, death, tdf, xdf, xind, ydf, yind):
        # tdf should always be age
        X, Y = [], []
        Xt, Yt = [], []
        n = min(self._sample_size, len(death))

        for i in range(n):

            x_vec, xt_vec = self._clean(self._x_rescale_range, death[i], tdf[i], xdf[i], xind)
            y_vec, yt_vec = self._clean(self._y_rescale_range, death[i], tdf[i], ydf[i], yind)

            if len(x_vec) and len(y_vec):
                X.append(x_vec)
                Y.append(y_vec)
                Xt.append(xt_vec)
                Yt.append(yt_vec)

        self._data_dict[self._X] = (X, Xt)
        self._data_dict[self._Y] = (Y, Yt)

    def _clean(self, rescale, death_i, tdf_i, df_i, ind):
        x_vec = []
        t_vec = []
        t_upper = min(
            100 if death_i < 0 or tdf_i[death_i - 1] < 0 else tdf_i[death_i - 1],
            rescale[1] if rescale else 100
        )
        if t_upper < rescale[1]: return x_vec, t_vec

        t_lower = rescale[0] if rescale else 0
        for k, v in zip(ind, df_i):
            if t_lower < tdf_i[k - 1] < t_upper and v > 0:
                x_vec.append(v)
                t_vec.append(self._rescale(rescale, tdf_i[k - 1]))
            elif tdf_i[k - 1] >= t_upper:
                break
        return x_vec, t_vec

    def _rescale(self, rescale_range, t):
        if rescale_range is None: return t
        begin, end = rescale_range
        return (t - begin) / (end - begin)

    @staticmethod
    def _get_np_data(data, col_start, col_end, data_name="age"):
        df = data.iloc[:, col_start:col_end]
        cols = list(df.columns)
        return df.to_numpy(), [int(e[len(data_name):]) for e in cols]

import os, numpy as np
server_specified_folder = "/Users/eric/Desktop/UCD/"
folder = server_specified_folder + "TransFD/py_code"
os.chdir(folder)
data_path = "../Data/Raw/RealData/FMdata.csv"
data = FraminghamDataset(data_path=data_path, X="hgt", Y="wgt", x_rescale_range=(30, 60), y_rescale_range=(30, 60))
total_t = np.array([item for sublist in data.sparseX[1] for item in sublist])
total_t = np.sort(np.unique(total_t))
a, b = 1/(total_t[-1] - total_t[0]), -total_t[0]/(total_t[-1] - total_t[0])

M_obs = [len(i) for i in data.sparseY[1]]
valid_sub = [M_obs[i] > 1 for i in range(len(M_obs))]
# M_obs2 = [len(i) for i in data.sparseY[1]]
T_obs = np.zeros((sum(valid_sub), len(total_t))) + np.nan
X_obs = np.zeros((sum(valid_sub), len(total_t) + 1)) + np.nan
sum_X = 0
max_X = -1
for counter, index in enumerate(np.where(valid_sub)[0]):
    X_now = (np.array(data.sparseY[0][index]) * 0.45359237) / np.power((np.array(data.sparseX[0][index]).mean() * 2.54 / 100), 2)
    sum_X += X_now.sum()
    max_X = np.max((max_X, X_now.__abs__().max()))
    X_obs[counter, np.arange(1, M_obs[index] + 1)] = X_now
    T_obs[counter, :M_obs[index]] = data.sparseY[1][index]

X_obs = (X_obs - np.floor(sum_X/sum(M_obs)))/(max_X - np.floor(sum_X/sum(M_obs)))
X_obs[:, 0] = [M_obs[index] for index in np.where(valid_sub)[0]]
T_obs = T_obs * a + b
np.random.seed(11252022)
new_order = list(range(sum(valid_sub)))
np.random.shuffle(new_order)

X_obs = X_obs[new_order, ]
T_obs = T_obs[new_order, ]
np.savetxt("../Data/IID/RealData/Framingham/X_obs.csv", X_obs, delimiter=",")
np.savetxt("../Data/IID/RealData/Framingham/T_obs.csv", T_obs, delimiter=",")

