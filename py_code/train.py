# shape of the tensor passed across all modules are kept as (batch, d, seq_len)
for name in dir():
    if not name.startswith("_"):
        del globals()[name]

import sys, torch, time, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.special import softmax
from pathlib import Path
from torch.optim import Adam
# server_specified_folder = "/Users/eric/Desktop/UCD"
# folder = server_specified_folder + "/TransFD/py_code_copy"
# os.chdir(folder)
from DataRelated import DataLoader
from ModelRelated import compute_loss, get_penalty
from PlotRelated import get_plot_scores_scatter, get_plot_scores_trend
from TransformerModel import Transformer

torch.manual_seed(321)
np.random.seed(321)

## Get the data. 
# Options are: "HighDim_E", "LowDim_G", "HighDim_G", "LowDim_E", "LowDim_T", "HighDim_T", "UK"
data_name = "/HighDim_E" 
real = (data_name == "/UK") # A useful indicator that records if the data is simulated
# Pick a kind of data to analyze.
# Options for (data_is_dense, error) are: case 1 (True, True), case 2 (False, True), case 3 (True, False), case 4 (False, False).
# Options for iidt are: True, False
data_is_dense, error = True, True
iidt = True

## Define the device
cuda_device = "cpu" # use "cuda:0" if gpu is avaliable
device = torch.device(cuda_device)

## Define the dataloader. 
# d: number of embedding layers. batch_size: batch size. split: proportions of training/validation/testing
dataloader_setting = {"d": 60, "batch_size": 256, "split": (90, 5, 5), "device": device}
dataloader_setting["batch_size"] = 250 if not real else dataloader_setting["batch_size"]

## Define the models. 
# Options are: "SAND", "Vanilla", "SelfAtt".
output_structure = "SAND"
# In a transformer, we set: the number of heads to be 2, recorded in "num_heads",
#                           the number of blocks in encoder and deocder to be 2 and 2, recorded in "num_layers",
#                           the number of queries, keys, values, hidden nodes to be 64, recorded in "num_q", "num_k", "num_v", and "num_hiddens"
#                           the learning rate to be 3e-4, recorded in "lr"
#                           the percentile for the check loss function to regress on to be be 0.1 and 0.9, recorded in "check_loss"
#                           the number of epoch to be 10000, recorded in "max_epoch"
#                           the dropout rate to be 0.15, recorded in "dropout"
#                           the percentile for the check loss function to regress on to be be 0.1 and 0.9, recorded in "check_loss"
# Specifically, if we use SAND, we set: the dropout rate to be 0.05, recorded in "dropout_SAND"
#                                       the number of queries, keys, values to be 64, recorded in "num_q_SAND", "num_k_SAND", "num_v_SAND".
model_settings = {"num_heads": 2, "num_layers": (2, 2), "num_q": 64, "num_k": 64, "num_v": 64, "num_hiddens": 64,\
                  "lr": 3e-4, "epoch_CV": 2, "max_epoch": 2, "dropout": 0.15, "check_loss": (0.1, 0.9),\
                  "dropout_SAND": 0.05, "num_q_SAND": 64, "num_k_SAND": 64, "num_v_SAND": 64,\
                  "f_in": (2 + dataloader_setting["d"], 1 + dataloader_setting["d"]), "batch_size": dataloader_setting["batch_size"],\
                  "penalty_max_iter": 2, "penalty_min_iter": 2}
denoise_method = "l2w" # if output_structure = "Vanilla", options for denoise_method are "l2w" and "l2o". Otherwise, it's "l2w"

## dataset is created using R. Here we just use the data.
data_type = "/dense" if data_is_dense else "/sparse"
sparsity_error_folder = (data_type + "/w_error" if error else data_type + "/wo_error") if not real else data_type
sparsity_error_folder = "" if data_name == "/Framingham" else sparsity_error_folder

X = pd.read_csv("../Data" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + sparsity_error_folder + "/X_obs.csv", header = None)
T = pd.read_csv("../Data" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + sparsity_error_folder + "/T_obs.csv", header = None)

## picking the best penalty
output_structure_folder = "/" + output_structure
batch_size = dataloader_setting["batch_size"]
if denoise_method != "None":
    dataLoader = DataLoader(X, T, dataloader_setting, output_structure = output_structure)
    model_settings["tptsTesting"], model_settings["tptsTraining"], model_settings["min_X"], model_settings["max_X"], model_settings["denoise_method"], model_settings["device"] = dataLoader.tptsTesting, dataLoader.tptsTraining, dataLoader.min_X, dataLoader.max_X, denoise_method, device
if denoise_method == "l2o" or denoise_method == "TVo":
    m, L = X.shape[1] - 1, model_settings["tptsTesting"]
    full_time_grid = np.arange(L)/(L - 1)
    if dataloader_setting["d"] == 0:
        d_T_full_tmp = torch.Tensor(full_time_grid).reshape(1, -1, 1)
        d_T_full = torch.repeat_interleave(d_T_full_tmp, model_settings["batch_size"], dim = 0).to(device)
    else:
        d_T_full_tmp = np.zeros([1, dataloader_setting["d"] + 1, L])
        d_T_full_tmp[:, 0, :] = full_time_grid
        for i in range(int(dataloader_setting["d"]/4)):
            d_T_full_tmp[0, 4*i+1, :] = np.sin(10 ** (- 4*(i + 1)/dataloader_setting["d"]) * full_time_grid * (L - 1))
            d_T_full_tmp[0, 4*i+2, :] = np.cos(10 ** (- 4*(i + 1)/dataloader_setting["d"]) * full_time_grid * (L - 1))
            d_T_full_tmp[0, 4*i+3, :] = np.sin(2 * np.pi * (i + 1) * full_time_grid)
            d_T_full_tmp[0, 4*i+4, :] = np.cos(2 * np.pi * (i + 1) * full_time_grid)

        d_T_full_tmp = torch.Tensor(d_T_full_tmp)
        d_T_full_full_batch = torch.repeat_interleave(d_T_full_tmp, model_settings["batch_size"], dim = 0).to(device)
if denoise_method == "l1w" or denoise_method == "l2w" or denoise_method == "l2o" or denoise_method == "TVo":
    loss_train_history = np.zeros((model_settings["penalty_max_iter"], model_settings["epoch_CV"])) + sys.maxsize
    loss_valid_history = np.zeros((model_settings["penalty_max_iter"], model_settings["epoch_CV"])) + sys.maxsize
    score_weights = np.exp(np.array(range(-model_settings["epoch_CV"], 0))/(model_settings["epoch_CV"]/10))
    score_weights[:int(model_settings["epoch_CV"]/4 * 3)] = 0
    score = []
    penalty_list = []

    repeat_obj = torch.tensor([model_settings["num_heads"]] * dataloader_setting["batch_size"]).to(device)
    for penalty_counter in range(model_settings["penalty_max_iter"]):
        # stop
        print(penalty_counter, flush = True)
        torch.manual_seed(321)
        model = Transformer(model_settings, output_structure).to(device)
        optimizer = Adam(model.parameters(), lr = model_settings["lr"])
        min_valid_loss = sys.maxsize
        penalty = get_penalty(penalty_counter, score, penalty_list)
        if (denoise_method == "l2o" or denoise_method == "TVo") and penalty_counter == 0:
            penalty = penalty * 0.01
        penalty_list.append(penalty)

        for k in range(model_settings["epoch_CV"]):
            loss_train = []
            loss_valid = []
            dataLoader.shuffle()
            model = model.train()
            for i, (emb_mask, e_m, d_m, x, y_t, y) in enumerate(dataLoader.get_train_batch()):
                optimizer.zero_grad()
                batch_size_now = x.shape[0]
                emb_m_mh = torch.repeat_interleave(emb_mask, model_settings["num_heads"], dim = 0)
                e_m_mh = torch.repeat_interleave(e_m, model_settings["num_heads"], dim = 0)
                d_m_mh = torch.repeat_interleave(d_m, model_settings["num_heads"], dim = 0)
                out = model.forward(x, y_t, emb_m_mh, e_m_mh, d_m_mh, e_m if output_structure == "SelfAtt" else d_m)
                loss = compute_loss(out, y, d_m, k, isTraining = True, check_loss = model_settings["check_loss"])

                if denoise_method == "l1w":
                    loss += torch.mean(torch.abs(torch.cat([param.view(-1) for param in model.parameters()]))) * penalty
                elif denoise_method == "l2w":
                    loss += torch.mean(torch.square(torch.cat([param.view(-1) for param in model.parameters()]))) * penalty
                else:
                    d_T_full = d_T_full_full_batch if batch_size_now == model_settings["batch_size"] else torch.repeat_interleave(d_T_full_tmp, batch_size_now, dim = 0).to(device)
                    out_diff1 = torch.diff(model.forward(x, d_T_full, emb_m_mh, emb_m_mh, None, None), 1)
                    loss += torch.mean(torch.square(torch.diff(out_diff1, 1))) * penalty if denoise_method == "l2o" else torch.mean(torch.abs(out_diff1)) * penalty

                loss_train.append(loss.item())
                # update parameters using backpropagation
                loss.backward()
                optimizer.step()
            loss_train_history[penalty_counter, k] = np.mean(loss_train)
            # model evaluation mode
            with torch.no_grad():
                model = model.eval()
                for emb_mask, e_m, d_m, x, y_t, y in dataLoader.get_valid_batch():
                    # stop
                    emb_m_mh = torch.repeat_interleave(emb_mask, model_settings["num_heads"], dim = 0)
                    e_m_mh = torch.repeat_interleave(e_m, model_settings["num_heads"], dim = 0)
                    d_m_mh = torch.repeat_interleave(d_m, model_settings["num_heads"], dim = 0)
                    valid_y = model.forward(x, y_t, emb_m_mh, e_m_mh, d_m_mh, e_m if output_structure == "SelfAtt" else d_m, isTraining = False)
                    [loss_valid_tmp, DAE_var_tmp] = compute_loss(valid_y, y, d_m, k, isTraining = False, check_loss = model_settings["check_loss"])
                    loss_valid.append(loss_valid_tmp.item())

            loss_valid_history[penalty_counter, k] = np.mean(loss_valid)
            if k % 20 == 0 or k < 20:
                print("penalty counter:", penalty_counter, "epoch:", k, "training loss = ", loss_train_history[penalty_counter, k], "validation loss = ", loss_valid_history[penalty_counter, k], flush = True)
            
        score.append(sum(loss_valid_history[penalty_counter, ] * score_weights))
        if penalty_counter >= model_settings["penalty_min_iter"] and np.abs(np.log10(penalty_list[penalty_counter]) - np.log10(penalty_list[penalty_counter - 1])) < 1e-2:
            break
    print(score, flush = True)
    print(penalty_list, flush = True)
    get_plot_scores_scatter(my_computer, penalty_list, score, loss_valid_history, iidt, real, data_name, output_structure_folder, sparsity_error_folder, denoise_method)
    get_plot_scores_trend(my_computer, penalty_list, score, model_settings, loss_valid_history, iidt, real, data_name, output_structure_folder, sparsity_error_folder, denoise_method)

## training
dataLoader = DataLoader(X, T, dataloader_setting, output_structure = output_structure)
model_settings["tptsTesting"], model_settings["tptsTraining"], model_settings["min_X"], model_settings["max_X"], model_settings["denoise_method"], model_settings["device"] = dataLoader.tptsTesting, dataLoader.tptsTraining, dataLoader.min_X, dataLoader.max_X, denoise_method, device
model = Transformer(model_settings, output_structure).to(device)
optimizer = Adam(model.parameters(), lr = model_settings["lr"])
loss_train_history = []
loss_valid_history = []
min_valid_loss = sys.maxsize
save_model_every = 1000

repeat_obj = torch.tensor([model_settings["num_heads"]] * dataloader_setting["batch_size"]).to(device)
for k in range(model_settings["max_epoch"]):
    if k % save_model_every == 0:
        checkpoint = {"model": Transformer(model_settings, output_structure), "state_dict": model.state_dict(), "optimizer" : optimizer.state_dict()}
        torch.save(checkpoint, "../Checkpoints/Tmp" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + output_structure_folder + sparsity_error_folder + "/ckpts_" + str(denoise_method) + "_" + str(k) + ".pth")

    loss_train = []
    loss_valid = []
    dataLoader.shuffle()
    model = model.train()
    # set model training state
    for i, (emb_mask, e_m, d_m, x, y_t, y) in enumerate(dataLoader.get_train_batch()):
        # e_m: mask for encoder, randomly mask some points; d_m: mask for decoder, only mask unobserved points
        # x: [B, 2, obs]: x and t, y_t: t, y: x
        optimizer.zero_grad()
        emb_m_mh = torch.repeat_interleave(emb_mask, model_settings["num_heads"], dim = 0)
        e_m_mh = torch.repeat_interleave(e_m, model_settings["num_heads"], dim = 0)
        d_m_mh = torch.repeat_interleave(d_m, model_settings["num_heads"], dim = 0)
        out = model.forward(x, y_t, emb_m_mh, e_m_mh, d_m_mh, e_m if output_structure == "SelfAtt" else d_m, iteration = k)
        loss = compute_loss(out, y, d_m, k, isTraining = True, check_loss = model_settings["check_loss"])
        
        if denoise_method == "l1w":
            loss += torch.mean(torch.abs(torch.cat([param.view(-1) for param in model.parameters()]))) * penalty
        elif denoise_method == "l2w":
            loss += torch.mean(torch.square(torch.cat([param.view(-1) for param in model.parameters()]))) * penalty
        elif denoise_method == "l2o" or denoise_method == "TVo":
            batch_size_now = x.shape[0]
            d_T_full = d_T_full_full_batch if batch_size_now == model_settings["batch_size"] else torch.repeat_interleave(d_T_full_tmp, batch_size_now, dim = 0).to(device)
            out_diff1 = torch.diff(model.forward(x, d_T_full, emb_m_mh, emb_m_mh, None, None), 1)
            loss += torch.mean(torch.square(torch.diff(out_diff1, 1))) * penalty if denoise_method == "l2o" else torch.mean(torch.abs(out_diff1)) * penalty
        loss_train.append(loss.item())

        # update parameters using backpropagation
        loss.backward()
        optimizer.step()
    loss_train_history.append(np.mean(loss_train))

    # model evaluation mode
    with torch.no_grad():
        model = model.eval()
        for emb_mask, e_m, d_m, x, y_t, y in dataLoader.get_valid_batch():
            emb_m_mh = torch.repeat_interleave(emb_mask, model_settings["num_heads"], dim = 0)
            e_m_mh = torch.repeat_interleave(e_m, model_settings["num_heads"], dim = 0)
            d_m_mh = torch.repeat_interleave(d_m, model_settings["num_heads"], dim = 0)
            valid_y = model.forward(x, y_t, emb_m_mh, e_m_mh, d_m_mh, e_m if output_structure == "SelfAtt" else d_m, isTraining = False)
            [loss_valid_tmp, DAE_var_tmp] = compute_loss(valid_y, y, d_m, k, isTraining = False, check_loss = model_settings["check_loss"])
            
            if denoise_method == "l1w":
                loss_valid_tmp += torch.mean(torch.abs(torch.cat([param.view(-1) for param in model.parameters()]))) * penalty
            elif denoise_method == "l2w":
                loss_valid_tmp += torch.mean(torch.square(torch.cat([param.view(-1) for param in model.parameters()]))) * penalty
            elif denoise_method == "l2o" or denoise_method == "TVo":
                batch_size_now = x.shape[0]
                d_T_full = d_T_full_full_batch if batch_size_now == model_settings["batch_size"] else torch.repeat_interleave(d_T_full_tmp, batch_size_now, dim = 0).to(device)
                out_diff1 = torch.diff(model.forward(x, d_T_full, emb_m_mh, emb_m_mh, None, None), 1)
                loss_valid_tmp += torch.mean(torch.square(torch.diff(out_diff1, 1))) * penalty if denoise_method == "l2o" else torch.mean(torch.abs(out_diff1)) * penalty
            loss_valid.append(loss_valid_tmp.item())

    if np.mean(loss_valid) < min_valid_loss:
        print("The best model is " + str(k) + ".", flush = True)
        checkpoint_best = {"model": Transformer(model_settings, output_structure), "state_dict": model.state_dict()} #, "optimizer" : optimizer.state_dict()}
        torch.save(checkpoint_best, "../Checkpoints" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + output_structure_folder + sparsity_error_folder + "/best_ckpts_" + str(denoise_method) + ".pth")
        min_valid_loss = np.mean(loss_valid)

    loss_valid_history.append(np.mean(loss_valid))
    if k % 20 == 0 or k < 10 or np.mean(loss_valid) == min_valid_loss:
        print("epoch:", k, "training loss = ", loss_train_history[-1], "validation loss = ", loss_valid_history[-1], flush = True)

plt.clf()
plt.subplots_adjust(left=0.2, right=0.9, top=0.95, bottom=0.15)
epoch_range = list(range(1, len(loss_train_history) + 1))
plt.plot(epoch_range, np.log10(loss_train_history), label = "train", lw = 1)
plt.plot(epoch_range, np.log10(loss_valid_history), label = "valid", lw = 1)
plt.xlabel("epoch vs. log(loss)", fontsize = 32)
plt.ylabel(output_structure + "/" + denoise_method, fontsize = 32);
plt.legend(fontsize = 15)
plt.savefig("../Plots" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + "/Loss" + output_structure_folder + "/" + str(denoise_method) + sparsity_error_folder.replace("/", "_") + "_loss.png")
plt.show()
plt.close()
