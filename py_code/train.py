# shape of the tensor passed across all modules are kept as (batch, d, seq_len)
for name in dir():
    if not name.startswith("_"):
        del globals()[name]

import sys, torch, time, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
import shelve
from scipy.special import softmax
from pathlib import Path
from torch.optim import Adam
torch.manual_seed(321)
np.random.seed(321)
my_computer = False

## Find data
iidt = False
if my_computer:
    data_name = "/HighDim_E" # "UK" "Framingham" "LowDim_G" "HighDim_G" "LowDim_E" "HighDim_E" "LowDim_T" "HighDim_T"
    real = (data_name == "/Framingham" or data_name == "/UK")
    output_structure = "Vanilla"  # "Vanilla" "SelfAtt" "DiffSelfAtt"
    # data_is_dense, error = False, False
    data_is_dense, error = False, True
    # data_is_dense, error = True, False
    # data_is_dense, error = True, True
    
    # denoise_method = "None"
    # denoise_method = "l2w"
    denoise_method = "l2o"
    
    server_specified_folder = "/Users/eric/Desktop/UCD"
    device = torch.device("cpu")
else:
    server_specified_folder = "/home/jshong96"
    data_name = "/" + sys.argv[1]
    output_structure = sys.argv[2]
    denoise_method = sys.argv[3]
    data_is_dense = (sys.argv[4] == "True")
    error = (sys.argv[5] == "True")
    denoise_method = denoise_method if output_structure == "Vanilla" else "l2w"
    real = (data_name == "/Framingham" or data_name == "/UK")
    cuda_device = "cuda:" + str(0 + data_is_dense + iidt * 2) if real else "cuda:" + (str(error * 2 + data_is_dense) if iidt else str(error * 2 + 1 - data_is_dense))
    cuda_device = "cuda:" + str(error * 2 - data_is_dense + 1) if denoise_method == "TVo" else cuda_device
    
    ## shared with Xiawei
    cuda_device = "cuda:" + (str(error + 0) if iidt else str(1 - error))
    # cuda_device = "cuda:" + (str(error + 0 + 2) if iidt else str(1 - error + 2)) # if she says yes
    device = torch.device(cuda_device)
    torch.cuda.manual_seed_all(321)

dataloader_setting = {"d": 60, "batch_size": 256, "split": (90, 5, 5), "device": device}
dataloader_setting["batch_size"] = 250 if not real else dataloader_setting["batch_size"]
dataloader_setting["batch_size"] = 32 if data_name == "/Framingham" else dataloader_setting["batch_size"]
model_settings = {"num_heads": 2, "num_layers": (2, 2), "num_q": 64, "num_k": 64, "num_v": 64, "num_hiddens": 64,\
                  "DAE": True, "lr": 2e-4, "epoch_CV": 1000, "max_epoch": 5000 + 1, "dropout": 0.15, "check_loss": (0.1, 0.9),\
                  "f_in": (2 + dataloader_setting["d"], 1 + dataloader_setting["d"]), "batch_size": dataloader_setting["batch_size"],\
                  "penalty_max_iter": 10, "penalty_min_iter": 5}
model_settings["lr"] = 1e-4 if real or output_structure == "DiffSelfAtt" else model_settings["lr"]

batch_size = dataloader_setting["batch_size"]
output_structure_folder = "/" + output_structure
folder = server_specified_folder + "/TransFD/py_code"
os.chdir(folder)
from DataRelated import DataLoader_allpts
from ModelRelated import compute_loss, get_penalty
from TransformerModel import Transformer

## dataset is created using R. Here we just use the data.
data_type = "/dense" if data_is_dense else "/sparse"
sparsity_error_folder = (data_type + "/w_error" if error else data_type + "/wo_error") if not real else data_type
sparsity_error_folder = "" if data_name == "/Framingham" else sparsity_error_folder

X = pd.read_csv("../Data" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + sparsity_error_folder + "/X_obs.csv", header = None)
T = pd.read_csv("../Data" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + sparsity_error_folder + "/T_obs.csv", header = None)

if my_computer:
    N_comp = 256
    X = X.iloc[:N_comp, :]
    T = T.iloc[:N_comp, :]
    dataloader_setting["batch_size"], model_settings["max_epoch"], model_settings["epoch_CV"] = 32, 2, 2

# if True:
#     model_settings["max_epoch"], model_settings["epoch_CV"] = 3, 2

## picking the best penalty
# denoise_method = "None"

if denoise_method != "None":
    dataLoader = DataLoader_allpts(X, T, dataloader_setting, output_structure = output_structure)
    model_settings["tptsTesting"], model_settings["tptsTraining"], model_settings["min_X"], model_settings["max_X"], model_settings["denoise_method"], model_settings["device"] = dataLoader.tptsTesting, dataLoader.tptsTraining, dataLoader.min_X, dataLoader.max_X, denoise_method, device
if denoise_method == "l2o" or denoise_method == "TVo":
    m, L = X.shape[1] - 1, model_settings["tptsTesting"]
    full_time_grid = np.arange(L)/(L - 1)
    if dataloader_setting["d"] == 0:
        d_T_full_tmp = torch.Tensor(full_time_grid).reshape(1, -1, 1)
        d_T_full = torch.repeat_interleave(d_T_full_tmp, model_settings["batch_size"], dim = 0).to(device)
    else:
        # full_T = np.array([full_time_grid[i] * 100 for i in range(L)])
        d_T_full_tmp = np.zeros([1, dataloader_setting["d"] + 1, L])
        d_T_full_tmp[:, 0, :] = full_time_grid
        for i in range(int(dataloader_setting["d"]/4)):
            d_T_full_tmp[0, 4*i+1, :] = np.sin(10 ** (2 - 4*(i + 1)/dataloader_setting["d"]) * full_time_grid)
            d_T_full_tmp[0, 4*i+2, :] = np.cos(10 ** (2 - 4*(i + 1)/dataloader_setting["d"]) * full_time_grid)
            d_T_full_tmp[0, 4*i+3, :] = np.sin(2 * np.pi * (i + 1) * full_time_grid)
            d_T_full_tmp[0, 4*i+4, :] = np.cos(2 * np.pi * (i + 1) * full_time_grid)
        
        d_T_full_tmp = torch.Tensor(d_T_full_tmp)
        d_T_full = torch.repeat_interleave(d_T_full_tmp, model_settings["batch_size"], dim = 0).to(device)
    d_m_de_full = 1 #torch.Tensor([[1 for _ in range(L)] for _ in range(model_settings["batch_size"])]).to(device) # de for denoise
    d_m_de_mh_full = 1 #torch.Tensor([[1 for _ in range(L)] for _ in range(model_settings["num_heads"] * model_settings["batch_size"])]).to(device) # de for denoise
if denoise_method == "l1w" or denoise_method == "l2w" or denoise_method == "l2o" or denoise_method == "TVo":
    loss_train_history = np.zeros((model_settings["penalty_max_iter"], model_settings["epoch_CV"])) + sys.maxsize
    loss_valid_history = np.zeros((model_settings["penalty_max_iter"], model_settings["epoch_CV"])) + sys.maxsize
    score_weights = np.exp(np.array(range(-model_settings["epoch_CV"], 0))/(model_settings["epoch_CV"]/10))
    score_weights[:int(model_settings["epoch_CV"]/5 * 3)] = 0
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
                # stop
                optimizer.zero_grad()
                batch_size_now = x.shape[0]
                emb_m_mh = torch.repeat_interleave(emb_mask, model_settings["num_heads"], dim = 0)
                e_m_mh = torch.repeat_interleave(e_m, model_settings["num_heads"], dim = 0)
                d_m_mh = torch.repeat_interleave(d_m, model_settings["num_heads"], dim = 0)
                x[:, 0, :] += torch.randn(x[:, 0, :].size(), device = model_settings["device"]) * DAE_sd if model_settings["DAE"] and k > 1 and output_structure != "Vanilla" else 0
                out = model.forward(x, y_t, emb_m_mh, e_m_mh, d_m_mh, d_m)
                loss = compute_loss(out, y, d_m, denoise_method, check_loss = model_settings["check_loss"])
                
                if denoise_method == "l1w":
                    loss += torch.mean(torch.abs(torch.cat([param.view(-1) for param in model.parameters()]))) * penalty
                elif denoise_method == "l2w":
                    loss += torch.mean(torch.square(torch.cat([param.view(-1) for param in model.parameters()]))) * penalty
                else:
                    d_T = d_T_full if batch_size_now == model_settings["batch_size"] else torch.repeat_interleave(d_T_full_tmp, batch_size_now, dim = 0).to(device)
                    d_m_de_mh = 1 #d_m_de_mh_full if batch_size_now == model_settings["batch_size"] else torch.Tensor([[1 for _ in range(L)] for _ in range(model_settings["num_heads"] * batch_size_now)]).to(device) # de for denoise
                    d_m_de = 1 #d_m_de_full if batch_size_now == model_settings["batch_size"] else torch.Tensor([[1 for _ in range(L)] for _ in range(batch_size_now)]).to(device)
                    out_diff1 = torch.diff(model.forward(x, d_T, emb_m_mh, emb_m_mh, d_m_de_mh, d_m_de), 1)
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
                    batch_size_now = x.shape[0]
                    emb_m_mh = torch.repeat_interleave(emb_mask, model_settings["num_heads"], dim = 0)
                    e_m_mh = torch.repeat_interleave(e_m, model_settings["num_heads"], dim = 0)
                    d_m_mh = torch.repeat_interleave(d_m, model_settings["num_heads"], dim = 0)
                    x[:, 0, :] += torch.randn(x[:, 0, :].size(), device = model_settings["device"]) * DAE_sd if model_settings["DAE"] and k > 1 and output_structure != "Vanilla" else 0
                    valid_y = model.forward(x, y_t, emb_m_mh, e_m_mh, d_m_mh, d_m, isTraining = False)
                    valid_loss = compute_loss(valid_y, y, d_m, isTraining = False, check_loss = model_settings["check_loss"])
                    loss_valid.append(valid_loss.item())
            loss_valid_history[penalty_counter, k] = np.mean(loss_valid)
            if model_settings["DAE"] and output_structure != "Vanilla":
                smo_detach = valid_y[0].detach()[:, :, 0]
                org_detach = valid_y[1].detach()[:, :, 0]
                DAE_sd = torch.pow(torch.mean(torch.pow(smo_detach - org_detach, 2)), 1/2).to(device)
                
        score.append(sum(loss_valid_history[penalty_counter, ] * score_weights))
        if penalty_counter >= model_settings["penalty_min_iter"] and np.abs(np.log10(penalty_list[penalty_counter]) - np.log10(penalty_list[penalty_counter - 1])) < 1e-2:
            break
    print(score, flush = True)
    print(penalty_list, flush = True)
    plt.clf()
    fig, ax = plt.subplots()
    ax.scatter(np.log10(penalty_list), score)
    for i in range(len(score)):
        ax.annotate(i, (np.log10(penalty_list)[i], score[i]))
    plt.show()
    plt.savefig("../Plots" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + "/Loss" + output_structure_folder + "/CV/" + sparsity_error_folder.replace("/", "_") + "_" + str(denoise_method) + "_trend_loss.png")
    loss_valid_history = loss_valid_history[:len(score), :]
    penalty = penalty_list[np.argmin(score)]
    rank_penalty_list = np.argsort(-np.array(penalty_list)).tolist()
    score = [score.copy()[rank_penalty_list[i]] for i in range(len(penalty_list))]
    penalty_list = [penalty_list.copy()[rank_penalty_list[i]] for i in range(len(penalty_list))]
    loss_valid_history = loss_valid_history[rank_penalty_list, :]
    plt.clf()
    plt.subplots_adjust(left = 0.2, right = 0.95, top = 0.9, bottom = 0.15)
    plt.plot(list(range(model_settings["epoch_CV"])), np.log10(loss_valid_history.T), lw = 1)
    powers = np.ceil(-np.log10(penalty_list))
    plt.legend([str(round(penalty_list[i] * np.power(10, powers[i]), 2)) + "e-" + str(int(powers[i])) + ", " + str(round(score[i], 4)) for i in range(len(penalty_list))], fontsize = 15)
    plt.xlabel("epoch vs. log(loss)", fontsize = 32)
    plt.ylabel("scores" + " (" + str(denoise_method) + ")", fontsize = 32)
    plt.title("penalty = " + str(round(penalty * np.power(10, np.ceil(-np.log10(penalty))), 2)) + "e-" + str(int(np.ceil(-np.log10(penalty)))), fontsize = 32)
    plt.savefig("../Plots" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + "/Loss" + output_structure_folder + "/CV/" + sparsity_error_folder.replace("/", "_") + "_" + str(denoise_method) + "_loss.png")
    if my_computer:
        plt.show()

## training
dataLoader = DataLoader_allpts(X, T, dataloader_setting, output_structure = output_structure)
model_settings["tptsTesting"], model_settings["tptsTraining"], model_settings["min_X"], model_settings["max_X"], model_settings["denoise_method"], model_settings["device"] = dataLoader.tptsTesting, dataLoader.tptsTraining, dataLoader.min_X, dataLoader.max_X, denoise_method, device
model = Transformer(model_settings, output_structure).to(device)
optimizer = Adam(model.parameters(), lr = model_settings["lr"])
loss_train_history = []
loss_valid_history = []
min_valid_loss = sys.maxsize
save_model_every = 1000

repeat_obj = torch.tensor([model_settings["num_heads"]] * dataloader_setting["batch_size"]).to(device)
for k in range(model_settings["max_epoch"]):
    if k % save_model_every == 0 or k == 100:
        checkpoint = {"model": Transformer(model_settings, output_structure), "state_dict": model.state_dict(), "optimizer" : optimizer.state_dict()}
        torch.save(checkpoint, "../Checkpoints/Tmp" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + output_structure_folder + sparsity_error_folder + "/ckpts_" + str(denoise_method) + "_" + str(k) + ".pth")

    loss_train = []
    loss_valid = []
    dataLoader.shuffle()
    model = model.train()
    # set model training state
    for i, (emb_mask, e_m, d_m, x, y_t, y) in enumerate(dataLoader.get_train_batch()):
        # stop
        # e_m: mask for encoder, randomly mask some points; d_m: mask for decoder, only mask unobserved points
        # x: [B, 2, obs]: x and t, y_t: t, y: x

        optimizer.zero_grad()
        batch_size_now = x.shape[0]
        emb_m_mh = torch.repeat_interleave(emb_mask, model_settings["num_heads"], dim = 0)
        e_m_mh = torch.repeat_interleave(e_m, model_settings["num_heads"], dim = 0)
        d_m_mh = torch.repeat_interleave(d_m, model_settings["num_heads"], dim = 0)
        # emb_m_mh = torch.repeat_interleave(emb_mask, repeat_obj if batch_size_now == batch_size else torch.tensor([model_settings["num_heads"]] * batch_size_now).to(device), dim = 0)
        # e_m_mh = torch.repeat_interleave(e_m, repeat_obj if batch_size_now == batch_size else torch.tensor([model_settings["num_heads"]] * batch_size_now).to(device), dim = 0)
        # d_m_mh = torch.repeat_interleave(d_m, repeat_obj if batch_size_now == batch_size else torch.tensor([model_settings["num_heads"]] * batch_size_now).to(device), dim = 0)
        x[:, 0, :] += torch.randn(x[:, 0, :].size(), device = model_settings["device"]) * DAE_sd if model_settings["DAE"] and k > 1 and output_structure != "Vanilla" else 0
        out = model.forward(x, y_t, emb_m_mh, e_m_mh, d_m_mh, d_m)
        loss = compute_loss(out, y, d_m, denoise_method, check_loss = model_settings["check_loss"])
        
        if denoise_method == "l1w":
            loss += torch.mean(torch.abs(torch.cat([param.view(-1) for param in model.parameters()]))) * penalty
        elif denoise_method == "l2w":
            loss += torch.mean(torch.square(torch.cat([param.view(-1) for param in model.parameters()]))) * penalty
        elif denoise_method == "l2o" or denoise_method == "TVo":
            d_T = d_T_full if batch_size_now == model_settings["batch_size"] else torch.repeat_interleave(d_T_full_tmp, batch_size_now, dim = 0).to(device)
            d_m_de_mh = 1 #d_m_de_mh_full if batch_size_now == model_settings["batch_size"] else torch.Tensor([[1 for _ in range(L)] for _ in range(model_settings["num_heads"] * batch_size_now)]).to(device) # de for denoise
            d_m_de = 1 # d_m_de_full if batch_size_now == model_settings["batch_size"] else torch.Tensor([[1 for _ in range(L)] for _ in range(batch_size_now)]).to(device)
            out_diff1 = torch.diff(model.forward(x, d_T, emb_m_mh, emb_m_mh, d_m_de_mh, d_m_de), 1)
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
            # stop
            batch_size_now = x.shape[0]
            emb_m_mh = torch.repeat_interleave(emb_mask, repeat_obj if batch_size_now == batch_size else torch.tensor([model_settings["num_heads"]] * batch_size_now).to(device), dim = 0)
            e_m_mh = torch.repeat_interleave(e_m, repeat_obj if batch_size_now == batch_size else torch.tensor([model_settings["num_heads"]] * batch_size_now).to(device), dim = 0)
            d_m_mh = torch.repeat_interleave(d_m, repeat_obj if batch_size_now == batch_size else torch.tensor([model_settings["num_heads"]] * batch_size_now).to(device), dim = 0)
            x[:, 0, :] += torch.randn(x[:, 0, :].size(), device = model_settings["device"]) * DAE_sd if model_settings["DAE"] and k > 1 and output_structure != "Vanilla" else 0
            valid_y = model.forward(x, y_t, emb_m_mh, e_m_mh, d_m_mh, d_m, isTraining = False)
            valid_loss = compute_loss(valid_y, y, d_m, isTraining = False, check_loss = model_settings["check_loss"])
            
            if denoise_method == "l1w":
                valid_loss += torch.mean(torch.abs(torch.cat([param.view(-1) for param in model.parameters()]))) * penalty
            elif denoise_method == "l2w":
                valid_loss += torch.mean(torch.square(torch.cat([param.view(-1) for param in model.parameters()]))) * penalty
            elif denoise_method == "l2o" or denoise_method == "TVo":
                d_T = d_T_full if batch_size_now == model_settings["batch_size"] else torch.repeat_interleave(d_T_full_tmp, batch_size_now, dim = 0).to(device)
                d_m_de_mh = d_m_de_mh_full if batch_size_now == model_settings["batch_size"] else torch.Tensor([[1 for _ in range(L)] for _ in range(model_settings["num_heads"] * batch_size_now)]).to(device) # de for denoise
                d_m_de = d_m_de_full if batch_size_now == model_settings["batch_size"] else torch.Tensor([[1 for _ in range(L)] for _ in range(batch_size_now)]).to(device)
                out_diff1 = torch.diff(model.forward(x, d_T, emb_m_mh, emb_m_mh, d_m_de_mh, d_m_de), 1)
                loss += torch.mean(torch.square(torch.diff(out_diff1, 1))) * penalty if denoise_method == "l2o" else torch.mean(torch.abs(out_diff1)) * penalty
            loss_valid.append(valid_loss.item())
    
    if np.mean(loss_valid) < min_valid_loss:
        print("The best model is " + str(k) + ".")
        checkpoint_best = {"model": Transformer(model_settings, output_structure), "state_dict": model.state_dict()} #, "optimizer" : optimizer.state_dict()}
        torch.save(checkpoint_best, "../Checkpoints" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + output_structure_folder + sparsity_error_folder + "/best_ckpts_" + str(denoise_method) + ".pth")
        min_valid_loss = np.mean(loss_valid)
    
    loss_valid_history.append(np.mean(loss_valid))
    if model_settings["DAE"] and output_structure != "Vanilla":
        smo_detach = valid_y[0].detach()[:, :, 0]
        org_detach = valid_y[1].detach()[:, :, 0]
        DAE_sd = torch.pow(torch.mean(torch.pow(smo_detach - org_detach, 2)), 1/2).to(device)
    
    if k % 1 == 0 and my_computer:
        print("epoch:", k, "training loss = ", loss_train_history[-1], "validation loss = ", loss_valid_history[-1], flush = True)
    
history = np.array([loss_train_history, loss_valid_history])
filename = "../LossHistory" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + output_structure_folder + sparsity_error_folder + "/" + str(denoise_method) + "_loss.npy"

with open(filename, 'wb') as file:
    np.save(file, history)

# with open(filename, 'rb') as file:
#     a = np.load(file)
# loss_train_history = a[[0]].squeeze(0)
# loss_valid_history = a[[1]].squeeze(0)

plt.clf()
plt.subplots_adjust(left=0.2, right=0.9, top=0.95, bottom=0.15)
epoch_range = list(range(1, len(loss_train_history) + 1))
plt.plot(epoch_range, np.log10(loss_train_history), label = "train", lw = 1)
plt.plot(epoch_range, np.log10(loss_valid_history), label = "valid", lw = 1)
plt.xlabel("epoch vs. log(loss)", fontsize = 32)
plt.ylabel(output_structure + "/" + denoise_method, fontsize = 32);
plt.legend(fontsize = 15)
plt.savefig("../Plots" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + "/Loss" + output_structure_folder + "/" + str(denoise_method) + sparsity_error_folder.replace("/", "_") + "_loss.png")
if my_computer:
    plt.show()
plt.close()

# filename = "../LossHistory" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + output_structure_folder + sparsity_error_folder + '/shelve.out'
# my_shelf = shelve.open(filename, 'n') # 'n' for new

# for key in dir():
#     try:
#         my_shelf[key] = globals()[key]
#     except TypeError:
#         print('ERROR shelving: {0}'.format(key))
# my_shelf.close()




# model = model.eval()
# for emb_mask, e_m, d_m, x, y_t, y in dataLoader.get_train_batch():
#     x[:, 0, :] += torch.randn(x[:, 0, :].size()) * DAE_sd if model_settings["DAE"] and k > 1e3 + 1 else 0
#     emb_m_mh = torch.repeat_interleave(emb_mask, torch.tensor([model_settings["num_heads"]] * e_m.shape[0]), dim = 0).to(device)
#     e_m_mh = torch.repeat_interleave(e_m, torch.tensor([model_settings["num_heads"]] * e_m.shape[0]), dim = 0).to(device)
#     d_m_mh = torch.repeat_interleave(d_m, torch.tensor([model_settings["num_heads"]] * d_m.shape[0]), dim = 0).to(device)
#     d_m, x, y_t, y = d_m.to(device), x.to(device), y_t.to(device), y.to(device)
#     y_pred = model.forward(x, y_t, emb_m_mh, e_m_mh, d_m_mh, d_m, isTraining = False)
#     # np.log(compute_loss(y_pred, y, d_m, isTraining = False, check_loss = model_settings["check_loss"]))
#     [y_pred, org] = y_pred if isinstance(y_pred, list) else [0, y_pred]
#     break
# 
# for i in range(y_t.size()[0]):
#     plt.clf()
#     plt.scatter(y_t[i, 0, :].detach().numpy(), y[i, :].detach().numpy())
#     plt.plot(y_t[i, 0, :].detach().numpy(), org[i, :].detach().numpy())
#     if output_structure != "Vanilla":
#         plt.plot(y_t[i, 0, :].detach().numpy(), y_pred[i, :].detach().numpy())
#     plt.show()
