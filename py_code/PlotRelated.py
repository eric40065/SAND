import numpy as np
import matplotlib.pyplot as plt
def get_plot_scores_scatter(my_computer, penalty_list, score, loss_valid_history, iidt, real, data_name, output_structure_folder, sparsity_error_folder, denoise_method):
    plt.clf()
    fig, ax = plt.subplots()
    ax.scatter(np.log10(penalty_list), score)
    for i in range(len(score)):
        ax.annotate(i, (np.log10(penalty_list)[i], score[i]))
    if my_computer:
        plt.show()
    plt.savefig("../Plots" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + "/Loss" + output_structure_folder + "/CV/" + sparsity_error_folder.replace("/", "_") + "_" + str(denoise_method) + "_trend_loss.png")
    loss_valid_history = loss_valid_history[:len(score), :]
    penalty = penalty_list[np.argmin(score)]
    rank_penalty_list = np.argsort(-np.array(penalty_list)).tolist()
    score = [score.copy()[rank_penalty_list[i]] for i in range(len(penalty_list))]
    penalty_list = [penalty_list.copy()[rank_penalty_list[i]] for i in range(len(penalty_list))]
    loss_valid_history = loss_valid_history[rank_penalty_list, :]

def get_plot_scores_trend(my_computer, penalty_list, score, model_settings, loss_valid_history, iidt, real, data_name, output_structure_folder, sparsity_error_folder, denoise_method):
    plt.clf()
    penalty = penalty_list[np.argmin(score)]
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
