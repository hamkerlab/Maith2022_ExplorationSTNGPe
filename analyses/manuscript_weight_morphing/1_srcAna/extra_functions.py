import numpy as np
import matplotlib.pylab as plt
from scipy import stats


def get_output(x, sim_id):
    """
    loads data of a simulation (selection_simX, output_simX)
    returns as list:
        trial_number
        correct[trials]
        decisions[trials]
        most_frequent_correct_action
        startTime[trial]
        dopamine_inputTime[trial]
    """
    trials = 0
    selection = np.load(x + "/selection_sim" + str(sim_id) + ".npy")
    while selection[trials, 1] != 0:
        trials += 1

    file = open(x + "/output_sim" + str(sim_id), "r")
    zeile = file.readline()
    correct = np.zeros(trials)
    decision = np.zeros(trials)
    start = np.zeros(trials)
    dopInp = np.zeros(trials)
    morph = np.zeros(trials)
    i = 0
    try:
        while 1:
            zeile = file.readline()
            liste = zeile.split("\t")
            correct[i] = liste[4]
            decision[i] = liste[5]
            start[i] = liste[1]
            dopInp[i] = liste[2]
            morph[i] = liste[8]
            i += 1
    except:
        file.close()

    frequentAction = [2, 3, 4, 5][np.histogram(correct, [2, 3, 4, 5, 10])[0].argmax()]

    return [trials, correct, decision, frequentAction, start, dopInp, morph]


def initial_visualization(morph_selections):
    """
    generates the initial overview of the data
    """
    ### get for each morph value a histogram of selections
    morph_val_arr = np.unique(morph_selections[:, 0])
    morph_hist_arr = np.zeros((morph_val_arr.size, 5))
    for morph_idx, morph_val in enumerate(morph_val_arr):
        mask = morph_selections[:, 0] == morph_val
        hist, _ = np.histogram(
            morph_selections[mask, 1], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        )
        morph_hist_arr[morph_idx] = hist

    ### get for each morph value nr in vs out cluster
    in_vs_out_arr = np.concatenate(
        [
            np.sum(morph_hist_arr[:, 1:4], 1)[:, None],
            (morph_hist_arr[:, 0] + morph_hist_arr[:, 4])[:, None],
        ],
        1,
    )
    in_vs_out_arr_norm = in_vs_out_arr / np.sum(in_vs_out_arr, 1)[:, None]

    plt.figure()
    x = morph_val_arr
    plt.fill_between(
        x,
        in_vs_out_arr_norm[:, 0],
        np.sum(in_vs_out_arr_norm, 1),
        label="out cluster",
        color="brown",
    )
    plt.fill_between(x, in_vs_out_arr_norm[:, 0], label="in cluster", color="green")
    plt.xlabel("weight change [%]")
    plt.ylabel("portion of selections")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../3_results/initial_vis_weight_morph.svg")


def make_correlation_plot(title, arr, plot=1, bold_font=None, mode="bars"):
    """
    arr: n*m arry with data
    first dim = vps, second dim = data over blocks

    returns correlation between blocks and data values
    """
    x = arr[:, 0]
    y = arr[:, 2]

    r, p_val = stats.spearmanr(x, y)
    n = len(x)
    df = n - 2
    ### calculate 95% CI using Z/norm-distribution
    z_crit = stats.norm.ppf(0.975)
    CI_fisher_transformed = np.arctanh(r) + np.array([-1, 1]) * z_crit * (
        1 / np.sqrt(n - 3)
    )
    CI = np.tanh(CI_fisher_transformed)

    if plot and mode == "bars":
        make_correlation_plot_bars(x, y, bold_font, p_val, r, title)
    elif plot and mode == "incluster":
        make_correlation_plot_incluster(x, y, bold_font, p_val, r, title)

    return [r, p_val, df, CI]


def make_correlation_plot_bars(x, y, bold_font, p_val, r, title):
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    bin_arr = np.arange(y_unique.max() + 2) - 0.5
    plot_y = np.zeros((x_unique.size, bin_arr.size - 1))
    for idx_x_val, x_val in enumerate(x_unique):
        hist, _ = np.histogram(y[x == x_val], bins=bin_arr)

        plot_y[idx_x_val, :] = np.array(
            [np.sum(hist[0 : idx + 1]) for idx in range(hist.size)]
        )

    plot_y_norm = plot_y / np.max(plot_y, 1)[:, None]

    plt.figure(figsize=(8.5 / 2.54, 7 / 2.54), dpi=500)
    for plot_idx in range(plot_y_norm.shape[1]):

        plot_y_1 = plot_y_norm[:, plot_y_norm.shape[1] - 1 - plot_idx]

        plt.bar(
            x_unique,
            plot_y_1,
            width=np.diff(np.sort(x_unique))[0],
            label=f"{plot_idx}",
            color=f"C{plot_idx}",
        )
    plt.xlabel("weight change [%]", **bold_font)
    plt.ylabel("portion of selections", **bold_font)
    if p_val >= 0.001:
        plt.text(
            0.97,
            0.97,
            "$r$ = "
            + str(round(r, 2))
            + ", $p$ = ."
            + str(round(p_val, 3)).split(".")[1],
            ha="right",
            va="top",
            transform=plt.gca().transAxes,
        )
    else:
        plt.text(
            0.97,
            0.97,
            "$r$ = " + str(round(r, 2)) + ", $p$ < .001",
            ha="right",
            va="top",
            transform=plt.gca().transAxes,
        )
    plt.legend()
    plt.tight_layout()
    plt.savefig(title)


def make_correlation_plot_incluster(x, y, bold_font, p_val, r, title):
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    bin_arr = np.arange(y_unique.max() + 2) - 0.5
    plot_y = np.zeros((x_unique.size, bin_arr.size - 1))
    for idx_x_val, x_val in enumerate(x_unique):
        hist, _ = np.histogram(y[x == x_val], bins=bin_arr)

        plot_y[idx_x_val, :] = hist

    plot_y_norm = plot_y / np.sum(plot_y, 1)[:, None]

    plt.figure(figsize=(8.5 / 2.54, 7 / 2.54), dpi=500)
    plt.bar(
        x_unique,
        height=plot_y_norm[:, 1],
        width=np.diff(np.sort(x_unique))[0],
        color="k",
    )
    plt.ylim(0, 1)
    plt.xlim(
        x_unique.min() - np.diff(np.sort(x_unique))[0] / 2,
        x_unique.max() + np.diff(np.sort(x_unique))[0] / 2,
    )
    plt.xlabel("weight change [%]", **bold_font)
    plt.ylabel("portion of in-cluster selections", **bold_font)

    if p_val >= 0.001:
        plt.text(
            1 - 0.97,
            0.97,
            "$r$ = "
            + str(round(r, 2))
            + ", $p$ = ."
            + str(round(p_val, 3)).split(".")[1],
            ha="left",
            va="top",
            transform=plt.gca().transAxes,
        )
    else:
        plt.text(
            1 - 0.97,
            0.97,
            "$r$ = " + str(round(r, 2)) + ", $p$ < .001",
            ha="left",
            va="top",
            transform=plt.gca().transAxes,
        )

    plt.tight_layout()
    plt.savefig(title)
