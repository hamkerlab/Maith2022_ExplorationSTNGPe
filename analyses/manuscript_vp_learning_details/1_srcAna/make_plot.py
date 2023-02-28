import numpy as np
import pylab as plt
from extra_functions import (
    get_output_vp,
    get_selected_cluster,
    get_centered_selected_out_cluster_m,
    make_regression_plot,
    make_single_vps_plot,
    make_correlation_plot,
)

import sys

experiment_idx = int(sys.argv[1])

rng = np.random.default_rng(0)

font = {"family": "Arial", "weight": "normal", "size": 8}
bold_font = {"family": "Arial", "weight": "bold", "size": 8}
large_bold_font = {"family": "Arial", "weight": "bold", "size": 10}
plt.rc("font", **font)

num_vps = 10
experiment = ["cluster", "rare"][experiment_idx]
folder = {
    "cluster": "../../../psychExp/exp1_final/4_dataEv/outputs_vps/",
    "rare": "../../../psychExp/exp_rev_1_final/4_dataEv/outputs_vps/",
}[experiment]
### num_valid_rs:
### cluster: 48 blocks --> 47 rule switches + 3 pauses --> 44 valid rule switches
### rare: 70 blocks -minus initial-> 69 rule switches + 4 pauses --> 65 valid rule switches
num_valid_rs = {"cluster": 44, "rare": 65}[experiment]
post_switch_trials = 7  # how many trials after switch should be analyzed
mode = "correlation"
fake_samples = 0

"""
Performance Plot of 10 vps:

initial learning blocks: x-axis trials from beginning, y-axis probability selecting rewarded action
reversal learning blocks: x-axis trials from last trial of previous block, y-axis probability of selecting previous and new rewarded action



"""


### LOAD DATA

selected_in_cluster = np.zeros((num_vps, num_valid_rs))
selected_out_cluster = np.zeros((num_vps, num_valid_rs))
for vp_id in range(num_vps):
    ### GET DATA OF TRIALS
    trials, correct, decision, _, start, dopInp, block, cluster = get_output_vp(
        folder, vp_id + 1
    )

    ### EXCLUDE DATA OF INITIAL TRAININGS BLOCKS
    familiarization_phase_blocks = {"cluster": 2, "rare": 1}[experiment]
    correct, decision, start, dopInp, block = (
        correct[block > familiarization_phase_blocks],
        decision[block > familiarization_phase_blocks],
        start[block > familiarization_phase_blocks],
        dopInp[block > familiarization_phase_blocks],
        block[block > familiarization_phase_blocks],
    )

    ### get how often in cluster/out cluster during exploration trials
    selected_in_cluster[vp_id], selected_out_cluster[vp_id] = get_selected_cluster(
        correct, decision, block, cluster, experiment
    )


### CHECK SINGLE VPS
make_single_vps_plot(
    f"../3_results/single_vps_{experiment}.svg", selected_out_cluster, bold_font
)

### GET CENTERED AVERAGE OF VPS
(
    selected_out_cluster_m,
    selected_out_cluster_sd,
    last_out_cluster_idx,
    selected_out_cluster_centered,
) = get_centered_selected_out_cluster_m(selected_out_cluster)


make_single_vps_plot(
    f"../3_results/single_vps_centered_{experiment}.svg",
    selected_out_cluster_centered,
    bold_font,
    mark_initial_nans=True,
)

if mode == "correlation":
    ### GET CORRELATION OF VPS
    corr_coef_vps, corr_p_val_vps, df_vps, CI = make_correlation_plot(
        f"../3_results/average_vps_classic_{experiment}.svg",
        selected_out_cluster,
        bold_font=bold_font,
    )
    corr_coef_vps, corr_p_val_vps, df_vps, CI = make_correlation_plot(
        f"../3_results/average_vps_scatter_bars_{experiment}.svg",
        selected_out_cluster,
        bold_font=bold_font,
        mode="scatter_bars",
    )
    corr_coef_vps, corr_p_val_vps, df_vps, CI = make_correlation_plot(
        f"../3_results/average_vps_scatter_bars_dif_align_{experiment}.svg",
        selected_out_cluster,
        bold_font=bold_font,
        mode="scatter_bars",
        bottom=1,
    )
    corr_coef_vps, corr_p_val_vps, df_vps, CI = make_correlation_plot(
        f"../3_results/average_vps_scatter_circles_{experiment}.svg",
        selected_out_cluster,
        bold_font=bold_font,
        mode="scatter_circles",
        bottom=1,
    )
    corr_coef_vps, corr_p_val_vps, df_vps, CI = make_correlation_plot(
        f"../3_results/average_vps_scatter_circles_size_{experiment}.svg",
        selected_out_cluster,
        bold_font=bold_font,
        mode="scatter_circles_size",
        bottom=1,
    )
    corr_coef_vps, corr_p_val_vps, df_vps, CI = make_correlation_plot(
        f"../3_results/average_vps_bars_{experiment}.svg",
        selected_out_cluster,
        bold_font=bold_font,
        mode="bars",
    )
    with open(f"../3_results/values_{experiment}.txt", "w") as f:
        print(
            "correlations results: r =",
            corr_coef_vps,
            " p =",
            corr_p_val_vps,
            " df =",
            df_vps,
            " 95%CI =",
            CI.round(2),
            file=f,
        )

if mode == "regression":
    ### MAKE REGRESSION FOR VPS AND GET SLOPE
    regression_results_vps = make_regression_plot(
        f"../3_results/average_vps_{experiment}.svg",
        selected_out_cluster_m,
        selected_out_cluster_sd,
        bold_font=bold_font,
    )
    with open(f"../3_results/values_{experiment}.txt", "w") as f:
        print("regression results:", regression_results_vps, file=f)
