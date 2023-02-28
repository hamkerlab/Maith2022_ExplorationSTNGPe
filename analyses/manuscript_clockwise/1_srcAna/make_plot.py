from extra_functions import (
    get_output_vp,
    get_exploration_selections,
    check_clockwise,
    change_to_one_hot_encode,
)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sys
from CompNeuroPy.extra_functions import create_cm

font = {"family": "Arial", "weight": "normal", "size": 8}
bold_font = {"family": "Arial", "weight": "bold", "size": 8}
large_bold_font = {"family": "Arial", "weight": "bold", "size": 10}
plt.rc("font", **font)

experiment_idx = int(sys.argv[1])
experiment = ["cluster", "rare"][experiment_idx]
folder = {
    "cluster": "../../../psychExp/exp1_final/4_dataEv/outputs_vps/",
    "rare": "../../../psychExp/exp_rev_1_final/4_dataEv/outputs_vps/",
}[experiment]
vpAnz = 10
### num_valid_rs:
### cluster: 48 blocks --> 47 rule switches + 3 pauses --> 44 valid rule switches
### rare: 70 blocks -minus initial-> 69 rule switches + 4 pauses --> 65 valid rule switches
num_valid_rs = {"cluster": 44, "rare": 65}[experiment]

clock_wise_arr = np.zeros((vpAnz, num_valid_rs))
clock_wise_arr_one_hot = np.zeros((vpAnz, num_valid_rs, 3))
for vp in range(vpAnz):
    output = get_output_vp(folder, vp + 1)

    ### GET DATA OF TRIALS
    trials, correct, decision, _, start, dopInp, block, cluster = get_output_vp(
        folder, vp + 1
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

    ### get for each valid rule switch the selections until exploration is finished
    ### valid = breaks removed
    ### if blocks not succesfull --> None value in list
    exploration_selection_list = get_exploration_selections(
        correctList=correct,
        decisionList=decision,
        blockList=block,
        experiment=experiment,
    )

    ### check if these selections are clockwise or anticlockwise or other
    ### if block was not successful and None value is given for exploration --> classify as other
    ### --> values 0,1,2
    clock_wise_arr[vp] = check_clockwise(exploration_selection_list)
    ### chagne to one hot encode --> 3 arrays for each explroation, 1 of 3 values is 1 others 0
    clock_wise_arr_one_hot[vp] = change_to_one_hot_encode(clock_wise_arr[vp])

### average over vps --> 3 arrays remain
### idx 0 = exploration periods
### idx 1 = 3 values for clockwise, anticlockwise, other, together sum to 1
clock_wise_arr_one_hot_mean = np.mean(clock_wise_arr_one_hot, 0)

### plot data of all exploration periods
### clockwise.svg
plt.figure()
plt.title("Search Strategy")
plt.ylabel("portion")
plt.xlabel("exploration phases")
x = np.arange(num_valid_rs)
clock = clock_wise_arr_one_hot_mean[:, 0]
clock_anti = np.sum(clock_wise_arr_one_hot_mean[:, :2], 1)
plt.bar(x, height=clock_anti, width=1, color="b", label="anti clockwise")
plt.bar(x, height=clock, width=1, color="r", label="clockwise")
plt.legend()
plt.tight_layout()
plt.savefig(f"../3_results/clockwise_{experiment}.svg")


### plot data of all exploration periods splitted
### clockwise_splitted.svg
plt.figure()
x = np.arange(num_valid_rs)

plt.subplot(311)
plt.title("Search Strategy")
clock = clock_wise_arr_one_hot_mean[:, 0]
plt.bar(x, height=clock, width=1, color="r")
plt.ylim(0, 1)
plt.ylabel("portion\nclockwise")
plt.gca().set_xticklabels([])

plt.subplot(312)
clock_anti = clock_wise_arr_one_hot_mean[:, 1]
plt.bar(x, height=clock_anti, width=1, color="b")
plt.ylim(0, 1)
plt.ylabel("portion\nanti clockwise")
plt.gca().set_xticklabels([])

plt.subplot(313)
plt.bar(x, height=clock + clock_anti, width=1, color="k")
plt.ylim(0, 1)
plt.ylabel("portion\nboth")
plt.xlabel("exploration phases")

plt.tight_layout()
plt.savefig(f"../3_results/clockwise_splitted_{experiment}.svg")


### split the 71 explroation periods into 7 blocks average data for blocks
nr_sub_blocks = 7
clock_wise_arr_one_hot_mean_clustered = np.zeros((nr_sub_blocks, 3))
for splitted_arr_idx, splitted_arr in enumerate(
    np.array_split(clock_wise_arr_one_hot_mean, nr_sub_blocks, axis=0)
):
    clock_wise_arr_one_hot_mean_clustered[splitted_arr_idx, :] = np.mean(
        splitted_arr, 0
    )

### plot data of clustered exploration periods
### clockwise_clustered.svg
plt.figure()
plt.title("Search Strategy")
plt.ylabel("portion")
plt.xlabel("experiment progress")
x = np.arange(nr_sub_blocks)
clock = clock_wise_arr_one_hot_mean_clustered[:, 0]
clock_anti = np.sum(clock_wise_arr_one_hot_mean_clustered[:, :2], 1)
plt.bar(x, height=clock_anti, width=1, color="b", label="anti clockwise")
plt.bar(x, height=clock, width=1, color="r", label="clockwise")
plt.legend()
plt.tight_layout()
plt.savefig(f"../3_results/clockwise_clustered_{experiment}.svg")


### plot data of all exploration periods splitted
### clockwise_clustered_splitted.svg
### attention: clock_anti from previous plot = clok + anti-clock selections
plt.figure()
plt.subplot(311)
plt.title("Search Strategy")
plt.bar(x, height=clock, width=1, color="r")
plt.ylim(0, 1)
plt.ylabel("portion\nclockwise")
plt.gca().set_xticklabels([])

plt.subplot(312)
plt.bar(x, height=clock_anti - clock, width=1, color="b")
plt.ylim(0, 1)
plt.ylabel("portion\nanti clockwise")
plt.gca().set_xticklabels([])

plt.subplot(313)
plt.bar(x, height=clock_anti, width=1, color="k")
plt.ylim(0, 1)
plt.ylabel("portion\nboth")
plt.xlabel("exploration phases")

plt.tight_layout()
plt.savefig(f"../3_results/clockwise_clustered_splitted_{experiment}.svg")


### plot data as matrix
color_clockwise = (216 / 255, 27 / 255, 96 / 255)
color_anti = (30 / 255, 136 / 255, 229 / 255)
color_other = "lightgray"
### prepare plot
horizontal_lines = False
matrix_ratio = [16, 9]
hist_ratio = 1 / 3
color_list = [color_clockwise, color_anti, color_other]  # clock, anticlock, other
cmap_clockwise = create_cm(color_list)
width_ratios = [matrix_ratio[0], hist_ratio * matrix_ratio[1]]
height_ratios = [hist_ratio * matrix_ratio[1], matrix_ratio[1]]
gs_kw = dict(width_ratios=width_ratios, height_ratios=height_ratios)
fig, axd = plt.subplot_mosaic(
    [["top", "legend"], ["main", "right"]],
    gridspec_kw=gs_kw,
    figsize=(8.5 / 2.54, (8.5 * sum(height_ratios) / sum(width_ratios)) / 2.54),
)

### plot matrix
axd["main"].imshow(
    clock_wise_arr,
    cmap=cmap_clockwise,
    aspect="auto",
    extent=[-0.5, num_valid_rs - 0.5, 0.5, vpAnz + 0.5],
    interpolation=None,
)
axd["main"].xaxis.set_major_locator(MaxNLocator(integer=True))
axd["main"].yaxis.set_major_locator(MaxNLocator(integer=True))
axd["main"].set_xlabel("exploration phases", **bold_font)
axd["main"].set_ylabel("participants", **bold_font)

### plot top histogram
x = np.arange(num_valid_rs)
clock = clock_wise_arr_one_hot_mean[:, 0]
clock_anti = clock_wise_arr_one_hot_mean[:, 1]
axd["top"].bar(
    x, height=1, width=1, facecolor=color_list[2], edgecolor=color_list[2], linewidth=0
)
axd["top"].bar(
    x,
    height=clock + clock_anti,
    width=1,
    facecolor=color_list[1],
    edgecolor=color_list[1],
    linewidth=0,
)
axd["top"].bar(
    x,
    height=clock,
    width=1,
    facecolor=color_list[0],
    edgecolor=color_list[0],
    linewidth=0,
)
axd["top"].set_xticks(axd["main"].get_xticks())
axd["top"].set_xticklabels([])
axd["top"].set_xlim(-0.5, num_valid_rs - 0.5)
axd["top"].set_ylim(0, 1)

### plot right histogram
### here mean over exploration phases not vps
clock_wise_arr_one_hot_mean_vps = np.mean(clock_wise_arr_one_hot, 1)
x = np.flip(np.arange(vpAnz) + 1)
clock = clock_wise_arr_one_hot_mean_vps[:, 0]
clock_anti = clock_wise_arr_one_hot_mean_vps[:, 1]
axd["right"].barh(
    x, width=1, height=1, facecolor=color_list[2], edgecolor=color_list[2], linewidth=0
)
axd["right"].barh(
    x,
    width=clock + clock_anti,
    height=1,
    facecolor=color_list[1],
    edgecolor=color_list[1],
    linewidth=0,
)
axd["right"].barh(
    x,
    width=clock,
    height=1,
    facecolor=color_list[0],
    edgecolor=color_list[0],
    linewidth=0,
)
axd["right"].set_yticks(axd["main"].get_yticks())
axd["right"].set_yticklabels([])
axd["right"].set_ylim(0.5, vpAnz + 0.5)
axd["right"].set_xlim(0, 1)

### plot horizontal lines
if horizontal_lines:
    for y_pos in range(vpAnz):
        y_pos = y_pos + 0.5
        axd["main"].axhline(y_pos, color="k", linewidth=0.4)
        axd["right"].axhline(y_pos, color="k", linewidth=0.4)

### set background to lightgray
axd["main"].set_facecolor("lightgray")
axd["top"].set_facecolor("lightgray")
axd["right"].set_facecolor("lightgray")

### create legend
axd["legend"].bar(0, 0, color=color_list[0], label="clockwise")
axd["legend"].bar(0, 0, color=color_list[1], label="counter-clockwise")
axd["legend"].bar(0, 0, color=color_list[2], label="other")
axd["legend"].axis("off")
axd["legend"].legend(loc=3)


plt.savefig(f"../3_results/clockwise_matrix_{experiment}.svg", dpi=300)
