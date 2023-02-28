import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LinearRegression
import warnings
from scipy import stats
from CompNeuroPy import create_cm


def get_output_vp(x, vpID):
    """
    loads data of a subject (output_vpX)
    returns as list:
        trial_number
        correct[trials]
        decisions[trials]
        most_frequent_correct_action
        startTime[trial]
        feedbackTime[trial]
    """
    trials = 0
    file = open(x + "/output_vp" + str(vpID), "r")
    try:
        while 1:
            zeile = file.readline()
            _ = zeile[0]
            trials += 1
    except:
        file.close()

    file = open(x + "/output_vp" + str(vpID), "r")
    correct = np.zeros(trials)
    decision = np.zeros(trials)
    start = np.zeros(trials)
    dopInp = np.zeros(trials)
    block = np.zeros(trials)
    i = 0
    try:
        while 1:
            zeile = file.readline()
            liste = zeile.split("\t")
            correct[i] = liste[4]
            decision[i] = liste[5]
            start[i] = liste[1]
            dopInp[i] = liste[2]
            block[i] = liste[8]
            i += 1
    except:
        file.close()

    frequentAction = [1, 2, 3, 4, 5][
        np.histogram(correct, [1, 2, 3, 4, 5, 10])[0].argmax()
    ]

    _ = np.histogram(correct, [1, 2, 3, 4, 5, 10])[0]
    _.sort()
    cluster = np.array([1, 2, 3, 4, 5])[
        np.histogram(correct, [1, 2, 3, 4, 5, 10])[0] > _[1]
    ]

    return [trials, correct, decision, frequentAction, start, dopInp, block, cluster]


# def get_response_times(start, dopInp, decision):
#     """
#     start: numpy array with start times of trials
#     dopInp: numpy array with feedback onset time of trials
#     decision: numpy array with selected actions

#     returns the response times of all trials
#     """

#     end = dopInp.copy()
#     end[decision == 0] = start[decision == 0] + 400
#     ret = end - start
#     return ret


# def get_block_trial_indizes(correct, block, post_switch_trials, experiment):
#     """
#     correct: numpy array with correct actions
#     block: numpy array with block of experiment
#     post_switch_trials: int, number of trials of block which should be returned

#     returns: matrix, rows = indizes of switches + following trials
#     """

#     ### detect rule switches
#     switch_trials = get_switch_trials(correct, block) - 1

#     ### exclude rule switches which are during breaks
#     if experiment == "cluster":
#         not_use_these_rule_switches = [11, 23, 35]
#     elif experiment == "rare":
#         not_use_these_rule_switches = [13, 27, 41, 55]
#     switch_trials = np.delete(switch_trials, not_use_these_rule_switches)

#     block_trial_indizes = np.array(
#         [switch_trials + i for i in range(post_switch_trials + 1)]
#     ).T  # matrix rows = indizes of switches + following trials
#     return block_trial_indizes


# def get_successful_block_list(correct, decision, block, experiment):
#     """
#     correct: numpy array with correct actions
#     decision: numpy array with selected actions
#     block: numpy array with block of experiment

#     returns: matrix, rows = indizes of switches + following trials
#     """

#     block_start_end_list = np.array(
#         [get_block_start_list(correct, block), get_block_end_list(correct, block)]
#     ).T

#     ### exclude rule switches which are during breaks
#     if experiment == "cluster":
#         ### not_use_these_rule_switches = [11, 23, 35]
#         not_use_these_blocks = [12, 24, 36]
#     elif experiment == "rare":
#         ### not_use_these_rule_switches = [13, 27,41,55]
#         not_use_these_blocks = [14, 28, 42, 56]
#     block_start_end_list = np.delete(block_start_end_list, not_use_these_blocks, 0)

#     ### generate list of arrays for each block if decision was correct
#     correct_decision_block = []
#     for idx, block_start_end in enumerate(block_start_end_list):
#         start = block_start_end[0]
#         end = block_start_end[1]
#         correct_decision_block.append(
#             1 * (correct[start : end + 1] == decision[start : end + 1])
#         )

#     ### get successful blocks
#     successful_block_list = np.zeros(len(correct_decision_block))
#     for idx, correct_decisions in enumerate(correct_decision_block):
#         successful_block_list[idx] = is_block_successful(correct_decisions)

#     return successful_block_list


def get_switch_trials(correct, block):
    """
    returns at which indizes the ruleswitch occurs (first trial with new rule)

    correct: numpy array with correct actions
    block: numpy array with block of experiment
    """
    ### detect rule switches
    switch_trials = np.where(np.diff(correct))[0]
    ### detect block switch (if not already rule switch)
    block_switch_list = np.where(np.diff(block))[0]
    for block_switch in block_switch_list:
        if not (block_switch in switch_trials):
            switch_trials = np.sort(np.append(block_switch, switch_trials))

    ret = switch_trials + 1
    return ret


def get_block_start_list(correct, block):
    """
    returns indizes of the beginnings of all blocks

    correct: numpy array with correct actions
    block: numpy array with block of experiment
    """
    ret = np.insert(get_switch_trials(correct, block), 0, 0)
    return ret


def get_block_end_list(correct, block):
    """
    returns indizes of the ends of all blocks

    correct: numpy array with correct actions
    block: numpy array with block of experiment
    """
    switch_trials = get_switch_trials(correct, block)
    ret = np.insert(switch_trials - 1, switch_trials.size, correct.size - 1)
    return ret


def is_block_successful(block_correct_decisions):
    """
    checks if block decisions: 7 consecutive correct decisions

    block_correct_decisions: array if decisions of block are correct

    returns if block is successful
    """
    diffs = np.where(np.diff(block_correct_decisions))[0] + 1
    diffs = np.insert(diffs, 0, 0)
    diffs = np.insert(diffs, diffs.size, block_correct_decisions.size)
    block_sequence_max_len = 0
    for j in range(diffs.size - 1):
        block_sequence = block_correct_decisions[diffs[j] : diffs[j + 1]]
        ### check if sequence = correct decisions
        if block_sequence[0] == 1:
            ### check if this is the new longest sequence
            if len(block_sequence) > block_sequence_max_len:
                block_sequence_max_len = len(block_sequence)
    ret = block_sequence_max_len >= 7
    return ret


def remove_invalid_blocks(arr_like, experiment):
    """
    arr_like: array or list, first index = blocks
    """
    ### define which blocks shouild be removed from arr_like
    ### exclude rule switches which are during breaks
    if experiment == "cluster":
        ### not_use_these_rule_switches = [11, 23, 35]
        not_use_these_blocks = [12, 24, 36]
    elif experiment == "rare":
        ### not_use_these_rule_switches = [13, 27,41,55]
        not_use_these_blocks = [14, 28, 42, 56]

    if isinstance(arr_like, list):
        ### it's a list
        for index in sorted(not_use_these_blocks, reverse=True):
            del arr_like[index]
    else:
        ### it should be an array
        arr_like = list(arr_like)
        for index in sorted(not_use_these_blocks, reverse=True):
            del arr_like[index]
        arr_like = np.array(arr_like)

    return arr_like


def remove_invalid_ruleswitches(arr_like, experiment):
    """
    arr_like: array or list, first index = blocks
    """
    ### define which blocks shouild be removed from arr_like
    ### exclude rule switches which are during breaks
    if experiment == "cluster":
        not_use_these_rule_switches = [11, 23, 35]
    elif experiment == "rare":
        not_use_these_rule_switches = [13, 27, 41, 55]

    if isinstance(arr_like, list):
        ### it's a list
        for index in sorted(not_use_these_rule_switches, reverse=True):
            del arr_like[index]
    else:
        ### it should be an array
        arr_like = list(arr_like)
        for index in sorted(not_use_these_rule_switches, reverse=True):
            del arr_like[index]
        arr_like = np.array(arr_like)

    return arr_like


# def get_how_long_until_rewarded(correct, decision, block, experiment):
#     """
#     correct: numpy array with correct actions
#     decision: numpy array with selected actions
#     block: numpy array with block of experiment

#     returns: for each valid block, how many trials(errors) before new correct consecutively selected
#     """

#     ### generate list of arrays for each block if decision was correct
#     correct_decision_block = get_correct_decision_block(correct, decision, block)

#     ### remove blocks after breaks
#     correct_decision_block = remove_invalid_blocks(correct_decision_block, experiment)

#     ### get errors of remaining successful blocks
#     errors_block_list = np.zeros(len(correct_decision_block))
#     for idx, correct_decisions in enumerate(correct_decision_block):
#         if is_block_successful(correct_decisions):
#             errors_block_list[idx] = get_error_block(correct_decisions)
#         else:
#             errors_block_list[idx] = np.nan

#     return errors_block_list


def get_correct_decision_block(correct, decision, block):
    """
    correct: numpy array with correct actions
    decision: numpy array with selected actions
    block: numpy array with block of experiment

    return: a list, for each block if decisions are correct
    """

    block_start_end_list = np.array(
        [get_block_start_list(correct, block), get_block_end_list(correct, block)]
    ).T

    correct_decision_block = []
    for _, block_start_end in enumerate(block_start_end_list):
        start = block_start_end[0]
        end = block_start_end[1]
        correct_decision_block.append(
            1 * (correct[start : end + 1] == decision[start : end + 1])
        )
    return correct_decision_block


def get_error_block(block_correct_decisions):
    """
    gets the number of errors until new correct decision is selected of the block

    block_correct_decisions: array if decisions of block are correct
    """
    diffs = np.where(np.diff(block_correct_decisions))[0] + 1
    diffs = np.insert(diffs, 0, 0)
    diffs = np.insert(diffs, diffs.size, block_correct_decisions.size)
    block_sequence_max_len = 0
    block_sequence_list = []
    for j in range(diffs.size - 1):
        block_sequence = block_correct_decisions[diffs[j] : diffs[j + 1]]
        block_sequence_list.append(block_sequence)
        ### check if sequence = correct decisions
        if block_sequence[0] == 1:
            ### check if this is the new longest sequence, if yes, save its index
            if len(block_sequence) > block_sequence_max_len:
                block_sequence_max_len = len(block_sequence)
                block_sequence_max_len_idx = j

    if block_sequence_max_len_idx == 0:
        ### no errors before consecutive rewarded sequence
        errors = 0
    else:
        ### only take not rewarded block sequences
        errors = 0
        for idx, block_sequence in enumerate(block_sequence_list):
            if block_sequence[0] == 0 and idx < block_sequence_max_len_idx:
                errors += block_sequence.size
    ret = errors
    return ret


def get_exploration_start_end_block(
    block_correct_decisions, prev_correct, decision_list
):
    """
    gets the start and end indices of the exploration trials of a block

    block_correct_decisions: array if decisions of a block are correct
    prev_correct: correct action of previous block
    decision_list: list with decisions of block
    """

    diffs = np.where(np.diff(block_correct_decisions))[0] + 1
    diffs = np.insert(diffs, 0, 0)
    diffs = np.insert(diffs, diffs.size, block_correct_decisions.size)

    block_sequence_max_len = 0
    block_sequence_list = []
    for j in range(diffs.size - 1):
        block_sequence = block_correct_decisions[diffs[j] : diffs[j + 1]]
        block_sequence_list.append(block_sequence)
        ### check if sequence = correct decisions
        if block_sequence[0] == 1:
            ### check if this is the new longest sequence, if yes, save its index
            if len(block_sequence) > block_sequence_max_len:
                block_sequence_max_len = len(block_sequence)
                block_sequence_max_len_idx = j

    ### get the end idx of the exploration phase
    ### --> count how many trials until the rewarded sequence is reached
    if block_sequence_max_len_idx == 0:
        ### no errors before consecutive rewarded sequence
        end_idx = 0
    else:
        ### only take not rewarded block sequences
        end_idx = 0
        for idx, block_sequence in enumerate(block_sequence_list):
            if block_sequence[0] == 0 and idx < block_sequence_max_len_idx:
                end_idx += block_sequence.size

    ### get start idx of exploration
    ### --> get idx where selecting prev correct finished
    ### it can also happen that the block starts with something
    ### else and than goes to prev correct --> than also use idx
    ### after prev correct because participants probably think prev
    ### correct is current correct and they only tried out randomly during block
    start_idx = 0
    found_prev_correct = False
    for idx, decision in enumerate(decision_list):
        if decision == prev_correct:
            if found_prev_correct:
                if idx == start_idx + 1:
                    start_idx = idx
                else:
                    break
            else:
                start_idx = idx
                found_prev_correct = True
    if found_prev_correct:
        start_idx = start_idx + 1

    ret = [start_idx, end_idx]
    return ret


# def get_block_trials(correct, decision, block, post_switch_trials, experiment):
#     """
#     correct: numpy array with correct actions
#     decision: numpy array with selected actions
#     block: numpy array with block of experiment
#     post_switch_trials: int, number of trials of block which should be returned

#     returns: pre and new correct and decisions for all switches/blocks
#     """
#     block_trial_indizes = get_block_trial_indizes(
#         correct, block, post_switch_trials, experiment
#     )

#     pre_correct = correct[block_trial_indizes][:, 0]
#     new_correct = correct[block_trial_indizes][:, -1]
#     ret = [pre_correct, new_correct, decision[block_trial_indizes]]
#     return ret


# def get_block_trials_are_correct(block_trials, pre_correct, new_correct):
#     """
#     block_trials: numpy array, rows=decisions during switches
#     pre_correct: numpy array, previous correct action for each switch
#     new_correct: numpy array, new correct action for each switch

#     returns: for each block/switch if decisions are pre/new correct --> two matrices
#     """

#     pre_correct_block = np.array(
#         [pre_correct for i in range(block_trials.shape[1])]
#     ).T  # extents the array to the dimension of block_trials
#     new_correct_block = np.array(
#         [new_correct for i in range(block_trials.shape[1])]
#     ).T  # extents the array to the dimension of block_trials

#     block_is_pre_correct = np.equal(
#         block_trials, pre_correct_block
#     )  # checks if decisions are pre correct
#     block_is_new_correct = np.equal(
#         block_trials, new_correct_block
#     )  # checks if decisions are new correct

#     ret = [block_is_pre_correct, block_is_new_correct]
#     return ret


# def get_initial_trials(correct, decision, post_switch_trials):
#     """
#     correct: numpy array with correct actions
#     decision: numpy array with selected actions
#     post_switch_trials: int, number of trials of block which should be returned

#     returns: correctness of initial trials
#     """
#     decision_is_correct = decision == correct
#     ret = np.concatenate(
#         [np.array([0]), decision_is_correct[:post_switch_trials]]
#     )  # prepend a 0 as "pre beginning" trial
#     return ret


def get_initial_weights(weights, post_switch_trials, correct):
    """
    weights: numpy array, for each trial 5 weights
    post_switch_trials: int, number of trials of block which should be returned
    correct: numpy array with correct actions

    returns: weights of initial trials, first column=rewarded action, other columns = other not-rewarded actions
    """

    temp = np.concatenate(
        [np.array([weights[0]]), weights[:post_switch_trials]], 0
    )  # prepend weights[0] as "pre beginning" trial
    correct_idx = int(correct[0] - 1)

    temp[:, 0], temp[:, correct_idx] = (
        temp[:, correct_idx],
        temp[:, 0].copy(),
    )  # switch weights of rewarded action with first action
    ret = temp
    return ret


def get_initial_weights_initial_learning(weights, post_switch_trials, correct):
    """
    weights: numpy array, for each trial 5 weights
    post_switch_trials: int, number of trials of block which should be returned
    correct: numpy array with correct actions

    returns: weights of initial trials, first column=rewarded action, other columns = other not-rewarded actions
    """

    temp = weights[
        : post_switch_trials + 1
    ]  # prepend weights[0] as "pre beginning" trial
    correct_idx = int(correct[0] - 1)

    temp[:, 0], temp[:, correct_idx] = (
        temp[:, correct_idx],
        temp[:, 0].copy(),
    )  # switch weights of rewarded action with first action
    ret = temp
    return ret


# def get_switch_weights(weights, post_switch_trials, correct):
#     """
#     weights: numpy array, for each trial 5 weights
#     post_switch_trials: int, number of trials of block which should be returned
#     correct: numpy array with correct actions

#     returns: averaged weights over all switches, first column=new correct, second column=pre correct, other columns other actions
#     """
#     block_trial_indizes = get_block_trial_indizes(correct, post_switch_trials)
#     pre_correct = correct[block_trial_indizes][:, 0]
#     new_correct = correct[block_trial_indizes][:, -1]

#     temp = weights[block_trial_indizes]  # weights for all switches

#     ### FOR EACH RS SWITCH WEIGTHS OF REWARDED ACTION WITH FIRST PLACE AND PREV REWARDED ACTION WITH SECOND PLACE
#     for rs_idx in range(temp.shape[0]):
#         ### SWITCH NEW CORRECT WITH FIRST COLUMN
#         new_correct_idx = int(new_correct[rs_idx] - 1)
#         temp[rs_idx, :, 0], temp[rs_idx, :, new_correct_idx] = (
#             temp[rs_idx, :, new_correct_idx],
#             temp[rs_idx, :, 0].copy(),
#         )

#         ### PUT PRE CORRECT INTO SECOND COLUMN
#         pre_correct_idx = int(pre_correct[rs_idx] - 1)
#         if (
#             pre_correct_idx == 0
#         ):  # if the pre_correct column was switched with the new_correct column
#             temp[rs_idx, :, 1], temp[rs_idx, :, new_correct_idx] = (
#                 temp[rs_idx, :, new_correct_idx],
#                 temp[rs_idx, :, 1].copy(),
#             )
#         else:
#             temp[rs_idx, :, 1], temp[rs_idx, :, pre_correct_idx] = (
#                 temp[rs_idx, :, pre_correct_idx],
#                 temp[rs_idx, :, 1].copy(),
#             )

#     ret = np.mean(temp, 0)
#     return ret


def plot_column(title, col, selections, post_switch_trials, bold_font, large_bold_font):
    """
    plots one column of the plot
    """
    trials = range(-1, post_switch_trials)
    if col == 0:
        selections = np.nanmean(selections, 0)
    else:
        selections_pre = np.mean(selections[0], 0)
        selections_new = np.mean(selections[1], 0)

    ### FIRST ROW
    ax = plt.subplot(3, 2, col + 1)
    plt.title(title, **large_bold_font)
    if col == 0:
        plt.plot(trials, selections, color="k")
    else:
        plt.plot(
            trials, selections_pre, color="k", ls="dotted", label="previously rewarded"
        )
        plt.plot(trials, selections_new, color="k", label="rewarded")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("trials", **bold_font)
    if col == 0:
        plt.ylabel("Performance", **bold_font)
    if col == 1:
        ax.set_yticklabels([])
        # plt.legend()

    """### SECOND ROW
    ax=plt.subplot(3,2,col+3)
    plt.plot(trials, weights_sd1[:,0], color='k')
    if col==0:
        plt.plot(trials, weights_sd1[:,1], color='k', ls='dashed')
    else:
        plt.plot(trials, weights_sd1[:,1], color='k', ls='dotted')
    plt.plot(trials, weights_sd1[:,2], color='k', ls='dashed')
    plt.plot(trials, weights_sd1[:,3], color='k', ls='dashed')
    plt.plot(trials, weights_sd1[:,4], color='k', ls='dashed')
    plt.ylim(0.00025,0.00105)
    ax.set_xticklabels([])
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    if col==0: plt.ylabel('Direct', **bold_font)
    if col==1: ax.set_yticklabels([])
    
    ### THIRD ROW
    ax=plt.subplot(3,2,col+5)
    plt.plot(trials, weights_sd2[:,0], color='k', label='rewarded')
    if col==0:
        plt.plot(trials, weights_sd2[:,1], color='k', ls='dashed')
    else:
        plt.plot(trials, weights_sd2[:,1], color='k', ls='dotted', label='previously rewarded')
    plt.plot(trials, weights_sd2[:,2], color='k', ls='dashed', label='others')
    plt.plot(trials, weights_sd2[:,3], color='k', ls='dashed')
    plt.plot(trials, weights_sd2[:,4], color='k', ls='dashed')
    plt.ylim(-0.00005,0.00115)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.xlabel('trials', **bold_font)
    if col==0: plt.ylabel('Indirect', **bold_font)
    if col==1:
        ax.set_yticklabels([])
        plt.legend()
    
    ### FOURTH ROW
    ax=plt.subplot(4,2,col+7)
    plt.plot(trials, weights_stn[:,0], color='k', label='rewarded')
    if col==0:
        plt.plot(trials, weights_stn[:,1], color='k', ls='dashed')
    else:
        plt.plot(trials, weights_stn[:,1], color='k', ls='dotted', label='old rewarded')
    plt.plot(trials, weights_stn[:,2], color='k', ls='dashed', label='others')
    plt.plot(trials, weights_stn[:,3], color='k', ls='dashed')
    plt.plot(trials, weights_stn[:,4], color='k', ls='dashed')
    #TODO plt.ylim(-0.5, 1.0)
    plt.xlabel('trials', **bold_font)
    if col==0: plt.ylabel('Hyperdirect', **bold_font)
    if col==1:
        ax.set_yticklabels([])
        plt.legend()"""


# def get_block_trials_are_cluster(block_trials, cluster):
#     """
#     block_trials: numpy array, rows=decisions during switches
#     cluster: numpy array, the three repeatedly rewarded responses

#     returns: for each block/switch if decisions are in/out cluster --> two matrices
#     """

#     out_cluster = [1, 2, 3, 4, 5]
#     for val in cluster:
#         out_cluster.remove(val)

#     in_cluster1_block = np.array([cluster[0] for i in range(block_trials.shape[1])]).T
#     in_cluster2_block = np.array([cluster[1] for i in range(block_trials.shape[1])]).T
#     in_cluster3_block = np.array([cluster[2] for i in range(block_trials.shape[1])]).T

#     out_cluster1_block = np.array(
#         [out_cluster[0] for i in range(block_trials.shape[1])]
#     ).T
#     out_cluster2_block = np.array(
#         [out_cluster[1] for i in range(block_trials.shape[1])]
#     ).T

#     block_is_in_cluster = (
#         np.equal(block_trials, in_cluster1_block).astype(int)
#         + np.equal(block_trials, in_cluster2_block).astype(int)
#         + np.equal(block_trials, in_cluster3_block).astype(int)
#     )

#     block_is_out_cluster = np.equal(block_trials, out_cluster1_block).astype(
#         int
#     ) + np.equal(block_trials, out_cluster2_block).astype(int)

#     ret = [block_is_in_cluster.astype(bool), block_is_out_cluster.astype(bool)]
#     return ret


def get_selected_cluster(correct, decision, block, cluster, experiment):
    """
    correct: numpy array with correct actions
    decision: numpy array with selected actions
    block: numpy array with block of experiment
    cluster: numpy array, the three repeatedly rewarded responses

    returns: for each block (also not successful), how often selected in/out cluster
    """

    out_cluster = [1, 2, 3, 4, 5]
    for val in cluster:
        out_cluster.remove(val)

    ### get start and end idx for the exploration period for each block
    ### not valid blocks/explorationperiods already removed
    ### initial block contains initial learning --> first entry is for second block/first exploration period
    ### for not succesfull blocks --> nan values
    exploration_start_end_list = get_exploration_start_end_arr(
        correct, decision, block, experiment
    )

    ### split decision trial array into decision block list
    decision_block_list = split_trial_arr_into_block_list(decision, correct, block)

    ### count for each block how often in/out cluster was selected
    in_cluster_selections = np.zeros(exploration_start_end_list.shape[0]) * np.nan
    out_cluster_selections = np.zeros(exploration_start_end_list.shape[0]) * np.nan
    for idx_exploration, exploration_start_end in enumerate(exploration_start_end_list):
        idx_block = idx_exploration + 1
        if np.isnan(exploration_start_end[0]):
            ### if start == nan --> skip this exploration
            pass
        else:
            decision_block = decision_block_list[idx_block]
            start = int(exploration_start_end[0])
            end = int(exploration_start_end[1])
            in_cluster_selections[idx_exploration] = np.sum(
                1 * (cluster[0] == decision_block[start : end + 1])
                + 1 * (cluster[1] == decision_block[start : end + 1])
                + 1 * (cluster[2] == decision_block[start : end + 1])
            )
            out_cluster_selections[idx_exploration] = np.sum(
                1 * (out_cluster[0] == decision_block[start : end + 1])
                + 1 * (out_cluster[1] == decision_block[start : end + 1])
            )

    return [in_cluster_selections, out_cluster_selections]


def split_trial_arr_into_block_list(trial_arr, correct, block):
    """
    trial_arr: any array with data for trials which will be splitted
    correct: numpy array with correct actions
    block: numpy array with block of experiment
    """
    switch_trials = get_switch_trials(correct, block)
    trial_arr_block_list = np.array_split(trial_arr, switch_trials)
    return trial_arr_block_list


def get_exploration_start_end_arr(correct, decision, block, experiment):
    """
    correct: numpy array with correct actions
    decision: numpy array with selected actions
    block: numpy array with block of experiment
    get a list: for each block start and end idx of exploration trials
    """

    ### generate list of arrays for each block if decision was correct
    correct_decision_block_list = get_correct_decision_block(correct, decision, block)

    ### get decisions splitted into blocks
    decision_block_list = split_trial_arr_into_block_list(decision, correct, block)

    ### remove blocks after breaks
    correct_decision_block_list = remove_invalid_blocks(
        correct_decision_block_list, experiment
    )
    decision_block_list = remove_invalid_blocks(decision_block_list, experiment)

    nr_blocks = len(correct_decision_block_list)

    ### get prev correct of each rule switch
    prev_correct_arr, _ = get_prev_and_new_correct(correct, block)

    ### remove ruleswitches during breaks
    prev_correct_arr = remove_invalid_ruleswitches(prev_correct_arr, experiment)

    ### get exploration start end indexes of successful and valid blocks
    start_end_arr = np.zeros((len(prev_correct_arr), 2))
    for idx_block in range(nr_blocks):
        ### we go over blocks but only want data for ruleswitches
        ### --> for first block no rule switch (at beginning is initial learning)
        ### --> skip
        idx_exploration = idx_block - 1
        if idx_exploration < 0:
            continue
        correct_decision_block = correct_decision_block_list[idx_block]
        decision_block = decision_block_list[idx_block]
        ### only analyze a block if its successful
        if is_block_successful(correct_decision_block):
            start_end_arr[idx_exploration] = get_exploration_start_end_block(
                correct_decision_block,
                prev_correct_arr[idx_exploration],
                decision_block,
            )
        else:
            start_end_arr[idx_exploration] = np.nan
    return start_end_arr


def get_prev_and_new_correct(correct, block):
    """
    correct: numpy array with correct actions
    block: numpy array with block of experiment
    post_switch_trials: int, number of trials of block which should be returned

    returns: matrix, rows = indizes of switches + following trials
    """

    ### detect rule switches
    ### --> trials at which ne action is correct
    switch_trials = get_switch_trials(correct, block)

    ### --> before switch trials == prev correct
    prev_correct_arr = correct[switch_trials - 1]
    ### and at swtich trials == new correct
    new_correct_arr = correct[switch_trials]

    return [prev_correct_arr, new_correct_arr]


def prepend_nans(arr):
    """
    arr = 2D array

    prepend nans to second dimension, doubling the length
    """
    a = arr.copy()
    for i in range(arr.shape[1]):
        a = np.insert(a.astype(float), 0, np.nan, 1)

    return a


def get_last_positive_idx(arr):
    """
    arr = 2D array

    get last index where entry >0 for dimension 1
    """
    ### replace nan values with zero
    ### because nan values should not be found by np.where
    arr = np.nan_to_num(arr)
    ret = np.zeros(arr.shape[0])
    for i in range(arr.shape[0]):
        ret[i] = np.where(arr[i])[0][-1]
    return ret


def center_at_last_idx(arr, idx_list):
    """
    arr = 2D array
    idx_list = where to center each array along dimension 1
    """

    new_arr = np.zeros((arr.shape[0], arr.shape[1] // 2))
    start_idx_list = (np.array(idx_list) - arr.shape[1] // 2).astype(int)
    end_idx_list = np.array(idx_list).astype(int)

    for i in range(arr.shape[0]):
        if start_idx_list[i] >= 0:
            new_arr[i] = arr[i, start_idx_list[i] : end_idx_list[i]]
        else:  # end_idx is before half of the array length --> just return nan values
            new_arr[i, :] = np.nan
    return new_arr


def get_fake_selections(
    block_n=45, repeat_prob=0.01, nothing_prob=0.02, bias=0, rng=np.random.default_rng()
):
    """
    get fake selections

    block_n: how many blocks
    repeat_prob: how likely it is that a resp can be selected again
    nothing_prob: how likely no selection
    bias: reduces out_cluster probabilities, between 0-1, 0=no bias, 1=out cluster resposnes probability is zero


    """
    selection_list = []
    max_len = 0
    for block in range(block_n):
        rew = 0
        available = [0, 1, 2, 3, 4]  # nothing, rew, in, out1, out2
        selection_list.append([])
        while rew == 0:
            ### define probabilities to select the differen actions or nothing
            selection_probs = get_p_list(available, bias)

            ### do selection
            selection = rng.choice(available, 1, p=selection_probs)[0]
            selection_list[block].append(selection)

            ### remove the selection from available options with high probability
            rand_val = rng.random()
            if rand_val > repeat_prob:
                available.remove(selection)

            if selection == 1:
                rew = 1
        max_len = max([len(selection_list[block]), max_len])

    # max_len=20
    selection_arr = np.ones((block_n, max_len))
    for block in range(block_n):
        selection_arr[block, : len(selection_list[block])] = selection_list[block]
    return selection_arr


def get_p_list(available, bias):

    return np.array([get_p(available, bias, resp) for resp in available])


def get_p(available, bias, resp):
    if resp in available:
        if resp == 0:
            prob = 0.02
        elif resp == 3 or resp == 4:
            prob = (
                (1 - get_p(available, bias, 0))
                / (len(available) - int(0 in available))
                * (1 - bias)
            )
        elif resp == 1 or resp == 2:
            prob = (
                1
                - get_p(available, bias, 0)
                - get_p(available, bias, 3)
                - get_p(available, bias, 4)
            ) / (
                2 * (int(1 in available) * int(2 in available))
                + 1 * int(not (bool(int(1 in available) * int(2 in available))))
            )

        return prob
    else:
        return 0


def center_arr_at_last_pos_idx(arr):
    ### prepend nans to array
    selected_out_cluster = prepend_nans(arr)

    ### get last index where selected_out_cluster > 0
    last_positive_idx = get_last_positive_idx(selected_out_cluster)

    ### center the arrays at this index (this is the last index.. it is not included in the new array because this would overestimate the frequencie for the last entry) and return to original length
    selected_out_cluster = center_at_last_idx(selected_out_cluster, last_positive_idx)

    return [selected_out_cluster, last_positive_idx]


def get_centered_selected_out_cluster_m(arr):

    selected_out_cluster, last_positive_idx = center_arr_at_last_pos_idx(arr)

    ### average over vps
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        selected_out_cluster_m = np.nanmean(selected_out_cluster, 0)
        selected_out_cluster_sd = np.nanstd(selected_out_cluster, 0)

    ### cut the nan values == blocks which are before the experiment for all vps, NOW LENGTH != VALID BLOCKS
    not_nan = np.logical_not(np.isnan(selected_out_cluster_m))
    selected_out_cluster_m = selected_out_cluster_m[not_nan]
    selected_out_cluster_sd = selected_out_cluster_sd[not_nan]
    return [
        selected_out_cluster_m,
        selected_out_cluster_sd,
        last_positive_idx,
        selected_out_cluster,
    ]


def make_regression_plot(
    title, selected_out_cluster_m, selected_out_cluster_sd, plot=1, bold_font=None
):
    ### linear regression
    x = np.array(range(selected_out_cluster_m.size)).reshape((-1, 1))
    y = selected_out_cluster_m
    model = LinearRegression().fit(x, y)

    if plot:
        y_pred = model.predict(x)
        ### PLOT
        plt.figure(figsize=(8.5 / 2.54, 7 / 2.54), dpi=300)
        x_plot = np.arange(-selected_out_cluster_m.size + 1, 1, 1)
        plt.errorbar(
            x_plot, selected_out_cluster_m, yerr=selected_out_cluster_sd, fmt="k."
        )
        plt.plot(x_plot, y_pred, color="red")
        plt.ylim(-0.4, 3.1)
        plt.xlabel("blocks", **bold_font)
        plt.ylabel("average response number", **bold_font)
        plt.tight_layout()
        plt.savefig(title)

    return {"r2": model.score(x, y), "b": model.intercept_, "a": model.coef_}


def make_single_vps_plot(title, arr, bold_font, mark_initial_nans=False):
    plt.figure(figsize=(8.5 / 2.54, 7 / 2.54))
    x = np.arange(arr.shape[1])
    x_rev = np.arange(-arr.shape[1], 0)
    for idx in range(arr.shape[0]):
        plt.subplot(arr.shape[0], 1, idx + 1)
        if np.isnan(arr[idx][0]) and mark_initial_nans:
            plt.bar(x_rev, arr[idx])
            until_nan_finished = 0  # np.where(np.isnan(arr[idx]))[0][-1]
            for nan_idx, arr_value in enumerate(arr[idx]):
                if np.isnan(arr_value) and nan_idx == until_nan_finished + 1:
                    until_nan_finished = nan_idx

            plt.axvspan(
                x_rev[0] - 1,
                x_rev[0] + until_nan_finished,
                color="grey",
                alpha=0.3,
            )
            plt.xlim(x_rev[0] - 1, x_rev[-1] + 1)
        else:
            plt.bar(x, arr[idx])
            plt.xlim(x[0] - 1, x[-1] + 1)
        plt.ylim(0, 3)
        plt.yticks([0, 3])
        if idx != arr.shape[0] - 1:
            plt.gca().set_yticklabels([])
            plt.gca().set_xticklabels([])
    plt.xlabel("blocks", **bold_font)
    plt.text(
        0,
        0.5,
        "number of responses",
        ha="center",
        va="center",
        rotation="vertical",
        transform=plt.gcf().transFigure,
        **bold_font,
    )

    plt.tight_layout()
    plt.savefig(title)


# def get_fake_data(
#     num_vps,
#     num_valid_rs,
#     learned_rule_range,
#     learned_rule_arr,
#     random_rule_learn_time=0,
#     rng=np.random.default_rng(),
# ):
#     ### create fake data
#     selected_out_cluster_fake = np.zeros((num_vps, num_valid_rs + 1))
#     ### create num_vps fake selection datasets (45 blocks)
#     for idx in range(num_vps):
#         ### when the vps should have learned the rule
#         if random_rule_learn_time:
#             rule_learned = rng.integers(
#                 low=learned_rule_range[0], high=learned_rule_range[1], endpoint=True
#             )
#         else:
#             rule_learned = learned_rule_arr[idx]

#         ### create the dataset, two phases 1:wihtout bias, 2: with bias = sudden rule learning
#         selections_phase_a = get_fake_selections(
#             block_n=rule_learned, repeat_prob=0.1, nothing_prob=0.02, bias=0, rng=rng
#         )
#         selections_phase_b = get_fake_selections(
#             block_n=num_valid_rs + 1 - rule_learned,
#             repeat_prob=0.1,
#             nothing_prob=0.02,
#             bias=1,
#             rng=rng,
#         )
#         selections_phase_a, selections_phase_b = make_same_len(
#             selections_phase_a, selections_phase_b
#         )
#         block_trials = np.concatenate([selections_phase_a, selections_phase_b])

#         ### get how often out cluster was selected for each block = block_is_out_cluster
#         cluster = [
#             1,
#             2,
#             5,
#         ]  # in fake selections the third in cluster response, i.e. the previously rewarded, does not exist, there are only [0,1,2,3,4] and 1,2 are cluster --> just use 5 as third

#         block_is_in_cluster, block_is_out_cluster = get_block_trials_are_cluster(
#             block_trials, cluster
#         )

#         selected_out_cluster_fake[idx] = np.sum(block_is_out_cluster, 1)

#     return selected_out_cluster_fake


def make_same_len(arr_a, arr_b):
    """
    arr_a and arr_b: 2D arrays

    retunr arrays but with same length of second dim (fill with ones)
    """

    while arr_a.shape[1] < arr_b.shape[1]:
        arr_a = np.insert(
            arr_a, np.shape(arr_a)[1], 1, 1
        )  # insert column of ones at the end
    while arr_b.shape[1] < arr_a.shape[1]:
        arr_b = np.insert(
            arr_b, np.shape(arr_b)[1], 1, 1
        )  # insert column of ones at the end

    return [arr_a, arr_b]


def make_correlation_plot(title, arr, plot=1, bold_font=None, mode="scatter", bottom=0):
    """
    arr: n*m arry with data
    first dim = vps, second dim = data over blocks

    returns correlation between blocks and data values
    """

    arr, _ = center_arr_at_last_pos_idx(arr)

    x = []
    y = []
    for vp_id in range(arr.shape[0]):
        for block_id in range(arr.shape[1]):
            if not (np.isnan(arr[vp_id, block_id])):
                x.append(-arr.shape[1] + block_id)
                y.append(arr[vp_id, block_id])
    x = np.array(x)
    y = np.array(y)

    r, p_val = stats.spearmanr(x, y)

    n = len(x)
    df = n - 2
    ### calculate 95% CI using Z/norm-distribution
    z_crit = stats.norm.ppf(0.975)
    CI_fisher_transformed = np.arctanh(r) + np.array([-1, 1]) * z_crit * (
        1 / np.sqrt(n - 3)
    )
    CI = np.tanh(CI_fisher_transformed)

    if plot and mode == "scatter":
        ### old plot version
        make_correlation_plot_scatter(x, y, arr, p_val, r, title, bold_font)
    elif plot and mode == "bars":
        make_correlation_plot_bars(x, y, arr, p_val, r, title, bold_font)
    elif plot and mode == "scatter_bars":
        make_correlation_plot_scatter_bars(
            x, y, arr, p_val, r, title, bold_font, bottom
        )
    elif plot and mode == "scatter_circles":
        make_correlation_plot_scatter_circles(x, y, arr, p_val, r, title, bold_font)
    elif plot and mode == "scatter_circles_size":
        make_correlation_plot_scatter_circles(
            x, y, arr, p_val, r, title, bold_font, False
        )

    return [r, p_val, df, CI]


def make_correlation_plot_bars(x, y, arr, p_val, r, title, bold_font):
    """
    instead of bar heights area between line
    one area for each number of selections (0, 1, 2, 3)
    """

    x_unique = np.unique(x)
    y_unique = np.unique(y)
    bin_arr = np.arange(y_unique.max() + 2) - 0.5
    plot_y = np.zeros((x_unique.size, bin_arr.size - 1))
    for idx_x_val, x_val in enumerate(x_unique):
        hist, _ = np.histogram(y[x == x_val], bins=bin_arr)

        plot_y[idx_x_val, :] = np.array(
            [np.sum(hist[0 : idx + 1]) for idx in range(hist.size)]
        )

    plot_y_norm = plot_y  # / np.max(plot_y, 1)[:, None]

    plt.figure(figsize=(8.5 / 2.54, 7 / 2.54), dpi=300)
    for plot_idx in range(plot_y_norm.shape[1] - 1, -1, -1):

        plot_y_1 = plot_y_norm[:, plot_idx]

        plt.bar(
            x_unique,
            plot_y_1,
            label=f"{plot_idx}",
            color=f"C{plot_idx}",
        )

    plt.legend()
    plt.tight_layout()
    plt.savefig(title)


def make_correlation_plot_scatter(x, y, arr, p_val, r, title, bold_font):
    """
    classic plot version with bar for each coordinate, height=nr of datapoints
    with linea regession line
    """
    ### linear regression for line
    model = LinearRegression().fit(x.reshape((-1, 1)), y)

    x_plot = np.linspace(-arr.shape[1], -1, 100)
    x_plot = np.linspace(x.min(), x.max(), 100)
    y_pred = model.predict(x_plot.reshape((-1, 1)))

    ### define sizes for scatter plot
    all_possible_x = np.unique(x)
    number_points = np.zeros(x.shape)
    for idx_x, x_val in enumerate(all_possible_x):
        values, counts = np.unique(y[x == x_val], return_counts=True)
        for idx_y, y_val in enumerate(values):
            mask_x = x == x_val
            mask_y = y == y_val
            mask = (mask_x * mask_y).astype(bool)
            number_points[mask] = counts[idx_y]
    area = (15 * number_points / number_points.max()) ** 2

    ### PLOT
    plt.figure(figsize=(8.5 / 2.54, 7 / 2.54), dpi=300)
    plt.scatter(x, y, s=area, marker="|", color="k")

    # for number in np.unique(number_points):
    #    plt.scatter(x[number_points==number],y[number_points==number],s=10,marker=(int(number), 2, 0), linewidth=0.5, color='k')
    plt.plot(x_plot, y_pred, color="red")
    plt.ylim(-0.6, 3.5)
    plt.xlim(x.min() - 1, x.max() + 1)
    plt.xlabel("blocks", **bold_font)
    plt.ylabel("never-rewarded selections", **bold_font)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
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
    plt.tight_layout()
    plt.savefig(title)


def make_correlation_plot_scatter_bars(x, y, arr, p_val, r, title, bold_font, bottom):
    """
    classic plot version with bar for each coordinate, height=nr of datapoints
    with linea regession line
    """
    ### linear regression for line
    model = LinearRegression().fit(x.reshape((-1, 1)), y)

    x_plot = np.linspace(-arr.shape[1], -1, 100)
    x_plot = np.linspace(x.min(), x.max(), 100)
    y_pred = model.predict(x_plot.reshape((-1, 1)))

    ### define sizes for scatter plot bars
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    bin_arr = np.arange(y_unique.max() + 2) - 0.5
    plot_y = np.zeros((x_unique.size, bin_arr.size - 1))
    for idx_x_val, x_val in enumerate(x_unique):
        hist, _ = np.histogram(y[x == x_val], bins=bin_arr)

        plot_y[idx_x_val, :] = hist

    plot_y_norm = (plot_y / np.max(plot_y)) * [1, 0.9][bottom]

    ### PLOT
    plt.figure(figsize=(8.5 / 2.54, 7 / 2.54), dpi=300)

    for plot_idx in range(plot_y.shape[1] - 1, -1, -1):
        plt.bar(
            x=x_unique,
            height=plot_y_norm[:, plot_idx],
            width=1,
            bottom=plot_idx - [plot_y_norm[:, plot_idx] / 2, 0][bottom],
            color="k",
            linewidth=0,
        )

        plt.axhline(plot_idx, color="k")

    # for number in np.unique(number_points):
    #    plt.scatter(x[number_points==number],y[number_points==number],s=10,marker=(int(number), 2, 0), linewidth=0.5, color='k')
    if bottom == 0:
        plt.plot(x_plot, y_pred, color="red")
    plt.ylim(-0.6, 3.5)
    plt.xlim(x.min() - 1, x.max() + 1)
    plt.xlabel("blocks", **bold_font)
    plt.ylabel("never-rewarded selections", **bold_font)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
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
    plt.tight_layout()
    plt.savefig(title)


def make_correlation_plot_scatter_circles(
    x, y, arr, p_val, r, title, bold_font, change_size=True
):
    """
    classic plot version with bar for each coordinate, height=nr of datapoints
    with linea regession line
    """
    ### linear regression for line
    model = LinearRegression().fit(x.reshape((-1, 1)), y)

    x_plot = np.linspace(-arr.shape[1], -1, 100)
    x_plot = np.linspace(x.min(), x.max(), 100)
    y_pred = model.predict(x_plot.reshape((-1, 1)))

    ### define sizes for scatter plot
    x_unique = np.unique(x)
    x_scatter = []
    y_scatter = []
    s_scatter = []
    for x_val in x_unique:
        values, counts = np.unique(y[x == x_val], return_counts=True)
        for idx_y, y_val in enumerate(values):
            x_scatter.append(x_val)
            y_scatter.append(y_val)
            s_scatter.append(counts[idx_y])
    x_scatter = np.array(x_scatter)
    y_scatter = np.array(y_scatter)
    s_scatter = np.array(s_scatter)

    ### PLOT
    my_cmap = create_cm(
        colors=[[180, 180, 180], [0, 0, 0]],
        name="gray_to_black",
        vmin=s_scatter.min(),
        vmax=s_scatter.max(),
        gamma=0.8,
    )
    if change_size:
        area = (15 * s_scatter / s_scatter.max()) ** 2
        edgecolors = my_cmap(s_scatter)
        facecolors = "none"
    else:
        area = 20
        edgecolors = "none"
        facecolors = my_cmap(s_scatter)

    plt.figure(figsize=(8.5 / 2.54, 7 / 2.54), dpi=300)
    plt.scatter(
        x_scatter,
        y_scatter,
        s=area,
        cmap=my_cmap,
        edgecolors=edgecolors,
        facecolors=facecolors,
    )

    # for number in np.unique(number_points):
    #    plt.scatter(x[number_points==number],y[number_points==number],s=10,marker=(int(number), 2, 0), linewidth=0.5, color='k')
    plt.plot(x_plot, y_pred, color="red")
    plt.ylim(-0.6, 3.5)
    plt.xlim(x.min() - 1, x.max() + 1)
    plt.xlabel("blocks", **bold_font)
    plt.ylabel("never-rewarded selections", **bold_font)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
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
    plt.tight_layout()
    plt.savefig(title)
