import numpy as np


def change_to_one_hot_encode(clockwise_arr):
    """
    gets arr with integer values between 0 and 2

    returns the same array as one hot encoded array
    """

    ret_arr = np.zeros((clockwise_arr.shape[0], 3))
    for idx in range(clockwise_arr.shape[0]):
        ret_arr[idx, int(clockwise_arr[idx])] = 1

    return ret_arr


def check_clockwise(exploration_selections_list):
    """
    exploration_selections_list: list of lists containing sleections for each ruleswitch (until exploration finished)

    check if these selections are ascending or descending with increment 1

    if the blcok was not successfull it is not analyzed --> classifies the coresponding exploration as "other"

    returns array with value for each ruleswitch:
        ascending/clockwise = 0
        descending/anti-clockwise = 1
        other = 2
    """
    clockwise_arr = np.zeros(len(exploration_selections_list))
    ### loop over blocks
    for idx in range(len(exploration_selections_list)):
        clockwise = True
        anticlockwise = True
        other = True
        if isinstance(exploration_selections_list[idx], type(None)):
            clockwise = False
            anticlockwise = False
            other = True
        else:
            ### go through all selections, check if differences are clockwise or anticlockwise
            for idx_idx in range(1, exploration_selections_list[idx].size):
                pre_sel = exploration_selections_list[idx][idx_idx - 1]
                now_sel = exploration_selections_list[idx][idx_idx]
                dif_sel = int(now_sel) - int(pre_sel)
                ### dif has to be ascending (increment 1) or from 5 to 1 to be clockwise
                if not dif_sel == 1 and not dif_sel == -4:
                    clockwise = False
                ### dif has to be descending (increment 1) or from 1 to 5 to be anticlockwise
                if not dif_sel == -1 and not dif_sel == 4:
                    anticlockwise = False
            ### if both are still True --> only 1 selection/no difference --> classify it as other
            if clockwise is True and anticlockwise is True:
                clockwise = False
                anticlockwise = False
            ### if clockwise or anticlockwise is still true --> set other to False
            if clockwise is True or anticlockwise is True:
                other = False
        ### store 0 for clockwise, 1 for anticlockwise and 2 for other
        clockwise_arr[idx] = np.array([0, 1, 2])[
            np.array([clockwise, anticlockwise, other])
        ]

    return clockwise_arr


def get_exploration_selections(correctList, decisionList, blockList, experiment):
    """
    return for each ruleswitch the exploration actions
    --> list (ruleswitches) of lists (explorationactions)
    """
    # exploration_idx_list = get_exploration_idx_list(
    #     correctList, decisionList, blockList, experiment
    # )

    # decision_block_list = get_decision_block_list(correctList, decisionList, blockList)

    # exploration_selections_list = []
    # for idx in range(len(exploration_idx_list)):
    #     exploration_selections_list.append(
    #         decision_block_list[idx][exploration_idx_list[idx].astype(bool)]
    #     )

    # return exploration_selections_list

    ### get start and end of exploration for each block valid block (breaks already removed)
    ### first block is initial learning --> first entry of array is second block
    exploration_start_end_list = get_exploration_start_end_arr(
        correctList, decisionList, blockList, experiment
    )

    ### get decisions in block shape
    decision_block_list = split_trial_arr_into_block_list(
        decisionList, correctList, blockList
    )
    ### remove blocks after breaks
    decision_block_list = remove_invalid_blocks(decision_block_list, experiment)

    nr_blocks = len(decision_block_list)

    ### list for each block the exploratory decisions (plus one trial before)
    exploration_decision_list = []
    for idx_block in range(nr_blocks):
        idx_exploration = idx_block - 1
        if idx_exploration < 0:
            pass
        else:
            ### save the decisions of the exploration trials
            ### and one trial before
            ### because we want to analyse the transitions (if they are clockwise etc.) --> also the transition to the first exploration trial
            start_idx, end_idx = exploration_start_end_list[idx_exploration]
            if np.isnan(start_idx):
                ### --> the block is not successfull
                exploration_decision_list.append(None)
            else:
                start_idx = int(start_idx)
                end_idx = int(end_idx)
                if start_idx == 0:
                    ### the first exploration trial is the first trial of the block
                    ### --> we need the last decision of the previous block
                    dec_last_block = decision_block_list[idx_block - 1][-1]
                    dec_this_block = decision_block_list[idx_block][: end_idx + 1]
                    [dec_last_block] + dec_this_block
                    exploration_decision_list.append([dec_last_block] + dec_this_block)
                else:
                    ### the first exploration trials is somewhere within the block
                    ### we can use the trial before which is also within the block
                    start_idx = start_idx - 1
                    exploration_decision_list.append(
                        decision_block_list[idx_block][start_idx : end_idx + 1]
                    )
    return exploration_decision_list


def get_exploration_idx_list(correct, decision, block, experiment):
    """
    returns for all trials if they are exploration trials, exploration trials = before the consecutive rewarded trials (including the first of the consecutive rewarded trials)

    normally the first error is not considered an exploration error but here we also set it to 1 (later the diff from this trial to the first exploration will be needed)

    here a list for each block is returned
    """

    block_start_end_list = np.array(
        [get_block_start_list(correct, block), get_block_end_list(correct, block)]
    ).T

    # ### TODO correct the indexes of exploration trials
    # exploration_start_end_list = get_exploration_start_end_arr(
    #     correct, decision, block, experiment
    # )

    ### generate list of arrays for each block if decision was correct
    correct_decision_block = []
    prev_correct = [np.nan]
    decision_block = []
    for idx, block_start_end in enumerate(block_start_end_list):
        start = block_start_end[0]
        end = block_start_end[1]
        correct_decision_block.append(
            1 * (correct[start : end + 1] == decision[start : end + 1])
        )
        decision_block.append(decision[start : end + 1])

        if start > 0:
            prev_correct.append(correct[start - 1])
    prev_correct = np.array(prev_correct)

    ### get trials before consecutive rewarded
    exploration_block = []
    for idx, correct_decisions in enumerate(correct_decision_block):
        exploration_block.append(
            get_exploration_block(
                correct_decisions, decision_block[idx], prev_correct[idx]
            )
        )

    ### return shape from block-wise to trial-wise
    ret = exploration_block
    return ret


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


def split_trial_arr_into_block_list(trial_arr, correct, block):
    """
    trial_arr: any array with data for trials which will be splitted
    correct: numpy array with correct actions
    block: numpy array with block of experiment
    """
    switch_trials = get_switch_trials(correct, block)
    trial_arr_block_list = np.array_split(trial_arr, switch_trials)
    return trial_arr_block_list


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


def get_decision_block_list(correct, decision, block):
    """
    returns list if lists with decisions of each block
    """

    block_start_end_list = np.array(
        [get_block_start_list(correct, block), get_block_end_list(correct, block)]
    ).T

    ### generate list of lists with decisions for each block
    decision_block = []
    for _, block_start_end in enumerate(block_start_end_list):
        start = block_start_end[0]
        end = block_start_end[1]
        decision_block.append(decision[start : end + 1])

    ret = decision_block
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


def get_switch_trials(correct, block):
    """
    returns at which indizes the ruleswitch occurs (first trial with new rule)

    correct: numpy array with correct actions
    block: numpy array with block of experiment
    """
    ### detect rule switches
    switch_trials = np.where(np.diff(correct))[0]
    ### detect block switch (if not already rule switch)
    for block_switch_trial in np.where(np.diff(block))[0]:
        if not (block_switch_trial in switch_trials):
            switch_trials = np.sort(np.append(block_switch_trial, switch_trials))

    ret = switch_trials + 1
    return ret


def get_exploration_block(block_correct_decisions, decision_block, prev_correct):
    """
    gets the indices of trials until new correct decision is selected of the block (consecutively)

    block_correct_decisions: array if decisions of block are correct

    this also returns the first error of the block (where the rule switch is not known --> not actually a exploration trial)
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

    if block_sequence_max_len == 0:
        ### there is no correct decision in whole block --> just return zeros
        exploration_trials = np.zeros(block_correct_decisions.size)
    else:
        if block_sequence_max_len_idx == 0:
            ### no errors before consecutive rewarded sequence
            errors = 0
        else:
            ### add all sequences before long consecutive rewarded one = epxloration "errors"
            errors = 0
            for idx, block_sequence in enumerate(block_sequence_list):
                if idx < block_sequence_max_len_idx:
                    errors += block_sequence.size

        ### all "errors" = exploration trials
        if block_sequence_max_len_idx == 0:
            ### first block, or vps anticipated the rule change... also somehow explorative
            exploration_trials = np.zeros(block_correct_decisions.size)
            exploration_trials[0] = 1
        else:
            exploration_trials = np.concatenate(
                [
                    np.ones(errors + 1),
                    np.zeros(block_correct_decisions.size - errors - 1),
                ]
            )
            ### do not count zeros as exploration trials
            exploration_trials[decision_block == 0] = 0

    ret = exploration_trials
    return ret


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
