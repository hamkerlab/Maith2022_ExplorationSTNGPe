import numpy as np
import pylab as plt
import random
from scipy.stats import sem
import matplotlib as mtl
import sys
from CompNeuroPy import create_dir

random.seed()
rng = np.random.default_rng(1)


##############################################################################################################################################################
###################################################################  FUNCTIONS SIMS  #########################################################################
##############################################################################################################################################################


def get_performance(sim, folder):
    failed = np.load(folder + "/failed_blocks_sim" + str(sim) + ".npy")
    performance = 1 - failed

    return performance


def get_output(x, simID):
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
    selection = np.load(x + "/selection_sim" + str(simID) + ".npy")
    while selection[trials, 1] != 0:
        trials += 1

    file = open(x + "/output_sim" + str(simID), "r")
    zeile = file.readline()
    correct = np.zeros(trials)
    decision = np.zeros(trials)
    start = np.zeros(trials)
    dopInp = np.zeros(trials)
    i = 0
    try:
        while 1:
            zeile = file.readline()
            liste = zeile.split("\t")
            correct[i] = liste[4]
            decision[i] = liste[5]
            start[i] = liste[1]
            dopInp[i] = liste[2]
            i += 1
    except:
        file.close()

    frequentAction = [2, 3, 4, 5][np.histogram(correct, [2, 3, 4, 5, 10])[0].argmax()]

    return [trials, correct, decision, frequentAction, start, dopInp]


def get_counts(correctList, decisionList):
    """
    loads correctList, decisionList

    counts for each rule switch how often actions were chosen
    all actions = [1,2,3,4,5,0]
    counted actions = [1,5,possibleAction,0]
    possibleAction = [2,3,4] without prev correct and new correct

    returns just first, mid and last counts (3x4 array)
    """
    counts = np.zeros((numRS, 6))

    rs = 0
    for trial in np.arange(1, correctList.shape[0]):
        if getRsIdx(trial, correctList, decisionList)[
            0
        ]:  # ruleswtich detected-->analyse
            rsIdx = getRsIdx(trial, correctList, decisionList)[1]
            prevcorrect = getRsIdx(trial, correctList, decisionList)[2]
            newcorrect = getRsIdx(trial, correctList, decisionList)[3]

            possibleAction = [1, 5, 2, 3, 4]
            possibleAction.remove(prevcorrect)
            possibleAction.remove(newcorrect)
            actions = possibleAction + [0]

            actions = np.array([1, 5, 2, 3, 4, 0])

            for idx in rsIdx:
                if decisionList[idx] != prevcorrect and decisionList[idx] != newcorrect:
                    counts[rs] += 1 * (decisionList[idx] == actions)
            rs += 1

    ### retirm first mid and last exploration
    ret = counts[[0, counts.shape[0] // 2, counts.shape[0] - 1], :]

    if simAnz == 10:
        # combine counts of six explorationperiods per time (early, mid, late)
        last = counts.shape[0]  # -13
        early = np.sum(counts[[0, 1, 2, 3, 4, 5]], 0)  # first 6
        mid = np.sum(
            counts[
                [
                    last // 2 - 3,
                    last // 2 - 2,
                    last // 2 - 1,
                    last // 2,
                    last // 2 + 1,
                    last // 2 + 2,
                ]
            ],
            0,
        )  # middle
        late = np.sum(
            counts[[last - 6, last - 5, last - 4, last - 3, last - 2, last - 1]], 0
        )  # last 6
        ret = np.array([early, mid, late])

    return ret


def get_counts_weighted(
    correctList, decisionList, numRS, in_cluster, out_cluster, blockList=None
):
    """
    correctList: correct action for each trial
    decisionList: selected action for each trial
    numRS: number of rule switches
    in_cluster: list with actions which are in cluster
    out_cluster: list with actions which are out cluster

    ret: array for each ruleswitch number of in-cluster and out-cluster exploration errors
        if blocks following the ruleswitches are not successfull --> Nan values
    """
    counts = np.zeros((numRS, 2))

    rs = 0
    for trial in np.arange(1, correctList.shape[0]):
        if getRsIdx(trial, correctList, decisionList, blockList)[
            0
        ]:  # ruleswtich detected-->analyse
            rsIdx = getRsIdx(trial, correctList, decisionList, blockList)[1]
            if isinstance(rsIdx, type(None)):
                counts[rs] = np.nan
            else:
                prevcorrect = getRsIdx(trial, correctList, decisionList, blockList)[2]
                newcorrect = getRsIdx(trial, correctList, decisionList, blockList)[3]

                ### due to prev and new correct in_cluster or out_cluster get smaller (some actions are no exploration errors)
                ### thus weight the in_cluster and out_cluster selections
                ### if there are 2 possible in_cluster errors and 1 possible out_cluster errors --> divide in_cluster selections by two etc.
                weight_in_cluster = len(in_cluster)
                weight_out_cluster = len(out_cluster)
                for do_not_count in [prevcorrect, newcorrect]:
                    if do_not_count in in_cluster:
                        weight_in_cluster -= 1
                    elif do_not_count in out_cluster:
                        weight_out_cluster -= 1

                for idx in rsIdx:
                    if (
                        decisionList[idx] != prevcorrect
                        and decisionList[idx] != newcorrect
                    ):
                        ### its an exploration error
                        if decisionList[idx] in in_cluster:
                            ### in_cluster
                            counts[rs, 0] += 1 / weight_in_cluster
                        elif decisionList[idx] in out_cluster:
                            ### out_cluster
                            counts[rs, 1] += 1 / weight_out_cluster
            rs += 1

    ret = counts

    return ret


def get_counts_weighted_sims(
    correctList, decisionList, numRS, in_cluster, out_cluster, mode="threetime"
):
    """
    correctList: correct action for each trial
    decisionList: selected action for each trial
    numRS: number of rule switches
    in_cluster: list with actions which are in cluster
    out_cluster: list with actions which are out cluster

    ret: array for first, mid and last ruleswitch number of in-cluster and out-cluster exploration errors
    """
    ### get for each ruleswitch counts of in-cluster/out-cluster
    ### if blocks not successfull --> nan values
    counts = get_counts_weighted(
        correctList, decisionList, numRS, in_cluster, out_cluster
    )

    if mode == "alltime":
        ### return counts of all times
        return counts

    ### get start mid and end counts
    ### skip not successfull ruleswitches
    counts_start, counts_mid, counts_end = get_start_mid_end(counts)

    ### return first mid and last exploration
    ret = np.array([counts_start, counts_mid, counts_end])

    return ret


def get_start_mid_end(arr):
    """

    get first mid and last entry of arr

    skip entries which contain nan values

    """

    ### do not use not successfull ruleswitches
    ### get arr of first time
    idx = 0
    arr_start = arr[idx]
    while contains_nan(arr_start):
        idx = idx + 1
        arr_start = arr[idx]

    ### get arr of mid time
    ### if not correct use ruleswitch before or after
    idx = 0
    arr_mid = arr[arr.shape[0] // 2]
    while contains_nan(arr_mid):
        idx = idx + 1
        ### first try later ruleswitch
        arr_mid = arr[arr.shape[0] // 2 + idx]
        ### if still nan -> try earlier ruleswitch
        if contains_nan(arr_mid):
            arr_mid = arr[arr.shape[0] // 2 - idx]

    ### get arr of last time
    idx = 0
    arr_end = arr[arr.shape[0] - 1 - idx]
    while contains_nan(arr_end):
        idx = idx + 1
        arr_end = arr[arr.shape[0] - 1 - idx]

    return [arr_start, arr_mid, arr_end]


def contains_nan(element):
    """
    element: either single value or array-like
    """

    return np.sum(np.isnan(np.array(element))) > 0


def get_trialsNeeded(correctList, decisionList):
    """
    loads correctList, decisionList

    counts for each ruleswitch/exploration period the number of trials

    returns trialsnumber just of first, middle and last exploration period
    """

    trialsNeeded = np.zeros(numRS)

    rs = 0
    for trial in np.arange(1, correctList.shape[0]):
        if getRsIdx(trial, correctList, decisionList)[0]:
            # ruleswtich detected-->analyse
            rsIdx = getRsIdx(trial, correctList, decisionList)[1]

            if isinstance(rsIdx, type(None)):
                ### if rsIdx == None --> not successfull block --> trialsNeeded = nan value
                trialsNeeded[rs] = np.nan
            else:
                # only count errors, if size==1 than the correct action was immediatly selected = no errors
                trialsNeeded[rs] = rsIdx.size - 1

            rs += 1

    trialsNeeded_start, trialsNeeded_mid, trialsNeeded_end = get_start_mid_end(
        trialsNeeded
    )

    ret = np.array([trialsNeeded_start, trialsNeeded_mid, trialsNeeded_end])

    return ret


def get_trialsNeeded_all(correctList, decisionList):
    """
    loads correctList, decisionList

    counts for each ruleswitch/exploration period the number of trials
    returns for each ruleswitch number of trials, nan value if block was not successfull
    """

    trialsNeeded = np.zeros(numRS)

    rs = 0
    for trial in np.arange(1, correctList.shape[0]):
        if getRsIdx(trial, correctList, decisionList)[0]:
            # ruleswtich detected-->analyse
            rsIdx = getRsIdx(trial, correctList, decisionList)[1]

            if isinstance(rsIdx, type(None)):
                ### if rsIdx None --> trialsNeeded = nan value
                trialsNeeded[rs] = np.nan
            else:
                # only count errors, if size==1 than the correct action was immediatly selected = no errors
                trialsNeeded[rs] = rsIdx.size - 1

            rs += 1

    ret = trialsNeeded

    return ret


##############################################################################################################################################################
###################################################################  FUNCTIONS VPS  ##########################################################################
##############################################################################################################################################################


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

    return [trials, correct, decision, frequentAction, start, dopInp, block]


def get_counts_vp(correctList, decisionList, blockList):
    """
    loads correctList, decisionList

    counts for each rule switch how often actions were chosen
    all actions = [1,2,3,4,5,0]
    for the first block:
        counted actions = [possibleActions,0]
        possibleActions = [1,2,3,4,5] without prev correct and new correct
    for the last two blocks:
        counted actions = [out1,out2,possibleAction,0]
        out1 and out2 = never rewarded actions in block 2 and 3 --> the two least rewarded actions (appearing in correctList)
        possibleAction = [1,2,3,4,5] without prev correct, new correct, out1 and out2

    returns just early, mid and late counts (3x4 array)
        early mid and late are combined counts of 6 exploration periods (which are early/mid/late)
    """
    counts = np.zeros((numRS, 4))

    _ = np.array([11, 23, 35, 47, 59])
    ### TODO make all rs values dependend on experiment
    ### TODO adjust analyses to new rare-experiment --> see analyses of simulations
    ### DID THESE THINGS --> NEW FUCNTION get_counts_weighted_vps
    ### THIS FUNCTION IS NOT USED ANYMORE!

    rs = 0
    for trial in np.arange(1, correctList.shape[0]):
        if getRsIdx(trial, correctList, decisionList, blockList)[
            0
        ]:  # ruleswtich detected-->analyse
            rsIdx = getRsIdx(trial, correctList, decisionList, blockList)[1]
            prevcorrect = getRsIdx(trial, correctList, decisionList, blockList)[2]
            newcorrect = getRsIdx(trial, correctList, decisionList, blockList)[3]

            if (
                rs == np.array([11, 23, 35, 47, 59])
            ).sum() == 0:  # nur wenn rs nicht einer von den "schlechten" ist auswerten, "schlecht" = rs von phase 12 auf 13 und rs zwischen bloecken
                if rs < 23:  # --> erstere Block
                    possibleAction = [1, 2, 3, 4, 5]
                    possibleAction.remove(prevcorrect)
                    possibleAction.remove(newcorrect)
                    actions = [
                        possibleAction[0],
                        possibleAction[1],
                        possibleAction[2],
                        0,
                    ]
                else:
                    reward_frequencies = np.histogram(correctList, [1, 2, 3, 4, 5, 6])[
                        0
                    ]
                    out1 = reward_frequencies.argmin()
                    reward_frequencies[out1] += 2 * reward_frequencies.max()
                    out2 = reward_frequencies.argmin() + 1
                    out1 += 1
                    possibleAction = [1, 2, 3, 4, 5]
                    possibleAction.remove(out1)
                    possibleAction.remove(out2)
                    possibleAction.remove(prevcorrect)
                    possibleAction.remove(newcorrect)
                    actions = [out1, out2, possibleAction[0], 0]

                for idx in rsIdx:
                    counts[rs] += 1 * (decisionList[idx] == actions)

                rs += 1
            else:
                rs += 1
    # combine counts of six explorationperiods per time (early, mid, late) --> 6*10(vps) = 60 explorationperiods per time
    early = np.sum(
        counts[[24, 25, 26, 27, 28, 29]], 0
    )  # first 6 ruleswitches of block 2
    mid = np.sum(
        counts[[44, 45, 46, 48, 49, 50]], 0
    )  # middle (of Blocks 2 and 3) 6 ruleswiches
    late = np.sum(counts[[65, 66, 67, 68, 69, 70]], 0)  # last 6 ruleswitches of block 3
    ret = np.array([early, mid, late])

    return ret


def get_counts_weighted_vps(
    correctList, decisionList, blockList, numRS, experiment, mode="threetime"
):

    ### get in_cluster, out_cluster
    reward_frequencies, _ = np.histogram(correctList, [1, 2, 3, 4, 5, 6])
    frequencies_idx = np.argsort(reward_frequencies)
    out_cluster = np.array([1, 2, 3, 4, 5])[frequencies_idx[:2]]
    in_cluster = np.array([1, 2, 3, 4, 5])[frequencies_idx[-3:]]

    ### get counts of all rule switches
    ### if blocks not successfull --> nan values
    counts = get_counts_weighted(
        correctList, decisionList, numRS, in_cluster, out_cluster, blockList=blockList
    )

    if mode == "alltime":
        ### return counts of all times
        ### of valid blocks (not initial familiarization and blocks after breaks)
        start_rs, mid_rs, end_rs, not_use_this_rule_switch_list = get_valid_idx_vps(
            experiment
        )
        all_idx_arr = np.arange(counts.shape[0]).astype(int)
        mask = (all_idx_arr >= start_rs).astype(int)
        ### set blocks after breaks to nan
        for not_use_this_rule_switch in not_use_this_rule_switch_list:
            counts[not_use_this_rule_switch] = np.nan
        return counts[mask.astype(bool)]

    ### get first mid and end values
    ### skip not successfull blocks and breaks
    counts_start, counts_mid, counts_end = get_start_mid_end_vps(
        counts, experiment, "sum"
    )

    ret = np.array([counts_start, counts_mid, counts_end])

    return ret


def get_valid_idx_vps(experiment):
    ### define which blocks shouild be removed from arr_like
    ### exclude rule switches which are during breaks
    if experiment == "001e" or experiment == "014a":
        ### idx 23 = first rule switch (actually initial learning)
        ### 12 later = idx 35 --> ruleswitch after/during pause and so on
        not_use_these_rule_switches = [35, 47, 59]
    elif experiment == "001f" or experiment == "014b":
        ### idx 14 = first rule switch (actually initial learning)
        ### 14 later = idx 28 --> ruleswitch after/during pause and so on
        not_use_these_rule_switches = [28, 42, 56, 70]

    ### define the ruleswitches where the three time points "start"
    if experiment == "001e" or experiment == "014a":
        ### rule switches for cluster experiment
        # first two blocks a 12 phases are familiarization
        # --> first two blocks = 23 rule switches --> last idx == 22
        # but next idx 23 is initial learning of actual experimental blocks
        # --> idx 24 is first rule switch in actual experimental blocks
        start_rs = 24
        # there are 71 ruleswitches --> 70 is last idx
        end_rs = 70
        mid_rs = (start_rs + end_rs) // 2
    elif experiment == "001f" or experiment == "014b":
        ### rule switches for rare experiment
        # first block a 15 phases is familiarization
        # --> first block = 14 rule switches --> last idx == 13
        # but next idx 14 is initial learning of actual experimental blocks
        # --> idx 15 is first rule switch in actual experimental blocks
        start_rs = 15
        # there are 84 ruleswitches --> 83 is last idx
        end_rs = 83
        mid_rs = (start_rs + end_rs) // 2
    return start_rs, mid_rs, end_rs, not_use_these_rule_switches


def get_start_mid_end_vps(arr, experiment, combine_mode):
    start_rs, mid_rs, end_rs, not_use_these_rule_switches = get_valid_idx_vps(
        experiment
    )

    ### combine counts of six explorationperiods per time (early, mid, late) --> 6*10(vps) = 60 explorationperiods per time
    ### skip not successfull blocks (nan values) and blocks after breaks
    start = []
    idx = 0
    while len(start) < 6:
        if (
            not contains_nan(arr[start_rs + idx])
            and (start_rs + idx) not in not_use_these_rule_switches
        ):
            start.append(arr[start_rs + idx])
        idx = idx + 1

    mid = []
    idx = 0
    while len(mid) < 6:
        if (
            not contains_nan(arr[mid_rs + idx])
            and (mid_rs + idx) not in not_use_these_rule_switches
        ):
            mid.append(arr[mid_rs + idx])
        if (
            not contains_nan(arr[mid_rs - idx])
            and len(mid) < 6
            and (mid_rs - idx) not in not_use_these_rule_switches
        ):
            mid.append(arr[mid_rs - idx])
        idx = idx + 1

    end = []
    idx = 0
    while len(end) < 6:
        if (
            not contains_nan(arr[end_rs - idx])
            and (end_rs - idx) not in not_use_these_rule_switches
        ):
            end.append(arr[end_rs - idx])
        idx = idx + 1

    start = np.array(start)
    mid = np.array(mid)
    end = np.array(end)

    if combine_mode == "sum":
        ret = [np.sum(start, 0), np.sum(mid, 0), np.sum(end, 0)]
    elif combine_mode == "mean":
        ret = [np.mean(start, 0), np.mean(mid, 0), np.mean(end, 0)]

    return ret


def get_trialsNeeded_vp(correctList, decisionList, blockList, numRS, experiment):
    """
    loads correctList, decisionList

    counts for each ruleswitch/exploration period the number of trials

    returns just early, mid and late trialsnumber
        early mid and late are the mean of 6 exploration periods (which are early/mid/late)
    """

    trialsNeeded = np.zeros(numRS)

    ### get for each rule switch the number of errors
    rs = 0
    for trial in np.arange(1, correctList.shape[0]):
        if getRsIdx(trial, correctList, decisionList, blockList)[0]:
            # ruleswtich detected-->analyse
            rsIdx = getRsIdx(trial, correctList, decisionList, blockList)[1]
            if isinstance(rsIdx, type(None)):
                ### if None --> not successful block --> set value to nan
                trialsNeeded[rs] = np.nan
            else:
                trialsNeeded[rs] = rsIdx.size - 1
            rs += 1

    ### get first mid and end values
    ### skip not successfull blocks and breaks
    trialsNeeded_start, trialsNeeded_mid, trialsNeeded_end = get_start_mid_end_vps(
        trialsNeeded, experiment, "mean"
    )

    ret = np.array([trialsNeeded_start, trialsNeeded_mid, trialsNeeded_end])

    return ret


def get_trialsNeeded_vp_all(correctList, decisionList, blockList, numRS, experiment):
    """
    loads correctList, decisionList

    counts for each ruleswitch/exploration period the number of trials

    returns just early, mid and late trialsnumber
        early mid and late are the mean of 6 exploration periods (which are early/mid/late)
    """

    trialsNeeded = np.zeros(numRS) * np.nan

    if experiment == "001e" or experiment == "014a":
        ### nur wenn rs nicht einer von den "schlechten" ist auswerten, "schlecht" = rs nach pause
        not_eval_rs = np.array([11, 23, 35, 47, 59])
        ### first 2 blocks = 24 phases --> 23 rule switches --> last idx = 22
        ### but next idx 23 is initial learning --> idx of first experimental exploration is 24
        experimental_rs_start_idx = 24
    elif experiment == "001f" or experiment == "014b":
        ### nur wenn rs nicht einer von den "schlechten" ist auswerten, "schlecht" = rs nach pause
        not_eval_rs = np.array([14, 28, 42, 56, 70])
        ### first block = 15 phases --> 14 rule switches --> last idx = 13
        ### but next idx 14 is initial learning --> idx of first experimental exploration is 15
        experimental_rs_start_idx = 15

    rs = 0
    for trial in np.arange(1, correctList.shape[0]):
        if getRsIdx(trial, correctList, decisionList, blockList)[0]:
            ### ruleswtich detected-->analyse
            rsIdx = getRsIdx(trial, correctList, decisionList, blockList)[1]
            if rs not in not_eval_rs and not isinstance(rsIdx, type(None)):
                ### rs is not after/during break and not None (block is successfull)
                ### --> check if errors have been made
                if rsIdx.size > 0:
                    ### only count errors, if size==1 than the correct action was immediatly selected = no errors
                    trialsNeeded[rs] = rsIdx.size - 1
                rs += 1
            else:
                rs += 1

    ret = trialsNeeded[experimental_rs_start_idx:]
    return ret


##############################################################################################################################################################
#################################################################  FUNCTIONS BOTH  ###########################################################################
##############################################################################################################################################################


def getRsIdx(trial, correctList, decisionList, blockList=None):
    """
    load the trial, correctList and decisionList

    checks if there is a ruleswitch (in correctList) at trial

    returns the indices of the following trials of the exploration period (without prevcorrect and zeros decisions but with the new correct)

    returns None as IDX if the block is not successfull
    """
    if trial < 1:
        return [False, False]
    if isinstance(blockList, type(None)):
        blockList = np.zeros(100000)
    if (
        correctList[trial] != correctList[trial - 1]
        or blockList[trial] != blockList[trial - 1]
    ):  # ruleswitch detected!
        prevCorrect = correctList[trial - 1]
        newCorrect = correctList[trial]
        start = trial
        i = trial
        # go over trials until the newCorrect gets selected two times = end of the exploration
        # also end if correct changes again or end of trials is reached
        while (
            (
                bool(
                    decisionList[i] == newCorrect and decisionList[i + 1] == newCorrect
                )
                is False
            )
            and (newCorrect == correctList[i + 1])
            and i < decisionList.shape[0] - 2
        ):
            i += 1
        end = i

        ### check if while ended because decisions are new correct
        if decisionList[i] == newCorrect and decisionList[i + 1] == newCorrect:
            ended_because_rewarded = True
        else:
            ### ended because i+1 reached end of decisionlist or because correct changed again --> next block
            ended_because_rewarded = False

        ### check if rewarded rewarded was selected at least 7 times in block following the rule switch
        ### if yes --> successful_block is True else it's False
        successful_block = False
        if ended_because_rewarded:
            while (newCorrect == correctList[i + 1]) and i < correctList.shape[0] - 2:
                i += 1
            if i == correctList.shape[0] - 2:
                ### if while ended because i reached end of array --> i needs to be incerased by 1
                i += 1
            end_block = i
            dec_is_correct = (
                decisionList[np.arange(start, end_block + 1, 1).astype(int)]
                == newCorrect
            )
            successful_block = (
                np.sum(
                    np.array(
                        list(
                            map(
                                sum,
                                np.split(
                                    dec_is_correct,
                                    np.where(np.diff(dec_is_correct))[0] + 1,
                                ),
                            )
                        )
                    )
                    > 6
                )
                > 0
            )

        if successful_block:
            ### only if the block is successfull --> analyse
            ### if not --> return -1 as idx
            IDX = np.arange(start, end + 1, 1)
            ### remove the prev correctfrom the exploration indizes
            IDX = IDX[decisionList[IDX] != prevCorrect]
            ### also dont take zero decisions (=invalid trials, dont get analyzed)
            IDX = IDX[decisionList[IDX] != 0]

            if count_repetitions == "without_rep":
                ### only count each selection once, remove repetitions of the same exploratory selection
                already_in = []
                new_IDX = []
                for IDX_val in IDX:
                    if not (decisionList[IDX_val] in already_in):
                        new_IDX.append(IDX_val)
                        already_in.append(decisionList[IDX_val])
                IDX = np.array(new_IDX)

            return [True, IDX, prevCorrect, newCorrect]
        else:
            return [True, None, prevCorrect, newCorrect]
    else:
        return [False, False]


def get_first_trial_idx(correctList, decisionList, blockList=None, experiment=None):

    if experiment == "001e" or experiment == "014a":
        ### first 2 blocks = 24 phases --> 23 rule switches --> last idx = 22
        ### but next idx 23 is initial learning --> idx of first experimental exploration is 24
        experimental_rs_start_idx = 24
    elif experiment == "001f" or experiment == "014b":
        ### first block = 15 phases --> 14 rule switches --> last idx = 13
        ### but next idx 14 is initial learning --> idx of first experimental exploration is 15
        experimental_rs_start_idx = 15
    else:
        experimental_rs_start_idx = 0

    nr_trials = len(correctList)
    first_trial_idx_list = []
    for trial_idx in range(nr_trials):
        if getRsIdx(trial_idx, correctList, decisionList, blockList)[0]:
            first_trial_idx_list.append(trial_idx)

    ### shift the trials so that first initial trial is zero
    ### thus number of trials = learning time
    first_trial_idx_list = np.array(first_trial_idx_list)
    first_trial_idx_list = first_trial_idx_list[experimental_rs_start_idx:]
    first_trial_idx_list = first_trial_idx_list - first_trial_idx_list[0]

    return first_trial_idx_list


##############################################################################################################################################################
#################################################################  GLOBAL PARAMETERS  ########################################################################
##############################################################################################################################################################
if len(sys.argv) == 7:
    error_analysis_idx = int(sys.argv[1])
    learning_on = int(sys.argv[2])
    which_experiment_idx = int(sys.argv[3])
    count_repetitions_idx = int(sys.argv[4])
    time_regression_mode_idx = int(sys.argv[5])
    stn_gpe_factor_idx = int(sys.argv[6])
else:
    error_analysis_idx = 0
    learning_on = 1
    which_experiment_idx = 0
    count_repetitions_idx = 0
    time_regression_mode_idx = 0
    stn_gpe_factor_idx = 0

error_analysis = ["ANOVA", "regression"][error_analysis_idx]
xlim = [0.5, 3.5]
xticks = [1, 2, 3]
probslim = [0, 60 * 1.5 / 2]  # [0,70]#[0,50]
yticksprobs = [0, 15, 30, 45]  # [0,2,4,6]#[0,15,30,45]
rtlim = [100, 650]
yticksrt = [150, 600]
triallim = [-0.5, 3]  # [0.5,5.5]
ytickstrial = [0, 1.5, 3]  # [1,3,5]
performancelim = [-0.05, 1.05]
yticksperformance = [0, 1]
xlabels = ["start", "mid", "end"]
caps = {"ANOVA": 4, "regression": 0}[error_analysis]
font = {}
font["axLabel"] = {"fontsize": 11, "fontweight": "normal"}
font["axTicks"] = {"fontsize": 9}
font["subplotLabels"] = {"fontsize": 14, "fontweight": "bold"}
font["legend"] = {"fontsize": 9, "fontweight": "normal"}
font["titles"] = {"fontsize": 14, "fontweight": "bold"}
simsDrittel = [[0], [14], [28]]
cols = [
    [87 / 255.0, 26 / 255.0, 16 / 255.0],
    [214 / 255.0, 74 / 255.0, 49 / 255.0],
    [106 / 255.0, 138 / 255.0, 38 / 255.0],
]  # never,frequent,rare
maxtolerance = 7  # wichtig fuer frequency_rare... wie weit duerfen timesteps von first,mid,last maximal weg sein
useSEM = True  # variation measurement for trials, either standard error of the mean or standard deviation
time_regression_mode = [
    "relative_exploration",
    "absolute_exploration",
    "absolute_trial",
][time_regression_mode_idx]
which_experiment = ["001e", "001f", "001g", "014a", "014b", "014c"][
    which_experiment_idx
]
count_repetitions = ["with_rep", "without_rep"][count_repetitions_idx]

frequency_plot = ["bar", "lines"][1]
ideal_errors = {
    "001e": [1.5, 0.5],
    "001f": [1.5, 0.857],
    "001g": [1.5, 1.5],
    "014a": [1.5, 0.5],
    "014b": [1.5, 0.857],
    "014c": [1.5, 1.5],
}[which_experiment]
detla_highlighting = 0.3
save_folder_img = f"../3_results/stn_gpe_factor_{stn_gpe_factor_idx}/"
save_folder_values = f"../2_dataEv//stn_gpe_factor_{stn_gpe_factor_idx}/"
create_dir(save_folder_img)
create_dir(save_folder_values)

##############################################################################################################################################################
###################################################################  SIMULATIONS  ############################################################################
##############################################################################################################################################################


folder_learning_on = {
    "001e": "../../../simulations/001e_Cluster_Experiment_PaperLearningRule_LearningON/4_dataEv",
    "001f": "../../../simulations/001f_rare_experiment/4_dataEv",
    "001g": "../../../simulations/001g_all_correct/4_dataEv",
    "014a": f"../../../simulations/014a_new_dopamine_cluster/4_dataEv/stn_gpe_factor_idx_{stn_gpe_factor_idx}",
    "014b": f"../../../simulations/014b_new_dopamine_rare/4_dataEv/stn_gpe_factor_idx_{stn_gpe_factor_idx}",
    "014c": f"../../../simulations/014c_new_dopamine_all/4_dataEv/stn_gpe_factor_idx_{stn_gpe_factor_idx}",
}[which_experiment]
folder_learning_off = {
    "001e": "../../../simulations/002e_Cluster_Experiment_PaperLearningRule_LearningOFF/4_dataEv",
    "001f": None,
    "001g": None,
    "014a": "../../../simulations/014d_new_dopamine_fixed_cluster/2_dataRaw/hubel_2",
    "014b": None,
    "014c": None,
}[which_experiment]
folder = [folder_learning_on, folder_learning_off]
print("##############################################")
print(folder)
print("##############################################")
numRS = {"001e": 59, "001f": 69, "001g": 59, "014a": 59, "014b": 69, "014c": 59}[
    which_experiment
]
maxSimNr = {"001e": 60, "001f": 60, "001g": 60, "014a": 60, "014b": 60, "014c": 60}[
    which_experiment
]
simAnz = {"001e": 60, "001f": 60, "001g": 60, "014a": 60, "014b": 60, "014c": 60}[
    which_experiment
]
loadedSims = np.arange(1, simAnz + 1, 1).astype(
    int
)  # rng.choice(np.arange(1, maxSimNr + 1), size=simAnz, replace=False)

# ###############################################################################
# #############################  PERFORMANCE  ###################################
# ###############################################################################
# fig = plt.figure(figsize=(184 / 25.4, 120 / 25.4), dpi=300)
# performance = np.zeros((numRS + 1, simAnz))

# ###########################  GET PERFORMANCE  #################################
# for sim in range(simAnz):
#     performance[:, sim] = get_performance(loadedSims[sim], folder[int(1 - learning_on)])

# #####################  MEAN PERFORMANCE OVER SIMS  ############################
# performanceMean = np.mean(performance, 1)
# print("mean performance:", performanceMean)


###############################################################################
###########################  Probabilities  ###################################
###############################################################################
fig = plt.figure(figsize=(184 / 25.4, 120 / 25.4), dpi=300)
probs = np.zeros((2, 3, 2, simAnz))
probs_alltime = np.zeros((2, numRS, 2, simAnz))
# 0.index= with vs without learning
# 1.index= 3 different times
# 2.index= 2 possible eploration error categories in_cluster, out_cluster

for learning_idx in range(2):
    if folder[learning_idx] == None:
        continue
    #############################  GET PROBS  #####################################
    for sim in range(simAnz):
        output = get_output(folder[learning_idx], loadedSims[sim])
        probs[learning_idx, :, :, sim] = get_counts_weighted_sims(
            correctList=output[1],
            decisionList=output[2],
            numRS=numRS,
            in_cluster=[2, 3, 4],
            out_cluster=[1, 5],
        )
        probs_alltime[learning_idx, :, :, sim] = get_counts_weighted_sims(
            correctList=output[1],
            decisionList=output[2],
            numRS=numRS,
            in_cluster=[2, 3, 4],
            out_cluster=[1, 5],
            mode="alltime",
        )

##############  Summieren UEBER SIMS + CHISQUARE SAVES  #######################
probs = np.sum(probs, 3)
probs_alltime = np.nanmean(probs_alltime, 3)

for learning_idx in range(2):
    if folder[learning_idx] == None:
        continue
    with open(
        f"{save_folder_values}frequencies_for_CHI2_simulation{['_on', '_off'][learning_idx]}_{which_experiment}_{count_repetitions}.txt",
        "w",
    ) as f:
        for timeIdx, time in enumerate(["first", "mid", "last"]):
            print(
                time,
                probs[learning_idx, timeIdx, 1],  # out_cluster
                probs[learning_idx, timeIdx, 0],  # in_cluster
                file=f,
            )


###############################  PLOT  ########################################
x = np.arange(3) + 1
ax_bar1 = plt.subplot(221)
if frequency_plot == "bar":
    width = 0.2
    plt.bar(
        x - 0.75 * width,
        probs[1 - learning_on, :, 1],  # out_cluster
        1.5 * width,
        color=cols[0],
        label="out_cluster",
    )
    plt.bar(
        x + 0.75 * width,
        probs[1 - learning_on, :, 0],  # in_cluster
        1.5 * width,
        color=cols[2],
        label="in_cluster",
    )
    # style
    plt.xlim(xlim)
    plt.xticks(xticks, [])
    plt.ylim(probslim)
    plt.yticks(yticksprobs, **font["axTicks"])
elif frequency_plot == "lines":
    plt.plot(probs_alltime[1 - learning_on, :, 1], ".", color=cols[0])
    plt.plot(probs_alltime[1 - learning_on, :, 0], ".", color=cols[2])
    plt.ylim(0, 1)

plt.ylabel("error frequencies", **font["axLabel"])
plt.text(
    -0.25,
    0.5,
    "A:",
    va="center",
    ha="center",
    transform=ax_bar1.transAxes,
    **font["subplotLabels"],
)
plt.title("Simulations", pad=15, **font["titles"])


###############################################################################
#############################  ANZ TRIALS  ####################################
###############################################################################
if error_analysis == "ANOVA":
    trialsNeeded = np.zeros((2, 3, simAnz))
elif error_analysis == "regression":
    trialsNeeded = np.zeros((2, numRS, simAnz))
first_trial_idx = np.zeros((2, numRS, simAnz))

############################  GET TRIALS  #####################################
for learing_idx in range(2):
    if folder[learing_idx] == None:
        continue
    for sim in range(simAnz):
        output = get_output(folder[learing_idx], loadedSims[sim])
        if error_analysis == "ANOVA":
            trialsNeeded[learing_idx, :, sim] = get_trialsNeeded(output[1], output[2])
        elif error_analysis == "regression":
            trialsNeeded[learing_idx, :, sim] = get_trialsNeeded_all(
                output[1], output[2]
            )
        first_trial_idx[learing_idx, :, sim] = get_first_trial_idx(
            correctList=output[1], decisionList=output[2]
        )

#######################  MITTELN UEBER SIMS  ##################################
trialsNeededMEAN = np.nanmean(trialsNeeded, 2)
if useSEM:
    trialsNeededSD = sem(trialsNeeded, axis=2, ddof=1, nan_policy="omit")
else:
    trialsNeededSD = np.nanstd(trialsNeeded, 2)

###############################  PLOT  ########################################
x = {"ANOVA": np.arange(3) + 1, "regression": np.arange(numRS)}[error_analysis]
ax_trials1 = plt.subplot(223)
if error_analysis == "ANOVA":
    plt.xlim(xlim)
    plt.xticks(xticks, xlabels, **font["axTicks"])
elif error_analysis == "regression":
    plt.axvspan(
        x[0] - detla_highlighting,
        x[0] + detla_highlighting,
        ymin=0,
        ymax=0.07,
        color="#ff8700ff",
    )
    plt.axvspan(
        x[-1] - detla_highlighting,
        x[-1] + detla_highlighting,
        ymin=0,
        ymax=0.07,
        color="#ff8700ff",
    )
    plt.axvspan(
        x[len(x) // 2] - detla_highlighting,
        x[len(x) // 2] + detla_highlighting,
        ymin=0,
        ymax=0.07,
        color="#ff8700ff",
    )
plt.axhline(ideal_errors[0], ls="--", color="k", alpha=0.5)
plt.axhline(ideal_errors[1], ls="--", color="k", alpha=0.5)
plt.errorbar(
    x,
    trialsNeededMEAN[1 - learning_on],
    yerr=trialsNeededSD[1 - learning_on],
    fmt=".",
    color="black",
    capsize=caps,
)
plt.yticks(ytickstrial, **font["axTicks"])
plt.ylim(triallim)
plt.ylabel("error trials", **font["axLabel"])
plt.xlabel("time", **font["axLabel"])
plt.text(
    -0.25,
    0.5,
    "C:",
    va="center",
    ha="center",
    transform=ax_trials1.transAxes,
    **font["subplotLabels"],
)


###############################################################################
##################  TRIALs FOR TTEST/ANOVA/REGRESSION  ########################
###############################################################################
if error_analysis == "ANOVA":
    for learning_idx in range(2):
        if folder[learning_idx] == None:
            continue
        filename = f"{save_folder_values}TRIALs_for_TTEST_simulation{['_on', '_off'][learning_idx]}_{which_experiment}_{count_repetitions}.txt"
        with open(filename, "w") as f:
            stringList = ["simID", "TRIALS", "TIME"]
            for string in stringList:
                print(string, end="\t", file=f)
            print("", file=f)

            for TIME in [0, 1, 2]:
                for simID in range(simAnz):

                    stringList = [
                        str(int(simID)),
                        str(round(trialsNeeded[learning_idx, TIME, simID], 3)),
                        str(int(TIME)),
                    ]
                    for string in stringList:
                        print(string, end="\t", file=f)
                    print("", file=f)
elif error_analysis == "regression":
    ### we previousyl made two ANOVAS
    ### 1st: plastic vs vps    x time
    ### 2nd: plastic vs fixed  x time
    ### --> for regression use plastic as baseline and two dummy variables for vps and fixed
    ### to get difference betwee groups for time->errors coefficient --> create interaction variables vps*time and fixed*time
    ### the coefficients of these two interaction variables represent the difference to the baseline group coefficient
    ### also perform 3 individual regression models for the three groups "palstic, vps, fixed" in which "time" predicts "errors"
    regression_data_interaction = {
        "vps": [],
        "fixed": [],
        "time": [],
        "vps*time": [],
        "fixed*time": [],
        "errors": [],
    }
    regression_data_learning = {
        "time": [],
        "errors": [],
    }
    regression_data_fixed = {
        "time": [],
        "errors": [],
    }

    ### fill regression data lists
    for learning_idx in range(2):
        if folder[learning_idx] == None:
            continue
        for rs_idx in range(numRS):
            for sim_idx in range(simAnz):
                if time_regression_mode == "relative_exploration":
                    time_regression = rs_idx / numRS
                elif time_regression_mode == "absolute_exploration":
                    time_regression = rs_idx
                elif time_regression_mode == "absolute_trial":
                    time_regression = first_trial_idx[learning_idx, rs_idx, sim_idx]
                regression_data_interaction["vps"].append(0)
                regression_data_interaction["fixed"].append(
                    learning_idx
                )  # if learning_idx == 1 --> learning off --> fixed=1
                regression_data_interaction["time"].append(time_regression)
                regression_data_interaction["vps*time"].append(0 * time_regression)
                regression_data_interaction["fixed*time"].append(
                    learning_idx * time_regression
                )
                regression_data_interaction["errors"].append(
                    trialsNeeded[learning_idx, rs_idx, sim_idx]
                )

                if learning_idx == 0:
                    ### learning on
                    regression_data_learning["time"].append(time_regression)
                    regression_data_learning["errors"].append(
                        trialsNeeded[learning_idx, rs_idx, sim_idx]
                    )

                elif learning_idx == 1:
                    ### learning off
                    regression_data_fixed["time"].append(time_regression)
                    regression_data_fixed["errors"].append(
                        trialsNeeded[learning_idx, rs_idx, sim_idx]
                    )

    ### covnert lists to arrays
    for key in regression_data_interaction.keys():
        regression_data_interaction[key] = np.array(regression_data_interaction[key])
    for key in regression_data_learning.keys():
        regression_data_learning[key] = np.array(regression_data_learning[key])
    for key in regression_data_fixed.keys():
        regression_data_fixed[key] = np.array(regression_data_fixed[key])

    ### save
    filename = f"{save_folder_values}TRIALs_for_regression_interaction_simulation_{which_experiment}_{count_repetitions}_{time_regression_mode}.npy"
    np.save(filename, regression_data_interaction)
    filename = f"{save_folder_values}TRIALs_for_regression_learning_simulation_{which_experiment}_{count_repetitions}_{time_regression_mode}.npy"
    np.save(filename, regression_data_learning)
    filename = f"{save_folder_values}TRIALs_for_regression_fixed_simulation_{which_experiment}_{count_repetitions}_{time_regression_mode}.npy"
    np.save(filename, regression_data_fixed)


##############################################################################################################################################################
############################################################  EYETRACKING EXPERIMENT  ########################################################################
##############################################################################################################################################################
folder = {
    "001e": "../../../psychExp/exp1_final/4_dataEv/outputs_vps/",
    "001f": "../../../psychExp/exp_rev_1_final/4_dataEv/outputs_vps/",
    "001g": None,
    "014a": "../../../psychExp/exp1_final/4_dataEv/outputs_vps/",
    "014b": "../../../psychExp/exp_rev_1_final/4_dataEv/outputs_vps/",
    "014c": None,
}[which_experiment]

### do nothing for vps if there is no folder/data
if folder == None:
    plt.savefig(
        f"{save_folder_img}manuscript_SRtask_results_{['learn_off','learn_on'][learning_on]}_{which_experiment}_{count_repetitions}_{error_analysis}.svg"
    )
    quit()

numRS = {
    "001e": 71,
    "001f": 84,
    "001g": None,
    "014a": 71,
    "014b": 84,
    "014c": None,
}[which_experiment]
vpAnz = 10
### num_analized_RS:
### cluster experiment: first two blocks a 12 phases are familiarization phase --> first 24 blocks i.e. 23 rs not analyzed
### rare experiment: first block a 15 phases are familiarization phase --> first 15 blocks i.e. 14 rs not analyzed
### additional -1 because first "exploration" of experimental blocks = initial learning
num_analized_RS = {
    "001e": numRS - 24,
    "001f": numRS - 15,
    "001g": None,
    "014a": numRS - 24,
    "014b": numRS - 15,
    "014c": None,
}[which_experiment]


###############################################################################
###########################  Probabilities  ###################################
###############################################################################
probs = np.zeros((3, 2, vpAnz))
probs_alltime = np.zeros((num_analized_RS, 2, vpAnz))
# 1.index= 3 different times
# 2.index= 2 possible eploration error categories in_cluster, out_cluster

#############################  GET PROBS  #####################################
for vp in range(vpAnz):
    output = get_output_vp(folder, vp + 1)
    probs[:, :, vp] = get_counts_weighted_vps(
        correctList=output[1],
        decisionList=output[2],
        blockList=output[6],
        numRS=numRS,
        experiment=which_experiment,
    )
    probs_alltime[:, :, vp] = get_counts_weighted_vps(
        correctList=output[1],
        decisionList=output[2],
        blockList=output[6],
        numRS=numRS,
        experiment=which_experiment,
        mode="alltime",
    )

##############  Summieren UEBER VPS + CHISQUARE SAVES  ########################
probs = np.sum(probs, 2)
probs_alltime = np.nanmean(probs_alltime, 2)

with open(
    f"{save_folder_values}frequencies_for_CHI2_eyetracking_{which_experiment}_{count_repetitions}.txt",
    "w",
) as f:
    for timeIdx, time in enumerate(["first", "mid", "last"]):
        print(
            time,
            probs[timeIdx, 1],  # out_cluster
            probs[timeIdx, 0],  # in_cluster
            file=f,
        )

# for i in range(probs.shape[0]):
#    probs[i]/=np.sum(probs[i])
# probsMEAN = probs

###############################  PLOT  ########################################
x = np.arange(3) + 1
ax = plt.subplot(222)
if frequency_plot == "bar":
    width = 0.2
    plt.bar(
        x - 0.75 * width,
        probs[:, 1],  # out_cluster
        1.5 * width,
        color=cols[0],
        label="out_cluster",
    )
    plt.bar(
        x + 0.75 * width,
        probs[:, 0],  # in_cluster
        1.5 * width,
        color=cols[2],
        label="in_cluster",
    )
    plt.xlim(xlim)
    plt.ylim(probslim)
    plt.xticks(xticks, [])
    plt.yticks(yticksprobs, **font["axTicks"])
elif frequency_plot == "lines":
    plt.plot(probs_alltime[:, 1], ".", color=cols[0])
    plt.plot(probs_alltime[:, 0], ".", color=cols[2])
    plt.ylim(0, 1)
plt.text(
    -0.25,
    0.5,
    "B:",
    va="center",
    ha="center",
    transform=ax.transAxes,
    **font["subplotLabels"],
)
plt.title("Experiments", pad=15, **font["titles"])


###############################################################################
###########################  TRIALS NEEDED  ###################################
###############################################################################
if error_analysis == "ANOVA":
    trialsNeeded = np.zeros((3, vpAnz))
elif error_analysis == "regression":
    trialsNeeded = np.zeros((num_analized_RS, vpAnz))
first_trial_idx = np.zeros((num_analized_RS, vpAnz))

##############################  GET RT  #######################################
for vp in range(vpAnz):
    output = get_output_vp(folder, vp + 1)
    if error_analysis == "ANOVA":
        trialsNeeded[:, vp] = get_trialsNeeded_vp(
            correctList=output[1],
            decisionList=output[2],
            blockList=output[6],
            numRS=numRS,
            experiment=which_experiment,
        )
    elif error_analysis == "regression":
        trialsNeeded[:, vp] = get_trialsNeeded_vp_all(
            correctList=output[1],
            decisionList=output[2],
            blockList=output[6],
            numRS=numRS,
            experiment=which_experiment,
        )
    first_trial_idx[:, vp] = get_first_trial_idx(
        correctList=output[1],
        decisionList=output[2],
        blockList=output[6],
        experiment=which_experiment,
    )

########################  MITTELN UEBER VPs  ##################################
trialsNeededMEAN = np.nanmean(trialsNeeded, 1)
if useSEM:
    trialsNeededSD = sem(trialsNeeded, axis=1, ddof=1, nan_policy="omit")
else:
    trialsNeededSD = np.nanstd(trialsNeeded, 1)

###############################  PLOT  ########################################
x = {"ANOVA": np.arange(3) + 1, "regression": np.arange(num_analized_RS)}[
    error_analysis
]
ax_trials2 = plt.subplot(224)
if error_analysis == "ANOVA":
    plt.xlim(xlim)
    plt.xticks(xticks, xlabels, **font["axTicks"])
elif error_analysis == "regression":
    plt.axvspan(
        x[0] - detla_highlighting,
        x[0] + 5 + detla_highlighting,
        ymin=0,
        ymax=0.07,
        color="#ff8700ff",
    )
    ### mark the mid 6 points
    ### very complecated to get the range...
    ### get mid and then increase range and ingore nan values
    mid = (x[0] + x[-1]) // 2
    mid_start = mid - detla_highlighting
    mid_end = mid + detla_highlighting
    while_loop_counter = 0
    nan_values = 0
    while (mid_end - mid_start) - nan_values < 6:
        if np.mod(while_loop_counter, 2) == 0:
            mid_end += 1
        else:
            mid_start -= 1
        idx_of_vals = []
        for range_val in range(
            np.ceil(mid_start).astype(int), np.ceil(mid_end).astype(int)
        ):
            idx_of_vals.append(np.where(x == range_val)[0][0])
        idx_of_vals = np.array(idx_of_vals).astype(int)
        nan_values = np.isnan(trialsNeededMEAN[idx_of_vals]).sum()
        while_loop_counter += 1
    plt.axvspan(mid_start, mid_end, ymin=0, ymax=0.07, color="#ff8700ff")
    plt.axvspan(
        x[-1] - 5 - detla_highlighting,
        x[-1] + detla_highlighting,
        ymin=0,
        ymax=0.07,
        color="#ff8700ff",
    )
plt.axhline(ideal_errors[0], ls="--", color="k", alpha=0.5)
plt.axhline(ideal_errors[1], ls="--", color="k", alpha=0.5)
plt.errorbar(
    x, trialsNeededMEAN, yerr=trialsNeededSD, fmt=".", color="black", capsize=caps
)
plt.yticks(ytickstrial, **font["axTicks"])
plt.ylim(triallim)
plt.xlabel("time", **font["axLabel"])
plt.text(
    -0.25,
    0.5,
    "D:",
    va="center",
    ha="center",
    transform=ax_trials2.transAxes,
    **font["subplotLabels"],
)


###############################################################################
###################  TRIALs FOR TTEST/ANOVA/REGRESSION  #######################
###############################################################################
if error_analysis == "ANOVA":
    filename = f"{save_folder_values}TRIALs_for_TTEST_eyetracking_{which_experiment}_{count_repetitions}.txt"
    with open(filename, "w") as f:
        stringList = ["subID", "TRIALS", "TIME"]
        for string in stringList:
            print(string, end="\t", file=f)
        print("", file=f)

        for TIME in [0, 1, 2]:
            for subID in range(vpAnz):

                stringList = [
                    str(int(subID)),
                    str(round(trialsNeeded[TIME, subID], 3)),
                    str(int(TIME)),
                ]
                for string in stringList:
                    print(string, end="\t", file=f)
                print("", file=f)
elif error_analysis == "regression":
    regression_data_interaction = {
        "vps": [],
        "fixed": [],
        "time": [],
        "vps*time": [],
        "fixed*time": [],
        "errors": [],
    }
    regression_data_vps = {
        "time": [],
        "errors": [],
    }

    ### fill the regression data lists
    for rs_idx in range(num_analized_RS):
        for vp_idx in range(vpAnz):
            if time_regression_mode == "relative_exploration":
                time_regression = rs_idx / num_analized_RS
            elif time_regression_mode == "absolute_exploration":
                time_regression = rs_idx
            elif time_regression_mode == "absolute_trial":
                time_regression = first_trial_idx[rs_idx, vp_idx]
            regression_data_interaction["vps"].append(1)
            regression_data_interaction["fixed"].append(0)
            regression_data_interaction["time"].append(time_regression)
            regression_data_interaction["vps*time"].append(1 * time_regression)
            regression_data_interaction["fixed*time"].append(0 * time_regression)
            regression_data_interaction["errors"].append(trialsNeeded[rs_idx, vp_idx])

            regression_data_vps["time"].append(time_regression)
            regression_data_vps["errors"].append(trialsNeeded[rs_idx, vp_idx])

    ### convert lists to arrays
    for key in regression_data_interaction.keys():
        regression_data_interaction[key] = np.array(regression_data_interaction[key])
    for key in regression_data_vps.keys():
        regression_data_vps[key] = np.array(regression_data_vps[key])

    ### save
    filename = f"{save_folder_values}TRIALs_for_regression_interaction_vps_{which_experiment}_{count_repetitions}_{time_regression_mode}.npy"
    np.save(filename, regression_data_interaction)
    filename = f"{save_folder_values}TRIALs_for_regression_vps_{which_experiment}_{count_repetitions}_{time_regression_mode}.npy"
    np.save(filename, regression_data_vps)


###############################################################################
##############################  LEGEND BOTTOM  ################################
###############################################################################
plt.subplots_adjust(top=0.9, bottom=0.2, right=0.99, left=0.12, wspace=0.4)
legend = {}
legend["width"] = 0.45
legend["height"] = 0.04
legend["bottom"] = 0.04
legend["rectangleSize"] = 0.03
legend["freespace"] = (legend["height"] - legend["rectangleSize"]) / 2.0

mid = (ax_trials1.get_position().x1 + ax_trials2.get_position().x0) / 2.0
xLeft = mid - legend["width"] / 2.0
legendField = mtl.patches.FancyBboxPatch(
    xy=(xLeft, legend["bottom"]),
    width=legend["width"],
    height=legend["height"],
    boxstyle=mtl.patches.BoxStyle.Round(pad=0.01),
    bbox_transmuter=None,
    mutation_scale=1,
    mutation_aspect=None,
    transform=fig.transFigure,
    **dict(linewidth=2, fc="w", ec="k", clip_on=False),
)
plt.gca().add_patch(legendField)
# never rewarded rectangle
(x0, y0) = (
    mid - legend["width"] / 2.0 + legend["freespace"],
    legend["bottom"] + legend["freespace"],
)
plt.gca().add_patch(
    mtl.patches.FancyBboxPatch(
        xy=(x0, y0),
        width=legend["rectangleSize"],
        height=legend["rectangleSize"],
        boxstyle=mtl.patches.BoxStyle.Round(pad=0),
        bbox_transmuter=None,
        mutation_scale=1,
        mutation_aspect=None,
        transform=fig.transFigure,
        **dict(linewidth=0, fc=cols[0], ec=cols[0], clip_on=False),
    )
)
plt.text(
    x0 + legend["rectangleSize"] + legend["freespace"],
    y0 + legend["rectangleSize"] / 2.0,
    "never rewarded",
    ha="left",
    va="center",
    transform=fig.transFigure,
    **font["legend"],
)
# repeatedly rewarded rectangle
(x0, y0) = (mid + legend["freespace"], legend["bottom"] + legend["freespace"])
plt.gca().add_patch(
    mtl.patches.FancyBboxPatch(
        xy=(x0, y0),
        width=legend["rectangleSize"],
        height=legend["rectangleSize"],
        boxstyle=mtl.patches.BoxStyle.Round(pad=0),
        bbox_transmuter=None,
        mutation_scale=1,
        mutation_aspect=None,
        transform=fig.transFigure,
        **dict(linewidth=0, fc=cols[2], ec=cols[2], clip_on=False),
    )
)
plt.text(
    x0 + legend["rectangleSize"] + legend["freespace"],
    y0 + legend["rectangleSize"] / 2.0,
    "repeatedly rewarded",
    ha="left",
    va="center",
    transform=fig.transFigure,
    **font["legend"],
)


###############################################################################
###############################  LEGEND TOP  ##################################
###############################################################################
# legend = {}
# legend["width"]       = 0.22
# legend["height"]      = 0.04
# legend["bottom"]      = ax_bar1.get_position().y1-0.06
# legend["rectangleSize"] = 0.03
# legend["freespace"]   = (legend["height"]-legend["rectangleSize"])/2.

# xLeft = (ax_bar1.get_position().x1 + ax_bar1.get_position().x0) / 2.-0.015
# mid = xLeft + legend["width"]/2.

# (x0, y0) = (mid-legend["width"]/2. + legend["freespace"], legend["bottom"] + legend["freespace"])

# mytext=plt.text(x0+legend["rectangleSize"]+legend["freespace"],y0+legend["rectangleSize"]/2.,r'STN$\rightarrow$GPe fixed',ha='left',va='center',transform=fig.transFigure, **font["legend"])

# figure coordinates for text box
# fig.canvas.draw()
# transf = fig.transFigure.inverted()
# bb = mytext.get_window_extent(renderer = fig.canvas.renderer)
# bbd = bb.transformed(transf)

# legendField=mtl.patches.FancyBboxPatch(xy=(bbd.x0-(legend["rectangleSize"]+legend["freespace"]),bbd.y0),width=bbd.x1-bbd.x0+legend["rectangleSize"]+legend["freespace"],height=bbd.y1-bbd.y0,boxstyle=mtl.patches.BoxStyle.Round(pad=0.01),bbox_transmuter=None,mutation_scale=1,mutation_aspect=None,transform=fig.transFigure,**dict(linewidth=0.7, fc='w',ec='k',clip_on=False))
# plt.gca().add_patch(legendField)

# plt.gca().add_patch(mtl.patches.FancyBboxPatch(xy=(x0,y0),width=legend["rectangleSize"],height=legend["rectangleSize"],boxstyle=mtl.patches.BoxStyle.Round(pad=0),bbox_transmuter=None,mutation_scale=1,mutation_aspect=None,transform=fig.transFigure,**dict(linewidth=1, fc=(0,0,0,0.2),ec=(0,0,0,1),clip_on=False)))


###############################################################################
###############################  LEGEND MID  ##################################
###############################################################################
# legend = {}
# legend["width"]       = 0.22
# legend["height"]      = 0.04
# legend["bottom"]      = ax_trials1.get_position().y1-0.06
# legend["rectangleSize"] = 0.03
# legend["freespace"]   = (legend["height"]-legend["rectangleSize"])/2.

# xLeft = (ax_trials1.get_position().x1 + ax_trials1.get_position().x0) / 2.-0.015
# mid = xLeft + legend["width"]/2.

# (x0, y0) = (mid-legend["width"]/2. + legend["freespace"], legend["bottom"] + legend["freespace"])

# mytext=plt.text(x0+legend["rectangleSize"]+legend["freespace"],y0+legend["rectangleSize"]/2.,r'STN$\rightarrow$GPe fixed',ha='left',va='center',transform=fig.transFigure, **font["legend"])

# figure coordinates for text box
# fig.canvas.draw()
# transf = fig.transFigure.inverted()
# bb = mytext.get_window_extent(renderer = fig.canvas.renderer)
# bbd = bb.transformed(transf)

# legendField=mtl.patches.FancyBboxPatch(xy=(bbd.x0-(legend["rectangleSize"]+legend["freespace"]),bbd.y0),width=bbd.x1-bbd.x0+legend["rectangleSize"]+legend["freespace"],height=bbd.y1-bbd.y0,boxstyle=mtl.patches.BoxStyle.Round(pad=0.01),bbox_transmuter=None,mutation_scale=1,mutation_aspect=None,transform=fig.transFigure,**dict(linewidth=0.7, fc=(1,1,1,0),ec='k',clip_on=False))
# plt.gca().add_patch(legendField)

# xInch=184/25.4
# xDots=xInch*300
# yInch=120/25.4
# yDots=yInch*300

# xP=(x0+legend["rectangleSize"]/2.)*xDots
# yP=(y0+legend["rectangleSize"]/2.)*yDots
# transf = ax_trials1.transData.inverted()
# xyP = transf.transform((xP,yP))

# ax_trials1.errorbar(xyP[0],xyP[1],yerr=0.15,fmt='.',color='grey',capsize=caps)


plt.savefig(
    f"{save_folder_img}manuscript_SRtask_results_{['learn_off','learn_on'][learning_on]}_{which_experiment}_{count_repetitions}_{error_analysis}.svg"
)
