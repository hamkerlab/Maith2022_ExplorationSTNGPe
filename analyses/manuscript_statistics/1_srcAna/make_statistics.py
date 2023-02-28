# from __future__ import print_function
import numpy as np

# import pyvttbl as pt
# from collections import namedtuple
import pylab as plt
from scipy import stats
from scipy.optimize import curve_fit

from pingouin import mixed_anova, rm_anova, homoscedasticity
from pandas import DataFrame
import pandas as pd
from CompNeuroPy import create_dir

from patsy import dmatrices
import statsmodels.api as sm
from itertools import combinations

font = {"family": "Arial", "weight": "normal", "size": 8}
bold_font = {"family": "Arial", "weight": "bold", "size": 8}
large_bold_font = {"family": "Arial", "weight": "bold", "size": 10}
plt.rc("font", **font)


def getData(datafile):
    """
    loads txt file with the IDs, dependent and independent variables / values

    returns for each of them a numpy array
    """
    file = open(datafile, "r")
    _ = file.readline()
    simID = []
    RT = []
    TIME = []
    PHASE = []
    i = 0
    try:
        while 1:
            zeile = file.readline()
            liste = zeile.split("\t")
            simID.append(int(liste[0]))
            RT.append(float(liste[1]))
            TIME.append(int(liste[2]))
            PHASE.append(int(liste[3]))
            i += 1
    except:
        file.close()

    return [np.array(simID), np.array(RT), np.array(TIME), np.array(PHASE)]


# def oldanova(datafile, experiment, names, saveFolder):
#     """
#     Two-way-repeated-meassures ANOVA

#     datafile = file with the IDs, dependent and independent variables / values

#     experiment = string for save files

#     names = list of 5 strings
#         Sim/Sub, simID/subID, DV, IV1, IV2

#     saveFolder = string with saveFolder

#     https://www.marsja.se/two-way-anova-repeated-measures-using-python/
#     """

#     [simID, RT, TIME, PHASE] = getData(datafile)

#     Sim = namedtuple(names[0], [names[1], names[2], names[3], names[4]])
#     df = pt.DataFrame()

#     for idx in xrange(len(simID)):
#         df.insert(Sim(simID[idx], RT[idx], TIME[idx], PHASE[idx])._asdict())

#     df.box_plot(
#         names[2],
#         factors=[names[3], names[4]],
#         fname=saveFolder + "/box_" + names[2] + "_" + experiment + ".png",
#     )

#     aov = df.anova(names[2], sub=names[1], wfactors=[names[3], names[4]])

#     filename = saveFolder + "/ANOVA_" + names[2] + "_" + experiment + ".txt"
#     with open(filename, "w") as f:
#         print(aov, file=f)

#     posthoctest(datafile, filename, names)


def rel_ttest(g1, g2, numTests):
    """
    loads two samples (1d arrays g1 and g2)

    calculates a two sided ttest for dependent samples

    resturns means, standard deviations, t, df, p and cohens d_z
    """
    n = np.sum(np.logical_not(np.isnan(g1)))
    df = n - 1
    M1 = np.nanmean(g1)
    M2 = np.nanmean(g2)
    SD1 = np.nanstd(g1)
    SD2 = np.nanstd(g2)
    Diff = g2 - g1
    MDiff = np.nanmean(Diff)
    SDDiff = np.nanstd(Diff)
    cohensD = MDiff / SDDiff  # Standardized difference scores d_z
    [tVal, pVal] = stats.ttest_rel(g2, g1, nan_policy="omit")
    ### calculate 95% CI for dependent samples
    t_crit = stats.t.ppf(
        0.975, df
    )  # 1-CI = 5% --> separated in two tails 2.5% in each tail --> either use right tail (0.975) or left tail (0.025), ppf=inverse of CDF
    CI = MDiff + np.array([-1, 1]) * t_crit * SDDiff / np.sqrt(n)
    # strings
    M1 = str(round(M1, 2))
    M2 = str(round(M2, 2))
    SD1 = str(round(SD1, 2))
    SD2 = str(round(SD2, 2))
    t = str(round(tVal, 2))
    df = str(int(round(df, 0)))
    CI = str(CI.round(2))
    p = str(round(pVal, 3))
    if pVal < 0.001:
        p = "<0.001"
    if pVal < (0.05 / numTests):
        p = p + "*"
    if pVal < (0.01 / numTests):
        p = p + "*"
    if pVal < (0.001 / numTests):
        p = p + "*"
    d = str(round(cohensD, 2))
    vals = [str(M1 + " (" + SD1 + ")"), str(M2 + " (" + SD2 + ")"), t, df, p, d, CI]
    return vals


def classical_cohensD(mean1, std1, count1, mean2, std2, count2):
    dof = count1 + count2 - 2
    cohens_d = (mean1 - mean2) / np.sqrt(
        ((count1 - 1) * std1**2 + (count2 - 1) * std2**2) / dof
    )
    return cohens_d


def ind_ttest(g1, g2, numTests):
    """
    loads two samples (1d arrays g1 and g2)

    calculates a two sided ttest for independent samples

    if variances are not equal use welch test

    resturns means, standard deviations, t, df, p and cohens d (classical with pooled SD)
    """
    n1 = np.sum(np.logical_not(np.isnan(g1)))
    n2 = np.sum(np.logical_not(np.isnan(g2)))
    df = n1 - 1 + n2 - 1
    M1 = np.nanmean(g1)
    M2 = np.nanmean(g2)
    SD1 = np.nanstd(g1)
    SD2 = np.nanstd(g2)
    # SD_pool = np.nanstd(np.concatenate([g1, g2]))
    cohensD = classical_cohensD(M1, SD1, n1, M2, SD2, n2)
    ### check equal variances and run one-way anova and post-hoc t-tests
    equal_var_dict = {
        "dv": np.concatenate([g1, g2]),
        "group": np.concatenate([np.ones(len(g1)), np.ones(len(g2)) * 2]).astype(int),
    }
    equal_var_df = pd.DataFrame(equal_var_dict)
    equal_var = homoscedasticity(data=equal_var_df, dv="dv", group="group").at[
        "levene", "equal_var"
    ]
    [tVal, pVal] = stats.ttest_ind(g1, g2, nan_policy="omit", equal_var=equal_var)
    ### 95% CI
    t_crit = stats.t.ppf(
        0.975, df
    )  # 1-CI = 5% --> separated in two tails 2.5% in each tail --> either use right tail (0.975) or left tail (0.025), ppf=inverse of CDF
    ### calculate pooled standard deviation, pooled vriance = pooled SD**2
    SDp = np.sqrt(((n1 - 1) * SD1**2 + (n2 - 1) * SD2**2) / (n1 + n2 - 2))
    CI = (M1 - M2) + np.array([-1, 1]) * t_crit * np.sqrt(
        (SDp**2 / n1) + (SDp**2 / n2)
    )
    # strings
    M1 = str(round(M1, 2))
    M2 = str(round(M2, 2))
    SD1 = str(round(SD1, 2))
    SD2 = str(round(SD2, 2))
    t = str(round(tVal, 2))
    if not (equal_var):
        t = t + "''"
    df = str(int(round(df, 0)))
    p = str(round(pVal, 3))
    CI = str(CI.round(2))
    if pVal < 0.001:
        p = "<0.001"
    if pVal < (0.05 / numTests):
        p = p + "*"
    if pVal < (0.01 / numTests):
        p = p + "*"
    if pVal < (0.001 / numTests):
        p = p + "*"
    d = str(round(cohensD, 2))
    vals = [str(M1 + " (" + SD1 + ")"), str(M2 + " (" + SD2 + ")"), t, df, p, d, CI]
    return vals


def posthoctest(datfile, filename, names):
    """
    multiple post hoc ttests

    datafile = file with the IDs, dependent and independent variables / values

    names = list of 5 strings
        Sim/Sub, simID/subID, DV, IV1, IV2

    makes ttests for the individual levels of factor 1
    """
    spacing = 20
    sep = ""
    with open(filename, "a") as f:

        print("\nPOSTHOC T-TESTS", file=f)

        [simID, RT, TIME, PHASE] = getData(datafile)

        TIMElevels = np.unique(TIME)
        PHASElevels = np.unique(PHASE)

        stringList = [
            names[4],
            names[3] + "=" + str(TIMElevels[0]),
            names[3] + "=" + str(TIMElevels[1]),
            "t",
            "df",
            "p",
            "d",
        ]
        for string in stringList:
            print(string.ljust(spacing, " "), end=sep, file=f)
        print("\n".ljust(spacing * len(stringList), "="), file=f)

        for PHASElevel in PHASElevels:
            ### ttest between the two times
            m0 = PHASE == PHASElevel
            m1 = TIME == TIMElevels[0]
            m2 = TIME == TIMElevels[1]
            group1 = RT[(m0 * m1).astype("bool")]
            group2 = RT[(m0 * m2).astype("bool")]
            ttestResults = rel_ttest(group1, group2, PHASElevels.size)
            print(str(PHASElevel).ljust(spacing, " "), end=sep, file=f)
            for string in ttestResults:
                print(string.ljust(spacing, " "), end=sep, file=f)
            print("", file=f)


def ttestForTrials(datafile, experiment, names, saveFolder):
    """
    datafile = file with the IDs, dependent and independent variable / values

    experiment = string for save files

    names = list of 4 strings
        Sim/Sub, simID/subID, DV, IV

    loads datafile with values for ID, DV and one IV (with two levels)

    calculates a two sided ttest for dependent samples to compare the groups of both IV levels

    prints the results in a txt file
    """

    if isinstance(
        datafile, list
    ):  # optional combine datasets e.g. simulations and vps combined --> which effect has time without separating groups (if no interaction --> groups are not separated)
        data = []
        for idx in range(len(datafile)):
            data.append(get_trials_data(datafile[idx]))

        simID = np.concatenate([data[idx][0] for idx in range(len(data))])
        TRIALS = np.concatenate([data[idx][1] for idx in range(len(data))])
        TIME = np.concatenate([data[idx][2] for idx in range(len(data))])

    else:
        [simID, TRIALS, TIME] = get_trials_data(datafile)

    TIMElevels = np.unique(TIME)
    comparisons = [[0, 1], [1, 2], [0, 2]]
    for comparisonIdx, comparison in enumerate(comparisons):
        m1 = TIME == TIMElevels[comparison[0]]
        m2 = TIME == TIMElevels[comparison[1]]
        group1 = TRIALS[m1]
        group2 = TRIALS[m2]

        spacing = 20
        sep = ""
        filename = saveFolder + "/TTEST_" + names[2] + "_" + experiment + ".txt"
        with open(filename, ["w", "a"][int(comparisonIdx > 0)]) as f:

            if comparisonIdx == 0:
                print("T-TEST " + names[2], file=f)
                stringList = [
                    names[3] + "=" + str(TIMElevels[comparison[0]]),
                    names[3] + "=" + str(TIMElevels[comparison[1]]),
                    "t",
                    "df",
                    "p",
                    "d_z",
                ]
                for string in stringList:
                    print(string.ljust(spacing, " "), end=sep, file=f)
                print("\n".ljust(spacing * len(stringList), "="), file=f)

            ### ttest between the two times
            ttestResults = rel_ttest(group1, group2, len(comparisons))
            for string in ttestResults:
                print(string.ljust(spacing, " "), end=sep, file=f)
            print("", file=f)


def ttestForTrials_vs_ideal(datafile, experiment, names, saveFolder):
    """
    loads error trials from data file

    calculates ttest for a single smaple, test if mean is different from ideal value (here 1.5 and 0.5)
    """

    [simID, TRIALS, TIME] = get_trials_data(datafile)

    TIMElevels = np.unique(TIME)
    comparison = [0, 2]
    m1 = TIME == TIMElevels[comparison[0]]
    m2 = TIME == TIMElevels[comparison[1]]
    group_list = [TRIALS[m1], TRIALS[m2]]

    spacing = 20
    sep = ""
    filename = saveFolder + "/TTEST_" + names[2] + "_" + experiment + ".txt"

    for ideal_idx, ideal in enumerate([1.5, 0.5]):
        with open(filename, ["w", "a"][ideal_idx]) as f:

            if ideal_idx == 0:
                print("T-TEST " + names[2] + " vs ideal", file=f)
                stringList = [
                    names[3],
                    "Mean (SD)",
                    "ideal",
                    "t",
                    "df",
                    "p",
                    "d",
                    "95%CI",
                ]

                for string in stringList:
                    print(string.ljust(spacing, " "), end=sep, file=f)
                print("\n".ljust(spacing * len(stringList), "="), file=f)

            ### ttest for sample vs ideal value
            ttestResults = ttest_vs_ideal(group_list[ideal_idx], ideal, 1)
            stringList = [str(comparison[ideal_idx])] + ttestResults
            for string in stringList:
                print(string.ljust(spacing, " "), end=sep, file=f)
            print("", file=f)


def ttest_exp_fit_vs_ideal(data_list, group_name_list, savefile):
    """
    checks if data of each group are smaller than 1

    prints results in save file
    """

    nr_group = len(data_list)
    ideal = 0

    for group_idx in range(nr_group):

        group = data_list[group_idx]
        group_name = group_name_list[group_idx]

        spacing = 20
        sep = ""

        with open(savefile, ["a", "w"][int(group_idx == 0)]) as f:

            if group_idx == 0:
                ### print header
                print("T-TEST vs ideal", file=f)
                stringList = [
                    "group_name",
                    "Mean (SD)",
                    "ideal",
                    "t",
                    "df",
                    "p",
                    "d",
                    "95%CI",
                ]

                for string in stringList:
                    print(string.ljust(spacing, " "), end=sep, file=f)
                print("\n".ljust(spacing * len(stringList), "="), file=f)

            ### ttest for sample vs ideal value
            ttestResults = ttest_vs_ideal(group, ideal, nr_group)
            stringList = [group_name] + ttestResults
            for string in stringList:
                print(string.ljust(spacing, " "), end=sep, file=f)
            print("", file=f)


def ttestForTimeouts_vs_ideal(datafile, experiment, names, saveFolder):
    """
    loads timeouts from data file

    calculates ttest for one sample for timeouts of second period against 0
    """

    [simID, TIMEOUTS, TIME] = get_trials_data(datafile)

    TIMElevels = np.unique(TIME)
    m = TIME == TIMElevels[1]
    group_list = [TIMEOUTS[m]]

    spacing = 20
    sep = ""
    filename = saveFolder + "/TTEST_" + names[2] + "_" + experiment + ".txt"

    for ideal_idx, ideal in enumerate([0]):
        with open(filename, ["w", "a"][ideal_idx]) as f:

            if ideal_idx == 0:
                print("T-TEST " + names[2] + " vs ideal", file=f)
                stringList = [names[3], "Mean (SD)", "ideal", "t", "df", "p", "d"]

                for string in stringList:
                    print(string.ljust(spacing, " "), end=sep, file=f)
                print("\n".ljust(spacing * len(stringList), "="), file=f)

            ### ttest for sample vs ideal value
            ttestResults = ttest_vs_ideal(group_list[ideal_idx], ideal, 1)
            stringList = [str(TIMElevels[1])] + ttestResults
            for string in stringList:
                print(string.ljust(spacing, " "), end=sep, file=f)
            print("", file=f)


def ttest_vs_ideal(g, val, numTests, alternative="greater"):
    """
    g: 1 sample
    val: H_0 population mean
    ideal=1.5
    unser=2
    --> wir sagen unser ist groesser, somit rejecten wir: H_0 = unser<=ideal --> H_1 = unser>ideal

    if alternative=="greater":
        H_0 = mean(g)<=val --> H_1 = mean(g)>val --> one-sided test
    if alternative=="less":
        H_0 = mean(g)<=val --> H_1 = mean(g)<val --> one-sided test

    test if H_0 can be rejceted with 1 sample ttest
    """
    n = np.sum(np.logical_not(np.isnan(g)))
    df = n - 1
    M = np.nanmean(g)
    SD = np.nanstd(g)
    cohensD = (np.nanmean(g) - val) / np.nanstd(g)  # standard cohens d
    [tVal, pVal] = stats.ttest_1samp(g, val, alternative=alternative, nan_policy="omit")
    ### calculate 95% CI
    t_crit = stats.t.ppf(
        0.975, df
    )  # 1-CI = 5% --> separated in two tails 2.5% in each tail --> either use right tail (0.975) or left tail (0.025), ppf=inverse of CDF
    CI = M + np.array([-1, 1]) * t_crit * SD / np.sqrt(n)
    # strings
    M = str(round(M, 2))
    SD = str(round(SD, 2))
    t = str(round(tVal, 2))
    df = str(int(round(df, 0)))
    p = str(round(pVal, 3))
    CI = str(CI.round(2))
    if pVal < 0.001:
        p = "<0.001"
    if pVal < (0.05 / numTests):
        p = p + "*"
    if pVal < (0.01 / numTests):
        p = p + "*"
    if pVal < (0.001 / numTests):
        p = p + "*"
    d = str(round(cohensD, 2))
    vals = [str(M + " (" + SD + ")"), str(val), t, df, p, d, CI]
    return vals


def ttestForTwoGroups(datafile, experiment, names, saveFolder, comparisons=[[0, 1]]):
    """
    datafile: directory of data file which contains 3 columns (ID, DV, IV) IV has two factors (e.g. two different phases)
    experiment: name of experiment
    names: list of two strings, name of [DV, IV]

    loads datafile with values

    calculates a two sided ttest for dependent samples to compare the groups of both IV levels

    prints the results in a txt file
    """

    [subID, RTs, PHASE] = get_trials_data(datafile)

    PHASElevels = np.unique(PHASE)
    for comparisonIdx, comparison in enumerate(comparisons):
        m1 = PHASE == PHASElevels[comparison[0]]
        m2 = PHASE == PHASElevels[comparison[1]]
        group1 = RTs[m1]
        group2 = RTs[m2]

        spacing = 20
        sep = ""
        filename = saveFolder + "/TTEST_" + names[0] + "_" + experiment + ".txt"
        with open(filename, ["w", "a"][int(comparisonIdx > 0)]) as f:

            if comparisonIdx == 0:
                print("T-TEST " + names[0], file=f)
                stringList = [
                    names[1] + "=" + str(PHASElevels[comparison[0]]),
                    names[1] + "=" + str(PHASElevels[comparison[1]]),
                    "t",
                    "df",
                    "p",
                    "d_z",
                    "95%CI",
                ]
                for string in stringList:
                    print(string.ljust(spacing, " "), end=sep, file=f)
                print("\n".ljust(spacing * len(stringList), "="), file=f)

            ### ttest between the two times
            ttestResults = rel_ttest(group1, group2, len(comparisons))
            for string in ttestResults:
                print(string.ljust(spacing, " "), end=sep, file=f)
            print("", file=f)


def ttest_exp_fit_n_groups(data_list, group_name_list, file_name):
    """
    data_list: list with daa for each group
    group_name_list: list with anme for each group

    calculates a two sided ttest for independent samples to compare the groups

    prints the results in a txt file
    """

    nr_groups = len(data_list)
    combinations_list = list(combinations(list(range(nr_groups)), 2))

    for comparison_idx, combination in enumerate(combinations_list):
        group1 = data_list[combination[0]]
        group2 = data_list[combination[1]]

        ### get and print ttest
        spacing = 20
        sep = ""
        with open(file_name, "a") as f:

            if comparison_idx == 0:
                ### print header
                print("\n\nT-TESTs", file=f)
                stringList = [
                    "group1",
                    "group2",
                    "m1(sd1)",
                    "m2(sd2)",
                    "t",
                    "df",
                    "p",
                    "d_z",
                    "95%CI",
                ]
                for string in stringList:
                    print(string.ljust(spacing, " "), end=sep, file=f)
                print("\n".ljust(spacing * len(stringList), "="), file=f)

            ### ttest between the two groups
            ttestResults = [
                group_name_list[combination[0]],
                group_name_list[combination[1]],
            ] + ind_ttest(group1, group2, len(combinations_list))
            for string in ttestResults:
                print(string.ljust(spacing, " "), end=sep, file=f)
            print("", file=f)


def welch_test_two_ind_groups_two_levels(datafile1, datafile2, names, saveFolder):
    """
    datafile1 and datafile2: directory of data file which contains 3 columns (ID, DV, IV) IV has two factors (e.g. two different phases)

    loads datafile with values

    calculates a two sided ttest (welch test) for independent samples to compare the two groups for each IV level (2 comparisons)

    prints the results in a txt file
    """

    [subID1, ERRORs1, PHASE1] = get_trials_data(datafile1)
    [subID2, ERRORs2, PHASE2] = get_trials_data(datafile2)

    PHASElevels = np.unique(np.concatenate([PHASE1, PHASE2]))
    comparisons = [[0, 0], [1, 1]]
    for comparisonIdx, comparison in enumerate(comparisons):
        m1 = PHASE1 == PHASElevels[comparison[0]]
        m2 = PHASE2 == PHASElevels[comparison[1]]
        group1 = ERRORs1[m1]
        group2 = ERRORs2[m2]

        spacing = 20
        sep = ""
        filename = saveFolder + "/TTEST_" + names[0] + ".txt"
        with open(filename, ["w", "a"][int(comparisonIdx > 0)]) as f:

            if comparisonIdx == 0:
                print("T-TEST " + names[0], file=f)
                stringList = [
                    names[1],
                    names[2],
                    names[3],
                    "t",
                    "df",
                    "p",
                    "d",
                    "95%CI",
                ]
                for string in stringList:
                    print(string.ljust(spacing, " "), end=sep, file=f)
                print("\n".ljust(spacing * len(stringList), "="), file=f)

            ### ttest between the two times
            ttestResults = ind_ttest(group1, group2, len(comparisons))
            stringList = [
                str(PHASElevels[comparison[0]])
                + " & "
                + str(PHASElevels[comparison[1]])
            ] + ttestResults
            for string in stringList:
                print(string.ljust(spacing, " "), end=sep, file=f)
            print("", file=f)


def welch_test_two_ind_groups_one_level(datafile1, datafile2, names, saveFolder):
    """
    datafile1 and datafile2: directory of data file which contains 3 columns (ID, DV, IV) IV has only one level (just a dummy variable)

    loads datafile with values

    calculates a two sided ttest (welch test) for independent samples to compare the two groups for each IV level (only 1 level --> 1 comparison)

    prints the results in a txt file
    """

    [_, DV1, _] = get_trials_data(datafile1)
    [_, DV2, _] = get_trials_data(datafile2)

    group1 = DV1
    group2 = DV2

    spacing = 20
    sep = ""
    filename = saveFolder + "/TTEST_" + names[0] + ".txt"
    with open(filename, "w") as f:

        print("T-TEST " + names[0], file=f)
        stringList = [
            names[1],
            names[2],
            names[3],
            "t",
            "df",
            "p",
            "d",
            "95%CI",
        ]
        for string in stringList:
            print(string.ljust(spacing, " "), end=sep, file=f)
        print("\n".ljust(spacing * len(stringList), "="), file=f)

        ### ttest between the two groups
        ttestResults = ind_ttest(group1, group2, 1)
        stringList = [" "] + ttestResults
        for string in stringList:
            print(string.ljust(spacing, " "), end=sep, file=f)
        print("", file=f)


def CHi2(datafile, saveFolder, name):
    """
    datafile = file with the frequencies for 3 times

    loads datafile and calculates test uniform distribution of frequencies of the 3 times

    prints the results in a txt file
    """

    file = open(datafile, "r")
    time = []
    f1 = []
    f2 = []
    i = 0
    try:
        while 1:
            zeile = file.readline()
            if len(zeile) > 1:
                liste = zeile.split(" ")
                time.append(str(liste[0]))
                f1.append(float(liste[1]))
                f2.append(float(liste[2]))
                i += 1
            else:
                quit()
    except:
        file.close()
    with open(saveFolder + "/Chi2_error_frequencies_results" + name + ".txt", "w") as f:
        print("time", "Chi2", "df", "p", file=f)
        for timeIdx, time in enumerate(time):
            if stats.chisquare([f1[timeIdx], f2[timeIdx]]).pvalue < 0.05 / 3.0:
                append = "*"
            else:
                append = ""
            if stats.chisquare([f1[timeIdx], f2[timeIdx]]).pvalue < 0.01 / 3.0:
                append = "**"
            if stats.chisquare([f1[timeIdx], f2[timeIdx]]).pvalue < 0.001 / 3.0:
                append = "***"

            print(
                time,
                round(stats.chisquare([f1[timeIdx], f2[timeIdx]]).statistic, 2),
                1,
                stats.chisquare([f1[timeIdx], f2[timeIdx]]).pvalue,
                append,
                file=f,
            )


def get_trials_data(datafile):
    file = open(datafile, "r")
    _ = file.readline()
    simID = []
    TRIALS = []
    TIME = []
    i = 0
    try:
        while 1:
            zeile = file.readline()
            liste = zeile.split("\t")
            simID.append(int(liste[0]))
            TRIALS.append(float(liste[1]))
            TIME.append(int(liste[2]))
            i += 1
    except:
        file.close()

    simID = np.array(simID)
    TRIALS = np.array(TRIALS)
    TIME = np.array(TIME)

    return [simID, TRIALS, TIME]


def anova_2between_3within(datafile_1, datafile_2, saveFolder, saveName):
    """
    do a 2x3 anova with factor1 = 2 groups (between factor) and factor2 = 3 different times (within factor)

    datafile: txt file with format: 3 columns [id, dependend var, time]
    """

    timeTransformation = [["0", "start"], ["1", "mid"], ["2", "end"]]

    [ID_1, TRIALS_1, TIME_1] = get_trials_data(datafile_1)
    GROUP_1 = np.array(["G1"] * ID_1.shape[0])

    [ID_2, TRIALS_2, TIME_2] = get_trials_data(datafile_2)
    GROUP_2 = np.array(["G2"] * ID_2.shape[0])

    for trans in timeTransformation:
        TIME_1 = np.where(TIME_1.astype(str) == trans[0], trans[1], TIME_1.astype(str))
        TIME_2 = np.where(TIME_2.astype(str) == trans[0], trans[1], TIME_2.astype(str))

    d = {
        "Trials": np.concatenate((TRIALS_1, TRIALS_2)),
        "TIME": np.concatenate((TIME_1, TIME_2)),
        "GROUP": np.concatenate((GROUP_1, GROUP_2)),
        "IDs": np.concatenate((ID_1, ID_2 + ID_1.max() + 1)),
    }
    df = DataFrame(data=d)

    anovaResults = mixed_anova(
        data=df,
        dv="Trials",
        between="GROUP",
        within="TIME",
        subject="IDs",
        effsize="np2",
    )
    anovaResults = anovaResults.round(3)

    with open(saveFolder + "/" + saveName + ".txt", "w") as f:
        print(anovaResults, file=f)

    ### calculate SS total
    grand_mean = np.mean(d["Trials"])
    SS_total = np.sum((d["Trials"] - grand_mean) ** 2)
    with open(saveFolder + "/" + saveName + ".txt", "a") as f:
        print("SS_total:", round(SS_total, 3), file=f)


def check_coefficients_stats(data_list, group_name_list, save_folder):
    """
    check for each group if mean smaller than zero with ttest

    than compare the groups with ttests

    data_list: list with number-of-groups lists containing the dependent variable values for all subjects of each group
    group_name_list: list with names of each group
    """

    save_file = f"{save_folder}/ttests_coefficients.txt"
    ### check if data of groups smaller 1
    ttest_exp_fit_vs_ideal(data_list, group_name_list, save_file)
    ### make ttests between all groups
    ttest_exp_fit_n_groups(data_list, group_name_list, save_file)


def anova_2between_2within(datafile_1, datafile_2, saveFolder, saveName):
    """
    do a 2x2 anova with factor1 = 2 groups (between factor) and factor2 = 2 different times (within factor)

    datafile: txt file with format: 3 columns [id, dependend var, time]
    """

    timeTransformation = [["0", "initial"], ["1", "reversal"]]

    [ID_1, TRIALS_1, TIME_1] = get_trials_data(datafile_1)
    GROUP_1 = np.array(["G1"] * ID_1.shape[0])

    [ID_2, TRIALS_2, TIME_2] = get_trials_data(datafile_2)
    GROUP_2 = np.array(["G2"] * ID_2.shape[0])

    for trans in timeTransformation:
        TIME_1 = np.where(TIME_1.astype(str) == trans[0], trans[1], TIME_1.astype(str))
        TIME_2 = np.where(TIME_2.astype(str) == trans[0], trans[1], TIME_2.astype(str))

    d = {
        "Trials": np.concatenate((TRIALS_1, TRIALS_2)),
        "TIME": np.concatenate((TIME_1, TIME_2)),
        "GROUP": np.concatenate((GROUP_1, GROUP_2)),
        "IDs": np.concatenate((ID_1, ID_2 + ID_1.max() + 1)),
    }
    df = DataFrame(data=d)

    # df=remove_nan_entries(df, "Trials", "IDs") wird doch nicht gebraucht, subjects mit nan in einer Bedingung werden automatisch ignoriert

    anovaResults = mixed_anova(
        data=df,
        dv="Trials",
        between="GROUP",
        within="TIME",
        subject="IDs",
        effsize="np2",
    )
    anovaResults = anovaResults.round(3)

    with open(saveFolder + "/" + saveName + ".txt", "w") as f:
        print(anovaResults, file=f)

    ### calculate SS total
    grand_mean = np.nanmean(d["Trials"])
    SS_total = np.nansum((d["Trials"] - grand_mean) ** 2)
    with open(saveFolder + "/" + saveName + ".txt", "a") as f:
        print("SS_total:", round(SS_total, 3), file=f)


def remove_nan_entries(df, variable, ID):
    """
    variable: string; name of column which may contain nan values
    ID: string; identifier of entries, if one entry with identifier contains nan all entries of this identifier are removed (each subject has nan value in one time --> remove all times of this participant)
    """
    nan_positions = np.where(np.isnan(df[variable]))[0]
    searched_id_list = [df.iloc[idx, :][ID] for idx in nan_positions]
    entries_with_searched_ids = (
        np.sum(
            np.array(
                [
                    np.array(df[ID] == searched_id).astype(int)
                    for searched_id in searched_id_list
                ]
            ),
            axis=0,
        )
        > 0
    )
    indizes_of_searched_ids = np.where(entries_with_searched_ids)[0]
    ret = df.drop(indizes_of_searched_ids)
    return ret


def anova_1between_3within(datafile_1, saveFolder, saveName):
    """
    do a 1x3 anova with factor1 = group (between factor) and factor2 = 3 different times (within factor)

    datafile: txt file with format: 3 columns [id, dependend var, time]
    """

    timeTransformation = [["0", "start"], ["1", "mid"], ["2", "end"]]

    [ID_1, TRIALS_1, TIME_1] = get_trials_data(datafile_1)
    # GROUP_1 = np.array(["G1"] * ID_1.shape[0])

    for trans in timeTransformation:
        TIME_1 = np.where(TIME_1.astype(str) == trans[0], trans[1], TIME_1.astype(str))

    d = {"Trials": TRIALS_1, "TIME": TIME_1, "IDs": ID_1}
    df = DataFrame(data=d)

    anovaResults = rm_anova(
        data=df, dv="Trials", within="TIME", subject="IDs", effsize="np2", detailed=True
    )
    anovaResults = anovaResults.round(3)

    with open(saveFolder + "/" + saveName + ".txt", "w") as f:
        print(anovaResults, file=f)

    ### TODO perform a post hoc t-test for dependend samples for start time vs last time
    ttestForTwoGroups(
        datafile_1,
        "post_hoc_" + saveName,
        ["Trials", "Time"],
        saveFolder,
        comparisons=[[0, 1], [0, 2], [1, 2]],
    )


saveFolder = "../3_results"


def regression_interaction_cluster(file_1, file_2, save_folder):
    """
    loads two files which have to be combined to a single dataframe

    performs multiple poisson regression with the combined data frame
    """

    dict_1 = np.load(file_1, allow_pickle=True).item()
    dict_2 = np.load(file_2, allow_pickle=True).item()

    dict_combined = {
        key: np.concatenate([dict_1[key], dict_2[key]]) for key in dict_1.keys()
    }

    df = pd.DataFrame(dict_combined)

    ### poisson regression with dummy variables vps and fixed and interaction terms vps*time and fixed*time
    ### interactions are relative to the baseline group which is learning model (both vps and fixed == 0)
    ### significant interactions --> significant different coefficient in rference and comparison group
    results_all = make_multi_poisson_regression(
        df=df,
        dv="errors",
        iv_list=["time", "vps", "fixed", "vps*time", "fixed*time"],
        save_folder=save_folder,
    )
    ### test predictions
    df_pred = pd.DataFrame(
        {
            "errors": results_all["mean"],
            "fixed": df["fixed"],
            "vps": df["vps"],
            "time": df["time"],
        }
    )
    plot_errors(
        df,
        "fixed",
        f"{save_folder}/all",
        "orig",
    )
    plot_errors(
        df,
        "vps",
        f"{save_folder}/all",
        "orig",
    )
    plot_errors(
        df,
        ["fixed", "vps"],
        f"{save_folder}/all",
        "orig",
    )
    plot_errors(
        df_pred,
        "fixed",
        f"{save_folder}/all",
        "pred",
    )
    plot_errors(
        df_pred,
        "vps",
        f"{save_folder}/all",
        "pred",
    )
    plot_errors(
        df_pred,
        ["fixed", "vps"],
        f"{save_folder}/all",
        "pred",
    )

    ### poisson regression with dummy variable fixed and interaction term fixed*time
    ### interaction is relative to the baseline group which is learning model (fixed == 0)
    ### siginificant interaction --> different effect of time onto errors in both groups
    df_plastic_vs_fixed = df.loc[
        df["vps"] == 0, ["errors", "fixed", "time", "fixed*time"]
    ]
    resutls_plastic_vs_fixed = make_multi_poisson_regression(
        df=df_plastic_vs_fixed,
        dv="errors",
        iv_list=["time", "fixed", "fixed*time"],
        save_folder=save_folder,
    )
    ### test predictions
    df_pred = pd.DataFrame(
        {
            "errors": resutls_plastic_vs_fixed["mean"],
            "fixed": df_plastic_vs_fixed["fixed"],
            "time": df_plastic_vs_fixed["time"],
        }
    )
    plot_errors(
        df_plastic_vs_fixed,
        "fixed",
        f"{save_folder}/plastic_vs_fixed",
        "orig",
    )
    plot_errors(
        df_plastic_vs_fixed,
        ["fixed"],
        f"{save_folder}/plastic_vs_fixed",
        "orig",
    )
    plot_errors(
        df_pred,
        "fixed",
        f"{save_folder}/plastic_vs_fixed",
        "pred",
    )
    plot_errors(
        df_pred,
        ["fixed"],
        f"{save_folder}/plastic_vs_fixed",
        "pred",
    )

    ### poisson regression with dummy variable vps and interaction term vps*time
    ### interaction is relative to the baseline group which is learning model (vps == 0)
    ### siginificant interaction --> different effect of time onto errors in both groups
    df_plastic_vs_vps = df.loc[df["fixed"] == 0, ["errors", "vps", "time", "vps*time"]]
    resutls_plastic_vs_vps = make_multi_poisson_regression(
        df=df_plastic_vs_vps,
        dv="errors",
        iv_list=["time", "vps", "vps*time"],
        save_folder=save_folder,
    )
    ### test predictions
    df_pred = pd.DataFrame(
        {
            "errors": resutls_plastic_vs_vps["mean"],
            "vps": df_plastic_vs_vps["vps"],
            "time": df_plastic_vs_vps["time"],
        }
    )
    plot_errors(
        df_plastic_vs_vps,
        "vps",
        f"{save_folder}/plastic_vs_vps",
        "orig",
    )
    plot_errors(
        df_plastic_vs_vps,
        ["vps"],
        f"{save_folder}/plastic_vs_vps",
        "orig",
    )
    plot_errors(
        df_pred,
        "vps",
        f"{save_folder}/plastic_vs_vps",
        "pred",
    )
    plot_errors(
        df_pred,
        ["vps"],
        f"{save_folder}/plastic_vs_vps",
        "pred",
    )

    ### poisson regression only for fixed
    df_fixed = df.loc[df["fixed"] == 1, ["fixed", "errors", "time"]]
    resutls_fixed = make_multi_poisson_regression(
        df=df_fixed,
        dv="errors",
        iv_list=["time"],
        save_folder=save_folder,
        append="_for_fixed",
    )
    ### test predictions
    df_pred = pd.DataFrame(
        {
            "errors": resutls_fixed["mean"],
            "time": df_fixed["time"],
            "fixed": df_fixed["fixed"],
        }
    )
    plot_errors(
        df_fixed,
        "fixed",
        f"{save_folder}/fixed",
        "orig",
    )
    plot_errors(
        df_pred,
        "fixed",
        f"{save_folder}/fixed",
        "pred",
    )

    ### poisson regression only for vps
    df_vps = df.loc[df["vps"] == 1, ["vps", "errors", "time"]]
    resutls_vps = make_multi_poisson_regression(
        df=df_vps,
        dv="errors",
        iv_list=["time"],
        save_folder=save_folder,
        append="_for_vps",
    )
    ### test predictions
    df_pred = pd.DataFrame(
        {
            "errors": resutls_vps["mean"],
            "time": df_vps["time"],
            "vps": df_vps["vps"],
        }
    )
    plot_errors(
        df_vps,
        "vps",
        f"{save_folder}/vps",
        "orig",
    )
    plot_errors(
        df_pred,
        "vps",
        f"{save_folder}/vps",
        "pred",
    )

    ### poisson regression only for plastic
    df_plastic = df.loc[
        ((df["fixed"] == 0).astype(int) * (df["vps"] == 0).astype(int)).astype(bool),
        ["fixed", "vps", "errors", "time"],
    ]
    resutls_plastic = make_multi_poisson_regression(
        df=df_plastic,
        dv="errors",
        iv_list=["time"],
        save_folder=save_folder,
        append="_for_plastic",
    )
    ### test predictions
    df_pred = pd.DataFrame(
        {
            "errors": resutls_plastic["mean"],
            "time": df_plastic["time"],
            "fixed": df_plastic["fixed"],
            "vps": df_plastic["vps"],
        }
    )
    plot_errors(
        df_plastic,
        ["fixed", "vps"],
        f"{save_folder}/plastic",
        "orig",
    )
    plot_errors(
        df_pred,
        ["fixed", "vps"],
        f"{save_folder}/plastic",
        "pred",
    )

    ### poisson regression with fixed as predictor using complete time TODO
    df_fixed = df.loc[df["fixed"] == 1, ["fixed", "errors", "time"]]
    resutls_fixed = make_multi_poisson_regression(
        df=df_fixed,
        dv="errors",
        iv_list=["time"],
        save_folder=save_folder,
        append="_for_fixed",
    )
    ### test predictions
    df_pred = pd.DataFrame(
        {
            "errors": resutls_fixed["mean"],
            "time": df_fixed["time"],
            "fixed": df_fixed["fixed"],
        }
    )
    plot_errors(
        df_fixed,
        "fixed",
        f"{save_folder}/fixed",
        "orig",
    )
    plot_errors(
        df_pred,
        "fixed",
        f"{save_folder}/fixed",
        "pred",
    )


def exp_fit_stats(file_1, file_2, save_folder, experiment):
    """
    loads two files which have to be combined to a single dataframe

    dataframe columns: dummy variables for groups, time (IV), interaction-terms for groups, errors (DV)
    for each group: calculate for all subjects exponential fits for errors(time)
    base group = if all other groups are zero

    after obtaining exp fits for each group --> ANOVA between groups
    ANOVA: 1 between factor with n levels, n = number of groups
    and post-hoc t-test between all pairs: independent, welch test for unequal group sizes
    """
    print("\n\n" + save_folder.split("exp_fit_")[-1])
    create_dir(save_folder)
    dict_1 = np.load(file_1, allow_pickle=True).item()
    dict_2 = np.load(file_2, allow_pickle=True).item()

    dict_combined = {
        key: np.concatenate([dict_1[key], dict_2[key]]) for key in dict_1.keys()
    }

    df = pd.DataFrame(dict_combined)

    ### normalize time over all groups for exponential fit
    df["time"] = df["time"] / df["time"].max()

    group_list = ["base"] + list(df.columns[0 : df.columns.get_loc("time")])
    group_size_dict = {"base": 60, "fixed": 60, "vps": 10}

    plt.figure()  # fig to check data of groups
    coeff = []
    used_group_name_list = []
    for group_idx, group_name in enumerate(group_list):
        ### get data of specific group
        if group_name == "base":
            mask = np.array([(df[tmp] == 0).astype(int) for tmp in group_list[1:]])
            mask = np.prod(mask, 0).astype(bool)
        else:
            mask = df[group_name] == 1
            if sum(mask) == 0:
                ### skip if there is no data for group
                continue
        print(group_name)
        used_group_name_list.append(group_name)
        df_group = df.loc[mask, ["time", "errors"]]
        df_group_arr = df_group.to_numpy()
        ### get group size
        group_size = group_size_dict[group_name]
        print(group_size)
        ### reshape 2D data array
        ### currently (time_size * group_size,2)
        ### new: idx 0 = time, idx 1 = subject, idx 2 = time or errors
        initial_shape = df_group_arr.shape[0]
        df_group_arr = df_group_arr.reshape(
            (initial_shape // group_size, group_size, 2)
        )

        ### create plot to check data
        plt.subplot(1, len(group_list), group_idx + 1)
        plt.title(group_name)
        plt.xlabel("time")
        plt.ylabel("errors")
        for sub_id in range(group_size):
            plt.plot(
                df_group_arr[:, sub_id, 0], df_group_arr[:, sub_id, 1] + sub_id * 4
            )

        ### fit exponential curves to all subs
        coeff.append([])
        for sub_id in range(group_size):
            coeff[group_idx].append(
                fit_exp(
                    x=df_group_arr[:, sub_id, 0],
                    y=df_group_arr[:, sub_id, 1],
                    exp=experiment,
                    group=group_name,
                    sub_id=sub_id,
                    save_folder=save_folder,
                    plot=True,
                )
            )

    ### save the check-data fig
    plt.tight_layout()
    plt.savefig(f"{save_folder}/check_data.png")
    plt.close()

    ### figure for cooeficients of groups
    plt.figure()
    ax = plt.subplot(111)
    ax.boxplot(coeff)
    # plt.ylim(0, 15)
    plt.xticks(np.arange(len(coeff)) + 1, used_group_name_list)
    plt.ylabel("coefficients")
    plt.xlabel("group")
    plt.tight_layout()
    plt.savefig(f"{save_folder}/coefficients.png")
    plt.close()

    ### do ANOVA and post-hoc t-tests
    check_coefficients_stats(coeff, used_group_name_list, save_folder)


def fit_exp(x, y, exp, group, sub_id, save_folder, plot=False):
    """
    x and y: arrays with same length
    exp: which experiment --> specifies the exponential curve limits

    ret: coefficient of exponential fit
    """

    if exp == "001e" or exp == "014a":
        start = 1.5
        end = 0.5
    if exp == "001f" or exp == "014b":
        start = 1.5
        end = 0.857

    ### remove value pairs with nan values
    not_nan_mask = (np.isnan(x).astype(int) + np.isnan(y).astype(int)) == 0
    x = x[not_nan_mask]
    y = y[not_nan_mask]

    ### fitted curve is exp decay or increase (= decay to symmetrical upper border)
    ### --> coefficients are positive or negative
    ### in the end return transform half-life time clipped at 1
    popt, _ = curve_fit(
        f=lambda x, tau: exp_func(x, start, end, tau),
        xdata=x,
        ydata=y,
        p0=[0],
        bounds=(-1000, 1000),
    )

    ### transform the time constant to half-life time
    ret = np.clip(np.log(2) / popt[0], -1, 1)
    ### than change scale of half-life time so that 1 and -1 is at the middle of the scale and -0 and +0 left and right
    if ret < 0:
        ret = -1 - ret
    else:
        ret = 1 - ret

    if plot:
        plt.figure(figsize=(2.3, 1.7))
        plt.title(
            f"{save_folder.split('exp_fit_')[-1]} {group} {sub_id}\n{popt[0]} {ret}"
        )
        plt.plot(x, y, "k.")
        plt.axhline(start, color="k")
        plt.axhline(end, color="k")
        plt.plot(x, exp_func(x, start, end, popt[0]), color="r")
        plt.savefig(f"{save_folder}/fit_{group}_{sub_id}.svg")
        plt.close()

    return ret


def exp_func(x, start, end, tau):
    """
    start: value at x==0
    end: value at x==inf
    tau: time constant
    """
    if tau > 0:
        b = end
    else:
        b = 2 * start - end
        tau = -tau
    a = start - b
    return a * np.exp(-tau * x) + b


def regression_interaction_rare(file_1, file_2, save_folder):
    """
    loads two files which have to be combined to a single dataframe

    performs multiple poisson regression with the combined data frame
    """

    dict_1 = np.load(file_1, allow_pickle=True).item()
    dict_2 = np.load(file_2, allow_pickle=True).item()

    dict_combined = {
        key: np.concatenate([dict_1[key], dict_2[key]]) for key in dict_1.keys()
    }

    df = pd.DataFrame(dict_combined)

    ### poisson regression with dummy variable vps and interaction term vps*time
    ### interaction is relative to the baseline group which is learning model (vps == 0)
    ### siginificant interaction --> different effect of time onto errors in both groups
    df_plastic_vs_vps = df.loc[df["fixed"] == 0, ["errors", "vps", "time", "vps*time"]]
    resutls_plastic_vs_vps = make_multi_poisson_regression(
        df=df_plastic_vs_vps,
        dv="errors",
        iv_list=["time", "vps", "vps*time"],
        save_folder=save_folder,
    )
    ### test predictions
    df_pred = pd.DataFrame(
        {
            "errors": resutls_plastic_vs_vps["mean"],
            "vps": df_plastic_vs_vps["vps"],
            "time": df_plastic_vs_vps["time"],
        }
    )
    plot_errors(
        df_plastic_vs_vps,
        "vps",
        f"{save_folder}/plastic_vs_vps",
        "orig",
    )
    plot_errors(
        df_plastic_vs_vps,
        ["vps"],
        f"{save_folder}/plastic_vs_vps",
        "orig",
    )
    plot_errors(
        df_pred,
        "vps",
        f"{save_folder}/plastic_vs_vps",
        "pred",
    )
    plot_errors(
        df_pred,
        ["vps"],
        f"{save_folder}/plastic_vs_vps",
        "pred",
    )

    ### poisson regression only for vps
    df_vps = df.loc[df["vps"] == 1, ["vps", "errors", "time"]]
    resutls_vps = make_multi_poisson_regression(
        df=df_vps,
        dv="errors",
        iv_list=["time"],
        save_folder=save_folder,
        append="_for_vps",
    )
    ### test predictions
    df_pred = pd.DataFrame(
        {
            "errors": resutls_vps["mean"],
            "time": df_vps["time"],
            "vps": df_vps["vps"],
        }
    )
    plot_errors(
        df_vps,
        "vps",
        f"{save_folder}/vps",
        "orig",
    )
    plot_errors(
        df_pred,
        "vps",
        f"{save_folder}/vps",
        "pred",
    )

    ### poisson regression only for plastic
    df_plastic = df.loc[
        ((df["fixed"] == 0).astype(int) * (df["vps"] == 0).astype(int)).astype(bool),
        ["fixed", "vps", "errors", "time"],
    ]
    resutls_plastic = make_multi_poisson_regression(
        df=df_plastic,
        dv="errors",
        iv_list=["time"],
        save_folder=save_folder,
        append="_for_plastic",
    )
    ### test predictions
    df_pred = pd.DataFrame(
        {
            "errors": resutls_plastic["mean"],
            "time": df_plastic["time"],
            "fixed": df_plastic["fixed"],
            "vps": df_plastic["vps"],
        }
    )
    plot_errors(
        df_plastic,
        ["fixed", "vps"],
        f"{save_folder}/plastic",
        "orig",
    )
    plot_errors(
        df_pred,
        ["fixed", "vps"],
        f"{save_folder}/plastic",
        "pred",
    )


def make_multi_poisson_regression(df, dv, iv_list, save_folder, append=""):
    """
    performs a multiple poisson regression

    Args:
        df: pandas dataframe
            a data frame containing the values for the dependend variable and independent variables (predictors)
        dv: string
            the name of the dependent variable
        iv_list: list of strings
            the names of the independent variables
        save_folder: string
            path to the save folder
    """
    create_dir(save_folder)

    ### create a patsy notation
    expr = f"{dv} ~ {' + '.join(iv_list)}"

    ### setup training data
    y_train, X_train = dmatrices(expr, df, return_type="dataframe")

    ### Using the statsmodels GLM class,
    ### train the Poisson regression model on the training data set
    poisson_training_results = sm.GLM(
        y_train, X_train, family=sm.families.Poisson()
    ).fit()

    ### print summary
    with open(f"{save_folder}/regression_{dv}_{'_'.join(iv_list)}{append}", "w") as f:
        print(poisson_training_results.summary(), file=f)

    ### return predictions
    poisson_predictions = poisson_training_results.get_prediction(X_train)
    predictions_summary_frame = poisson_predictions.summary_frame()

    return predictions_summary_frame


def plot_errors(df, name, save_folder, append):
    """
    gets a dataframe from regression and a independent dummy varaible name
    (defining a group, e.g. vps) and plots the errors like in the manuscript_SRtask analysis

    name can also be a list of all group names which should not be plotted
    """
    create_dir(save_folder)
    if isinstance(name, str):
        ### take the values there name == 1
        data_mask = df[name] == 1
    elif isinstance(name, list):
        ### take the values there none of the names is 1
        mask = np.zeros((len(name), df.shape[0])).astype(int)
        for name_idx, name_val in enumerate(name):
            mask[name_idx] = df[name_val] == 1
        data_mask = (np.sum(mask, 0) == 0).astype(bool)

    df = df.loc[data_mask, ["time", "errors"]]
    time_arr = np.zeros(np.unique(df["time"]).size)
    error_arr = np.zeros((2, np.unique(df["time"]).size))
    time_unique = np.sort(np.unique(df["time"]))

    tol = np.mean(np.diff(time_unique)) / 2

    for idx_time, time in enumerate(time_unique):
        mask_time = (
            (df["time"] > time - tol).astype(int)
            * (df["time"] < time + tol).astype(int)
        ).astype(bool)
        time_arr[idx_time] = time
        error_arr[0, idx_time] = np.nanmean(df["errors"][mask_time])
        error_arr[1, idx_time] = stats.sem(
            df["errors"][mask_time], axis=0, ddof=1, nan_policy="omit"
        )

    plt.figure()
    plt.errorbar(time_arr, error_arr[0], yerr=error_arr[1], fmt=".", color="black")
    plt.xlabel("time")
    plt.ylabel("errors")
    plt.ylim(-0.5, 3.7)
    if isinstance(name, str):
        plt.savefig(f"{save_folder}/errors_{append}_{name}.png", dpi=300)
    elif isinstance(name, list):
        plt.savefig(f"{save_folder}/errors_{append}_not_{'_'.join(name)}.png", dpi=300)


##############################################################################################################################################################
####################################################################   SIMULATIONS   #########################################################################
##############################################################################################################################################################
sim_chi2 = True
sim_anova = False
sim_ttest = False

if sim_chi2:

    def chi2_simulations(rep):
        ### classic model/dopa
        ###############################  CLUSTER LEARNING ON  ##############################
        datafile = f"../../manuscript_SRtask_results/2_dataEv/stn_gpe_factor_3/frequencies_for_CHI2_simulation_on_001e_{rep}.txt"
        CHi2(datafile, saveFolder, f"_classic_CLUSTER_SIMULATIONS_on_{rep}")

        ##############################  CLUSTER LEARNING OFF  ##############################
        datafile = f"../../manuscript_SRtask_results/2_dataEv/stn_gpe_factor_3/frequencies_for_CHI2_simulation_off_001e_{rep}.txt"
        CHi2(datafile, saveFolder, f"_classic_CLUSTER_SIMULATIONS_off_{rep}")

        ###############################  RARE LEARNING ON  ##############################
        datafile = f"../../manuscript_SRtask_results/2_dataEv/stn_gpe_factor_3/frequencies_for_CHI2_simulation_on_001f_{rep}.txt"
        CHi2(datafile, saveFolder, f"_classic_RARE_SIMULATIONS_on_{rep}")

        ### new model/dopa
        ###############################  CLUSTER LEARNING ON  ##############################
        datafile = f"../../manuscript_SRtask_results/2_dataEv/stn_gpe_factor_3/frequencies_for_CHI2_simulation_on_014a_{rep}.txt"
        CHi2(datafile, saveFolder, f"_new_CLUSTER_SIMULATIONS_on_{rep}")

        ##############################  CLUSTER LEARNING OFF  ##############################
        datafile = f"../../manuscript_SRtask_results/2_dataEv/stn_gpe_factor_3/frequencies_for_CHI2_simulation_off_014a_{rep}.txt"
        CHi2(datafile, saveFolder, f"_new_CLUSTER_SIMULATIONS_off_{rep}")

        ###############################  RARE LEARNING ON  ##############################
        datafile = f"../../manuscript_SRtask_results/2_dataEv/stn_gpe_factor_3/frequencies_for_CHI2_simulation_on_014b_{rep}.txt"
        CHi2(datafile, saveFolder, f"_new_RARE_SIMULATIONS_on_{rep}")

        ###############################  ALL LEARNING ON  ##############################
        datafile = f"../../manuscript_SRtask_results/2_dataEv/stn_gpe_factor_3/frequencies_for_CHI2_simulation_on_014c_{rep}.txt"
        CHi2(datafile, saveFolder, f"_new_ALL_SIMULATIONS_on_{rep}")

        return datafile

    # datafile = chi2_simulations("with_rep")
    datafile = chi2_simulations("without_rep")

#################################  TRIALS  ####################################
if sim_anova:
    ##########################  MIXED ANOVAS FOR CLUSTER EXPERIMENT ########################
    ##########################  MIXED ANOVA ON VS OFF WITH REP #############################
    datafile_simulations_on = "../../manuscript_SRtask_results/2_dataEv/TRIALs_for_TTEST_simulation_on_001e_with_rep.txt"
    datafile_simulations_off = "../../manuscript_SRtask_results/2_dataEv/TRIALs_for_TTEST_simulation_off_001e_with_rep.txt"
    anova_2between_3within(
        datafile_simulations_on,
        datafile_simulations_off,
        saveFolder,
        "anova_trials_cluster_on_vs_off_with_rep",
    )

    ##########################  MIXED ANOVA ON VS OFF WITHOUT REP #############################
    datafile_simulations_on = "../../manuscript_SRtask_results/2_dataEv/TRIALs_for_TTEST_simulation_on_001e_without_rep.txt"
    datafile_simulations_off = "../../manuscript_SRtask_results/2_dataEv/TRIALs_for_TTEST_simulation_off_001e_without_rep.txt"
    anova_2between_3within(
        datafile_simulations_on,
        datafile_simulations_off,
        saveFolder,
        "anova_trials_cluster_on_vs_off_without_rep",
    )

if sim_ttest:
    ################################  TTEST ON VS IDEAL  WITH REP  ##########################
    datafile_simulations_on = "../../manuscript_SRtask_results/2_dataEv/TRIALs_for_TTEST_simulation_on_001e_with_rep.txt"
    ttestForTrials_vs_ideal(
        datafile_simulations_on,
        "CLUSTER_SIMULATIONS_VS_IDEAL_with_rep",
        ["Sim", "simID", "TRIALS", "TIME"],
        saveFolder,
    )
    ################################  TTEST ON VS IDEAL  WITHOUT REP  ##########################
    datafile_simulations_on = "../../manuscript_SRtask_results/2_dataEv/TRIALs_for_TTEST_simulation_on_001e_without_rep.txt"
    ttestForTrials_vs_ideal(
        datafile_simulations_on,
        "CLUSTER_SIMULATIONS_VS_IDEAL_without_rep",
        ["Sim", "simID", "TRIALS", "TIME"],
        saveFolder,
    )


##############################################################################################################################################################
####################################################################   EXPERIMENTS   #########################################################################
##############################################################################################################################################################
exp_chi2 = True
exp_anova = False
exp_regression = False
exp_exp_fit = True
exp_ttest = True

if exp_chi2:

    def chi2_experiments(rep):
        for experiment_names in [["001e", "CLUSTER"], ["014b", "RARE"]]:
            ###############################  FREQUENCIES CLUSTER  #################################
            datafile = f"../../manuscript_SRtask_results/2_dataEv/stn_gpe_factor_3/frequencies_for_CHI2_eyetracking_{experiment_names[0]}_{rep}.txt"
            CHi2(datafile, saveFolder, f"_vps_{experiment_names[1]}_EXPERIMENTS_{rep}")
        return datafile

    datafile = chi2_experiments("without_rep")

###############################  TRIALs  ######################################
if exp_anova:
    ######  MIXED ANOVA EXP VS SIM EXPLORATION ERRORS START VS MID VS END  WITH REP  ########
    datafile_simulations_on = "../../manuscript_SRtask_results/2_dataEv/TRIALs_for_TTEST_simulation_on_001e_with_rep.txt"
    datafile_participants = "../../manuscript_SRtask_results/2_dataEv/TRIALs_for_TTEST_eyetracking_001e_with_rep.txt"
    anova_2between_3within(
        datafile_simulations_on,
        datafile_participants,
        saveFolder,
        "anova_trials_cluster_sim_vs_exp_with_rep",
    )

    ######  MIXED ANOVA EXP VS SIM EXPLORATION ERRORS START VS MID VS END  WITHOUT REP  ########
    datafile_simulations_on = "../../manuscript_SRtask_results/2_dataEv/TRIALs_for_TTEST_simulation_on_001e_without_rep.txt"
    datafile_participants = "../../manuscript_SRtask_results/2_dataEv/TRIALs_for_TTEST_eyetracking_001e_without_rep.txt"
    anova_2between_3within(
        datafile_simulations_on,
        datafile_participants,
        saveFolder,
        "anova_trials_cluster_sim_vs_exp_without_rep",
    )

if exp_regression:
    ######  REGRESSION CLUSTER PLASTIC-MODEL / FIXED-MODEL / VPS / TIME --> ERRORS  #######
    for regression_time_mode in [
        "relative_exploration",
        "absolute_exploration",
        "absolute_trial",
    ]:
        ### without rep
        datafile_regr_sims = f"../../manuscript_SRtask_results/2_dataEv/TRIALs_for_regression_interaction_simulation_001e_without_rep_{regression_time_mode}.npy"
        datafile_regr_vps = f"../../manuscript_SRtask_results/2_dataEv/TRIALs_for_regression_interaction_vps_001e_without_rep_{regression_time_mode}.npy"
        regression_interaction_cluster(
            datafile_regr_sims,
            datafile_regr_vps,
            f"../3_results/regression_interaction_cluster_without_rep_{regression_time_mode}",
        )

        ### with rep
        datafile_regr_sims = f"../../manuscript_SRtask_results/2_dataEv/TRIALs_for_regression_interaction_simulation_001e_with_rep_{regression_time_mode}.npy"
        datafile_regr_vps = f"../../manuscript_SRtask_results/2_dataEv/TRIALs_for_regression_interaction_vps_001e_with_rep_{regression_time_mode}.npy"
        regression_interaction_cluster(
            datafile_regr_sims,
            datafile_regr_vps,
            f"../3_results/regression_interaction_cluster_with_rep_{regression_time_mode}",
        )

if exp_exp_fit:
    ###############################   EXPONENTIAL FITS   ######################################
    for exp in ["001e", "001f", "014a", "014b"]:
        for regression_time_mode in [
            "absolute_exploration",
        ]:
            for rep in ["without_rep"]:

                datafile_exp_fit_sims = f"../../manuscript_SRtask_results/2_dataEv/stn_gpe_factor_3/TRIALs_for_regression_interaction_simulation_{exp}_{rep}_{regression_time_mode}.npy"
                datafile_exp_fit_vps = f"../../manuscript_SRtask_results/2_dataEv/stn_gpe_factor_3/TRIALs_for_regression_interaction_vps_{exp}_{rep}_{regression_time_mode}.npy"
                exp_fit_stats(
                    datafile_exp_fit_sims,
                    datafile_exp_fit_vps,
                    f"../3_results/exp_fit_{exp}_{rep}_{regression_time_mode}",
                    experiment=exp,
                )

if exp_ttest:
    # ###############################  TTEST EXP VS IDEAL WITH REP  #############################
    # datafile_participants = "../../manuscript_SRtask_results/2_dataEv/TRIALs_for_TTEST_eyetracking_001e_with_rep.txt"
    # ttestForTrials_vs_ideal(
    #     datafile_participants,
    #     "CLUSTER_EXPERIMENTS_VS_IDEAL_with_rep",
    #     ["?", "?", "TRIALS", "TIME"],
    #     saveFolder,
    # )
    # ###############################  TTEST EXP VS IDEAL WITHOUT REP  #############################
    # datafile_participants = "../../manuscript_SRtask_results/2_dataEv/TRIALs_for_TTEST_eyetracking_001e_without_rep.txt"
    # ttestForTrials_vs_ideal(
    #     datafile_participants,
    #     "CLUSTER_EXPERIMENTS_VS_IDEAL_without_rep",
    #     ["?", "?", "TRIALS", "TIME"],
    #     saveFolder,
    # )

    #########################  GLOBAL RESPONSE TIMES TTEST  ##############################
    datafile_global_rts_participants_cluster = "../../manuscript_global_performance_vps/2_dataEv/response_times_per_vp_cluster.txt"
    ttestForTwoGroups(
        datafile_global_rts_participants_cluster,
        "CLUSTER_EXPERIMENTS",
        ["RTs", "Phase"],
        saveFolder,
    )
    datafile_global_rts_participants_rare = "../../manuscript_global_performance_vps/2_dataEv/response_times_per_vp_rare.txt"
    ttestForTwoGroups(
        datafile_global_rts_participants_rare,
        "RARE_EXPERIMENTS",
        ["RTs", "Phase"],
        saveFolder,
    )

    ### new dopa model
    ##########################  GLOBAL ERRORS TTEST NEVER-EXPERIMENT ##############################
    datafile_global_errors_never_participants = "../../manuscript_global_performance_vps/2_dataEv/number_of_errors_vps_cluster.txt"
    datafile_global_errors_never_sims = (
        "../../manuscript_global_performance/2_dataEv/number_of_errors_sims_014a.txt"
    )
    welch_test_two_ind_groups_two_levels(
        datafile_global_errors_never_sims,
        datafile_global_errors_never_participants,
        ["global_errors_014a", "Phase", "Simulations", "Participants"],
        saveFolder,
    )
    ##########################  GLOBAL ERRORS TTEST RARE-EXPERIMENT ##############################
    datafile_global_errors_rare_participants = (
        "../../manuscript_global_performance_vps/2_dataEv/number_of_errors_vps_rare.txt"
    )
    datafile_global_errors_rare_sims = (
        "../../manuscript_global_performance/2_dataEv/number_of_errors_sims_014b.txt"
    )
    welch_test_two_ind_groups_two_levels(
        datafile_global_errors_rare_sims,
        datafile_global_errors_rare_participants,
        ["global_errors_014b", "Phase", "Simulations", "Participants"],
        saveFolder,
    )

    ### original dopa model
    ##########################  GLOBAL ERRORS TTEST NEVER-EXPERIMENT ##############################
    datafile_global_errors_never_participants = "../../manuscript_global_performance_vps/2_dataEv/number_of_errors_vps_cluster.txt"
    datafile_global_errors_never_sims = (
        "../../manuscript_global_performance/2_dataEv/number_of_errors_sims_001e.txt"
    )
    welch_test_two_ind_groups_two_levels(
        datafile_global_errors_never_sims,
        datafile_global_errors_never_participants,
        ["global_errors_001e", "Phase", "Simulations", "Participants"],
        saveFolder,
    )
    ##########################  GLOBAL ERRORS TTEST RARE-EXPERIMENT ##############################
    datafile_global_errors_rare_participants = (
        "../../manuscript_global_performance_vps/2_dataEv/number_of_errors_vps_rare.txt"
    )
    datafile_global_errors_rare_sims = (
        "../../manuscript_global_performance/2_dataEv/number_of_errors_sims_001f.txt"
    )
    welch_test_two_ind_groups_two_levels(
        datafile_global_errors_rare_sims,
        datafile_global_errors_rare_participants,
        ["global_errors_001f", "Phase", "Simulations", "Participants"],
        saveFolder,
    )

    #################### PUPIL SIZES TTESTs FOR RARE EXP #############################

    # ### tests for individual pupils sizes for each vp
    # vp_anz = 10
    # for vp in range(vp_anz):
    #     datafile_pupil_in = (
    #         f"../../manuscript_pupil_size/2_dataEv/pupil_size_individual_in_{vp}.txt"
    #     )
    #     datafile_pupil_out = (
    #         f"../../manuscript_pupil_size/2_dataEv/pupil_size_individual_out_{vp}.txt"
    #     )
    #     welch_test_two_ind_groups_one_level(
    #         datafile_pupil_in,
    #         datafile_pupil_out,
    #         [f"pupil_size_{vp}", "_", "in cluster", "out cluster"],
    #         saveFolder,
    #     )

    # ### test for average pupil size, dependent (two values for each vp)
    # datafile_pupil_avg = "../../manuscript_pupil_size/2_dataEv/pupil_size_avg.txt"
    # ttestForTwoGroups(
    #     datafile_pupil_avg,
    #     "RARE_EXPERIMENTS",
    #     ["PUPIL", "RESPONSE_TYPE"],
    #     saveFolder,
    # )

print("\n")
