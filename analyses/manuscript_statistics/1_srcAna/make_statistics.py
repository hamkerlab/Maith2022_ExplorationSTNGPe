#from __future__ import print_function
import numpy as np
#import pyvttbl as pt
#from collections import namedtuple
import pylab as plt
from scipy import stats
from pingouin import mixed_anova, rm_anova
from pandas import DataFrame
import pandas as pd
from CompNeuroPy import print_df


def getData(datafile):
    """
    loads txt file with the IDs, dependent and independent variables / values
    
    returns for each of them a numpy array
    """
    file = open(datafile, 'r')
    header = file.readline()
    simID=[]
    RT=[]
    TIME=[]
    PHASE=[]
    i=0
    try:
        while 1:
            zeile = file.readline()
            liste = zeile.split('\t')
            simID.append(int(liste[0]))
            RT.append(float(liste[1]))
            TIME.append(int(liste[2]))
            PHASE.append(int(liste[3]))
            i+=1
    except:
        file.close()
        
    return [np.array(simID),np.array(RT),np.array(TIME),np.array(PHASE)]


def oldanova(datafile,experiment,names,saveFolder):
    """
    Two-way-repeated-meassures ANOVA
    
    datafile = file with the IDs, dependent and independent variables / values
    
    experiment = string for save files
    
    names = list of 5 strings
        Sim/Sub, simID/subID, DV, IV1, IV2
        
    saveFolder = string with saveFolder
    
    https://www.marsja.se/two-way-anova-repeated-measures-using-python/
    """

    [simID,RT,TIME,PHASE]=getData(datafile)
    
    
    Sim = namedtuple(names[0], [names[1],names[2],names[3],names[4]])               
    df = pt.DataFrame()
    
    for idx in xrange(len(simID)):
        df.insert(Sim(simID[idx],RT[idx], TIME[idx],PHASE[idx])._asdict())
     
    
    df.box_plot(names[2], factors=[names[3],names[4]], fname=saveFolder+'/box_'+names[2]+'_'+experiment+'.png')
    
    aov = df.anova(names[2], sub=names[1], wfactors=[names[3],names[4]])

    filename=saveFolder+'/ANOVA_'+names[2]+'_'+experiment+'.txt'
    with open(filename, 'w') as f:
        print(aov, file=f)
        
    posthoctest(datafile,filename,names)
        
def rel_ttest(g1,g2,numTests):
    """
    loads two samples (1d arrays g1 and g2)
    
    calculates a two sided ttest for dependent samples
    
    resturns means, standard deviations, t, df, p and cohens d_z
    """    
    n=np.sum(np.logical_not(np.isnan(g1)))
    df=n-1
    M1=np.nanmean(g1)
    M2=np.nanmean(g2)
    SD1=np.nanstd(g1)
    SD2=np.nanstd(g2)
    Diff=g2-g1
    MDiff=np.nanmean(Diff)
    SDDiff=np.nanstd(Diff)
    cohensD=MDiff/SDDiff# Standardized difference scores d_z
    [tVal,pVal]=stats.ttest_rel(g2,g1,nan_policy='omit')
    ### calculate 95% CI for dependent samples
    t_crit = stats.t.ppf(0.975, df)# 1-CI = 5% --> separated in two tails 2.5% in each tail --> either use right tail (0.975) or left tail (0.025), ppf=inverse of CDF
    CI = MDiff + np.array([-1,1])*t_crit*SDDiff/np.sqrt(n)
    #strings
    M1=str(round(M1,2))
    M2=str(round(M2,2))
    SD1=str(round(SD1,2))
    SD2=str(round(SD2,2))
    t=str(round(tVal,2))
    df=str(int(round(df,0)))
    CI=str(CI.round(2))
    p=str(round(pVal,3))
    if pVal<0.001:
        p='<0.001'
    if pVal<(0.05/numTests):
        p=p+'*'
    if pVal<(0.01/numTests):
        p=p+'*'
    if pVal<(0.001/numTests):
        p=p+'*'
    d=str(round(cohensD,2))
    vals=[str(M1+' ('+SD1+')'),str(M2+' ('+SD2+')'),t,df,p,d,CI]
    return vals
    

def classical_cohensD(mean1,std1,count1,mean2,std2,count2):
    dof = (count1 + count2 - 2)
    cohens_d = (mean1 - mean2) / np.sqrt(((count1 - 1) * std1 ** 2 + (count2 - 1) * std2 ** 2) / dof)
    return cohens_d

def ind_ttest(g1,g2,numTests):
    """
    loads two samples (1d arrays g1 and g2)
    
    calculates a two sided ttest for independent samples
    
    if sample sizes are not equal use welch test
    
    resturns means, standard deviations, t, df, p and cohens d (classical with pooled SD)
    """    
    n1=np.sum(np.logical_not(np.isnan(g1)))
    n2=np.sum(np.logical_not(np.isnan(g2)))
    df=n1-1 + n2-1
    M1=np.nanmean(g1)
    M2=np.nanmean(g2)
    SD1=np.nanstd(g1)
    SD2=np.nanstd(g2)
    SD_pool=np.nanstd(np.concatenate([g1,g2]))
    cohensD=classical_cohensD(M1,SD1,n1,M2,SD2,n2)
    [tVal,pVal]=stats.ttest_ind(g1,g2,nan_policy='omit', equal_var=[False,True][int(n1==n2)])
    ### 95% CI
    t_crit = stats.t.ppf(0.975, df)# 1-CI = 5% --> separated in two tails 2.5% in each tail --> either use right tail (0.975) or left tail (0.025), ppf=inverse of CDF
    ### calculate pooled standard deviation, pooled vriance = pooled SD**2
    SDp  = np.sqrt(((n1-1)*SD1**2 + (n2-1)*SD2**2)/(n1+n2-2))
    CI = (M1-M2) + np.array([-1,1])*t_crit*np.sqrt((SDp**2/n1)+(SDp**2/n2))
    #strings
    M1=str(round(M1,2))
    M2=str(round(M2,2))
    SD1=str(round(SD1,2))
    SD2=str(round(SD2,2))
    t=str(round(tVal,2))
    df=str(int(round(df,0)))
    p=str(round(pVal,3))
    CI=str(CI.round(2))
    if pVal<0.001:
        p='<0.001'
    if pVal<(0.05/numTests):
        p=p+'*'
    if pVal<(0.01/numTests):
        p=p+'*'
    if pVal<(0.001/numTests):
        p=p+'*'
    d=str(round(cohensD,2))
    vals=[str(M1+' ('+SD1+')'),str(M2+' ('+SD2+')'),t,df,p,d,CI]
    return vals
        

def posthoctest(datfile,filename,names):
    """
    multiple post hoc ttests
    
    datafile = file with the IDs, dependent and independent variables / values
    
    names = list of 5 strings
        Sim/Sub, simID/subID, DV, IV1, IV2
    
    makes ttests for the individual levels of factor 1 
    """
    spacing=20
    sep=''
    with open(filename, 'a') as f:
        
        print('\nPOSTHOC T-TESTS', file=f)
        
        [simID,RT,TIME,PHASE]=getData(datafile)
        
        TIMElevels=np.unique(TIME)
        PHASElevels=np.unique(PHASE)
        
        stringList=[names[4],names[3]+'='+str(TIMElevels[0]),names[3]+'='+str(TIMElevels[1]),'t','df','p','d']
        for string in stringList:
            print(string.ljust(spacing,' '),end=sep,file=f)
        print('\n'.ljust(spacing*len(stringList),'='),file=f)
        
        for PHASElevel in PHASElevels:
            ### ttest between the two times
            m0=PHASE==PHASElevel
            m1=TIME==TIMElevels[0]
            m2=TIME==TIMElevels[1]
            group1=RT[(m0*m1).astype('bool')]
            group2=RT[(m0*m2).astype('bool')]
            ttestResults=rel_ttest(group1,group2,PHASElevels.size)
            print(str(PHASElevel).ljust(spacing,' '),end=sep,file=f)
            for string in ttestResults:
                print(string.ljust(spacing,' '),end=sep,file=f)
            print('',file=f)
            
            
def ttestForTrials(datafile,experiment,names,saveFolder):
    """    
    datafile = file with the IDs, dependent and independent variable / values
    
    experiment = string for save files
    
    names = list of 4 strings
        Sim/Sub, simID/subID, DV, IV
    
    loads datafile with values for ID, DV and one IV (with two levels)
    
    calculates a two sided ttest for dependent samples to compare the groups of both IV levels
    
    prints the results in a txt file
    """
    
    if isinstance(datafile,list):#optional combine datasets e.g. simulations and vps combined --> which effect has time without separating groups (if no interaction --> groups are not separated)
        data=[]
        for idx in range(len(datafile)):
            data.append(get_trials_data(datafile[idx]))
        
        simID = np.concatenate([data[idx][0] for idx in range(len(data))])
        TRIALS = np.concatenate([data[idx][1] for idx in range(len(data))])
        TIME = np.concatenate([data[idx][2] for idx in range(len(data))])
        
    else:
        [simID, TRIALS, TIME] = get_trials_data(datafile)

    TIMElevels=np.unique(TIME)
    comparisons = [[0,1],[1,2],[0,2]]
    for comparisonIdx,comparison in enumerate(comparisons):
        m1=TIME==TIMElevels[comparison[0]]
        m2=TIME==TIMElevels[comparison[1]]
        group1=TRIALS[m1]
        group2=TRIALS[m2]
            
        spacing=20
        sep=''
        filename=saveFolder+'/TTEST_'+names[2]+'_'+experiment+'.txt'
        with open(filename, ['w','a'][int(comparisonIdx>0)]) as f:
            
            if comparisonIdx==0:
                print('T-TEST '+names[2], file=f)
                stringList=[names[3]+'='+str(TIMElevels[comparison[0]]),names[3]+'='+str(TIMElevels[comparison[1]]),'t','df','p','d_z']
                for string in stringList:
                    print(string.ljust(spacing,' '),end=sep,file=f)
                print('\n'.ljust(spacing*len(stringList),'='),file=f)
            
            ### ttest between the two times
            ttestResults=rel_ttest(group1,group2,len(comparisons))
            for string in ttestResults:
                print(string.ljust(spacing,' '),end=sep,file=f)
            print('',file=f)
            
            
            
def ttestForTrials_vs_ideal(datafile,experiment,names,saveFolder):
    """    
    loads error trials from data file
    
    calculates ttest for a single smaple, test if mean is different from ideal value (here 1.5 and 0.5)
    """

    [simID, TRIALS, TIME] = get_trials_data(datafile)


    TIMElevels=np.unique(TIME)
    comparison = [0,2]
    m1=TIME==TIMElevels[comparison[0]]
    m2=TIME==TIMElevels[comparison[1]]
    group_list=[TRIALS[m1],TRIALS[m2]]
        
    spacing=20
    sep=''
    filename=saveFolder+'/TTEST_'+names[2]+'_'+experiment+'.txt'
    
    for ideal_idx, ideal in enumerate([1.5,0.5]):
        with open(filename, ['w','a'][ideal_idx]) as f:
            
            if ideal_idx==0:
                print('T-TEST '+names[2]+' vs ideal', file=f)
                stringList=[names[3],'Mean (SD)','ideal','t','df','p','d','95%CI']
                
                for string in stringList:
                    print(string.ljust(spacing,' '),end=sep,file=f)
                print('\n'.ljust(spacing*len(stringList),'='),file=f)
            
            ### ttest for sample vs ideal value
            ttestResults=ttest_vs_ideal(group_list[ideal_idx],ideal,1)
            stringList=[str(comparison[ideal_idx])]+ttestResults
            for string in stringList:
                print(string.ljust(spacing,' '),end=sep,file=f)
            print('',file=f)
            
            
def ttestForTimeouts_vs_ideal(datafile,experiment,names,saveFolder):
    """    
    loads timeouts from data file
    
    calculates ttest for one sample for timeouts of second period against 0
    """

    [simID, TIMEOUTS, TIME] = get_trials_data(datafile)


    TIMElevels=np.unique(TIME)
    m=TIME==TIMElevels[1]
    group_list=[TIMEOUTS[m]]
        
    spacing=20
    sep=''
    filename=saveFolder+'/TTEST_'+names[2]+'_'+experiment+'.txt'
    
    for ideal_idx, ideal in enumerate([0]):
        with open(filename, ['w','a'][ideal_idx]) as f:
            
            if ideal_idx==0:
                print('T-TEST '+names[2]+' vs ideal', file=f)
                stringList=[names[3],'Mean (SD)','ideal','t','df','p','d']
                
                for string in stringList:
                    print(string.ljust(spacing,' '),end=sep,file=f)
                print('\n'.ljust(spacing*len(stringList),'='),file=f)
            
            ### ttest for sample vs ideal value
            ttestResults=ttest_vs_ideal(group_list[ideal_idx],ideal,1)
            stringList=[str(TIMElevels[1])]+ttestResults
            for string in stringList:
                print(string.ljust(spacing,' '),end=sep,file=f)
            print('',file=f)


def ttest_vs_ideal(g,val,numTests):
    """
    g: 1 sample
    val: H_0 population mean
    ideal=1.5
    unser=2
    --> wir sagen unser ist groesser, somit rejecten wir: H_0 = unser<=ideal --> H_1 = unser>ideal
    H_0 = mean(g)<=val --> H_1 = mean(g)>val --> one-sided test
    
    test if H_0 can be rejceted with 1 sample ttest
    """    
    n=np.sum(np.logical_not(np.isnan(g)))
    df=n-1
    M=np.nanmean(g)
    SD=np.nanstd(g)
    cohensD=(np.nanmean(g)-val)/np.nanstd(g)# standard cohens d
    [tVal,pVal]=stats.ttest_1samp(g,val, alternative='greater', nan_policy='omit')
    ### calculate 95% CI
    t_crit = stats.t.ppf(0.975, df)# 1-CI = 5% --> separated in two tails 2.5% in each tail --> either use right tail (0.975) or left tail (0.025), ppf=inverse of CDF
    CI = M + np.array([-1,1])*t_crit*SD/np.sqrt(n)
    #strings
    M=str(round(M,2))
    SD=str(round(SD,2))
    t=str(round(tVal,2))
    df=str(int(round(df,0)))
    p=str(round(pVal,3))
    CI=str(CI.round(2))
    if pVal<0.001:
        p='<0.001'
    if pVal<(0.05/numTests):
        p=p+'*'
    if pVal<(0.01/numTests):
        p=p+'*'
    if pVal<(0.001/numTests):
        p=p+'*'
    d=str(round(cohensD,2))
    vals=[str(M+' ('+SD+')'),str(val),t,df,p,d,CI]
    return vals
    

def ttestForTwoGroups(datafile,experiment,names,saveFolder, comparisons=[[0,1]]):
    """   
        datafile: directory of data file which contains 3 columns (ID, DV, IV) IV has two factors (e.g. two different phases)
        experiment: name of experiment
        names: list of two strings, name of [DV, IV]
    
        loads datafile with values
        
        calculates a two sided ttest for dependent samples to compare the groups of both IV levels
        
        prints the results in a txt file
    """
    
    [subID, RTs, PHASE] = get_trials_data(datafile)


    PHASElevels=np.unique(PHASE)
    for comparisonIdx,comparison in enumerate(comparisons):
        m1=PHASE==PHASElevels[comparison[0]]
        m2=PHASE==PHASElevels[comparison[1]]
        group1=RTs[m1]
        group2=RTs[m2]
            
        spacing=20
        sep=''
        filename=saveFolder+'/TTEST_'+names[0]+'_'+experiment+'.txt'
        with open(filename, ['w','a'][int(comparisonIdx>0)]) as f:
            
            if comparisonIdx==0:
                print('T-TEST '+names[0], file=f)
                stringList=[names[1]+'='+str(PHASElevels[comparison[0]]),names[1]+'='+str(PHASElevels[comparison[1]]),'t','df','p','d_z','95%CI']
                for string in stringList:
                    print(string.ljust(spacing,' '),end=sep,file=f)
                print('\n'.ljust(spacing*len(stringList),'='),file=f)
            
            ### ttest between the two times
            ttestResults=rel_ttest(group1,group2,len(comparisons))
            for string in ttestResults:
                print(string.ljust(spacing,' '),end=sep,file=f)
            print('',file=f)
            
            
def welch_test_two_ind_groups(datafile1,datafile2, names,saveFolder):
    """   
        datafile1 and datafile2: directory of data file which contains 3 columns (ID, DV, IV) IV has two factors (e.g. two different phases)

        loads datafile with values
        
        calculates a two sided ttest (welch test) for independent samples to compare the two groups for each IV level (2 comparisons)
        
        prints the results in a txt file
    """
    
    [subID1, ERRORs1, PHASE1] = get_trials_data(datafile1)
    [subID2, ERRORs2, PHASE2] = get_trials_data(datafile2)



    PHASElevels=np.unique(np.concatenate([PHASE1,PHASE2]))
    comparisons = [[0,0], [1,1]]
    for comparisonIdx,comparison in enumerate(comparisons):
        m1=PHASE1==PHASElevels[comparison[0]]
        m2=PHASE2==PHASElevels[comparison[1]]
        group1=ERRORs1[m1]
        group2=ERRORs2[m2]
        
            
        spacing=20
        sep=''
        filename=saveFolder+'/TTEST_'+names[0]+'.txt'
        with open(filename, ['w','a'][int(comparisonIdx>0)]) as f:
            
            if comparisonIdx==0:
                print('T-TEST '+names[0], file=f)
                stringList=[names[1],names[2],names[3],'t','df','p','d','95%CI']
                for string in stringList:
                    print(string.ljust(spacing,' '),end=sep,file=f)
                print('\n'.ljust(spacing*len(stringList),'='),file=f)
            
            ### ttest between the two times
            ttestResults=ind_ttest(group1,group2,len(comparisons))
            stringList=[str(PHASElevels[comparison[0]])+' & '+str(PHASElevels[comparison[1]])]+ttestResults
            for string in stringList:
                print(string.ljust(spacing,' '),end=sep,file=f)
            print('',file=f)

def CHi2(dataFile, saveFolder, name):
    """    
    datafile = file with the frequencies for 3 times
    
    loads datafile and calculates test uniform distribution of frequencies of the 3 times
    
    prints the results in a txt file
    """
    
    file = open(datafile, 'r')
    time=[]
    f1=[]
    f2=[]
    i=0
    try:
        while 1:
            zeile = file.readline()
            if len(zeile)>1:
                liste = zeile.split(' ')
                time.append(str(liste[0]))
                f1.append(float(liste[1]))
                f2.append(float(liste[2]))
                i+=1
            else:
                stop
    except:
        file.close()
    with open(saveFolder+'/Chi2_error_frequencies_results'+name+'.txt', 'w') as f:
        print('time', 'Chi2', 'df', 'p', file = f)
        for timeIdx, time in enumerate(time):
            if stats.chisquare([f1[timeIdx], f2[timeIdx]]).pvalue < 0.05/3.:
                append = '*'
            else:
                append = ''
            if stats.chisquare([f1[timeIdx], f2[timeIdx]]).pvalue < 0.01/3.:
                append = '**'
            if stats.chisquare([f1[timeIdx], f2[timeIdx]]).pvalue < 0.001/3.:
                append = '***'

            print(time, round(stats.chisquare([f1[timeIdx], f2[timeIdx]]).statistic, 2), 1, stats.chisquare([f1[timeIdx], f2[timeIdx]]).pvalue, append, file = f)
      
def get_trials_data(datafile):
    file = open(datafile, 'r')
    header = file.readline()
    simID=[]
    TRIALS=[]
    TIME=[]
    i=0
    try:
        while 1:
            zeile = file.readline()
            liste = zeile.split('\t')
            simID.append(int(liste[0]))
            TRIALS.append(float(liste[1]))
            TIME.append(int(liste[2]))
            i+=1
    except:
        file.close()
        
    simID=np.array(simID)
    TRIALS=np.array(TRIALS)
    TIME=np.array(TIME)

    return [simID, TRIALS, TIME]
  
def anova_2between_3within(datafile_1, datafile_2, saveFolder, saveName):
    """
        do a 2x3 anova with factor1 = 2 groups (between factor) and factor2 = 3 different times (within factor)
        
        datafile: txt file with format: 3 columns [id, dependend var, time]
    """

    timeTransformation = [['0','start'],['1','mid'],['2','end']]

    [ID_1, TRIALS_1, TIME_1] = get_trials_data(datafile_1)
    GROUP_1 = np.array(["G1"]*ID_1.shape[0])

    [ID_2, TRIALS_2, TIME_2] = get_trials_data(datafile_2)
    GROUP_2 = np.array(["G2"]*ID_2.shape[0])

    for trans in timeTransformation:
        TIME_1=np.where(TIME_1.astype(str)==trans[0], trans[1], TIME_1.astype(str))
        TIME_2=np.where(TIME_2.astype(str)==trans[0], trans[1], TIME_2.astype(str))

    d = {   'Trials' : np.concatenate((TRIALS_1,TRIALS_2)),
            'TIME'   : np.concatenate((TIME_1,TIME_2)),
            'GROUP'  : np.concatenate((GROUP_1,GROUP_2)),
            'IDs'    : np.concatenate((ID_1,ID_2+ID_1.max()+1))}
    df = DataFrame(data=d)

    anovaResults = df.mixed_anova(dv='Trials', between='GROUP', within='TIME', subject='IDs', effsize="np2")
    anovaResults = anovaResults.round(3)
    
    with open(saveFolder+'/'+saveName+'.txt', 'w') as f:    
        print(anovaResults, file=f)
    
    ### calculate SS total
    grand_mean=np.mean(d['Trials'])
    SS_total=np.sum((d['Trials']-grand_mean)**2)
    with open(saveFolder+'/'+saveName+'.txt', 'a') as f:    
        print('SS_total:',round(SS_total,3), file=f)
        
        
def anova_2between_2within(datafile_1, datafile_2, saveFolder, saveName):
    """
        do a 2x2 anova with factor1 = 2 groups (between factor) and factor2 = 2 different times (within factor)
        
        datafile: txt file with format: 3 columns [id, dependend var, time]
    """

    timeTransformation = [['0','initial'],['1','reversal']]

    [ID_1, TRIALS_1, TIME_1] = get_trials_data(datafile_1)
    GROUP_1 = np.array(["G1"]*ID_1.shape[0])

    [ID_2, TRIALS_2, TIME_2] = get_trials_data(datafile_2)
    GROUP_2 = np.array(["G2"]*ID_2.shape[0])

    for trans in timeTransformation:
        TIME_1=np.where(TIME_1.astype(str)==trans[0], trans[1], TIME_1.astype(str))
        TIME_2=np.where(TIME_2.astype(str)==trans[0], trans[1], TIME_2.astype(str))

    d = {   'Trials' : np.concatenate((TRIALS_1,TRIALS_2)),
            'TIME'   : np.concatenate((TIME_1,TIME_2)),
            'GROUP'  : np.concatenate((GROUP_1,GROUP_2)),
            'IDs'    : np.concatenate((ID_1,ID_2+ID_1.max()+1))}
    df = DataFrame(data=d)

    #df=remove_nan_entries(df, "Trials", "IDs") wird doch nicht gebraucht, subjects mit nan in einer Bedingung werden automatisch ignoriert

    anovaResults = df.mixed_anova(dv='Trials', between='GROUP', within='TIME', subject='IDs', effsize="np2")
    anovaResults = anovaResults.round(3)
    
    with open(saveFolder+'/'+saveName+'.txt', 'w') as f:    
        print(anovaResults, file=f)
    
    ### calculate SS total
    grand_mean=np.nanmean(d['Trials'])
    SS_total=np.nansum((d['Trials']-grand_mean)**2)
    with open(saveFolder+'/'+saveName+'.txt', 'a') as f:    
        print('SS_total:',round(SS_total,3), file=f)

def remove_nan_entries(df, variable, ID):
    """
        variable: string; name of column which may contain nan values
        ID: string; identifier of entries, if one entry with identifier contains nan all entries of this identifier are removed (each subject has nan value in one time --> remove all times of this participant)
    """
    nan_positions = np.where(np.isnan(df[variable]))[0]
    searched_id_list = [df.iloc[idx,:][ID] for idx in nan_positions]
    entries_with_searched_ids = np.sum(np.array([np.array(df[ID]==searched_id).astype(int) for searched_id in searched_id_list]),axis=0)>0
    indizes_of_searched_ids = np.where(entries_with_searched_ids)[0]
    ret=df.drop(indizes_of_searched_ids)
    return ret
        
def anova_1between_3within(datafile_1, saveFolder, saveName):
    """
        do a 1x3 anova with factor1 = group (between factor) and factor2 = 3 different times (within factor)
        
        datafile: txt file with format: 3 columns [id, dependend var, time]
    """

    timeTransformation = [['0','start'],['1','mid'],['2','end']]

    [ID_1, TRIALS_1, TIME_1] = get_trials_data(datafile_1)
    GROUP_1 = np.array(["G1"]*ID_1.shape[0])


    for trans in timeTransformation:
        TIME_1=np.where(TIME_1.astype(str)==trans[0], trans[1], TIME_1.astype(str))

    d = {   'Trials' : TRIALS_1,
            'TIME'   : TIME_1,
            'IDs'    : ID_1}
    df = DataFrame(data=d)

    anovaResults = df.rm_anova(dv='Trials', within='TIME', subject='IDs', effsize="np2", detailed=True)
    anovaResults = anovaResults.round(3)
    
    with open(saveFolder+'/'+saveName+'.txt', 'w') as f:    
        print(anovaResults, file=f)
        
    ### TODO perform a post hoc t-test for dependend samples for start time vs last time
    ttestForTwoGroups(datafile_1,'post_hoc_'+saveName, ['Trials','Time'],saveFolder, comparisons=[[0,1],[0,2],[1,2]])


saveFolder='../3_results'



##############################################################################################################################################################
####################################################################   SIMULATIONS   #########################################################################
##############################################################################################################################################################

###############################  FREQUENCIES ON  ##############################
datafile='../../manuscript_SRtask_results/2_dataEv/frequencies_for_CHI2_simulation_on.txt'
CHi2(datafile, saveFolder, 'CLUSTER_SIMULATIONS_on')

##############################  FREQUENCIES OFF  ##############################
datafile='../../manuscript_SRtask_results/2_dataEv/frequencies_for_CHI2_simulation_off.txt'
CHi2(datafile, saveFolder, 'CLUSTER_SIMULATIONS_off')

#################################  TRIALS  ####################################

##########################  MIXED ANOVA ON VS OFF #############################
datafile_simulations_on ='../../manuscript_SRtask_results/2_dataEv/TRIALs_for_TTEST_simulation_on.txt'
datafile_simulations_off='../../manuscript_SRtask_results/2_dataEv/TRIALs_for_TTEST_simulation_off.txt'
anova_2between_3within(datafile_simulations_on, datafile_simulations_off, saveFolder, 'anova_trials_on_off')

#############################  ONE-WAY ANOVA  #################################
anova_1between_3within(datafile_simulations_on, saveFolder, 'anova_trials_on')
anova_1between_3within(datafile_simulations_off, saveFolder, 'anova_trials_off')

################################  TTEST ON  ###################################
#ttestForTrials(datafile_simulations_on, 'CLUSTER_SIMULATIONS_on', ['Sim','simID','TRIALS','TIME'], saveFolder)
ttestForTrials_vs_ideal(datafile_simulations_on, 'CLUSTER_SIMULATIONS_VS_IDEAL', ['Sim','simID','TRIALS','TIME'], saveFolder)



##############################################################################################################################################################
####################################################################   EXPERIMENTS   #########################################################################
##############################################################################################################################################################

###############################  FREQUENCIES  #################################
datafile='../../manuscript_SRtask_results/2_dataEv/frequencies_for_CHI2_eyetracking.txt'
CHi2(datafile, saveFolder, 'CLUSTER_EXPERIMENTS')

###############################  TRIALs  ######################################

######  MIXED ANOVA EXP VS SIM EXPLORATION ERRORS START VS MID VS END  ########
datafile_participants='../../manuscript_SRtask_results/2_dataEv/TRIALs_for_TTEST_eyetracking.txt'
anova_2between_3within(datafile_simulations_on, datafile_participants, saveFolder, 'anova_trials_sim_exp')

#######  MIXED ANOVA EXP VS SIM GENEREAL ERRORS INTIAL VS REVERSAL ############
datafile_global_errors_participants='../../manuscript_global_performance_vps/2_dataEv/number_of_errors_vps.txt'
datafile_global_errors_sims='../../manuscript_global_performance/2_dataEv/number_of_errors_sims.txt'
#anova_2between_2within(datafile_global_errors_sims, datafile_global_errors_participants, saveFolder, 'anova_global_errors_sim_exp')

###############################  TTEST EXP  ###################################
#ttestForTrials([datafile_simulations_on, datafile_participants], 'CLUSTER_EXPERIMENTS_AND_SIMS', ['Sim','simID','TRIALS','TIME'], saveFolder)
#ttestForTrials(datafile_participants, 'CLUSTER_EXPERIMENTS', ['Sim','simID','TRIALS','TIME'], saveFolder)
ttestForTrials_vs_ideal(datafile_participants, 'CLUSTER_EXPERIMENTS_VS_IDEAL', ['?','?','TRIALS','TIME'], saveFolder)

#########################  RESPONSE TIMES TTEST  ##############################
datafile_participants='../../manuscript_global_performance_vps/2_dataEv/response_times_per_vp.txt'
ttestForTwoGroups(datafile_participants, 'CLUSTER_EXPERIMENTS', ['RTs','Phase'], saveFolder)

#############################  TIMEOUTS TTEST  ################################
#datafile_participants='../../manuscript_global_performance_vps/2_dataEv/time_outs_per_vp.txt'
#ttestForTimeouts_vs_ideal(datafile_participants, 'CLUSTER_EXPERIMENTS_VS_IDEAL', ['?','?','TIMEOUTS','PHASE'], saveFolder)

##########################  GLOBAL ERRORS TTEST  ##############################
welch_test_two_ind_groups(datafile_global_errors_sims, datafile_global_errors_participants, ['global_errors', 'Phase', 'Participants', 'Simulations'], saveFolder)















print('\n')
