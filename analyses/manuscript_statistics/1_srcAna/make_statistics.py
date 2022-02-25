#from __future__ import print_function
import numpy as np
#import pyvttbl as pt
#from collections import namedtuple
import pylab as plt
from scipy import stats
from pingouin import mixed_anova, rm_anova
from pandas import DataFrame


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
    
    resturns means, standard deviations, t, df, p and cohens d
    """    
    n=np.sum(np.logical_not(np.isnan(g1)))
    df=n-1
    M1=np.nanmean(g1)
    M2=np.nanmean(g2)
    SD1=np.nanstd(g1)
    SD2=np.nanstd(g2)
    Diff=g2-g1
    cohensD=np.nanmean(Diff)/np.nanstd(Diff)
    [tVal,pVal]=stats.ttest_rel(g2,g1,nan_policy='omit')
    #strings
    M1=str(round(M1,2))
    M2=str(round(M2,2))
    SD1=str(round(SD1,2))
    SD2=str(round(SD2,2))
    t=str(round(tVal,2))
    df=str(int(round(df,0)))
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
    vals=[str(M1+' ('+SD1+')'),str(M2+' ('+SD2+')'),t,df,p,d]
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
    
    if isinstance(datafile,list):
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
                stringList=[names[3]+'='+str(TIMElevels[comparison[0]]),names[3]+'='+str(TIMElevels[comparison[1]]),'t','df','p','d']
                for string in stringList:
                    print(string.ljust(spacing,' '),end=sep,file=f)
                print('\n'.ljust(spacing*len(stringList),'='),file=f)
            
            ### ttest between the two times
            ttestResults=rel_ttest(group1,group2,len(comparisons))
            for string in ttestResults:
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

    anovaResults = df.mixed_anova(dv='Trials', between='GROUP', within='TIME', subject='IDs', effsize="n2")
    anovaResults = anovaResults.round(3)
    
    with open(saveFolder+'/'+saveName+'.txt', 'w') as f:    
        print(anovaResults, file=f)


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

    anovaResults = df.rm_anova(dv='Trials', within='TIME', subject='IDs', effsize="n2", detailed=True)
    anovaResults = anovaResults.round(3)
    
    with open(saveFolder+'/'+saveName+'.txt', 'w') as f:    
        print(anovaResults, file=f)


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
ttestForTrials(datafile_simulations_on, 'CLUSTER_SIMULATIONS_on', ['Sim','simID','TRIALS','TIME'], saveFolder)



##############################################################################################################################################################
####################################################################   EXPERIMENTS   #########################################################################
##############################################################################################################################################################

###############################  FREQUENCIES  #################################
datafile='../../manuscript_SRtask_results/2_dataEv/frequencies_for_CHI2_eyetracking.txt'
CHi2(datafile, saveFolder, 'CLUSTER_EXPERIMENTS')

###############################  TRIALs  ######################################

##########################  MIXED ANOVA EXP VS SIM ############################
datafile_participants='../../manuscript_SRtask_results/2_dataEv/TRIALs_for_TTEST_eyetracking.txt'
anova_2between_3within(datafile_simulations_on, datafile_participants, saveFolder, 'anova_trials_sim_exp')

###############################  TTEST EXP  ###################################
ttestForTrials([datafile_simulations_on, datafile_participants], 'CLUSTER_EXPERIMENTS', ['Sim','simID','TRIALS','TIME'], saveFolder)















print('\n')
