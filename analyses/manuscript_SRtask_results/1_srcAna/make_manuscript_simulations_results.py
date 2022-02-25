import numpy as np
import pylab as plt
import os
import random
from scipy.stats import chisquare, sem
import matplotlib as mtl
random.seed()
rng = np.random.default_rng(1)


##############################################################################################################################################################
###################################################################  FUNCTIONS SIMS  #########################################################################
##############################################################################################################################################################

def get_performance(sim):
    performance = np.zeros(3)
    failed = np.load(folder+'/failed_blocks_sim'+str(sim)+'.npy')
    performance[0] = (1-failed[0]) / 1.
    performance[1] = ((failed[:failed.size//2]).size-np.sum(failed[:failed.size//2])) / (failed[:failed.size//2]).size
    performance[2] = (failed.size-np.sum(failed)) / failed.size
    return performance

def get_output(x,simID):
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
    trials=0    
    selection = np.load(x+"/selection_sim"+str(simID)+".npy")
    while(selection[trials,1]!=0):
        trials+=1

    file = open(x+'/output_sim'+str(simID), 'r')
    zeile = file.readline()
    correct = np.zeros(trials)
    decision = np.zeros(trials)
    start = np.zeros(trials)
    dopInp = np.zeros(trials)
    i=0
    try:
        while 1:
            zeile = file.readline()
            liste = zeile.split('\t')
            correct[i] = liste[4]
            decision[i] = liste[5]
            start[i] = liste[1]
            dopInp[i] = liste[2]
            i+=1
    except:
        file.close()

    frequentAction=[2,3,4,5][np.histogram(correct,[2,3,4,5,10])[0].argmax()]

    return [trials,correct,decision,frequentAction,start,dopInp]

def get_counts(correctList,decisionList):
    """
    loads correctList, decisionList
    
    counts for each rule switch how often actions were chosen
    all actions = [1,2,3,4,5,0]
    counted actions = [1,5,possibleAction,0]
    possibleAction = [2,3,4] without prev correct and new correct
    
    returns just first, mid and last counts (3x4 array)
    """
    counts=np.zeros((numRS,4))    

    rs=0
    for trial in np.arange(1,correctList.shape[0]):
        if getRsIdx(trial, correctList, decisionList)[0]:#ruleswtich detected-->analyse
            rsIdx=getRsIdx(trial, correctList, decisionList)[1]
            prevcorrect=getRsIdx(trial, correctList, decisionList)[2]
            newcorrect=getRsIdx(trial, correctList, decisionList)[3]
           
            possibleAction=[2,3,4]
            possibleAction.remove(prevcorrect)
            possibleAction.remove(newcorrect)
            actions=[1,5,possibleAction[0],0]

            for idx in rsIdx:
                counts[rs]+=1*(decisionList[idx]==actions)   
            rs+=1

    ret=counts[[0,counts.shape[0]//2,counts.shape[0]-2],:]

    if simAnz==10:
        #combine counts of six explorationperiods per time (early, mid, late)
        last=counts.shape[0]#-13
        early = np.sum(counts[[0,1,2,3,4,5]],0)#first 6
        mid = np.sum(counts[[last//2-3,last//2-2,last//2-1,last//2,last//2+1,last//2+2]],0)#middle
        late = np.sum(counts[[last-6,last-5,last-4,last-3,last-2,last-1]],0)#last 6
        ret=np.array([early,mid,late])

    return ret
  
def get_trialsNeeded(correctList,decisionList):
    """
    loads correctList, decisionList
    
    counts for each ruleswitch/exploration period the number of trials
    
    returns trialsnumber just of first, middle and last exploration period
    """
    
    trialsNeeded=np.zeros(numRS)    

    rs=0
    for trial in np.arange(1,correctList.shape[0]):
        if getRsIdx(trial, correctList, decisionList)[0]:#ruleswtich detected-->analyse
            rsIdx=getRsIdx(trial, correctList, decisionList)[1]
            prevcorrect=getRsIdx(trial, correctList, decisionList)[2]
            newcorrect=getRsIdx(trial, correctList, decisionList)[3]

            trialsNeeded[rs]=rsIdx.size-1#only count errors, if size==1 than the correct action was immediatly selected = no errors
            
            rs+=1

    ret=trialsNeeded[[0,trialsNeeded.shape[0]//2,trialsNeeded.shape[0]-2]]#get_first_mid_last_NotNan(trialsNeeded)[0]
    
    if simAnz==10:
        #combine trialsNeeded of six explorationperiods per time (early, mid, late)
        last = trialsNeeded.shape[0]#-13
        early = np.nanmean(trialsNeeded[[0,1,2,3,4,5]],0)#first 6
        mid = np.nanmean(trialsNeeded[[last//2-3,last//2-2,last//2-1,last//2,last//2+1,last//2+2]],0)#middle
        late = np.nanmean(trialsNeeded[[last-6,last-5,last-4,last-3,last-2,last-1]],0)#last 6
        ret=np.array([early,mid,late])

    return ret



##############################################################################################################################################################
###################################################################  FUNCTIONS VPS  ##########################################################################
##############################################################################################################################################################

def get_output_vp(x,vpID):
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
    trials=0
    file = open(x+'/output_vp'+str(vpID), 'r')
    try:
        while 1:
            zeile = file.readline()
            buf = zeile[0]
            trials+=1
    except:
        file.close()

    file = open(x+'/output_vp'+str(vpID), 'r')
    correct = np.zeros(trials)
    decision = np.zeros(trials)
    start = np.zeros(trials)
    dopInp = np.zeros(trials)
    block = np.zeros(trials)
    i=0
    try:
        while 1:
            zeile = file.readline()
            liste = zeile.split('\t')
            correct[i] = liste[4]
            decision[i] = liste[5]
            start[i] = liste[1]
            dopInp[i] = liste[2]
            block[i] = liste[8]
            i+=1
    except:
        file.close()

    frequentAction=[1,2,3,4,5][np.histogram(correct,[1,2,3,4,5,10])[0].argmax()]

    return [trials,correct,decision,frequentAction,start,dopInp,block]


def get_counts_vp(correctList,decisionList,blockList):
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
    counts=np.zeros((numRS,4))
    
    rs=0
    for trial in np.arange(1,correctList.shape[0]):
        if getRsIdx(trial, correctList, decisionList,blockList)[0]:#ruleswtich detected-->analyse
            rsIdx=getRsIdx(trial, correctList, decisionList,blockList)[1]
            prevcorrect=getRsIdx(trial, correctList, decisionList,blockList)[2]
            newcorrect=getRsIdx(trial, correctList, decisionList,blockList)[3]
            
            if (rs==np.array([11,23,35,47,59])).sum()==0:#nur wenn rs nicht einer von den "schlechten" ist auswerten, "schlecht" = rs von phase 12 auf 13 und rs zwischen bloecken
                if rs < 23:#--> erstere Block
                    possibleAction=[1,2,3,4,5]
                    possibleAction.remove(prevcorrect)
                    possibleAction.remove(newcorrect)
                    actions=[possibleAction[0],possibleAction[1],possibleAction[2],0]
                else:
                    reward_frequencies=np.histogram(correctList,[1,2,3,4,5,6])[0]
                    out1=reward_frequencies.argmin()
                    reward_frequencies[out1]+=2*reward_frequencies.max()
                    out2=reward_frequencies.argmin()+1
                    out1+=1
                    possibleAction=[1,2,3,4,5]
                    possibleAction.remove(out1)
                    possibleAction.remove(out2)
                    possibleAction.remove(prevcorrect)
                    possibleAction.remove(newcorrect)
                    actions=[out1,out2,possibleAction[0],0]
                    
                for idx in rsIdx:
                    counts[rs]+=1*(decisionList[idx]==actions)
                    
                rs+=1
            else:
                rs+=1
    #combine counts of six explorationperiods per time (early, mid, late) --> 6*10(vps) = 60 explorationperiods per time
    early = np.sum(counts[[24,25,26,27,28,29]],0)#first 6 ruleswitches of block 2
    mid = np.sum(counts[[44,45,46,48,49,50]],0)#middle (of Blocks 2 and 3) 6 ruleswiches
    late = np.sum(counts[[65,66,67,68,69,70]],0)#last 6 ruleswitches of block 3
    ret=np.array([early,mid,late])

    return ret

def get_trialsNeeded_vp(correctList,decisionList,blockList):
    """
    loads correctList, decisionList
    
    counts for each ruleswitch/exploration period the number of trials
    
    returns just early, mid and late trialsnumber
        early mid and late are the mean of 6 exploration periods (which are early/mid/late)
    """
    
    trialsNeeded=np.zeros(numRS)    

    rs=0
    for trial in np.arange(1,correctList.shape[0]):
        if getRsIdx(trial, correctList, decisionList,blockList)[0]:#ruleswtich detected-->analyse
            rsIdx=getRsIdx(trial, correctList, decisionList,blockList)[1]
            if (rs==np.array([11,23,35,47,59])).sum()==0:#nur wenn rs nicht einer von den "schlechten" ist auswerten, "schlecht" = rs von phase 12 auf 13 und rs zwischen bloecken
                if rsIdx.size==0:
                    trialsNeeded[rs]=None
                else:
                    trialsNeeded[rs]=rsIdx.size-1#only count errors, if size==1 than the correct action was immediatly selected = no errors
                
                rs+=1
            else:
                rs+=1
                
    #combine trialsNeeded of six explorationperiods per time (early, mid, late) --> 6*10(vps) = 60 explorationperiods per time
    early = np.nanmean(trialsNeeded[[24,25,26,27,28,29]])#first 6 ruleswitches of block 2
    mid = np.nanmean(trialsNeeded[[44,45,46,48,49,50]])#middle (of Blocks 2 and 3) 6 ruleswiches
    late = np.nanmean(trialsNeeded[[65,66,67,68,69,70]])#last 6 ruleswitches of block 3
    ret=np.array([early,mid,late])

    return ret

        

##############################################################################################################################################################
#################################################################  FUNCTIONS BOTH  ###########################################################################
##############################################################################################################################################################

def getRsIdx(trial, correctList, decisionList, blockList=np.zeros(100000)):
    """
    load the trial, correctList and decisionList
    
    checks if there is a ruleswitch (in correctList) at trial
    
    returns the indices of the following trials of the exploration period (without prevcorrect and zeros decisions)
    """
    if correctList[trial]!=correctList[trial-1] or blockList[trial]!=blockList[trial-1]:#ruleswitch detected!
        prevCorrect=correctList[trial-1]
        newCorrect=correctList[trial]
        start=trial
        i=trial
        #go over trials until the newCorrect gets selected two times = end of the exploration
        while (decisionList[i]==newCorrect and decisionList[i+1]==newCorrect)==False and newCorrect==correctList[i]:
            i+=1
        end=i
        
        IDX=np.arange(start,end+1,1)
        IDX=IDX[decisionList[IDX]!=prevCorrect]#take only the IDX of the trials where the prevCorrect wasn't selected (=common errors)
        IDX=IDX[decisionList[IDX]!=0]#also dont take zero decisions (=invalid trials, dont get analyzed)
              
        return [True, IDX, prevCorrect, newCorrect]
    else:
        return [False,False]



##############################################################################################################################################################
#################################################################  GLOBAL PARAMETERS  ########################################################################
##############################################################################################################################################################
learning_on=1
xlim=[0.5,3.5]
xticks=[1,2,3]
probslim=[0,50]#[0,1.1]
yticksprobs=[0,15,30,45]#[0.0,0.5,1.0]
rtlim=[100,650]
yticksrt=[150,600]
triallim=[-0.5, 3.7]#[0.5,5.5]
ytickstrial=[0,1.5,3]#[1,3,5]
performancelim=[-0.05,1.05]
yticksperformance=[0,1]
xlabels = ['start', 'mid', 'end']
caps=4
font = {}
font["axLabel"]         = {'fontsize': 11, 'fontweight' : 'normal'}
font["axTicks"]         = {'fontsize': 9}
font["subplotLabels"]   = {'fontsize': 14, 'fontweight' : 'bold'}
font["legend"]          = {'fontsize': 9, 'fontweight' : 'normal'}
font["titles"]          = {'fontsize': 14, 'fontweight' : 'bold'}
simsDrittel=[[0],[14],[28]]
cols=[[87/255.,26/255.,16/255.],[214/255.,74/255.,49/255.],[106/255.,138/255.,38/255.]]#never,frequent,rare
maxtolerance=7#wichtig fuer frequency_rare... wie weit duerfen timesteps von first,mid,last maximal weg sein
useSEM=True# variation measurement for trials, either standard error of the mean or standard deviation


##############################################################################################################################################################
###################################################################  SIMULATIONS  ############################################################################
##############################################################################################################################################################
folder1='../../../simulations/001e_Cluster_Experiment_PaperLearningRule_LearningON/4_dataEv'
folder2='../../../simulations/002e_Cluster_Experiment_PaperLearningRule_LearningOFF/4_dataEv'
folder=[folder1, folder2]
numRS=59
maxSimNr=60
simAnz=60
loadedSims = rng.choice(np.arange(1,maxSimNr+1), size=simAnz, replace=False)

"""###############################################################################
#############################  PERFORMANCE  ###################################
###############################################################################
fig=plt.figure(figsize=(184/25.4,120/25.4), dpi=300)
performance = np.zeros((3,simAnz))
#index = 3 different times

###########################  GET PERFORMANCE  #################################
for sim in range(simAnz):
    performance[:,sim] = get_performance(loadedSims[sim])

#####################  MEAN PERFORMANCE OVER SIMS  ############################
performanceMean = np.mean(performance,1)

###############################  PLOT  ########################################
x=np.arange(3)+1
plt.subplot(111)
width=0.2
plt.bar(x,performanceMean,width,color=cols[0])
print(performanceMean)
plt.xlim(xlim)
plt.ylim(performancelim)
plt.xticks(xticks,**font["axTicks"])
plt.yticks(yticksperformance,**font["axTicks"])

plt.savefig('../3_results/performance.svg')
plt.close()"""



###############################################################################
###########################  Probabilities  ###################################
###############################################################################
fig=plt.figure(figsize=(184/25.4,120/25.4), dpi=300)
probs=np.zeros((2,3,4,simAnz))
#0.index= with vs without learning
#1.index= 3 different times
#2.index= 4 possible error actions + zeros, 0,1=never rewarded, 2=other cluster action, 3=zeros 

for learning_idx in range(2):
    #############################  GET PROBS  #####################################
    for sim in range(simAnz):
        output=get_output(folder[learning_idx],loadedSims[sim])
        probs[learning_idx,:,:,sim]=get_counts(output[1],output[2])

##############  Summieren UEBER SIMS + CHISQUARE SAVES  #######################
probs=np.sum(probs,3)

for learning_idx in range(2):
    with open('../2_dataEv/frequencies_for_CHI2_simulation'+['_on','_off'][learning_idx]+'.txt', 'w') as f:
        for timeIdx, time in enumerate(['first', 'mid', 'last']):
            print(time, (probs[learning_idx,timeIdx,0]+probs[learning_idx,timeIdx,1])/2.0, probs[learning_idx,timeIdx,2], file = f)

#for i in range(probs.shape[0]):
#    probs[i]/=np.sum(probs[i])
probsMEAN=probs

###############################  PLOT  ########################################
x=np.arange(3)+1
ax_bar1=plt.subplot(221)
width=0.2
#learning on
plt.bar(x-0.75*width,(probsMEAN[1-learning_on,:,0]+probsMEAN[1-learning_on,:,1])/2.0,1.5*width,color=cols[0],label='out_cluster')
plt.bar(x+0.75*width,probsMEAN[1-learning_on,:,2],1.5*width,color=cols[2],label='in_cluster')
#style
plt.xlim(xlim)
plt.ylim(probslim)
plt.xticks(xticks,[])
plt.yticks(yticksprobs,**font["axTicks"])
plt.ylabel('error frequencies',**font["axLabel"])
plt.text(-0.25, 0.5, 'A:', va='center', ha='center', transform=ax_bar1.transAxes, **font["subplotLabels"])
plt.title('Simulations', pad=15, **font["titles"])


###############################################################################
#############################  ANZ TRIALS  ####################################
###############################################################################
trialsNeeded=np.zeros((2,3,simAnz))

############################  GET TRIALS  #####################################
for learing_idx in range(2):
    for sim in range(simAnz):
        output=get_output(folder[learing_idx],loadedSims[sim])
        trialsNeeded[learing_idx,:,sim]=get_trialsNeeded(output[1],output[2])

#######################  MITTELN UEBER SIMS  ##################################
trialsNeededMEAN=np.mean(trialsNeeded,2)
if useSEM:
    trialsNeededSD=sem(trialsNeeded, axis=2, ddof=1, nan_policy='raise')
else:
    trialsNeededSD=np.std(trialsNeeded,2)

###############################  PLOT  ########################################
ax_trials1=plt.subplot(223)
plt.axhline(1.5, ls = '--', color = 'k', alpha = 0.5)
plt.axhline(0.5, ls = '--', color = 'k', alpha = 0.5)
plt.errorbar(x,trialsNeededMEAN[1-learning_on],yerr=trialsNeededSD[1-learning_on],fmt='.',color='black',capsize=caps)
plt.xlim(xlim)
plt.xticks(xticks,xlabels,**font["axTicks"])
plt.yticks(ytickstrial,**font["axTicks"])
plt.ylim(triallim)
plt.ylabel('error trials',**font["axLabel"])
plt.xlabel('time',**font["axLabel"])
plt.text(-0.25, 0.5, 'C:', va='center', ha='center', transform=ax_trials1.transAxes, **font["subplotLabels"])
              
                
###############################################################################
###########################  TRIALs FOR TTEST  ################################
###############################################################################
for learning_idx in range(2):
    filename='../2_dataEv/TRIALs_for_TTEST_simulation'+['_on','_off'][learning_idx]+'.txt'
    with open(filename, 'w') as f:
        stringList=['simID','TRIALS','TIME']
        for string in stringList:
            print(string,end='\t',file=f)
        print('',file=f)
        
        for TIME in [0,1,2]:
            for simID in range(simAnz):
                                
                stringList=[str(int(simID)),str(round(trialsNeeded[learning_idx,TIME,simID],3)),str(int(TIME))]
                for string in stringList:
                    print(string,end='\t',file=f)
                print('',file=f)





##############################################################################################################################################################
############################################################  EYETRACKING EXPERIMENT  ########################################################################
##############################################################################################################################################################
folder='../../../psychExp/exp1_final/4_dataEv/outputs_vps/'
numRS=71
vpAnz=10


###############################################################################
###########################  Probabilities  ###################################
###############################################################################
probs=np.zeros((3,4,vpAnz))#3 possible error actions + zeros
#2.index: 0,1 never rewarded, 2 other cluster action, 3 zeros 

#############################  GET PROBS  #####################################
for vp in range(vpAnz):
    output=get_output_vp(folder,vp+1)
    probs[:,:,vp]=get_counts_vp(output[1],output[2],output[6])

##############  Summieren UEBER VPS + CHISQUARE SAVES  ########################
probs=np.sum(probs,2)

with open('../2_dataEv/frequencies_for_CHI2_eyetracking.txt', 'w') as f:
    for timeIdx, time in enumerate(['first', 'mid', 'last']):
        print(time, (probs[timeIdx,0] + probs[timeIdx,1])/2.0, probs[timeIdx,2], file = f)

#for i in range(probs.shape[0]):
#    probs[i]/=np.sum(probs[i])
probsMEAN=probs

###############################  PLOT  ########################################
x=np.arange(3)+1
ax=plt.subplot(222)
width=0.2
plt.bar(x-0.75*width,(probsMEAN[:,0]+probsMEAN[:,1])/2.0,1.5*width,color=cols[0],label='out_cluster')#,yerr=probsSD[:,0],capsize=caps)
#plt.bar(x,probsMEAN[:,1],width,color=cols[0])#,yerr=probsSD[:,1],capsize=caps)
plt.bar(x+0.75*width,probsMEAN[:,2],1.5*width,color=cols[2],label='in_cluster')#,yerr=probsSD[:,2],capsize=caps)
plt.xlim(xlim)
plt.ylim(probslim)
plt.xticks(xticks,[])
plt.yticks(yticksprobs,**font["axTicks"])
plt.text(-0.25, 0.5, 'B:', va='center', ha='center', transform=ax.transAxes, **font["subplotLabels"])
plt.title('Experiments', pad=15, **font["titles"])


###############################################################################
###########################  TRIALS NEEDED  ###################################
###############################################################################
trialsNeeded=np.zeros((3,vpAnz))

##############################  GET RT  #######################################
for vp in range(vpAnz):
    output=get_output_vp(folder,vp+1)
    trialsNeeded[:,vp]=get_trialsNeeded_vp(output[1],output[2],output[6])

########################  MITTELN UEBER VPs  ##################################
trialsNeededMEAN=np.mean(trialsNeeded,1)
if useSEM:
    trialsNeededSD=sem(trialsNeeded, axis=1, ddof=1, nan_policy='raise')
else:
    trialsNeededSD=np.std(trialsNeeded,1)

###############################  PLOT  ########################################
ax_trials2=plt.subplot(224)
plt.axhline(1.5, ls = '--', color = 'k', alpha = 0.5)
plt.axhline(0.5, ls = '--', color = 'k', alpha = 0.5)
plt.errorbar(x,trialsNeededMEAN,yerr=trialsNeededSD,fmt='.',color='black',capsize=caps)
plt.xlim(xlim)
plt.xticks(xticks,xlabels,**font["axTicks"])
plt.yticks(ytickstrial,**font["axTicks"])
plt.ylim(triallim)
plt.xlabel('time',**font["axLabel"])
plt.text(-0.25, 0.5, 'D:', va='center', ha='center', transform=ax_trials2.transAxes, **font["subplotLabels"])

                
                
###############################################################################
###########################  TRIALs FOR TTEST  ################################
###############################################################################
filename='../2_dataEv/TRIALs_for_TTEST_eyetracking.txt'
with open(filename, 'w') as f:
    stringList=['subID','TRIALS','TIME']
    for string in stringList:
        print(string,end='\t',file=f)
    print('',file=f)
    
    for TIME in [0,1,2]:
        for subID in range(vpAnz):
                            
            stringList=[str(int(subID)),str(round(trialsNeeded[TIME,subID],3)),str(int(TIME))]
            for string in stringList:
                print(string,end='\t',file=f)
            print('',file=f)



###############################################################################
##############################  LEGEND BOTTOM  ################################
###############################################################################
plt.subplots_adjust(top=0.9,bottom=0.2,right=0.99,left=0.12, wspace=0.4)
legend = {}
legend["width"]       = 0.45
legend["height"]      = 0.04
legend["bottom"]      = 0.04
legend["rectangleSize"] = 0.03
legend["freespace"]   = (legend["height"]-legend["rectangleSize"])/2.

mid = (ax_trials1.get_position().x1 + ax_trials2.get_position().x0) / 2.
xLeft = mid-legend["width"]/2.
legendField=mtl.patches.FancyBboxPatch(xy=(xLeft,legend["bottom"]),width=legend["width"],height=legend["height"],boxstyle=mtl.patches.BoxStyle.Round(pad=0.01),bbox_transmuter=None,mutation_scale=1,mutation_aspect=None,transform=fig.transFigure,**dict(linewidth=2, fc='w',ec='k',clip_on=False))
plt.gca().add_patch(legendField)
# never rewarded rectangle
(x0, y0) = (mid-legend["width"]/2. + legend["freespace"], legend["bottom"] + legend["freespace"])
plt.gca().add_patch(mtl.patches.FancyBboxPatch(xy=(x0,y0),width=legend["rectangleSize"],height=legend["rectangleSize"],boxstyle=mtl.patches.BoxStyle.Round(pad=0),bbox_transmuter=None,mutation_scale=1,mutation_aspect=None,transform=fig.transFigure,**dict(linewidth=0, fc=cols[0],ec=cols[0],clip_on=False)))
plt.text(x0+legend["rectangleSize"]+legend["freespace"],y0+legend["rectangleSize"]/2.,'never rewarded',ha='left',va='center',transform=fig.transFigure, **font["legend"])
# repeatedly rewarded rectangle
(x0, y0) = (mid+legend["freespace"], legend["bottom"] + legend["freespace"])
plt.gca().add_patch(mtl.patches.FancyBboxPatch(xy=(x0,y0),width=legend["rectangleSize"],height=legend["rectangleSize"],boxstyle=mtl.patches.BoxStyle.Round(pad=0),bbox_transmuter=None,mutation_scale=1,mutation_aspect=None,transform=fig.transFigure,**dict(linewidth=0, fc=cols[2],ec=cols[2],clip_on=False)))
plt.text(x0+legend["rectangleSize"]+legend["freespace"],y0+legend["rectangleSize"]/2.,'repeatedly rewarded',ha='left',va='center',transform=fig.transFigure, **font["legend"])



###############################################################################
###############################  LEGEND TOP  ##################################
###############################################################################
#legend = {}
#legend["width"]       = 0.22
#legend["height"]      = 0.04
#legend["bottom"]      = ax_bar1.get_position().y1-0.06
#legend["rectangleSize"] = 0.03
#legend["freespace"]   = (legend["height"]-legend["rectangleSize"])/2.

#xLeft = (ax_bar1.get_position().x1 + ax_bar1.get_position().x0) / 2.-0.015
#mid = xLeft + legend["width"]/2.

#(x0, y0) = (mid-legend["width"]/2. + legend["freespace"], legend["bottom"] + legend["freespace"])

#mytext=plt.text(x0+legend["rectangleSize"]+legend["freespace"],y0+legend["rectangleSize"]/2.,r'STN$\rightarrow$GPe fixed',ha='left',va='center',transform=fig.transFigure, **font["legend"])

#figure coordinates for text box
#fig.canvas.draw()
#transf = fig.transFigure.inverted()
#bb = mytext.get_window_extent(renderer = fig.canvas.renderer)
#bbd = bb.transformed(transf)

#legendField=mtl.patches.FancyBboxPatch(xy=(bbd.x0-(legend["rectangleSize"]+legend["freespace"]),bbd.y0),width=bbd.x1-bbd.x0+legend["rectangleSize"]+legend["freespace"],height=bbd.y1-bbd.y0,boxstyle=mtl.patches.BoxStyle.Round(pad=0.01),bbox_transmuter=None,mutation_scale=1,mutation_aspect=None,transform=fig.transFigure,**dict(linewidth=0.7, fc='w',ec='k',clip_on=False))
#plt.gca().add_patch(legendField)

#plt.gca().add_patch(mtl.patches.FancyBboxPatch(xy=(x0,y0),width=legend["rectangleSize"],height=legend["rectangleSize"],boxstyle=mtl.patches.BoxStyle.Round(pad=0),bbox_transmuter=None,mutation_scale=1,mutation_aspect=None,transform=fig.transFigure,**dict(linewidth=1, fc=(0,0,0,0.2),ec=(0,0,0,1),clip_on=False)))



###############################################################################
###############################  LEGEND MID  ##################################
###############################################################################
#legend = {}
#legend["width"]       = 0.22
#legend["height"]      = 0.04
#legend["bottom"]      = ax_trials1.get_position().y1-0.06
#legend["rectangleSize"] = 0.03
#legend["freespace"]   = (legend["height"]-legend["rectangleSize"])/2.

#xLeft = (ax_trials1.get_position().x1 + ax_trials1.get_position().x0) / 2.-0.015
#mid = xLeft + legend["width"]/2.

#(x0, y0) = (mid-legend["width"]/2. + legend["freespace"], legend["bottom"] + legend["freespace"])

#mytext=plt.text(x0+legend["rectangleSize"]+legend["freespace"],y0+legend["rectangleSize"]/2.,r'STN$\rightarrow$GPe fixed',ha='left',va='center',transform=fig.transFigure, **font["legend"])

#figure coordinates for text box
#fig.canvas.draw()
#transf = fig.transFigure.inverted()
#bb = mytext.get_window_extent(renderer = fig.canvas.renderer)
#bbd = bb.transformed(transf)

#legendField=mtl.patches.FancyBboxPatch(xy=(bbd.x0-(legend["rectangleSize"]+legend["freespace"]),bbd.y0),width=bbd.x1-bbd.x0+legend["rectangleSize"]+legend["freespace"],height=bbd.y1-bbd.y0,boxstyle=mtl.patches.BoxStyle.Round(pad=0.01),bbox_transmuter=None,mutation_scale=1,mutation_aspect=None,transform=fig.transFigure,**dict(linewidth=0.7, fc=(1,1,1,0),ec='k',clip_on=False))
#plt.gca().add_patch(legendField)

#xInch=184/25.4
#xDots=xInch*300
#yInch=120/25.4
#yDots=yInch*300

#xP=(x0+legend["rectangleSize"]/2.)*xDots
#yP=(y0+legend["rectangleSize"]/2.)*yDots
#transf = ax_trials1.transData.inverted()
#xyP = transf.transform((xP,yP))

#ax_trials1.errorbar(xyP[0],xyP[1],yerr=0.15,fmt='.',color='grey',capsize=caps)

            
plt.savefig('../3_results/manuscript_SRtask_results.svg')








