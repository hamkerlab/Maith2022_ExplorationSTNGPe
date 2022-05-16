import numpy as np
import pylab as plt
from scipy import stats

def get_output(x,sim_id):
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
    selection = np.load(x+"/selection_sim"+str(sim_id)+".npy")
    while(selection[trials,1]!=0):
        trials+=1

    file = open(x+'/output_sim'+str(sim_id), 'r')
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

def get_block_trial_indizes(correct, post_switch_trials):
    """
        correct: numpy array with correct actions
        post_switch_trials: int, number of trials of block which should be returned
        
        returns: matrix, rows = indizes of switches + following trials
    """
    switch_trials = np.where(np.diff(correct))[0]# indizes before switch
    block_trial_indizes = np.array([switch_trials+i for i in range(post_switch_trials+1)]).T# matrix rows = indizes of switches + following trials
    return block_trial_indizes

def get_block_trials(correct, decision, post_switch_trials):
    """
        correct: numpy array with correct actions
        decision: numpy array with selected actions
        post_switch_trials: int, number of trials of block which should be returned
        
        returns: pre and new correct and decisions for all switches/blocks
    """
    block_trial_indizes = get_block_trial_indizes(correct, post_switch_trials)
    pre_correct = correct[block_trial_indizes][:,0]
    new_correct = correct[block_trial_indizes][:,-1]
    ret = [pre_correct, new_correct, decision[block_trial_indizes]]
    return ret
    
def get_block_trials_are_correct(block_trials, pre_correct, new_correct):
    """
        block_trials: numpy array, rows=decisions during switches
        pre_correct: numpy array, previous correct action for each switch
        new_correct: numpy array, new correct action for each switch
        
        returns: for each block/switch if decisions are pre/new correct --> two matrices
    """
    
    pre_correct_block = np.array([pre_correct for i in range(block_trials.shape[1])]).T # extents the array to the dimension of block_trials
    new_correct_block = np.array([new_correct for i in range(block_trials.shape[1])]).T # extents the array to the dimension of block_trials
    
    block_is_pre_correct = np.equal(block_trials, pre_correct_block) # checks if decisions are pre correct
    block_is_new_correct = np.equal(block_trials, new_correct_block) # checks if decisions are new correct
    
    ret = [block_is_pre_correct, block_is_new_correct]
    return ret
    
def get_initial_trials(correct, decision, post_switch_trials):
    """
        correct: numpy array with correct actions
        decision: numpy array with selected actions
        post_switch_trials: int, number of trials of block which should be returned
        
        returns: correctness of initial trials
    """
    decision_is_correct = decision==correct
    ret = np.concatenate([np.array([0]), decision_is_correct[:post_switch_trials]]) # prepend a 0 as "pre beginning" trial
    return ret
    
def get_initial_weights(weights, post_switch_trials, correct):
    """
        weights: numpy array, for each trial 5 weights
        post_switch_trials: int, number of trials of block which should be returned
        correct: numpy array with correct actions
        
        returns: weights of initial trials, first column=rewarded action, other columns = other not-rewarded actions
    """
       
    temp = np.concatenate([np.array([weights[0]]), weights[:post_switch_trials]],0) # prepend weights[0] as "pre beginning" trial
    correct_idx = int(correct[0]-1)
    
    temp[:, 0], temp[:, correct_idx] = temp[:, correct_idx], temp[:, 0].copy() # switch weights of rewarded action with first action
    ret = temp
    return ret
    
def get_initial_weights_initial_learning(weights, post_switch_trials, correct):
    """
        weights: numpy array, for each trial 5 weights
        post_switch_trials: int, number of trials of block which should be returned
        correct: numpy array with correct actions
        
        returns: weights of initial trials, first column=rewarded action, other columns = other not-rewarded actions
    """
       
    temp = weights[:post_switch_trials+1] # prepend weights[0] as "pre beginning" trial
    correct_idx = int(correct[0]-1)
    
    temp[:, 0], temp[:, correct_idx] = temp[:, correct_idx], temp[:, 0].copy() # switch weights of rewarded action with first action
    ret = temp
    return ret
    
def get_switch_weights(weights, post_switch_trials, correct):
    """
        weights: numpy array, for each trial 5 weights
        post_switch_trials: int, number of trials of block which should be returned
        correct: numpy array with correct actions
        
        returns: averaged weights over all switches, first column=new correct, second column=pre correct, other columns other actions
    """
    block_trial_indizes = get_block_trial_indizes(correct, post_switch_trials)
    pre_correct = correct[block_trial_indizes][:,0]
    new_correct = correct[block_trial_indizes][:,-1]
    
    temp = weights[block_trial_indizes] # weights for all switches
    
    ### FOR EACH RS SWITCH WEIGTHS OF REWARDED ACTION WITH FIRST PLACE AND PREV REWARDED ACTION WITH SECOND PLACE
    for rs_idx in range(temp.shape[0]):
        ### SWITCH NEW CORRECT WITH FIRST COLUMN
        new_correct_idx = int(new_correct[rs_idx]-1)
        temp[rs_idx, :, 0], temp[rs_idx, :, new_correct_idx] = temp[rs_idx, :, new_correct_idx], temp[rs_idx, :, 0].copy()
        
        ### PUT PRE CORRECT INTO SECOND COLUMN
        pre_correct_idx = int(pre_correct[rs_idx]-1)
        if pre_correct_idx == 0: # if the pre_correct column was switched with the new_correct column
            temp[rs_idx, :, 1], temp[rs_idx, :, new_correct_idx] = temp[rs_idx, :, new_correct_idx], temp[rs_idx, :, 1].copy()
        else:
            temp[rs_idx, :, 1], temp[rs_idx, :, pre_correct_idx] = temp[rs_idx, :, pre_correct_idx], temp[rs_idx, :, 1].copy()
            
    ret = np.mean(temp,0)
    return ret


def selections_to_scatter(selections, trials, size=10 ,return_counts=False, max_nr=None):
    """
        gets 2D array with selection data
        first dimension = vps
        second dimension = trials (in scatter plot x-axis)
        trials: future x-axis values, same size as selections.shape[1]!
        
        returns x,y and size for scatter plot
        if return_counts==True --> returns data point counts instead of size
    """
    trials = np.array(trials)
    assert trials.size == selections.shape[1], 'ERROR: selections_to_scatter, trials size does not fit to selections array'
    ### convert selections for each vp into x and y where x is number of trial
    x = np.array([trials for _ in range(selections.shape[0])]).flatten()
    y = selections.flatten()
    ### remove nan value pairs
    mask = np.logical_not(np.isnan(y))
    x = x[mask]; y = y[mask]
    ### get numbers of unique x-y pairs
    x_y = np.array([x,y]).T
    unique_x_y, nr_unique_x_y = np.unique(x_y, axis=0, return_counts=True)
    x_scatter = unique_x_y[:,0]
    y_scatter = unique_x_y[:,1]
    if max_nr!=None:
        s_scatter = size * (nr_unique_x_y/max_nr)
    else:
        s_scatter = size * (nr_unique_x_y/nr_unique_x_y.max())
    
    if return_counts:
        return [x_scatter,y_scatter,s_scatter,nr_unique_x_y]
    else:
        return [x_scatter,y_scatter,s_scatter]


def plot_column(title, col, selections, weights_sd1, weights_sd2, weights_stn, post_switch_trials, bold_font, large_bold_font,mode='scatter'):
    """
        plots one column of the plot
    """
    trials=range(-1,post_switch_trials)
    
    weights_sd1 = np.mean(weights_sd1,0)
    weights_sd2 = np.mean(weights_sd2,0)
    weights_stn = np.mean(weights_stn,0)
    
    classic_plot=1
    if classic_plot:
    
    
        if mode=='scatter':
            if col==0:
                x_scatter,y_scatter,s_scatter,c_scatter = selections_to_scatter(selections, trials, size=10, return_counts=True, max_nr=60)
                selections_raw = selections[np.logical_not(np.isnan(selections[:,0])),:]
                selections = np.nanmean(selections,0)
            else:
                x_scatter_pre,y_scatter_pre,s_scatter_pre,c_scatter_pre = selections_to_scatter(selections[0], trials, size=10, return_counts=True, max_nr=60)
                x_scatter_new,y_scatter_new,s_scatter_new,c_scatter_new = selections_to_scatter(selections[1], trials, size=10, return_counts=True, max_nr=60)
                selections_pre = np.mean(selections[0],0)
                selections_new = np.mean(selections[1],0)
                
            ### FIRST ROW
            ax=plt.subplot(3,2,col+1)
            plt.title(title, **large_bold_font)
            if col==0:
                """plt.scatter(x_scatter,y_scatter,s=s_scatter,facecolor=(0,0,0,0), edgecolor=(0,0,0,1), lw=0.2*np.sqrt(s_scatter))
                ### add annotation
                idx_max=np.argmax(c_scatter)
                s_max = s_scatter[idx_max]
                c_max = c_scatter[idx_max]
                x_annot = list(trials)[int(len(list(trials))*0.6)]
                y_annot = 0.5
                plt.scatter(x_annot,y_annot,s=s_max,facecolor=(0,0,0,0), edgecolor=(0,0,0,1), lw=0.2*np.sqrt(s_max))
                plt.text(x_annot+0.3, y_annot, '= '+str(c_max), ha='left', va='center')
                #for s in s_scatter:
                #    plt.scatter(x_scatter[s_scatter==s],y_scatter[s_scatter==s],s=5,marker=(int(s), 2, 0), lw=0.1, color='k')
                """
                axo = plt.gca()
                axo.plot(trials, selections, color='k',alpha=0.6)
                axo.set_ylim(-0.05,1.05)
                
                axc=plt.gca().twinx()
                ### initial learning --> plot matrix
                selections_com = np.append(selections_raw,selections).reshape(selections_raw.shape[0]+1,selections_raw.shape[1])
                yticks=np.arange(selections_raw.shape[0])[[0,-1]]
                yticklabels=[str(yticks[0])]+[str(yticks[-1])]
                selections_sort=selections_raw[np.argsort(np.sum(selections_raw,1))]
                plt.imshow(1-selections_raw, cmap='GnBu', aspect="auto", extent=[-1.5,6.5,-0.5,selections_raw.shape[0]-0.5], alpha=0.5)
                #for xpos in np.array([0,1,2,3,4,5,6])-0.5:
                #    plt.axvline(xpos, color='k', lw=0.3)
                plt.yticks(yticks,yticklabels)
                plt.yticks([])
                
                axo.set_zorder(2)
                axo.set_frame_on(False)
                axc.set_zorder(1)
                
                
            else:
                plt.scatter(x_scatter_pre,y_scatter_pre,s=s_scatter_pre,facecolor=(0,0,0,0), edgecolor=(1,0,0,1), lw=0.2*np.sqrt(s_scatter_pre))
                plt.scatter(x_scatter_new,y_scatter_new,s=s_scatter_new,facecolor=(0,0,0,0), edgecolor=(0,0,0,1), lw=0.2*np.sqrt(s_scatter_new))
                
                ### add annotation
                idx_max=np.argmax(c_scatter_pre)
                s_max = s_scatter_pre[idx_max]
                c_max = c_scatter_pre[idx_max]
                x_annot = list(trials)[int(len(list(trials))*0.2)]
                y_annot = 0.6
                plt.scatter(x_annot,y_annot,s=s_max,facecolor=(0,0,0,0), edgecolor=(1,0,0,1), lw=0.2*np.sqrt(s_max))
                plt.text(x_annot+0.3, y_annot, '= '+str(c_max), ha='left', va='center')
                
                ### add annotation
                idx_max=np.argmax(c_scatter_new)
                s_max = s_scatter_new[idx_max]
                c_max = c_scatter_new[idx_max]
                x_annot = list(trials)[int(len(list(trials))*0.2)]
                y_annot = 0.4
                plt.scatter(x_annot,y_annot,s=s_max,facecolor=(0,0,0,0), edgecolor=(0,0,0,1), lw=0.2*np.sqrt(s_max))
                plt.text(x_annot+0.3, y_annot, '= '+str(c_max), ha='left', va='center')
                
                #for s in s_scatter_pre:
                #    plt.scatter(x_scatter_pre[s_scatter_pre==s],y_scatter_pre[s_scatter_pre==s],s=5,marker=(int(s), 2, 0), lw=0.1, color='gray')
                #for s in s_scatter_new:
                #    plt.scatter(x_scatter_new[s_scatter_new==s],y_scatter_new[s_scatter_new==s],s=5,marker=(int(s), 2, 0), lw=0.1, color='k')
                plt.plot(trials, selections_pre, color='red', label='previously rewarded',alpha=0.6)
                plt.plot(trials, selections_new, color='k', label='rewarded',alpha=0.6)
                
                plt.ylim(-0.05,1.05)
                
            plt.xticks([0,2,4,6])
            plt.xlim(-1.2,post_switch_trials-0.8)
            ax.set_xticklabels([])
            if col==0: axo.set_ylabel('Performance', **bold_font)
            if col==1: ax.set_yticklabels([])
                
                
        else:
            if col==0:
                selections_er = stats.sem(selections, axis=0, ddof=0, nan_policy='omit')
                selections = np.nanmean(selections,0)
            else:
                selections_pre_er = stats.sem(selections[0], axis=0, ddof=0, nan_policy='omit')
                selections_new_er = stats.sem(selections[1], axis=0, ddof=0, nan_policy='omit')
                selections_pre = np.mean(selections[0],0)
                selections_new = np.mean(selections[1],0)
          
        
            ### FIRST ROW
            ax=plt.subplot(3,2,col+1)
            plt.title(title, **large_bold_font)
            if col==0:
                plt.errorbar(trials, selections, yerr=selections_er, color='k')
            else:
                plt.errorbar(trials, selections_pre, yerr=selections_pre_er, color='gray', label='previously rewarded')
                plt.errorbar(trials, selections_new, yerr=selections_new_er, color='k', label='rewarded')
            plt.ylim(-0.05,1.05)
            ax.set_xticklabels([])
            if col==0: plt.ylabel('Performance', **bold_font)
            if col==1: ax.set_yticklabels([])
    
    
    
    
        """if col==0:
            selections_er = stats.sem(selections, axis=0, ddof=0, nan_policy='omit')
            selections = np.nanmean(selections,0)
        else:
            selections_pre_er = stats.sem(selections[0], axis=0, ddof=0, nan_policy='omit')
            selections_new_er = stats.sem(selections[1], axis=0, ddof=0, nan_policy='omit')
            selections_pre = np.mean(selections[0],0)
            selections_new = np.mean(selections[1],0)
    
        ### FIRST ROW
        ax=plt.subplot(3,2,col+1)
        plt.title(title, **large_bold_font)
        if col==0:
            plt.errorbar(trials, selections, yerr=selections_er, color='k')
        else:
            plt.errorbar(trials, selections_pre, yerr=selections_pre_er, color='gray', label='previously rewarded')
            plt.errorbar(trials, selections_new, yerr=selections_new_er, color='k', label='rewarded')
        plt.ylim(-0.05,1.05)
        ax.set_xticklabels([])
        if col==0: plt.ylabel('Performance', **bold_font)
        if col==1: ax.set_yticklabels([])"""
        
        ### SECOND ROW
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
            #plt.legend()
    else:       
        ### FIRST ROW
        ax=plt.subplot(3,2,col+1)
        plt.title(title, **large_bold_font)
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
        
        ### SECOND ROW
        ax=plt.subplot(3,2,col+3)
        plt.plot(trials, weights_sd2[:,0], color='k', label='rewarded')
        if col==0:
            plt.plot(trials, weights_sd2[:,1], color='k', ls='dashed')
        else:
            plt.plot(trials, weights_sd2[:,1], color='k', ls='dotted', label='previously rewarded')
        plt.plot(trials, weights_sd2[:,2], color='k', ls='dashed', label='others')
        plt.plot(trials, weights_sd2[:,3], color='k', ls='dashed')
        plt.plot(trials, weights_sd2[:,4], color='k', ls='dashed')
        plt.ylim(-0.00005,0.00115)
        ax.set_xticklabels([])
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        if col==0: plt.ylabel('Indirect', **bold_font)
        if col==1: ax.set_yticklabels([])
    
        ### THIRD ROW
        ax=plt.subplot(3,2,col+5)
        plt.plot(trials, weights_stn[:,0], color='k', label='rewarded')
        if col==0:
            plt.plot(trials, weights_stn[:,1], color='k', ls='dashed')
        else:
            plt.plot(trials, weights_stn[:,1], color='k', ls='dotted', label='previously rewarded')
        plt.plot(trials, weights_stn[:,2], color='k', ls='dashed', label='others')
        plt.plot(trials, weights_stn[:,3], color='k', ls='dashed')
        plt.plot(trials, weights_stn[:,4], color='k', ls='dashed')
        plt.ylim(-0.00005, 0.00055)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        plt.xlabel('trials', **bold_font)
        if col==0: plt.ylabel('Hyperdirect', **bold_font)
        if col==1:
            ax.set_yticklabels([])
            plt.legend()
    
    
def save_2_within_arrays(filename, DV, data1, data2):
    """
        filenam: name of text file
        DV: str, name of dependent variable
        data1: data points for exploitation
        data2: data points for exploration
        len(data1)==len(data2) !
    """
    mask_without_nan = np.logical_not(np.isnan(data1))*np.logical_not(np.isnan(data2))
    data=np.array([data1, data2])
    save_file = '../2_dataEv/'+filename
    with open(save_file, 'w') as f:
        stringList=['subID',DV,'PHASE']
        for string in stringList:
            print(string,end='\t',file=f)
        print('',file=f)
        
        for PHASE in [0,1]:
            for subID in range(len(data1)):
                if mask_without_nan[subID]:
                    stringList=[str(int(subID)),str(data[PHASE,subID]),str(int(PHASE))]
                    for string in stringList:
                        print(string,end='\t',file=f)
                    print('',file=f)
    
    
    
    
    
    
    
    
    
    
    
    
    
    





















