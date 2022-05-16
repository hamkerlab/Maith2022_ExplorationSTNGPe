import numpy as np
import pylab as plt
from scipy import stats


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


def get_response_times(start, dopInp, decision):
    """
        start: numpy array with start times of trials
        dopInp: numpy array with feedback onset time of trials
        decision: numpy array with selected actions
        
        returns the response times of all trials
    """
    
    end = dopInp.copy()
    end[decision==0] = start[decision==0]+400
    ret = end-start
    return ret
    
    
    

def get_block_trial_indizes(correct, block, post_switch_trials):
    """
        correct: numpy array with correct actions
        block: numpy array with block of experiment
        post_switch_trials: int, number of trials of block which should be returned
        
        returns: matrix, rows = indizes of switches + following trials
    """
    
    
    ### detect rule switches
    switch_trials=np.where(np.diff(correct))[0]
    ### detect block switch (if not already rule switch)
    if not(np.where(np.diff(block))[0][0] in switch_trials):
        switch_trials=np.sort(np.append(np.where(np.diff(block))[0][0], switch_trials))
    ### exclude rule switches which are shortly before breaks or between the two blocks
    not_use_these_rule_switches = [11,23,35]
    switch_trials=np.delete(switch_trials, not_use_these_rule_switches)

    block_trial_indizes = np.array([switch_trials+i for i in range(post_switch_trials+1)]).T# matrix rows = indizes of switches + following trials
    return block_trial_indizes
    
def get_successful_block_list(correct, decision, block):
    """
        correct: numpy array with correct actions
        decision: numpy array with selected actions
        block: numpy array with block of experiment
        
        returns: matrix, rows = indizes of switches + following trials
    """
    
    block_start_end_list = np.array([get_block_start_list(correct, block), get_block_end_list(correct, block)]).T
    
    ### exclude rule switches which are shortly before breaks or between the two blocks
    not_use_these_rule_switches = [11,23,35]
    not_use_these_blocks        = [12,24,36] 
    block_start_end_list = np.delete(block_start_end_list, not_use_these_blocks, 0)

    ### generate list of arrays for each block if decision was correct
    correct_decision_block=[]
    for idx, block_start_end in enumerate(block_start_end_list):
        start = block_start_end[0]
        end   = block_start_end[1]
        correct_decision_block.append(1*(correct[start:end+1]==decision[start:end+1]))

    ### get successful blocks
    successful_block_list=np.zeros(len(correct_decision_block))
    for idx, correct_decisions in enumerate(correct_decision_block):
        successful_block_list[idx] = is_block_successful(correct_decisions)

    return successful_block_list

def get_switch_trials(correct, block):
    """
        returns at which indizes the ruleswitch occurs (first trial with new rule)
        
        correct: numpy array with correct actions
        block: numpy array with block of experiment
    """
    ### detect rule switches
    switch_trials=np.where(np.diff(correct))[0]
    ### detect block switch (if not already rule switch)
    if not(np.where(np.diff(block))[0][0] in switch_trials):
        switch_trials=np.sort(np.append(np.where(np.diff(block))[0][0], switch_trials))
        
    ret=switch_trials+1
    return ret

def get_block_start_list(correct, block):
    """
        returns indizes of the beginnings of all blocks
        
        correct: numpy array with correct actions
        block: numpy array with block of experiment
    """
    ret=np.insert(get_switch_trials(correct, block),0,0)
    return ret

def get_block_end_list(correct, block):
    """
        returns indizes of the ends of all blocks
        
        correct: numpy array with correct actions
        block: numpy array with block of experiment
    """
    switch_trials=get_switch_trials(correct, block)
    ret=np.insert(switch_trials-1,switch_trials.size,correct.size-1)
    return ret
   
def is_block_successful(block_correct_decisions):
    """
        checks if block decisions: 7 consecutive correct decisions
        
        block_correct_decisions: array if decisions of block are correct
        
        returns if block is successful
    """
    diffs=np.where(np.diff(block_correct_decisions))[0]+1
    diffs=np.insert(diffs,0,0)
    diffs=np.insert(diffs,diffs.size,block_correct_decisions.size)
    block_sequence_max_len = 0
    for j in range(diffs.size-1):
        block_sequence=block_correct_decisions[diffs[j]:diffs[j+1]]
        ### check if sequence = correct decisions
        if block_sequence[0]==1:
            ### check if this is the new longest sequence
            if len(block_sequence)>block_sequence_max_len:
                block_sequence_max_len=len(block_sequence)
    ret=block_sequence_max_len>=7
    return ret
    
def get_how_long_until_rewarded(correct, decision, block):
    """
        correct: numpy array with correct actions
        decision: numpy array with selected actions
        block: numpy array with block of experiment
        
        returns: for each valid block, how many trials(errors) before new correct consecutively selected
    """
    
    block_start_end_list = np.array([get_block_start_list(correct, block), get_block_end_list(correct, block)]).T
    
    ### exclude rule switches which are shortly before breaks or between the two blocks
    not_use_these_rule_switches = [11,23,35]
    not_use_these_blocks        = [12,24,36] 
    block_start_end_list = np.delete(block_start_end_list, not_use_these_blocks, 0)

    ### generate list of arrays for each block if decision was correct
    correct_decision_block=[]
    for idx, block_start_end in enumerate(block_start_end_list):
        start = block_start_end[0]
        end   = block_start_end[1]
        correct_decision_block.append(1*(correct[start:end+1]==decision[start:end+1]))

    ### get errors of successful blocks
    errors_block_list=np.zeros(len(correct_decision_block))
    for idx, correct_decisions in enumerate(correct_decision_block):
        if is_block_successful(correct_decisions):
            errors_block_list[idx] = get_error_block(correct_decisions)
        else:
            errors_block_list[idx] = np.nan

    return errors_block_list
    
def get_error_block(block_correct_decisions):
    """
        gets the number of errors until new correct decision is selected of the block
        
        block_correct_decisions: array if decisions of block are correct
    """
    diffs=np.where(np.diff(block_correct_decisions))[0]+1
    diffs=np.insert(diffs,0,0)
    diffs=np.insert(diffs,diffs.size,block_correct_decisions.size)
    block_sequence_max_len = 0
    block_sequence_list=[]
    for j in range(diffs.size-1):
        block_sequence=block_correct_decisions[diffs[j]:diffs[j+1]]
        block_sequence_list.append(block_sequence)
        ### check if sequence = correct decisions
        if block_sequence[0]==1:
            ### check if this is the new longest sequence, if yes, save its index
            if len(block_sequence)>block_sequence_max_len:
                block_sequence_max_len=len(block_sequence)
                block_sequence_max_len_idx = j
            
    if block_sequence_max_len_idx==0:
        ### no errors before consecutive rewarded sequence
        errors=0
    else:
        ### only take not rewarded block sequences
        errors=0
        for idx,block_sequence in enumerate(block_sequence_list):
            if block_sequence[0]==0 and idx<block_sequence_max_len_idx:
                errors+=block_sequence.size
    ret=errors
    return ret
    

def get_block_trials(correct, decision, block, post_switch_trials):
    """
        correct: numpy array with correct actions
        decision: numpy array with selected actions
        block: numpy array with block of experiment
        post_switch_trials: int, number of trials of block which should be returned
        
        returns: pre and new correct and decisions for all switches/blocks
    """
    block_trial_indizes = get_block_trial_indizes(correct, block, post_switch_trials)
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

def plot_column(title, col, selections, post_switch_trials, bold_font, large_bold_font, mode='scatter'):
    """
        plots one column of the plot
    """
    trials=range(-1,post_switch_trials)
    
    if mode=='scatter':
        if col==0:
            x_scatter,y_scatter,s_scatter,c_scatter = selections_to_scatter(selections, trials, size=10, return_counts=True, max_nr=10)
            selections_raw = selections[np.logical_not(np.isnan(selections[:,0])),:]
            selections = np.nanmean(selections,0)
        else:
            x_scatter_pre,y_scatter_pre,s_scatter_pre,c_scatter_pre = selections_to_scatter(selections[0], trials, size=10, return_counts=True, max_nr=10)
            x_scatter_new,y_scatter_new,s_scatter_new,c_scatter_new = selections_to_scatter(selections[1], trials, size=10, return_counts=True, max_nr=10)
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
            x_annot = list(trials)[int(len(list(trials))*0.65)]
            y_annot = 0.6
            plt.scatter(x_annot,y_annot,s=s_max,facecolor=(0,0,0,0), edgecolor=(1,0,0,1), lw=0.2*np.sqrt(s_max))
            plt.text(x_annot+0.3, y_annot, '= '+str(c_max), ha='left', va='center')
            
            ### add annotation
            idx_max=np.argmax(c_scatter_new)
            s_max = s_scatter_new[idx_max]
            c_max = c_scatter_new[idx_max]
            x_annot = list(trials)[int(len(list(trials))*0.65)]
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
        plt.xlabel('trials', **bold_font)
        if col==0: axo.set_ylabel('Performance', **bold_font)
        if col==1: 
            ax.set_yticklabels([])
            
            
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
        plt.xlabel('trials', **bold_font)
        if col==0: plt.ylabel('Performance', **bold_font)
        if col==1: 
            ax.set_yticklabels([])
        #plt.legend()
    
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
    
    
def get_exploration_idx(correct, decision, block):
    """
        returns for all trials if they are exploration trials, exploration trials = before the consecutive rewarded trials (including the first of the consecutive rewarded trials)
    """
    
    
    block_start_end_list = np.array([get_block_start_list(correct, block), get_block_end_list(correct, block)]).T

    ### generate list of arrays for each block if decision was correct
    correct_decision_block=[]
    prev_correct=[np.nan]
    decision_block=[]
    for idx, block_start_end in enumerate(block_start_end_list):
        start = block_start_end[0]
        end   = block_start_end[1]
        correct_decision_block.append(1*(correct[start:end+1]==decision[start:end+1]))
        decision_block.append(decision[start:end+1])

        if start>0: prev_correct.append(correct[start-1])
    prev_correct=np.array(prev_correct)

    ### get trials before consecutive rewarded
    exploration_block=[]
    for idx, correct_decisions in enumerate(correct_decision_block):
        exploration_block.append(get_exploration_block(correct_decisions, decision_block[idx], prev_correct[idx]))

    ### return shape from block-wise to trial-wise
    ret=np.concatenate(exploration_block)
    return ret
    
    
def get_reversal_learning_idx(correct, decision, block):
    """
        returns for all trials if they are reversal learening trials, trials = before the consecutive rewarded trials (including the first of the consecutive rewarded trials)
    """
    
    
    block_start_end_list = np.array([get_block_start_list(correct, block), get_block_end_list(correct, block)]).T

    ### generate list of arrays for each block if decision was correct
    correct_decision_block=[]
    prev_correct=[np.nan]
    decision_block=[]
    for idx, block_start_end in enumerate(block_start_end_list):
        start = block_start_end[0]
        end   = block_start_end[1]
        correct_decision_block.append(1*(correct[start:end+1]==decision[start:end+1]))
        decision_block.append(decision[start:end+1])

        if start>0: prev_correct.append(correct[start-1])
    prev_correct=np.array(prev_correct)

    ### get trials before consecutive rewarded
    reversal_block=[]
    for idx, correct_decisions in enumerate(correct_decision_block):
        reversal_block.append(get_reversal_block(correct_decisions, decision_block[idx], prev_correct[idx]))

    ### return shape from block-wise to trial-wise
    ret=np.concatenate(reversal_block)
    return ret
    
    
def get_consecutive_rewarded_idx(correct, decision, block):
    """
        returns for all trials if they are consecutive rewarded trials
    """
    
    
    block_start_end_list = np.array([get_block_start_list(correct, block), get_block_end_list(correct, block)]).T

    ### generate list of arrays for each block if decision was correct
    correct_decision_block=[]
    prev_correct=[np.nan]
    decision_block=[]
    for idx, block_start_end in enumerate(block_start_end_list):
        start = block_start_end[0]
        end   = block_start_end[1]
        correct_decision_block.append(1*(correct[start:end+1]==decision[start:end+1]))
        decision_block.append(decision[start:end+1])

        if start>0: prev_correct.append(correct[start-1])
    prev_correct=np.array(prev_correct)

    ### get trials before consecutive rewarded
    consecutive_rewarded_block=[]
    for idx, correct_decisions in enumerate(correct_decision_block):
        consecutive_rewarded_block.append(get_consecutive_rewarded_block(correct_decisions, decision_block[idx], prev_correct[idx]))

    ### return shape from block-wise to trial-wise
    ret=np.concatenate(consecutive_rewarded_block)
    return ret
    
    
def get_exploration_block(block_correct_decisions, decision_block, prev_correct):
    """
        gets the indices of trials until new correct decision is selected of the block (consecutively)
        
        block_correct_decisions: array if decisions of block are correct
    """
    diffs=np.where(np.diff(block_correct_decisions))[0]+1
    diffs=np.insert(diffs,0,0)
    diffs=np.insert(diffs,diffs.size,block_correct_decisions.size)
    block_sequence_max_len = 0
    block_sequence_list=[]
    for j in range(diffs.size-1):
        block_sequence=block_correct_decisions[diffs[j]:diffs[j+1]]
        block_sequence_list.append(block_sequence)
        ### check if sequence = correct decisions
        if block_sequence[0]==1:
            ### check if this is the new longest sequence, if yes, save its index
            if len(block_sequence)>block_sequence_max_len:
                block_sequence_max_len=len(block_sequence)
                block_sequence_max_len_idx = j
            
            
    if block_sequence_max_len_idx==0:
        ### no errors before consecutive rewarded sequence
        errors=0
    else:
        ### add all sequences before long consecutive rewarded one = epxloration "errors"
        errors=0
        for idx,block_sequence in enumerate(block_sequence_list):
            if idx<block_sequence_max_len_idx:
                errors+=block_sequence.size
                
    ### all "errors" = exploration trials
    if block_sequence_max_len_idx==0:
        ### first block, or vps anticipated the rule change... also somehow explorative
        exploration_trials=np.zeros(block_correct_decisions.size)
        exploration_trials[0]=1
    else:
        exploration_trials=np.concatenate([np.ones(errors+1),np.zeros(block_correct_decisions.size-errors-1)])
        ### do not use first error, because their the rule change is not known
        exploration_trials[0]=0
        
                
    ret=exploration_trials
    return ret
    
    
def get_reversal_block(block_correct_decisions, decision_block, prev_correct):
    """
        gets the indices of trials until new correct decision is selected of the block (consecutively)
        
        block_correct_decisions: array if decisions of block are correct
    """
    diffs=np.where(np.diff(block_correct_decisions))[0]+1
    diffs=np.insert(diffs,0,0)
    diffs=np.insert(diffs,diffs.size,block_correct_decisions.size)
    block_sequence_max_len = 0
    block_sequence_list=[]
    for j in range(diffs.size-1):
        block_sequence=block_correct_decisions[diffs[j]:diffs[j+1]]
        block_sequence_list.append(block_sequence)
        ### check if sequence = correct decisions
        if block_sequence[0]==1:
            ### check if this is the new longest sequence, if yes, save its index
            if len(block_sequence)>block_sequence_max_len:
                block_sequence_max_len=len(block_sequence)
                block_sequence_max_len_idx = j
            
            
    if block_sequence_max_len_idx==0:
        ### no errors before consecutive rewarded sequence
        errors=0
    else:
        ### add all sequences before long consecutive rewarded one = epxloration "errors"
        errors=0
        for idx,block_sequence in enumerate(block_sequence_list):
            if idx<block_sequence_max_len_idx:
                errors+=block_sequence.size
                
    ### all "errors" = reversal trials
    if block_sequence_max_len_idx==0:
        ### first block, or vps anticipated the rule change... also somehow explorative
        reversal_trials=np.zeros(block_correct_decisions.size)
        reversal_trials[0]=1
    else:
        reversal_trials=np.concatenate([np.ones(errors+1),np.zeros(block_correct_decisions.size-errors-1)])
        
                
    ret=reversal_trials
    return ret
    
    
def get_consecutive_rewarded_block(block_correct_decisions, decision_block, prev_correct):
    """
        gets the indices of trials in which new correct decision is selected of the block (consecutively)
        
        block_correct_decisions: array if decisions of block are correct
    """
    diffs=np.where(np.diff(block_correct_decisions))[0]+1
    diffs=np.insert(diffs,0,0)
    diffs=np.insert(diffs,diffs.size,block_correct_decisions.size)
    block_sequence_max_len = 0
    block_sequence_list=[]
    for j in range(diffs.size-1):
        block_sequence=block_correct_decisions[diffs[j]:diffs[j+1]]
        block_sequence_list.append(block_sequence)
        ### check if sequence = correct decisions
        if block_sequence[0]==1:
            ### check if this is the new longest sequence, if yes, save its index
            if len(block_sequence)>block_sequence_max_len:
                block_sequence_max_len=len(block_sequence)
                block_sequence_max_len_idx = j
            
    ### set all sequences to zero, except the consecutive rewarded sequence
    consecutive_rewarded_trials = [np.zeros(block_sequence_list[i].size).astype(int) for i in range(len(block_sequence_list))]
    consecutive_rewarded_trials[block_sequence_max_len_idx] = np.ones(block_sequence_list[block_sequence_max_len_idx].size).astype(int)
    consecutive_rewarded_trials = np.concatenate(consecutive_rewarded_trials)        
                
    ret=consecutive_rewarded_trials
    return ret
    
    
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
                if mask_without_nan[subID] or 1:#initially if one time of this subject has a "nan" than do not use this subject... but I don't know why...
                    stringList=[str(int(subID)),str(data[PHASE,subID]),str(int(PHASE))]
                    for string in stringList:
                        print(string,end='\t',file=f)
                    print('',file=f)
    

    
    
    
    
    
    





















