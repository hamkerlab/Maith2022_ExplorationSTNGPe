import numpy as np
import pylab as plt

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

def plot_column(title, col, selections, weights_sd1, weights_sd2, weights_stn, post_switch_trials, bold_font, large_bold_font):
    """
        plots one column of the plot
    """
    trials=range(-1,post_switch_trials)
    if col==0:
        selections = np.mean(selections,0)
    else:
        selections_pre = np.mean(selections[0],0)
        selections_new = np.mean(selections[1],0)
    weights_sd1 = np.mean(weights_sd1,0)
    weights_sd2 = np.mean(weights_sd2,0)
    weights_stn = np.mean(weights_stn,0)
        
    
    ### FIRST ROW
    ax=plt.subplot(3,2,col+1)
    plt.title(title, **large_bold_font)
    if col==0: plt.plot(trials, selections, color='k')
    else:
        plt.plot(trials, selections_pre, color='k', ls='dotted')
        plt.plot(trials, selections_new, color='k')
    plt.ylim(-0.05,1.05)
    ax.set_xticklabels([])
    if col==0: plt.ylabel('Performance', **bold_font)
    if col==1: ax.set_yticklabels([])
    
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
        plt.legend()
    
    """### FOURTH ROW
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





















