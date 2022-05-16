import numpy as np
import pylab as plt
from extra_functions import get_output, get_block_trials, get_block_trials_are_correct, get_initial_trials, get_initial_weights, get_switch_weights, plot_column, get_initial_weights_initial_learning, save_2_within_arrays

font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 8}
bold_font = {'family' : 'Arial', 'weight' : 'bold', 'size'   : 8}
large_bold_font = {'family' : 'Arial', 'weight' : 'bold', 'size'   : 10}
plt.rc('font', **font)

num_sims=60
num_blocks=60
folder='../../../simulations/001e_Cluster_Experiment_PaperLearningRule_LearningON/4_dataEv'
post_switch_trials=7 # how many trials after switch should be analyzed

"""
Performance Plot of 60 Simulations:

initial learning blocks: x-axis trials from beginning, y-axis probability selecting rewarded action
reversal learning blocks: x-axis trials from last trial of previous block, y-axis probability of selecting previous and new rewarded action

Below performance Plot: weights of direct and indirect pathway (maybe hyperdirect?)
x-axis: trials as in performacne plot
y-axis: 5 weights

in general: plot with 2 columns and 3 (or 4) rows
"""


### LOAD DATA
pre_correct_number    = np.zeros((num_sims, post_switch_trials+1))
new_correct_number    = np.zeros((num_sims, post_switch_trials+1))
initial_trials        = np.zeros((num_sims, post_switch_trials+1))
number_errors         = np.zeros((num_sims, num_blocks-1))
number_initial_errors = np.zeros(num_sims)
initial_weights_sd1   = np.zeros((num_sims, post_switch_trials+1, 5))
initial_weights_sd2   = np.zeros((num_sims, post_switch_trials+1, 5))
initial_weights_stn   = np.zeros((num_sims, post_switch_trials+1, 5))
switch_weights_sd1    = np.zeros((num_sims, post_switch_trials+1, 5))
switch_weights_sd2    = np.zeros((num_sims, post_switch_trials+1, 5))
switch_weights_stn    = np.zeros((num_sims, post_switch_trials+1, 5))
for sim_id in range(num_sims):
    ### GET DATA OF TRIALS
    trials, correct, decision, frequentAction, start, dopInp = get_output(folder,sim_id+1)
    ### LOAD MEAN WEIGHTS OF PATHWAYS, INDEX 0=BEFORE FIRST TRIALS, INDEX 1=AFTER FIRST TRIAL --> CUT FIRST ELEMENT
    mw_sd1 = np.load(folder+"/mw_c_sd1_sim"+str(sim_id+1)+".npy")
    mw_sd2 = np.load(folder+"/mw_c_sd2_sim"+str(sim_id+1)+".npy")
    mw_stn = np.load(folder+"/mw_c_stn_sim"+str(sim_id+1)+".npy")
    
    ### FIRST: REVERSAL DATA
    
    ### SELECTIONS
    ### GET DECISIONS AFTER SWITCHES AN PRE AND NEW CORRECTS ACTIONS
    pre_correct, new_correct, block_trials = get_block_trials(correct, decision, post_switch_trials)
    ### CHECK IF DECISIONS ARE PRE/NEW CORRECT ACTIONS
    block_is_pre_correct, block_is_new_correct = get_block_trials_are_correct(block_trials, pre_correct, new_correct)
    ### COLLECT OVER ALL BLOCKS HOW OFTEN PRE/NEW CORRECT ACTION WAS SELECTED DURING SWITCH
    pre_correct_number[sim_id] = np.mean(block_is_pre_correct,0)
    new_correct_number[sim_id] = np.mean(block_is_new_correct,0)
    ### GET HOW MANY TRIALS SWITCH TOOK
    number_errors[sim_id] = np.sum(np.logical_not(block_is_new_correct),1)-1 # sum of trials which are not new correct (-1 because of pre switch trial)

    
    ### WEIGHTS
    switch_weights_sd1[sim_id] = get_switch_weights(mw_sd1[1:trials+1].reshape(trials,5), post_switch_trials, correct)
    switch_weights_sd2[sim_id] = get_switch_weights(mw_sd2[1:trials+1].reshape(trials,5), post_switch_trials, correct)
    switch_weights_stn[sim_id] = get_switch_weights(mw_stn[1:trials+1].reshape(trials,5), post_switch_trials, correct)
    
    ### SECOND: INITIAL DATA
    
    ### SELECTIONS
    initial_trials[sim_id] = get_initial_trials(correct, decision, post_switch_trials)
    number_initial_errors[sim_id] = np.sum(np.logical_not(initial_trials[sim_id]))-1
    
    ### WEIGHTS
    initial_weights_sd1[sim_id] = get_initial_weights_initial_learning(mw_sd1[:trials].reshape(trials,5), post_switch_trials, correct)
    initial_weights_sd2[sim_id] = get_initial_weights_initial_learning(mw_sd2[:trials].reshape(trials,5), post_switch_trials, correct)
    initial_weights_stn[sim_id] = get_initial_weights_initial_learning(mw_stn[:trials].reshape(trials,5), post_switch_trials, correct)
  
number_errors_per_sim=np.mean(number_errors,1) 

### PLOT
plt.figure(figsize=(8.5/2.54,8.5/2.54),dpi=500)

## INITIAL BLOCKS
plot_column('Initial blocks', 0, initial_trials, initial_weights_sd1, initial_weights_sd2, initial_weights_stn, post_switch_trials, bold_font, large_bold_font)
## REVERSAL BLOCKS
plot_column('Reversal blocks', 1, [pre_correct_number, new_correct_number], switch_weights_sd1, switch_weights_sd2, switch_weights_stn, post_switch_trials, bold_font, large_bold_font)

plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.2)
plt.savefig('../3_results/global_performance.svg')


### TEXT FILE
with open('../3_results/quantities.txt', 'w') as f:
    ### AVERAGE ERRORS
    print('number of errors:', file=f)
    print('initial:', np.mean(number_initial_errors),np.std(number_initial_errors), file=f)
    print('reversal:', np.mean(number_errors_per_sim),np.std(number_errors_per_sim), '\n', file=f)
    ### HOW MANY TRIALS UNTIL 100%
    print('number of trials until 100%:', file=f)
    print('(first trial is pre block beginning!)', file=f)
    print('initial:', np.mean(initial_trials,0), file=f)
    print('reversal:', np.mean(pre_correct_number,0), np.mean(new_correct_number,0), file=f)
    

### SAVES FOR STATS
save_2_within_arrays('number_of_errors_sims.txt', 'ERRORS', number_initial_errors, number_errors_per_sim)

