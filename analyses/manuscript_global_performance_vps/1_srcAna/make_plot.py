import numpy as np
import pylab as plt
from extra_functions import get_output_vp, get_block_trials, get_block_trials_are_correct, get_initial_trials, get_initial_weights, get_switch_weights, plot_column, get_initial_weights_initial_learning, get_successful_block_list, get_how_long_until_rewarded, get_response_times, get_exploration_idx, save_2_within_arrays, get_reversal_learning_idx, get_consecutive_rewarded_idx

font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 8}
bold_font = {'family' : 'Arial', 'weight' : 'bold', 'size'   : 8}
large_bold_font = {'family' : 'Arial', 'weight' : 'bold', 'size'   : 10}
plt.rc('font', **font)

num_vps=10
num_valid_rs=44# 48 blocks --> 47 rule switches + 3 pauses --> 44 valid rule switches
folder='../../../psychExp/exp1_final/4_dataEv/outputs_vps/'
post_switch_trials=7 # how many trials after switch should be analyzed

"""
Performance Plot of 10 vps:

initial learning blocks: x-axis trials from beginning, y-axis probability selecting rewarded action
reversal learning blocks: x-axis trials from last trial of previous block, y-axis probability of selecting previous and new rewarded action



"""


### LOAD DATA
pre_correct_number          = np.zeros((num_vps, post_switch_trials+1))
new_correct_number          = np.zeros((num_vps, post_switch_trials+1))
initial_trials              = np.zeros((num_vps, post_switch_trials+1))*np.nan
number_errors               = np.zeros((num_vps, num_valid_rs))
number_initial_errors       = np.zeros(num_vps)
initial_weights_sd1         = np.zeros((num_vps, post_switch_trials+1, 5))
initial_weights_sd2         = np.zeros((num_vps, post_switch_trials+1, 5))
initial_weights_stn         = np.zeros((num_vps, post_switch_trials+1, 5))
switch_weights_sd1          = np.zeros((num_vps, post_switch_trials+1, 5))
switch_weights_sd2          = np.zeros((num_vps, post_switch_trials+1, 5))
switch_weights_stn          = np.zeros((num_vps, post_switch_trials+1, 5))
response_times_all          = []
response_times_reversal  = []
response_times_exploitation = []
successful_block_list       = []
for vp_id in range(num_vps):
    ### GET DATA OF TRIALS
    trials, correct, decision, _, start, dopInp, block = get_output_vp(folder,vp_id+1)
    
    ### EXCLUDE DATA OF INITIAL TRAININGS BLOCKS
    correct, decision, start, dopInp, block = correct[block>2], decision[block>2], start[block>2], dopInp[block>2], block[block>2]

    ### FIRST: REVERSAL DATA
    ### SELECTIONS
    ### GET DECISIONS AFTER SWITCHES AN PRE AND NEW CORRECTS ACTIONS
    pre_correct, new_correct, block_trials = get_block_trials(correct, decision, block, post_switch_trials)
    ### CHECK IF DECISIONS ARE PRE/NEW CORRECT ACTIONS
    block_is_pre_correct, block_is_new_correct = get_block_trials_are_correct(block_trials, pre_correct, new_correct)
    
    ### GET SUCCESSFUL BLOCKS
    successful_blocks = get_successful_block_list(correct, decision, block)
    successful_block_list.append(successful_blocks)
    
    ### COLLECT OVER ONLY SUCCESSFUL BLOCKS HOW OFTEN PRE/NEW CORRECT ACTION WAS SELECTED DURING SWITCH
    pre_correct_number[vp_id] = np.mean(block_is_pre_correct[successful_blocks[1:].astype(bool),:],0)
    new_correct_number[vp_id] = np.mean(block_is_new_correct[successful_blocks[1:].astype(bool),:],0)
    
    ### FOR EACH VALID BLOCK HOW MANY ERRORS WERE MADE BEFORE NEW CORRECT WAS SELECTED CONSECUTIVELY
    how_long_until_rewarded = get_how_long_until_rewarded(correct, decision, block)
    number_errors[vp_id] = how_long_until_rewarded[1:]

           
    ### SECOND: INITIAL DATA
    ### SELECTIONS
    if successful_blocks[0]:
        initial_trials[vp_id] = get_initial_trials(correct, decision, post_switch_trials)
    number_initial_errors[vp_id] = how_long_until_rewarded[0]

    ### THIRD: RESPONSE TIMES
    response_times = get_response_times(start, dopInp, decision)
    exploration_idx = get_exploration_idx(correct, decision, block)
    reversal_idx = get_reversal_learning_idx(correct, decision, block)
    consecutive_rewarded_idx = get_consecutive_rewarded_idx(correct, decision, block)

    response_times_all.append(response_times)
    response_times_reversal.append(response_times[reversal_idx.astype(bool)])
    response_times_exploitation.append(response_times[consecutive_rewarded_idx.astype(bool)])


successful_block_list = np.array(successful_block_list)

### GET DATA PER VP
number_errors_per_vp = np.nanmean(number_errors,1)
response_times_all_per_vp = np.array([np.median(rt_list[rt_list<400]) for rt_list in response_times_all])# per vp the median of RTs
time_outs_all_per_vp = [100*np.mean(rt_list>=400) for rt_list in response_times_all]# in percent
response_times_exploitation_per_vp = np.array([np.median(rt_list[rt_list<400]) for rt_list in response_times_exploitation])# per vp the median of RTs
time_outs_exploitation_per_vp = [100*np.mean(rt_list>=400) for rt_list in response_times_exploitation]# in percent
response_times_reversal_per_vp = np.array([np.median(rt_list[rt_list<400]) for rt_list in response_times_reversal])# per vp the median of RTs
time_outs_exploration_per_vp = [100*np.mean(rt_list>=400) for rt_list in response_times_reversal]# in percent

### PLOT
plt.figure(figsize=(8.5/2.54,8.5/2.54),dpi=500)

## INITIAL BLOCKS
plot_column('Initial blocks', 0, initial_trials, post_switch_trials, bold_font, large_bold_font)
## REVERSAL BLOCKS
plot_column('Reversal blocks', 1, [pre_correct_number, new_correct_number], post_switch_trials, bold_font, large_bold_font)

plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.2)
plt.savefig('../3_results/global_performance.svg')


### TEXT FILE
with open('../3_results/quantities.txt', 'w') as f:
    ### SUCCESSFUL BLOCKS
    print('successful_block_percent:', np.mean(successful_block_list), '\n', file=f)
    print('successful_blocks_absolute:', np.sum(successful_block_list), '\n', file=f)
    print('successful_block_percent mean over vps:', np.mean(np.mean(successful_block_list,1)),'\nSD over vps:',np.std(np.mean(successful_block_list,1)), '\n', file=f)
    ### AVERAGE ERRORS
    print('number of errors:', file=f)
    print('initial:', np.nanmean(number_initial_errors,0),np.nanstd(number_initial_errors,0), file=f)
    print('reversal:', np.nanmean(number_errors_per_vp),np.nanstd(number_errors_per_vp), '\n', file=f)
    ### RESPONSE TIMES ALL
    print('response times all:', file=f)
    print('mean:    ', np.mean(response_times_all_per_vp), file=f)
    print('std:     ', np.std(response_times_all_per_vp), file=f)
    print('median:  ', np.median(response_times_all_per_vp), file=f)
    print('timeouts: m =', round(np.mean(time_outs_all_per_vp),2), '%, sd =', round(np.std(time_outs_all_per_vp),2),'%\n', file=f)
    ### RESPONSE TIMES EXPLOIATION
    print('response times exploitation:', file=f)
    print('mean:    ', np.mean(response_times_exploitation_per_vp), file=f)
    print('std:     ', np.std(response_times_exploitation_per_vp), file=f)
    print('median:  ', np.median(response_times_exploitation_per_vp), file=f)
    print('timeouts: m =', round(np.mean(time_outs_exploitation_per_vp),2), '%, sd =', round(np.std(time_outs_exploitation_per_vp),2),'%\n', file=f)
    ### RESPONSE TIMES REVERSAL
    print('response times reversal:', file=f)
    print('mean:    ', np.mean(response_times_reversal_per_vp), file=f)
    print('std:     ', np.std(response_times_reversal_per_vp), file=f)
    print('median:  ', np.median(response_times_reversal_per_vp), file=f)
    print('timeouts: m =', round(np.mean(time_outs_exploration_per_vp),2), '%, sd =', round(np.std(time_outs_exploration_per_vp),2),'%\n', file=f)
    ### HOW MANY TRIALS UNTIL 100%
    print('number of trials until 100%: VPs do not reach 100%', file=f)
    print('(first trial is pre block beginning!)', file=f)
    print('initial:', np.nanmean(initial_trials,0), file=f)
    print('reversal:', np.mean(pre_correct_number,0), file=f)
    
    
### SAVE DATA PER VP FOR STATISTICS
save_2_within_arrays('response_times_per_vp.txt', 'RT', response_times_exploitation_per_vp, response_times_reversal_per_vp)
save_2_within_arrays('time_outs_per_vp.txt', 'RT', time_outs_exploitation_per_vp, time_outs_exploration_per_vp)
save_2_within_arrays('number_of_errors_vps.txt', 'ERRORS', number_initial_errors, number_errors_per_vp)




