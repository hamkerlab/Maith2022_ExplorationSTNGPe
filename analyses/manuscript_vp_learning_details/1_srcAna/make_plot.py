import numpy as np
import pylab as plt
from extra_functions import get_output_vp, get_block_trials, get_block_trials_are_correct, get_initial_trials, get_initial_weights, get_switch_weights, plot_column, get_initial_weights_initial_learning, get_successful_block_list, get_how_long_until_rewarded, get_response_times, get_block_trials_are_cluster, get_selected_cluster, prepend_nans, get_last_positive_idx, center_at_last_idx, get_centered_selected_out_cluster_m, make_regression_plot, make_single_vps_plot, get_fake_data, make_correlation_plot
from tqdm import tqdm
from scipy.stats import norm

rng = np.random.default_rng(0)

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
mode='correlation'
fake_samples=0

"""
Performance Plot of 10 vps:

initial learning blocks: x-axis trials from beginning, y-axis probability selecting rewarded action
reversal learning blocks: x-axis trials from last trial of previous block, y-axis probability of selecting previous and new rewarded action



"""


### LOAD DATA
pre_correct_number    = np.zeros((num_vps, post_switch_trials+1))
new_correct_number    = np.zeros((num_vps, post_switch_trials+1))
initial_trials        = np.zeros((num_vps, post_switch_trials+1))*np.nan
number_errors         = np.zeros((num_vps, num_valid_rs))
number_initial_errors = np.zeros(num_vps)
initial_weights_sd1   = np.zeros((num_vps, post_switch_trials+1, 5))
initial_weights_sd2   = np.zeros((num_vps, post_switch_trials+1, 5))
initial_weights_stn   = np.zeros((num_vps, post_switch_trials+1, 5))
switch_weights_sd1    = np.zeros((num_vps, post_switch_trials+1, 5))
switch_weights_sd2    = np.zeros((num_vps, post_switch_trials+1, 5))
switch_weights_stn    = np.zeros((num_vps, post_switch_trials+1, 5))
response_times        = []
successful_block_list = []
selected_in_cluster   = np.zeros((num_vps, num_valid_rs+1))
selected_out_cluster  = np.zeros((num_vps, num_valid_rs+1))
for vp_id in range(num_vps):
    ### GET DATA OF TRIALS
    trials, correct, decision, _, start, dopInp, block, cluster = get_output_vp(folder,vp_id+1)
    
    ### EXCLUDE DATA OF INITIAL TRAININGS BLOCKS
    correct, decision, start, dopInp, block = correct[block>2], decision[block>2], start[block>2], dopInp[block>2], block[block>2]

    ### FIRST: REVERSAL DATA
    ### SELECTIONS
    ### GET DECISIONS AFTER SWITCHES AN PRE AND NEW CORRECTS ACTIONS
    pre_correct, new_correct, block_trials = get_block_trials(correct, decision, block, post_switch_trials)
    ### CHECK IF DECISIONS ARE PRE/NEW CORRECT ACTIONS
    block_is_pre_correct, block_is_new_correct = get_block_trials_are_correct(block_trials, pre_correct, new_correct)
    
    ### check if actions are in or out cluster
    block_is_in_cluster, block_is_out_cluster = get_block_trials_are_cluster(block_trials, cluster)
    
    ### GET SUCCESSFUL BLOCKS
    successful_blocks = get_successful_block_list(correct, decision, block)
    successful_block_list.append(successful_blocks)
    
    ### COLLECT OVER ONLY SUCCESSFUL BLOCKS HOW OFTEN PRE/NEW CORRECT ACTION WAS SELECTED DURING SWITCH
    pre_correct_number[vp_id] = np.mean(block_is_pre_correct[successful_blocks[1:].astype(bool),:],0)
    new_correct_number[vp_id] = np.mean(block_is_new_correct[successful_blocks[1:].astype(bool),:],0)
    
    ### FOR EACH VALID BLOCK HOW MANY ERRORS WERE MADE BEFORE NEW CORRECT WAS SELECTED CONSECUTIVELY
    how_long_until_rewarded = get_how_long_until_rewarded(correct, decision, block)
    number_errors[vp_id] = how_long_until_rewarded[1:]
    
    ### similar to get_how_long_until_rewarded but get how often in cluster/out cluster BEFORE NEW CORRECT WAS SELECTED CONSECUTIVELY
    selected_in_cluster[vp_id], selected_out_cluster[vp_id] = get_selected_cluster(correct, decision, block, cluster)
           
    ### SECOND: INITIAL DATA
    ### SELECTIONS
    if successful_blocks[0]:
        initial_trials[vp_id] = get_initial_trials(correct, decision, post_switch_trials)
    number_initial_errors[vp_id] = how_long_until_rewarded[0]

    ### THIRD: RESPONSE TIMES
    response_times.append(get_response_times(start, dopInp, decision))

successful_block_list = np.array(successful_block_list)



### CHECK SINGLE VPS
make_single_vps_plot('../3_results/single_vps.svg',selected_out_cluster, bold_font)

### GET CENTERED AVERAGE OF VPS
selected_out_cluster_m, selected_out_cluster_sd, last_out_cluster_idx, selected_out_cluster_centered = get_centered_selected_out_cluster_m(selected_out_cluster)
make_single_vps_plot('../3_results/single_vps_centered.svg',selected_out_cluster_centered, bold_font)

if mode=='correlation':
    ### GET CORRELATION OF VPS
    corr_coef_vps, corr_p_val_vps, df_vps, CI = make_correlation_plot('../3_results/average_vps.svg', selected_out_cluster, bold_font=bold_font)
    with open('../3_results/values.txt', 'w') as f:
        print('correlations results: r =',corr_coef_vps, ' p =', corr_p_val_vps, ' df =', df_vps, ' 95%CI =', CI.round(2), file=f)

if mode=='regression':
    ### MAKE REGRESSION FOR VPS AND GET SLOPE
    regression_results_vps = make_regression_plot('../3_results/average_vps.svg', selected_out_cluster_m, selected_out_cluster_sd, bold_font=bold_font)
    with open('../3_results/values.txt', 'w') as f:
        print('regression results:',regression_results_vps, file=f)

### when (which range) did vps learn rule - USED FOR FAKE DATA
learned_rule_arr = (last_out_cluster_idx-selected_out_cluster.shape[1]).astype(int)
learned_rule_range = np.array([learned_rule_arr.min(),learned_rule_arr.max()])

### CREATE FAKE DATA
selected_out_cluster_fake = get_fake_data(selected_out_cluster.shape[0], num_valid_rs, learned_rule_range, learned_rule_arr, rng=rng)

### GET SINGLE PLOTS
make_single_vps_plot('../3_results/single_fakes.svg',selected_out_cluster_fake, bold_font)

if mode=='correlation':
    ### GET CORRELATION OF VPS
    make_correlation_plot('../3_results/average_fakes.svg', selected_out_cluster_fake, bold_font=bold_font)
if mode=='regression':
    ### GET AVERAGE AND REGRESSION
    selected_out_cluster_m, selected_out_cluster_sd, _, _ = get_centered_selected_out_cluster_m(selected_out_cluster_fake)
    make_regression_plot('../3_results/average_fakes.svg', selected_out_cluster_m, selected_out_cluster_sd, bold_font=bold_font)

if fake_samples>0:
    ### GET MANY SLOPES OF FAKE DATASETS
    slope_arr = np.zeros(fake_samples)*np.nan
    if mode=='correlation': p_val_arr = np.zeros(fake_samples)*np.nan
    for idx in tqdm(range(slope_arr.size)):
        ### create fake data
        selected_out_cluster_fake = get_fake_data(selected_out_cluster.shape[0], num_valid_rs, learned_rule_range, learned_rule_arr, rng=rng)
        if mode=='regression':
            ### get average and regression
            selected_out_cluster_m, selected_out_cluster_sd, _, _ = get_centered_selected_out_cluster_m(selected_out_cluster_fake)
            regression_results = make_regression_plot('../3_results/average_fakes.svg', selected_out_cluster_m, selected_out_cluster_sd, plot=0)
            ### get slope
            slope_arr[idx] = regression_results['a']
        elif mode=='correlation':
            ### GET CORRELATION OF fakes
            corr_coef, corr_p_val, _, _ = make_correlation_plot('../3_results/average_vps.svg', selected_out_cluster_fake, bold_font=bold_font, plot=0)
            ### get slope
            slope_arr[idx] = corr_coef
            p_val_arr[idx] = corr_p_val

    if mode=='correlation':
        plt.figure(dpi=300)
        plt.plot(slope_arr,p_val_arr,'k.')
        plt.axhline(0.05, color='k')
        plt.xlabel('correlaiton')
        plt.ylabel('p value')
        plt.savefig('../3_results/correlation_vs_pval.png')

    ### CHECK SLOPES WITH HISTOGRAM
    ### SEEMS TO BE NORMAL DISTRIBUTION --> Means + Standard Deviation
    slopes_m = np.mean(slope_arr)
    slopes_sd = np.std(slope_arr)
    x_plot = np.linspace(norm.ppf(0.001, loc=slopes_m, scale=slopes_sd),norm.ppf(0.999, loc=slopes_m, scale=slopes_sd), 100)
    y_plot = norm.pdf(x_plot, loc=slopes_m, scale=slopes_sd)
    x_fill = np.linspace(norm.ppf(0.001, loc=slopes_m, scale=slopes_sd),norm.ppf(0.05, loc=slopes_m, scale=slopes_sd), 100)
    y_fill = norm.pdf(x_fill, loc=slopes_m, scale=slopes_sd)
    
    if mode=='regression':
        vp_val=regression_results_vps['a'][0]
    if mode=='correlation':
        vp_val=corr_coef_vps


    plt.figure(figsize=(8.5/2.54,7/2.54))
    _ = plt.hist(slope_arr, bins='auto', density=True, zorder=1)
    plt.plot(x_plot, y_plot, color='orange', zorder=2)
    plt.fill_between(x_fill, y_fill, ec=None, fc='orange', alpha=0.5, zorder=2)
    x_val=vp_val
    y_val=norm.pdf(vp_val, loc=slopes_m, scale=slopes_sd)
    max_val = y_plot.max()
    max_fig = max_val*1.05
    plt.ylim(0,max_fig)
    marker_height=np.max([y_val/max_fig, 0.1])
    plt.axvline(vp_val, ymin=0, ymax=marker_height, color='r', zorder=2)
    
    p_val=norm.cdf(vp_val, loc=slopes_m, scale=slopes_sd)
    if p_val>=0.001:
        plt.text(vp_val, marker_height*max_fig+0.01*max_fig, '$p$ = .'+str(round(p_val,3)).split('.')[1], va='bottom', ha='left')
    else:
        plt.text(vp_val, marker_height*max_fig+0.01*max_fig, '$p$ < .001', va='bottom', ha='left')
    if mode=='correlation':
        plt.xlabel('Pearson correlation', **bold_font)
    if mode=='regression':
        plt.xlabel('Slope', **bold_font)
    plt.ylabel('Probability density', **bold_font)
    plt.tight_layout()
    plt.savefig('../3_results/fake_slopes_distribution.svg')
    
    with open('../3_results/values.txt', 'a') as f:
        print('p_val of vps slope:',norm.cdf(vp_val, loc=slopes_m, scale=slopes_sd), file=f)


    





















































