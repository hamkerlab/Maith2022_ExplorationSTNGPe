import numpy as np
import pylab as plt
#import matplotlib.cm as cm
import matplotlib as mtl
from matplotlib.colors import LinearSegmentedColormap
import sys
from extra_functions import get_output

def get_lasttrial(dataFolder,simID):
    f = open(dataFolder+'/output_sim'+str(simID), 'r')
    length=10
    lasttrial=0
    while length>1:
        zeile=f.readline()
        length=len(zeile)
        lasttrial+=1
    lasttrial-=3
    return lasttrial

minSTD=1
for useSIM in [2]:
    ###################################################################################
    ##############################   PARAMETERS   #####################################
    ###################################################################################
    ### Loading
    dataFolder = {}
    dataFolder["shortSimulation"]   = "../../../simulations/007a_plotSimulation_learningrule_PaperLearningRule/2_dataRaw"
    dataFolder["wholeSimulation"]   = "../../../simulations/001e_Cluster_Experiment_PaperLearningRule_LearningON/4_dataEv"  
    SimIdx = {}
    SimIdx["wholeSimulation"]       = useSIM
    SimIdx["shortSimulation"]       = 11
    initialWeights                  = 0.00063
    dwMax                           = 0.00145
    ### Calculations
    trialIdx = {} # over which trials the mw changes should be calculated, took from output file
    trialIdx["2_before"]                = 49
    trialIdx["2_after"]                 = 59
    trialIdx["3_before"]                = 8
    trialIdx["3_after"]                 = 18
    trialIdx["4_before"]                = 28
    trialIdx["4_after"]                 = 38
    trialIdx["wholeSimulation_before"]  = 0
    trialIdx["wholeSimulation_after"]   = get_lasttrial(dataFolder["wholeSimulation"],SimIdx["wholeSimulation"])
    activityTime = {} # monitor start end, after selection start end, to compare activity after selection with global activitie
    activityTime["start_global"]  = 13111.8
    activityTime["end_global"]    = 127502.5
    activityTime["delta_global"]  = activityTime["end_global"] - activityTime["start_global"]
    activityTime["start_afterRS"] = 125802.5 # about when dopamine starts
    activityTime["delta_afterRS"] = 300
    activityTime["end_afterRS"]   = activityTime["start_afterRS"] + activityTime["delta_afterRS"] # about when dopamine ends
    subPopsSize                   = 100
    ### Figure
    fig = {}
    fig["B"]                        = 180 # mm
    fig["H"]                        = 200 # mm
    colorbarTickPosition        = 0.7 # 1 would be at the top/bottom of colorbar
    colorbarH_weights           = 0.35
    colorbarH_activities        = 0.2
    font = {}
    font["colorBarTicks"]   = {'labelsize': 9}
    font["axLabel"]         = {'fontsize': 9, 'fontweight' : 'normal'}
    font["axTicks"]         = {'labelsize': 9}
    font["subplotLabels"]   = {'fontsize': 14, 'fontweight' : 'bold'}
    font["STNGPeLabels"]    = {'fontsize': 9, 'fontweight' : 'bold'}
    font["colorBarLabels"]  = {'fontsize': 9, 'fontweight' : 'bold'}
    font["legend"]          = {'fontsize': 9, 'fontweight' : 'normal'}
    subplotLabels           = ["B", "C", "D", "E"]
    cmap_activities         = 'cool'#'winter'
    circles = {}
    circles["angles"]     = np.linspace(0, 2 * np.pi, 100) # angles for circles
    circles["radius"]     = 0.2 # radius for circles
    circles["gridxMax"]   = 1
    circles["gridxMin"]   = -circles["gridxMax"]
    circles["gridyMax"]   = 0.5
    circles["gridyMin"]   = -circles["gridyMax"]
    circles["gab"]        = 3.6   
    circles["randomDots"] = np.array([np.array([0.25,0.25,0.25,0.6,0.6,0.6,0.83,0.83,0.83,0.83])*circles["radius"], np.array([0,0.4,0.8,0.2,0.7,0.9,0.1,0.4,0.6,0.8])*2*np.pi])
    rectangle = {}
    rectangle["freespace"] = 0.1
    rectangle["x0"]        = circles["gridxMin"] - circles["radius"] - rectangle["freespace"]
    rectangle["x1"]        = circles["gridxMax"] + circles["radius"] + rectangle["freespace"]
    rectangle["y0"]        = circles["gridyMin"] - circles["radius"] - rectangle["freespace"]
    rectangle["y1"]        = circles["gridyMin"] + circles["radius"] + rectangle["freespace"]  
    activitiesPlotBorder   = 0.05 # axis scale          
    legend = {}
    legend["freespace"]   = 0.05
    legend["width"]       = 1.5
    legend["height"]      = 0.2
    legend["bottom"]      = -1.3 # relative to subplot A coordinates
    legend["arrowlength"] = 0.2
    
    cmap_dw = LinearSegmentedColormap.from_list("mycmap", ['red','white','lime'])
    cmap_activities = LinearSegmentedColormap.from_list("mycmap", ['blue','white','orange'])




    ###################################################################################
    ###############################   LOAD DATA   #####################################
    ###################################################################################
    mw = {}
    mw["wholeSimulation"]                  = np.load(dataFolder["wholeSimulation"]+"/mw_stn_gpe_sim"+str(SimIdx["wholeSimulation"])+".npy")
    mw["shortSimulation"]                  = np.load(dataFolder["shortSimulation"]+"/mw_stn_gpe_sim"+str(SimIdx["shortSimulation"])+".npy")
    mw["wholeSimulationTransformation"]    = mw["wholeSimulation"][0]/initialWeights
    mw["shortSimulationTransformation"]    = mw["shortSimulation"][0]/initialWeights
    spikeTimes = {}
    spikeTimes["GPe"]       = np.load(dataFolder["shortSimulation"]+"/t_gpe_sim"+str(SimIdx["shortSimulation"])+".npy")
    spikeTimes["STN"]       = np.load(dataFolder["shortSimulation"]+"/t_stn_sim"+str(SimIdx["shortSimulation"])+".npy")
    spikeCounts = {}
    spikeCounts["GPe"]      = np.load(dataFolder["shortSimulation"]+"/n_gpe_sim"+str(SimIdx["shortSimulation"])+".npy")
    spikeCounts["STN"]      = np.load(dataFolder["shortSimulation"]+"/n_stn_sim"+str(SimIdx["shortSimulation"])+".npy")


    ###################################################################################
    #########################   CALCULATE PLOTTING DATA   #############################
    ###################################################################################
    ### Meanweight changes of 10 trials
    mwChanges = {}
    for rewardedAction in ["2","3","4"]:
        mwChanges[rewardedAction] = (mw["shortSimulation"][trialIdx[rewardedAction+"_after"]]/mw["shortSimulationTransformation"] - mw["shortSimulation"][trialIdx[rewardedAction+"_before"]]/mw["shortSimulationTransformation"]).transpose()/dwMax

    ### Meanweight changes of whole simulation
    mwChanges["wholeSimulation"] = (mw["wholeSimulation"][trialIdx["wholeSimulation_after"]]/mw["wholeSimulationTransformation"] - mw["wholeSimulation"][trialIdx["wholeSimulation_before"]]/mw["wholeSimulationTransformation"]).transpose()/dwMax

    if mwChanges["wholeSimulation"].std()<minSTD:
        minSTD=mwChanges["wholeSimulation"].std()
        print(useSIM)

    ### Min max changes for color symmetrical code
    mw_max = np.amax([np.amax(np.abs(changes)) for changes in list(mwChanges.values())])
    mw_min = -mw_max
    print(mw_max)
    
    
    ### activitities
    activities = {}
    ### get the time points of the dopamine inputs, where action 2 was selected
    trials, correct, decision, frequentAction, start, dopInp = get_output(dataFolder["shortSimulation"],SimIdx["shortSimulation"])
    starting_trial = 0 # which trial from the period where 2 was correctly selected should be the first trial (ignore trials directly after RS)
    selected_action = 3 # which response was selected in the trials (comments here all say 2 but it can be adjusted here)
    after_selection_times = dopInp[(decision==selected_action)*(correct==selected_action)][starting_trial:] # all times of dop inputs where action 2 was correctly selected
    print('trials:',after_selection_times.shape[0])
    ### + 300 ms --> activity after selection of action 2
    ### obtain all spikes done in this time periods and divide it by the summed up time period
    timeAfterSelection=300#ms
    activities['afterRS'] = np.zeros((5,2))
    for popIdx, popName in enumerate(["GPe","STN"]):
        allTimes = np.array([spikeTimes[popName] for i in range(after_selection_times.shape[0])]) # duplicate spike times array, to compare it with after_selection_times
        spike_time_windows = np.sum((allTimes>after_selection_times[:,None]) * (allTimes<after_selection_times[:,None]+timeAfterSelection),0) # get for each trial the spiketimes after selection
        spikeCountsMasked = spikeCounts[popName][np.array(spike_time_windows, dtype=bool)] # all spikes which occured in all this time (after selection periods of all trials)
        for subPop in range(5): # deviding spikes in 5 sub populations
            activities['afterRS'][subPop,popIdx] = np.sum((spikeCountsMasked>=100*subPop) * (spikeCountsMasked<100*(subPop+1))) / (timeAfterSelection*after_selection_times.shape[0]/1000.) / subPopsSize
    ### then obtain activity over complete simulation time
    global_time = np.array([start[(decision==selected_action)*(correct==selected_action)][starting_trial],after_selection_times[-1]]) # the complete time range beginning to end of aciton 2 was correctly selected
    complete_simulation_time = global_time[1]-global_time[0]
    activities['global'] = np.zeros((5,2))
    for popIdx, popName in enumerate(["GPe","STN"]):
        mask = (spikeTimes[popName] > global_time[0]) * (spikeTimes[popName] < global_time[1])
        spikeCountsMasked = spikeCounts[popName][np.array(mask, dtype=bool)]  
        for subPop in range(5): # deviding spikes in 5 sub populations
            activities['global'][subPop,popIdx] = np.sum((spikeCountsMasked>=100*subPop) * (spikeCountsMasked<100*(subPop+1))) / (complete_simulation_time/1000.) / subPopsSize
    ### compare both activity vectors
    activities["afterRS"] = (activities["afterRS"] - activities["global"]) / activities["global"]
    print(activities["afterRS"])
    ### rescale activities for colormap
    activities["afterRS"] = (activities["afterRS"] - (-1)) / (1 - (-1))
    print(activities["afterRS"])



    ###################################################################################
    ##################################   PLOTS   ######################################
    ###################################################################################
    fg = plt.figure(figsize=(fig["B"]/25.4, fig["H"]/25.4), dpi = 300)
    plt.subplots_adjust(bottom = 0.05, top = 0.95, right = 0.8, hspace = 0.4, wspace = 0.6)

    ### activities after trial plot
    #cmap = plt.cm.get_cmap(cmap_activities)
    cmap = cmap_activities


    ax_A = plt.subplot(3,1,1)
    ax_A.axis('equal')
    ax_A.axis('off')
    plt.text(ax_A.get_position().x0-0.1, (ax_A.get_position().y0 + ax_A.get_position().y1) / 2., "A:", transform = plt.gcf().transFigure, **font["subplotLabels"])

    ### draw circles
    gridX = np.linspace(circles["gridxMin"], circles["gridxMax"], 5)
    grid  = np.concatenate((gridX,gridX,np.ones(gridX.shape[0])*circles["gridyMax"],np.ones(gridX.shape[0])*circles["gridyMin"])).reshape((2,2*gridX.shape[0])).T
    colorList = cmap(activities["afterRS"].T.reshape((10,)))
    print(activities["afterRS"].T.reshape((10,)))
    for idx,point in enumerate(grid): # left
        plt.plot(point[0] - circles["gab"]/2. + circles["radius"] * np.cos(circles["angles"]), point[1] + circles["radius"] * np.sin(circles["angles"]), color = colorList[idx])
        plt.plot(point[0] - circles["gab"]/2. + circles["randomDots"][0] * np.cos(circles["randomDots"][1]), point[1] + circles["randomDots"][0] * np.sin(circles["randomDots"][1]), '.', color = colorList[idx])
    for idx,point in enumerate(grid): # right
        plt.plot(point[0] + circles["gab"]/2. + circles["radius"] * np.cos(circles["angles"]), point[1] + circles["radius"] * np.sin(circles["angles"]), color = colorList[idx])
        plt.plot(point[0] + circles["gab"]/2. + circles["randomDots"][0] * np.cos(circles["randomDots"][1]), point[1] + circles["randomDots"][0] * np.sin(circles["randomDots"][1]), '.', color = colorList[idx])

    ### draw 4 rectangles
    for rec in range(4):
        offsetX = [circles["gab"]/2., -circles["gab"]/2.][rec//2]
        offsetY = [0, circles["gridyMax"] - circles["gridyMin"]][rec%2]
        plt.plot(
        [rectangle["x0"] + offsetX, rectangle["x1"] + offsetX, rectangle["x1"] + offsetX, rectangle["x0"] + offsetX, rectangle["x0"] + offsetX], 
        [rectangle["y0"] + offsetY, rectangle["y0"] + offsetY, rectangle["y1"] + offsetY, rectangle["y1"] + offsetY, rectangle["y0"] + offsetY], color = 'k')

    ### write STN, GPe
    plt.text(0, circles["gridyMax"], 'GPe', va = 'center', ha = 'center', **font["STNGPeLabels"])
    plt.text(0, circles["gridyMin"], 'STN', va = 'center', ha = 'center', **font["STNGPeLabels"])

    ### draw arrows
    def sizedArrow(x0, y0, x1, y1, width, color, style):
        if style == 'LTP':
            sizedArrowStyle = mtl.patches.ArrowStyle.Simple(tail_width=1*width,head_width=3*width,head_length=5*width)
            arrow = mtl.patches.FancyArrowPatch((x0,y0), (x1,y1), connectionstyle='arc3,rad=0', shrinkA = 1, shrinkB = 1,clip_on=False, **dict(arrowstyle=sizedArrowStyle, linewidth=0.5+0.5*width,color=color))
            plt.gca().add_patch(arrow)
        elif style == 'LTD':
            xHead = x1
            yHead = y1 - (width/30.) * 1.4
            plt.gca().add_patch(mtl.patches.Circle((xHead,yHead), width/30., fill = True, color = color,clip_on=False))
            sizedArrowStyle = mtl.patches.ArrowStyle.Simple(tail_width=1*width,head_width=1*width,head_length=0.01*width)
            arrow = mtl.patches.FancyArrowPatch((x0,y0), (xHead,yHead), connectionstyle='arc3,rad=0', shrinkA = 1, shrinkB = 2,clip_on=False, **dict(arrowstyle=sizedArrowStyle, linewidth=0.5+0.5*width,color=color))
            plt.gca().add_patch(arrow)
        elif style == 'LTD_legend':
            xHead = x1 - (width/30.)
            yHead = y1
            plt.gca().add_patch(mtl.patches.Circle((xHead,yHead), width/30., fill = True, color = color,clip_on=False))
            sizedArrowStyle = mtl.patches.ArrowStyle.Simple(tail_width=1*width,head_width=1*width,head_length=0.01*width)
            arrow = mtl.patches.FancyArrowPatch((x0,y0), (xHead,yHead), connectionstyle='arc3,rad=0', shrinkA = 1, shrinkB = 2,clip_on=False, **dict(arrowstyle=sizedArrowStyle, linewidth=0.5+0.5*width,color=color))
            plt.gca().add_patch(arrow)

    for idx,point in enumerate(grid[5:10]): # left / LTP
        x0 = point[0] - circles["gab"]/2.
        y0 = point[1] + circles["radius"]
        x1List = grid[2,0] - circles["gab"]/2. + circles["radius"] * np.cos(circles["angles"])
        y1List = grid[2,1] + circles["radius"] * np.sin(circles["angles"])
        minDistancePoint = np.argmin(np.sqrt((x1List - x0)**2 + (y1List - y0)**2))
        x1 = x1List[minDistancePoint]
        y1 = y1List[minDistancePoint]
        sizedArrow(x0,y0,x1,y1,1,'k','LTP')
    for idx,point in enumerate(grid[:5]): # right / LTD
        x1 = point[0] + circles["gab"]/2.
        y1 = point[1] - circles["radius"]
        x0List = grid[7,0] + circles["gab"]/2. + circles["radius"] * np.cos(circles["angles"])
        y0List = grid[7,1] + circles["radius"] * np.sin(circles["angles"])
        minDistancePoint = np.argmin(np.sqrt((x0List - x1)**2 + (y0List - y1)**2))
        x0 = x0List[minDistancePoint]
        y0 = y0List[minDistancePoint]
        sizedArrow(x0,y0,x1,y1,1,'k','LTD')

    ### legend
    xLeft = -legend["width"]/2.
    legendField=mtl.patches.FancyBboxPatch(xy=(xLeft,legend["bottom"]),width=legend["width"],height=legend["height"],boxstyle=mtl.patches.BoxStyle.Round(pad=0.1),bbox_transmuter=None,mutation_scale=1,mutation_aspect=None,**dict(linewidth=2, fc='w',ec='k',clip_on=False))
    plt.gca().add_patch(legendField)
    # LTP arrow
    (x0, y0, x1, y1, textdist) = (-legend["width"]/2. + legend["freespace"], legend["bottom"] + legend["height"]/2., -legend["width"]/2. + legend["freespace"] + legend["arrowlength"], legend["bottom"] + legend["height"]/2., legend["freespace"])
    sizedArrow(x0,y0,x1,y1,1,'k','LTP')
    plt.text(x1+textdist,y1,'LTP',ha='left',va='center', **font["legend"])
    # LTD arrow
    (x0, y0, x1, y1, textdist) = (legend["freespace"], legend["bottom"] + legend["height"]/2., legend["freespace"] + legend["arrowlength"], legend["bottom"] + legend["height"]/2., legend["freespace"])
    sizedArrow(x0,y0,x1,y1,1,'k','LTD_legend')
    plt.text(x1+textdist,y1,'LTD',ha='left',va='center', **font["legend"])

    plt.xlim(rectangle["x0"] - circles["gab"]/2. - activitiesPlotBorder,rectangle["x1"] + circles["gab"]/2. + activitiesPlotBorder)
    plt.ylim(rectangle["y0"], rectangle["y1"] + circles["gridyMax"] - circles["gridyMin"])


    ### four weightchange plots
    axes = {}
    for idx, weightchange in enumerate(["2","3","4","wholeSimulation"]):
        ax = plt.subplot(3,2,idx+3)
        image = ax.matshow(mwChanges[weightchange], cmap=cmap_dw, vmin=mw_min, vmax=mw_max)
        ax.set_xlabel('GPe', **font["axLabel"])    
        ax.xaxis.set_label_position('top')
        ax.xaxis.set_ticks_position('top')
        ax.set_ylabel('STN', **font["axLabel"])
        ax.tick_params(axis='both', which='both', **font["axTicks"])
        #ax.set_xticklabels([0,1,2,3,4,5])
        #ax.set_yticklabels([0,1,2,3,4,5])
        plt.xticks([0,1,2,3,4],[1,2,3,4,5])
        plt.yticks([0,1,2,3,4],[1,2,3,4,5])
        axes[weightchange] = ax
        plt.text(ax.get_position().x0-0.1, (ax.get_position().y0 + ax.get_position().y1) / 2., subplotLabels[idx], transform = plt.gcf().transFigure, **font["subplotLabels"])
      

    ### colorbar weights
    cax_weights = plt.axes([0.85, (axes["2"].get_position().y1 - axes["4"].get_position().y0 - colorbarH_weights) / 2. + axes["4"].get_position().y0, 0.03, colorbarH_weights])
    cbar_weights = plt.colorbar(image, cax = cax_weights)
    cax_weights.tick_params(axis='both', which='both', **font["colorBarTicks"])
    cax_weights.set_ylabel('Mean weight changes [%]', **font["colorBarLabels"])
    #cbar_weights.formatter.set_powerlimits((0, 0))
    plt.yticks(cax_weights.get_yticks(),np.round(np.array(cax_weights.get_yticks())*100,0))
    #cbar_weights.update_ticks()
    #print(cax_weights.get_yticks())

    ### colorbar activities
    cax_activities = plt.axes([0.85, (ax_A.get_position().y1 - ax_A.get_position().y0 - colorbarH_activities) / 2. + ax_A.get_position().y0, 0.03, colorbarH_activities])
    plt.colorbar(mtl.cm.ScalarMappable(norm=mtl.colors.Normalize(vmin=-1, vmax=1), cmap=cmap), cax=cax_activities)
    cax_activities.tick_params(axis='both', which='both', **font["colorBarTicks"])
    cax_activities.set_ylabel('Normalized mean firing rate', **font["colorBarLabels"])

    plt.savefig("../3_results/activities_and_weightchanges.svg")


















































