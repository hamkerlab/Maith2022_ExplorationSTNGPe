from ANNarchy import *
import pylab as plt
import random
from timeit import default_timer as timer
import sys
import math

setup(dt=0.1)
setup(num_threads=4 )

random.seed()
np.random.seed()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--simID", help="simulation id")
args = parser.parse_args()
sim_id = int(args.simID)


#General parameters
num_stimulus              = 1
num_actions               = 5
population_size           = 100
input_rate                = 25
presentation_time         = 1000.
dopamine_rate             = 0.025e-5 #0.9e-5
t_dop                     = 60
inter_trial               = 1700
dop_decay = True


 
stn_gpe_synapse = 'fixed' #fixed,plastic
stn_gpe_mw = 0.00063 #0.00055 start mean weights

stn_gpe_pattern = 'normal'
stn_gpe_mw_list = np.array([    [stn_gpe_mw,stn_gpe_mw,stn_gpe_mw,stn_gpe_mw,stn_gpe_mw],
                [stn_gpe_mw,stn_gpe_mw,stn_gpe_mw,stn_gpe_mw,stn_gpe_mw],
                [stn_gpe_mw,stn_gpe_mw,stn_gpe_mw,stn_gpe_mw,stn_gpe_mw],
                [stn_gpe_mw,stn_gpe_mw,stn_gpe_mw,stn_gpe_mw,stn_gpe_mw],
                [stn_gpe_mw,stn_gpe_mw,stn_gpe_mw,stn_gpe_mw,stn_gpe_mw]    ])

#---------------------------------------------------------------------------------------Neuron models---------------------------------------------------------------------------------------
Hybrid_neuron = Neuron(
parameters="""

    a = 0.0   : population
    b = 0.0   : population
    c = 0.0   : population
    d = 0.0   : population
    n0 = 140. : population
    n1 = 5.0  : population
    n2 = 0.04 : population
    I = 0.0   : population
    tau_ampa = 80 : population
    tau_gaba = 80 : population
    E_ampa = 0.0 : population
    E_gaba = -90.0 : population
""",

equations="""
    dg_ampa/dt = -g_ampa/tau_ampa : init = 0
    dg_gaba/dt = -g_gaba/tau_gaba : init = 0
    dv/dt = n2*v*v+n1*v+n0 - u  + I - g_ampa*(v-E_ampa) - g_gaba*(v-E_gaba): init = -70.
    du/dt = a*(b*(v)-u) : init = -18.55
""",

spike = """
v>=30
""",

reset = """
    v = c
    u = u+d
"""

)

Striatum_neuron = Neuron(
parameters="""

    a = 0.05       : population
    b = -20.0      : population
    c = -55.0      : population
    d = 377        : population
    n0 = 61.65119  : population
    n1 = 2.594639  : population
    n2 = 0.022799  : population
    I = 0.0        : population
    tau_ampa = 80  : population
    tau_gaba = 80  : population
    E_ampa = 0.0   : population
    E_gaba = -90.0 : population
    Vr = -80.      : population
    C  = 50.       : population
""",

equations="""
    dg_ampa/dt = -g_ampa/tau_ampa : init = 0
    dg_gaba/dt = -g_gaba/tau_gaba : init = 0
    dv/dt = n2*v*v+n1*v+n0 - u/C  + I/C - g_ampa*(v-E_ampa) - g_gaba*(v-E_gaba): init = -70.
    du/dt = a*(b*(v-Vr)-u) : init = -18.55
""",

spike = """
v>=40
""",

reset = """
    v = c
    u = u+d
"""

)

Integrator_neuron = Neuron(
    parameters = """
        decision  = 0     :population
        tau       = 3000.  :population 
        threshold = 7.5    :population
        id        = -1
    """,
    equations = """
        dg_ampa/dt = -g_ampa/tau
    """,
    spike = """
        g_ampa>=threshold
    """,
    reset = """
        decision = id
    """
)


Poisson_neuron = Neuron(
    parameters = """
    rates = 0.0
    """,
    equations = """
    p = Uniform(0.0, 1.0) * 1000.0 / dt
    """,
    spike = """
    p <= rates
    """,
   reset = """
   p=0.0
   """
)

#---------------------------------------------------------------------------------------Synapse models---------------------------------------------------------------------------------------
FixedSynapse = Synapse(
parameters = """
    max_trans = 1.5 : postsynaptic
    max_weight = 1.0  : postsynaptic
    min_weight = 0.0  : postsynaptic
    
""",
equations = "",
pre_spike = """
g_target += w : max=max_trans
"""
)


IzhikevichSynapse = Synapse(
parameters = """
    tau_E    = 80     : postsynaptic
    tau_pre  = 60     : postsynaptic
    tau_post = 20     : postsynaptic
    dApre    = 0.03   : postsynaptic
    dApost   = -0.01  : postsynaptic
    factor   = 1      : postsynaptic
    tau_dop  = 50     : postsynaptic
    max_weight = 0.1  : postsynaptic
    min_weight = 0.0  : postsynaptic
    gm         = 1    : postsynaptic
    delta      = 0.5e-7 : postsynaptic
    max_E         = 0 : postsynaptic
    min_E         = -50 : postsynaptic
    Apre_max         = 1 : postsynaptic
    Apre_min         = -50 : postsynaptic
""",
equations = """
dE/dt     = -E/tau_E
dw/dt     =  E*dop*factor
dApre/dt  = -Apre/tau_pre
dApost/dt = -Apost/tau_post
ddop/dt   = -dop/tau_dop 
""",
pre_spike="""
w = clip(w -  delta,min_weight,max_weight)
g_target += gm*w
E = clip(E + Apost,min_E,max_E)
Apre = clip(Apre + dApre,Apre_min,Apre_max)
""",
post_spike ="""
w = clip(w,min_weight,max_weight)
E = clip(E + Apre,min_E,max_E)
Apost += dApost
"""

)

IzhikevichSynapse_D2 = Synapse(
parameters = """
    tau_E    = 80     : postsynaptic
    tau_pre  = 60     : postsynaptic
    tau_post = 20     : postsynaptic
    dApre    = 0.03   : postsynaptic
    dApost   = -0.01  : postsynaptic
    factor   = 1      : postsynaptic
    tau_dop  = 50     : postsynaptic
    max_weight = 0.1  : postsynaptic
    min_weight = 0.0  : postsynaptic
    gm         = 1    : postsynaptic
    delta      = 0.5e-7 : postsynaptic
    tw         = 0.001 : postsynaptic
    max_E         = 0 : postsynaptic
    min_E         = -50 : postsynaptic
""",
equations = """
dE/dt     = -E/tau_E
dw/dt     =  E*dop*factor
dApre/dt  = -Apre/tau_pre
dApost/dt = -Apost/tau_post
ddop/dt   = -dop/tau_dop
""",
pre_spike="""
w =  clip(w -  delta,min_weight,max_weight)
g_target += gm*w
E = clip(E + Apost,min_E,max_E)
Apre +=  dApre
""",
post_spike ="""
w = clip(w,min_weight,max_weight)
E = clip(E + Apre,min_E,max_E)
Apost += dApost
"""
)

IzhikevichSynapse_STN = Synapse(
parameters = """
    tau_E    = 80     : postsynaptic
    tau_pre  = 60     : postsynaptic
    tau_post = 20     : postsynaptic
    dApre    = 0.03   : postsynaptic
    dApost   = -0.01  : postsynaptic
    factor   = 1      : postsynaptic
    tau_dop  = 50     : postsynaptic
    max_weight = 0.1  : postsynaptic
    min_weight = 0.0  : postsynaptic
    gm         = 1    : postsynaptic
    delta      = 0.5e-7 : postsynaptic
""",
equations = """
dE/dt     = -E/tau_E
dw/dt     =  E*dop*factor
dApre/dt  = -Apre/tau_pre
dApost/dt = -Apost/tau_post
ddop/dt   = -dop/tau_dop 
""",
pre_spike="""
w = clip(w -  delta,min_weight,max_weight)
g_target += gm*w
E += Apost
Apre +=  dApre
""",
post_spike ="""
w = clip(w,min_weight,max_weight)
E += Apre
Apost += dApost
"""
)

STDP_online = Synapse(
parameters = """
tau_pre = 30.0 : postsynaptic
tau_post = 3.0 : postsynaptic
cApre = 1.8e-6 : postsynaptic
cApost = -2e-7 : postsynaptic
max_weight = 0.00006 : postsynaptic
min_weight = 0 : postsynaptic
delta = 6.0e-7
""",
equations = """
tau_pre * dApre/dt = - Apre
tau_post * dApost/dt = - Apost
""",
pre_spike = """
w = clip(w + Apost-delta, 0.0 , max_weight)
g_target += w
Apre += cApre
""",
post_spike = """
w = clip(w + Apre, 0.0 , max_weight)
Apost += cApost
"""
)

#dw/dt     =  if (dop>(0.05e-6)): con*E*dop*factor else: E*dop*factor
#This is my initial implementation for a learning rule for the STN-GPe connections.
#I tested this with only 1 rule switch and did what I expected.
#However, it needs to be tested with more rule switch and for this we need to fix the indirect pathway
STN_learning = Synapse(
parameters = """
    tau_dop           = 19       : postsynaptic
    min_weight        = 0.0      : postsynaptic
    max_weight        = 0.00145   : postsynaptic
    factor            = 0.06     : postsynaptic
    post_factor       = 8.0      : postsynaptic
    pre_factor        = 1.0      : postsynaptic
    dopamine_rate     = 0.025e-5 : postsynaptic
    tau_preTr         = 50.0     : postsynaptic
    tau_postTr        = 50.0     : postsynaptic
    dpreTr            = 1e-3     : postsynaptic
    dpostTr           = 1e-3     : postsynaptic
    beta              = 0.1      : postsynaptic
    tau_preTr_smooth  = 10000.0  : postsynaptic
    tau_postTr_smooth = 10000.0  : postsynaptic
""",
equations = """
    dpreTr/dt  = -preTr/tau_preTr : init = 0.75e-3
    dpostTr/dt = -postTr/tau_postTr : init = 1.8e-3

    dpreTr_smooth/dt  = (preTr-preTr_smooth)/tau_preTr_smooth : init = 0.75e-3
    dpostTr_smooth/dt = (postTr-postTr_smooth)/tau_postTr_smooth : init = 1.8e-3

    LTP = pre_factor * pos((1-beta) * preTr_smooth - preTr) / preTr_smooth + post_factor * pos((1-beta) * postTr_smooth - postTr) / postTr_smooth
    LTD = pre_factor * neg((1+beta) * preTr_smooth - preTr) / preTr_smooth + post_factor * neg((1+beta) * postTr_smooth - postTr) / postTr_smooth

    alpha = w / max_weight
    w_change = LTP * (1 - alpha) + LTD * alpha
    dw/dt = pos(dop)*factor*w_change : min=min_weight, max=max_weight
    ddop/dt   = -dop/tau_dop
""",
pre_spike = """
    preTr+=dpreTr
    w = clip(w,min_weight,max_weight)
    g_target += w
""",
post_spike = """
    postTr+=dpostTr
"""
)


#---------------------------------------------------------------------------------------Layerspp---------------------------------------------------------------------------------------
SD1      = Population(num_actions*population_size,Striatum_neuron)
SD2      = Population(num_actions*population_size,Striatum_neuron)
GPi      = Population(num_actions*population_size,Hybrid_neuron)
GPe      = Population(num_actions*(population_size),Hybrid_neuron)
STN      = Population(num_actions*population_size,Hybrid_neuron)
Thalamus = Population(num_actions*population_size,Hybrid_neuron)
Cortex   = Population(num_stimulus*population_size, Poisson_neuron) 
TI       = Population(num_actions*population_size//2,Hybrid_neuron)
ThalamusI = Population(num_actions*population_size,Hybrid_neuron)

#Noise
GPeE     = Population(num_actions*population_size, Poisson_neuron) 
GPiE     = Population(num_actions*population_size, Poisson_neuron) 
STNE     = Population(num_actions*population_size,Poisson_neuron)
SDN      = Population(num_actions*population_size,Poisson_neuron)
ThalN    = Population(num_actions*population_size,Poisson_neuron)
num_freqs = 50
max_freqs = 50 
GPeE.rates = 100
GPiE.rates = 100#700
STNE.rates = 100#500
SDN.rates = 25
ThalN.rates = 100


#Integrators
Integrators = Population(num_actions,Integrator_neuron,stop_condition=""" decision>0 """)


#GPi parameters
GPi.a = 0.005
GPi.b = 0.585
GPi.c = -65.
GPi.d = 4.0
GPi.I = 10

#GPe parameters
GPe.a = 0.005
GPe.b = 0.585
GPe.c = -65
GPe.d = 4
GPe.I = 0

#STN parameters
STN.a = 0.005
STN.b = 0.265
STN.c = -65
STN.d = 2.0
STN.I = 0.0 #6.0

#Thalamus parameters (TC)
Thalamus.a = 0.02
Thalamus.b = 0.25
Thalamus.c = -65 
Thalamus.d = 0.05
Thalamus.I = 0

#ThalamusI parameters (FS)
ThalamusI.a = 0.1
ThalamusI.b = 0.2
ThalamusI.c = -65
ThalamusI.d = 2
ThalamusI.I = 0

#TI parameters
TI.a = 0.002
TI.b = 0.25
TI.c = -65
TI.d = 0.05
TI.I = 0


#Integrator set up
Integrators.id = np.arange(1,num_actions+1)

#---------------------------------------------------------------------------------------Functions to create the connectivity patterns-------------------------------------------------------
def probabilistic_pattern(pre, post, mean_weight, sd_weight, probability):
    
    # Create a compressed sparse row (CSR) structure for the connectivity matrix
    synapses = CSR()
    # For all neurons in the post-synaptic population
    for post_rank in range(post.size):
    # Decide which pre-synaptic neurons should form synapses
        ranks = []
        for pre_rank in range(pre.size):
            pre_population  = pre_rank//population_size
            post_population = post_rank//population_size
            prob            = probability[1]
            if(pre_population==post_population):
                prob = probability[0]
            if random.random() < prob:
                ranks.append(pre_rank)
        # Create weights and delays arrays of the same size
        values = list(mean_weight  + sd_weight*np.random.normal(size=len(ranks)))
        delays = [0 for i in range(len(ranks)) ]
        # Add this information to the CSR matrix
        synapses.add(post_rank, ranks, values, delays)
    return synapses

def probabilistic_pattern_STNGPe(pre, post, mean_weight_list, sd_weight, probability):
    
    # Create a compressed sparse row (CSR) structure for the connectivity matrix
    synapses = CSR()
    # For all neurons in the post-synaptic population
    for post_rank in range(post.size):
    # Decide which pre-synaptic neurons should form synapses
        ranks = []
        values = []
        for pre_rank in range(pre.size):
            pre_population  = pre_rank//population_size
            post_population = post_rank//population_size
            prob            = probability[1]
            if(pre_population==post_population):
                prob = probability[0]
            if random.random() < prob:
                ranks.append(pre_rank)
                mean_weight=mean_weight_list[pre_population,post_population]
                values.append(mean_weight  + sd_weight*np.random.normal())
            

        # Create weights and delays arrays of the same size
        delays = [0 for i in range(len(ranks)) ]
        # Add this information to the CSR matrix
        synapses.add(post_rank, ranks, values, delays)
    
    return synapses
    

def integrator_pattern(pre, post, mean_weight, sd_weight):
    # Create a compressed sparse row (CSR) structure for the connectivity matrix
    synapses = CSR()
    # For all neurons in the post-synaptic population
    for post_rank in range(post.size):
    # Decide which pre-synaptic neurons should form synapses
        ranks = []
        for pre_rank in range(pre.size):
            pre_population  = pre_rank//population_size
            post_population = post_rank
            if(pre_population==post_population):
                ranks.append(pre_rank)
        # Create weights and delays arrays of the same size
        values = list(mean_weight  + sd_weight*np.random.normal(size=len(ranks)))
        delays = [0 for i in range(len(ranks)) ]
        # Add this information to the CSR matrix
        synapses.add(post_rank, ranks, values, delays)
    return synapses

def fixed_association_pattern(pre, post,max_weight):
    # Create a compressed sparse row (CSR) structure for the connectivity matrix
    synapses = CSR()
    # For all neurons in the post-synaptic population
    for post_rank in range(post.size):
    # Decide which pre-synaptic neurons should form synapses
        ranks = []
        values = list()
        for pre_rank in range(pre.size):
            pre_population  = pre_rank//population_size
            post_population = post_rank//population_size
            ranks.append(pre_rank)
            #if(pre_population == post_population):
            if(post_population==2):
                values.append(max_weight)
            else:
                values.append(0.0)
        # Create weights and delays arrays of the same size
        delays = [0 for i in range(len(ranks)) ]
        # Add this information to the CSR matrix
        synapses.add(post_rank, ranks, values, delays)
    return synapses

def frequency_pattern(pre, post, mean_weight):
    # Create a compressed sparse row (CSR) structure for the connectivity matrix
    synapses = CSR()
    # For all neurons in the post-synaptic population
    values = list()
    for post_rank in range(post.size):
    # Decide which pre-synaptic neurons should form synapses
        ranks = []
        for pre_rank in range(pre.size):
            pre_population  = pre_rank//num_freqs
            post_population = post_rank
            if(pre_population==post_population):
                ranks.append(pre_rank)
                values.append(mean_weight)
        # Create weights and delays arrays of the same size
        delays = [0 for i in range(len(ranks)) ]
        # Add this information to the CSR matrix
        synapses.add(post_rank, ranks, values, delays)
    return synapses

#different nuclei interconnection
def limit_exploratory_pattern(pre, post, mean_weight, sd_weight, probability):
    # Create a compressed sparse row (CSR) structure for the connectivity matrix
    synapses = CSR()
    # For all neurons in the post-synaptic population
    for post_rank in range(post.size):
    # Decide which pre-synaptic neurons should form synapses
        ranks = []
        for pre_rank in range(pre.size):
            pre_population  = pre_rank//population_size
            post_population = post_rank//population_size


            if(pre_population==0 or pre_population==4):
                #first and last population connect to all different populations.
                prob = probability[1]
                if(pre_population==post_population):
                    prob = probability[0]
                if random.random() < prob:
                    ranks.append(pre_rank)
            else:
                #the three other actions connect only between them
                prob = probability[1]
                if(pre_population==post_population):
                    prob = probability[0]
                if (post_population!=0 and post_population!=4) and random.random() < prob:
                    ranks.append(pre_rank)
        # Create weights and delays arrays of the same size
        values = list(mean_weight  + sd_weight*np.random.normal(size=len(ranks)))
        delays = [0 for i in range(len(ranks)) ]
        # Add this information to the CSR matrix
        synapses.add(post_rank, ranks, values, delays)
    return synapses


#---------------------------------------------------------------------------------------Projection definitions---------------------------------------------------------------------------------------


#Plastic cortex synapses
csd1_min_weight=0.0003#0.00035
CSD1 = Projection(
    pre = Cortex,
    post = SD1,
    target = 'ampa',
    synapse = IzhikevichSynapse
).connect_all_to_all(weights =  Normal(0.0004,0.00004), delays = 0) 

CSD2 = Projection(
    pre = Cortex,
    post = SD2,
    target = 'ampa',
    synapse = IzhikevichSynapse_D2
).connect_all_to_all(weights = 0.0, delays = 0)

CSTN = Projection(
    pre = Cortex,
    post = STN,
    target = 'ampa',
    synapse = IzhikevichSynapse_STN
).connect_all_to_all(weights = 0, delays = 0)

CT = Projection(
        pre  = Cortex,
        post = Thalamus,
    target = 'ampa',
    synapse = STDP_online
).connect_all_to_all(weights = 0, delays = 0)

#GPi STN GPe

D1GPi = Projection( 
    pre    = SD1, 
    post   = GPi, 
    target = 'gaba',
    synapse = FixedSynapse
).connect_with_func(method=probabilistic_pattern, mean_weight=0.0085, sd_weight=0, probability=[0.35,0.0]) #oli0.0085, 0.35,0.0

D2GPe = Projection( 
    pre    = SD2, 
    post   = GPe, 
    target = 'gaba',
    synapse = FixedSynapse
).connect_with_func(method=probabilistic_pattern, mean_weight=0.005, sd_weight=0.0, probability=[0.25,0.0]) #oli 0.004

GPeSTN = Projection( 
    pre    = GPe, 
    post   = STN, 
    target = 'gaba',
    synapse = FixedSynapse
).connect_with_func(method=probabilistic_pattern, mean_weight=0.007, sd_weight=0, probability=[0.35,0.05]) #0.0022


if(stn_gpe_synapse == 'fixed'):
    STNGPe = Projection( 
        pre    = STN, 
        post   = GPe, 
        target = 'ampa',
        synapse = FixedSynapse
    ).connect_with_func(method=probabilistic_pattern_STNGPe, mean_weight_list=stn_gpe_mw_list, sd_weight=0.0, probability=[0.05,0.3])
if(stn_gpe_synapse == 'plastic'):
    STNGPe = Projection( 
        pre    = STN, 
        post   = GPe, 
        target = 'ampa',
        synapse = STN_learning
    ).connect_with_func(method=probabilistic_pattern_STNGPe, mean_weight_list=stn_gpe_mw_list, sd_weight=0.0, probability=[0.05,0.3])


STNGPi = Projection( 
    pre    = STN, 
    post   = GPi, 
    target = 'ampa',
    synapse = FixedSynapse
).connect_with_func(method=probabilistic_pattern, mean_weight=0.0008, sd_weight=0, probability=[0.3,0.0]) #0.0016

GPeGPi = Projection( 
    pre    = GPe, 
    post   = GPi, 
    target = 'gaba',
    synapse = FixedSynapse
).connect_with_func(method=probabilistic_pattern, mean_weight=0.0055, sd_weight=0, probability=[0.35,0.0])  #0.008

GPiThal = Projection( 
    pre    = GPi, 
    post   = Thalamus, 
    target = 'gaba',
    synapse = FixedSynapse
).connect_with_func(method=probabilistic_pattern, mean_weight=0.0007, sd_weight=0, probability=[0.35,0.0])#0.005

#feedback synapses
ThalSD2 = Projection( 
    pre    = Thalamus, 
    post   = SD2, 
    target = 'ampa',
    synapse = FixedSynapse
).connect_with_func(method=probabilistic_pattern, mean_weight=0.001, sd_weight=0, probability=[0.4,0.0])#0.0015

ThalSD1 = Projection( 
    pre    = Thalamus, 
    post   = SD1, 
    target = 'ampa',
    synapse = FixedSynapse
).connect_with_func(method=probabilistic_pattern, mean_weight=0.001, sd_weight=0, probability=[0.4,0.0])#0.0015

#noise / tonic activation
STNESTN = Projection(
    pre  = STNE,
    post = STN,
    target = 'ampa',
    synapse = FixedSynapse
).connect_one_to_one( weights = 0.19) 

GPiEGPi = Projection(
    pre  = GPiE,
    post = GPi,
    target = 'ampa',
    synapse = FixedSynapse
).connect_one_to_one( weights = 0.11) 

GPeEGPe = Projection(
    pre  = GPeE,
    post = GPe,
    target = 'ampa',
    synapse = FixedSynapse
).connect_one_to_one( weights = 0.015 )

SDNSD1 = Projection(
    pre  = SDN,
    post = SD1,
    target = 'ampa',
    synapse = FixedSynapse
).connect_one_to_one( weights = 0.015 )

SDNSD1 = Projection(
    pre  = SDN,
    post = SD1,
    target = 'gaba',
    synapse = FixedSynapse
).connect_one_to_one( weights = 0.015 )

SDNSD2 = Projection(
    pre  = SDN,
    post = SD2,
    target = 'ampa',
    synapse = FixedSynapse
).connect_one_to_one( weights = 0.015 )

SDNSD2 = Projection(
    pre  = SDN,
    post = SD2,
    target = 'gaba',
    synapse = FixedSynapse
).connect_one_to_one( weights = 0.015 )

ThalNThal = Projection(
    pre  = ThalN,
    post = Thalamus,
    target = 'ampa',
    synapse = FixedSynapse
).connect_one_to_one( weights = 0.005 )




#local competition

SD1SD1 = Projection( 
    pre    = SD1, 
    post   = SD1, 
    target = 'gaba',
    synapse = FixedSynapse
).connect_with_func(method=probabilistic_pattern, mean_weight=0.015, sd_weight=0, probability=[0.0,0.35]) #0.035

SD1SD12 = Projection( 
    pre    = SD1, 
    post   = SD1, 
    target = 'gaba',
    synapse = FixedSynapse
).connect_with_func(method=probabilistic_pattern, mean_weight=0.00004, sd_weight=0, probability=[0.35,0.0]) #0.035

ThalThalI = Projection( #oli
    pre    = Thalamus, #oli
    post   = ThalamusI, #oli
    target = 'ampa',#oli
    synapse = FixedSynapse #oli
).connect_with_func(method=probabilistic_pattern, mean_weight=0.0025, sd_weight=0, probability=[0.35,0.0]) #oli 0.00225

ThalIThal = Projection( #oli
    pre    = ThalamusI, #oli
    post   = Thalamus, #oli
    target = 'gaba',#oli
    synapse = FixedSynapse #oli
).connect_with_func(method=probabilistic_pattern, mean_weight=0.0008, sd_weight=0, probability=[0.0,0.35]) #oli 0.0035


"""
#self excitation
STNSTN = Projection( 
    pre    = STN, 
    post   = STN, 
    target = 'ampa',
    synapse = FixedSynapse
).connect_with_func(method=probabilistic_pattern, mean_weight=0.0009, sd_weight=0, probability=[0.4,0.0])
"""
#Thalamus to integrators
ThalInt = Projection(
    pre     = Thalamus,
    post    = Integrators,
    target  = 'ampa',
    synapse = FixedSynapse
).connect_with_func(method=integrator_pattern,mean_weight = 0.02,sd_weight = 0)
ThalInt.max_trans = 1000




CSD1.factor     = 3#5.7 
CSD1.dApre      = 0.8#0.1
CSD1.dApost     = -0.1#-0.006
CSD1.tau_E      =  120#150
CSD1.tau_pre    = 300#150
CSD1.delta      = 0.000001#0.0000015
CSD1.max_weight = 0.001
CSD1.min_weight = csd1_min_weight#0.0006#0.00045
CSD1.w = np.clip(CSD1.w,CSD1.min_weight,CSD1.max_weight)
CSD1.Apre_max    = 1.5
CSD1.tau_dop    = 150
CSD1.max_E    = 2.5

CSD2.factor     = -0.9#-11 
CSD2.dApre      =  0.2#0.1
CSD2.dApost     = -0.006
CSD2.tau_E      = 1600#150
CSD2.tau_pre    =  150*3 
CSD2.gm         = 1#2
CSD2.max_weight = 0.0011
CSD2.tau_dop    = 75*4
CSD2.max_E    = 8
CSD2.delta      = 0.000005

CSTN.factor     = 700#3
CSTN.dApost     = -0.02
CSTN.max_weight = 0.0005#0.000012
CSTN.dApre      = 0.03 
CSTN.tau_E      = 90 
CSTN.tau_pre    = 30
CSTN.tau_post   = 20 
CSTN.delta = 0


#---------------------------------------------------------------------------------------Compile---------------------------------------------------------------------------------------
wie_viele_parallel=20
if(sim_id%wie_viele_parallel==0):
    nummer=wie_viele_parallel
else:
    nummer=sim_id%wie_viele_parallel

compile(directory="annarchy_sim"+str(sim_id))#+str(sim_id)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--simID", help="simulation id")
args = parser.parse_args()
sim_id = int(args.simID)


#---------------------------------------------------------------------------------------Monitors---------------------------------------------------------------------------------------
"""
m_cor  = Monitor(Cortex,'spike')

m_d1   = Monitor(SD1,'spike')
m_gpi  = Monitor(GPi,'spike')

m_d2   = Monitor(SD2,'spike')
m_gpe  = Monitor(GPe,'spike')
m_stn  = Monitor(STN,'spike')

m_thal = Monitor(Thalamus,'spike')
m_thalI = Monitor(ThalamusI,'spike')
m_int  = Monitor(Integrators,'g_ampa')

m_sd1_v=Monitor(SD1,'v')

m_thal_v=Monitor(Thalamus,'v')



#STNGPe Details Monitore#####################################################################################STNGPe Details Monitore
monitor_neurons = [50]#range(50,60)+range(150,160)+range(230,280)+range(350,360)+range(450,460)
monitors = ['preTr','LTD']

dendrite = np.array([None] * len(monitor_neurons))
m = np.array([None] * len(monitor_neurons) * len(monitors)).reshape((len(monitor_neurons),len(monitors)))

neuron_nr=0
for monitor_neuron in monitor_neurons:
    dendrite[neuron_nr] = STNGPe.dendrite(monitor_neuron)
    monitor_nr=0
    for monitor in monitors:
        m[neuron_nr,monitor_nr] = Monitor(dendrite[neuron_nr], monitor, period=1,start=False)
        monitor_nr+=1
    neuron_nr+=1


rf_counter = np.zeros((len(monitor_neurons),num_actions))

neuron_nr=0
for monitor_neuron in monitor_neurons:
    mon_neuron_rf = dendrite[neuron_nr].receptive_field(variable='tau_dop', fill=0.0)
    for i in range(num_actions*population_size):
        if(mon_neuron_rf[i]!=0.0):
            rf_counter[neuron_nr,i/population_size]+=1
    neuron_nr+=1
#STNGPe Details Monitore#####################################################################################STNGPe Details Monitore

dendriteD1 = CSD1.dendrite(150)
m1 = Monitor(dendriteD1,'Apre')
m2 = Monitor(dendriteD1,'Apost')
m3 = Monitor(dendriteD1,'E')
m4 = Monitor(dendriteD1,'dop')
m5 = Monitor(dendriteD1,'w')
"""


#---------------------------------------------------------------------------------------extra functions---------------------------------------------------------------------------------------
def positive_dopamine(num_correct):
    if(dop_decay):
        return dopamine_rate*np.exp(-num_correct/t_dop);
    else:
        return dopamine_rate
        
def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def mean_weights(synapses):
    return rebin(np.clip(np.array(synapses.w),synapses.min_weight[0],synapses.max_weight[0]),[num_actions,num_stimulus])

def mean_weights_sg(synapses,prob):

    m = np.clip(synapses.connectivity_matrix(),synapses.min_weight[0],synapses.max_weight[0])
    means = np.zeros((num_actions,num_actions))
    for i in range(num_actions):
        for j in range(num_actions):
            means[i,j] = np.mean(m[i*population_size:(i+1)*population_size,j*population_size:(j+1)*population_size])
    return means

def raw_mean_weights_sg(synapses):

    m = synapses.connectivity_matrix()
    means = np.zeros((num_actions,num_actions))
    for i in range(num_actions):
        for j in range(num_actions):
            connections = m[i*population_size:(i+1)*population_size,j*population_size:(j+1)*population_size]
            means[i,j] = np.mean(connections[connections>0])

def make_correct_list(num_blocks):
    test=[]
    for block in range(num_blocks):
    
        if(block%2==1):
            correct=2
        else:
            if( np.random.choice(2) > 0):
                correct=1
            else:
                correct=3
        if(block==num_blocks-2):
            if( np.random.choice(2) > 0):
                correct=0
            else:
                correct=4
        test.append(correct)
    return test

def weights(means, stds):

    weights=np.zeros((num_actions*population_size,population_size))
    for corNeuron in range(population_size):

        for pop in range(num_actions):

            weights[pop*population_size:(pop+1)*population_size,corNeuron]=np.random.normal(means[pop], stds[pop], population_size)
    return weights.tolist()


#---------------------------------------------------------------------------------------SIMULATION---------------------------------------------------------------------------------------
#Initial simulation to let the network run to a stable state
simulate(2000.0)

"""
#STNGPe Details Monitore starten#####################################################################################STNGPe Details Monitore starten
neuron_nr=0
for monitor_neuron in monitor_neurons:
    monitor_nr=0
    for monitor in monitors:
        m[neuron_nr,monitor_nr].start()
        monitor_nr+=1
    neuron_nr+=1
#STNGPe Details Monitore starten#####################################################################################STNGPe Details Monitore starten
"""







#prev_correct=3
#print "prev_correct = ",prev_correct+1




#CSD1.w = weights( [0.00030012,0.00031612,0.00031535,0.00098771,0.00030016], [1e-20,1e-20,1e-20,1e-20,1e-20] )
#CSD2.w = weights( [0,0,0,0,0], [1e-20,1e-20,1e-20,1e-20,1e-20] )
#CSTN.w = weights( [0.00037616,0.00034358,0.00035935,0.0004931,0.00036849], [1e-20,1e-20,1e-20,1e-20,1e-20] )
#CT.w = weights( [1.06147598e-7,8.38542006e-8,1.00262385e-7,5.46554415e-5,7.64999230e-8], [1e-20,1e-20,1e-20,1e-20,1e-20] )

num_blocks=60

num_correct = np.zeros(num_stimulus)
mw_c_sd1 = np.zeros((num_blocks*100,num_actions,num_stimulus))
mw_c_sd2 = np.zeros((num_blocks*100,num_actions,num_stimulus))
mw_c_stn = np.zeros((num_blocks*100,num_actions,num_stimulus))
mw_c_thal = np.zeros((num_blocks*100,num_actions,num_stimulus))
mw_stn_gpe = np.zeros((num_blocks*100,num_actions,num_actions))
mw_c_sd1[0]   = mean_weights(CSD1)
mw_c_sd2[0]   = mean_weights(CSD2)
mw_c_stn[0]   = mean_weights(CSTN)
mw_c_thal[0]  = mean_weights(CT)
mw_stn_gpe[0] = mean_weights_sg(STNGPe,[0.05,0.3])
initial_times  = np.zeros(num_blocks*100)
selection = np.zeros((num_blocks*100,2))
failed_blocks = np.zeros(num_blocks)
output=0

#correct_list=[1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1]
#correct_list=make_correct_list(num_blocks)
trial = 0
correct=np.random.randint(low=1,high=4)
for block in range(num_blocks):
    num_trials = 0
    prev_correct = correct
    while(prev_correct==correct):
        correct=np.random.randint(low=1,high=4)
    print("new_correct = ",correct+1,"\n")
    
    probabilities_reward = np.zeros(5)
    probabilities_reward[correct] = 1.0
    num_correct[0] = 0
    num_zeros = 0
    #this will run trials until 15 correct responses are given or for a maximum of 30 trials
    while(num_correct[0]<15 and num_trials<30 and num_zeros<3):
        
        #if(num_correct[0]==1 and block==0):
            
            #starten
            
        #if(num_correct[0]==9 and block==1):
            
            #speichern stoppen
        
        start = timer()
        st = get_time()
        stim_id = random.randint(0,num_stimulus-1)
        Cortex[stim_id*population_size:(stim_id+1)*population_size].rates = input_rate
        Integrators.decision = 0
        Integrators.g_ampa = 0    
        r = simulate_until(max_duration=presentation_time, population=Integrators)
        Cortex[stim_id*population_size:(stim_id+1)*population_size].rates  = 0
        decision = int(Integrators.decision)
        print(decision)
        dopamine_level = 0    
        if(decision>0):
            num_zeros = 0
            pr = probabilities_reward[decision-1]
            ran = random.random()
            if(ran<pr):
                dopamine_level = positive_dopamine(num_correct[stim_id])
                num_correct[stim_id] +=1
            else:
                if(decision-1 == prev_correct):
                    dopamine_level = -dopamine_rate
                else:
                    dopamine_level = -dopamine_rate
                num_correct[stim_id] = 0
        else:
            num_zeros += 1
            num_correct[stim_id] = 0
            dopamine_level = -dopamine_rate/2.



        simulate(130)#dopamin delay

        CSD1.dop = dopamine_level
        CSD2.dop = dopamine_level
        CSTN.dop = dopamine_level

        STNGPe.dop = dopamine_level
        dop_input_time = get_time()

        simulate(inter_trial)

        #This is what it saves on every trial
        #You should set it according to what you need
        #The more you save the slower the simulation is 
        mw_c_sd1[trial+1]   = mean_weights(CSD1)
        mw_c_sd2[trial+1]   = mean_weights(CSD2)
        mw_c_stn[trial+1]   = mean_weights(CSTN)
        mw_c_thal[trial+1]  = mean_weights(CT)
        mw_stn_gpe[trial+1] = mean_weights_sg(STNGPe,[0.05,0.3])
        end = timer()

        if(output==0):
            output=[[trial,st,dop_input_time,get_time(),correct+1,decision,dopamine_level,end-start]]
        else:
            output.append([trial,st,dop_input_time,get_time(),correct+1,decision,dopamine_level,end-start])



        selection[trial] = [decision,correct+1]
        #dop_lvl[trial] = dopamine_level
        num_trials+=1
        trial +=1
    
    if(num_trials==30 or num_zeros==3):
        failed_blocks[block]=1





#global
np.save("selection_sim"+str(sim_id)+".npy",selection)
np.save("failed_blocks_sim"+str(sim_id)+".npy",failed_blocks)
np.save("mw_c_sd1_sim"+str(sim_id)+".npy",mw_c_sd1)
np.save("mw_c_sd2_sim"+str(sim_id)+".npy",mw_c_sd2)
np.save("mw_c_stn_sim"+str(sim_id)+".npy",mw_c_stn)
np.save("mw_c_thal_sim"+str(sim_id)+".npy",mw_c_thal)
np.save("mw_stn_gpe_sim"+str(sim_id)+".npy",mw_stn_gpe)
#np.save("dop_lvl_sim"+str(sim_id)+".npy",dop_lvl)

output = np.array(output)
np.savetxt("output_sim"+str(sim_id),output,delimiter="\t",newline="\n",fmt=["%1d", "%1.1f", "%1.1f", "%1.1f", "%1d", "%1d", "%1e", "%1.2f"], header="Trial\tStart\tDop_Input\tEnd\tcorrect\tdecision\tdopamine\ttime\t\tstn_gpe:"+stn_gpe_pattern)



"""
# cortex spikes
spikes_cor = m_cor.get('spike')
t_cor, n_cor = m_cor.raster_plot(spikes_cor)
np.save("t_cor_sim"+str(sim_id)+".npy",t_cor)
np.save("n_cor_sim"+str(sim_id)+".npy",n_cor)

# SD1 spikes
spikes_sd1 = m_d1.get('spike')
t_sd1, n_sd1 = m_d1.raster_plot(spikes_sd1)
np.save("t_sd1_sim"+str(sim_id)+".npy",t_sd1)
np.save("n_sd1_sim"+str(sim_id)+".npy",n_sd1)
# GPi spikes
spikes_gpi = m_gpi.get('spike')
t_gpi, n_gpi = m_gpi.raster_plot(spikes_gpi)
np.save("t_gpi_sim"+str(sim_id)+".npy",t_gpi)
np.save("n_gpi_sim"+str(sim_id)+".npy",n_gpi)

# SD2 spikes
spikes_sd2 = m_d2.get('spike')
t_sd2, n_sd2 = m_d2.raster_plot(spikes_sd2)
np.save("t_sd2_sim"+str(sim_id)+".npy",t_sd2)
np.save("n_sd2_sim"+str(sim_id)+".npy",n_sd2)
# GPe spikes
spikes_gpe = m_gpe.get('spike')
t_gpe, n_gpe = m_gpe.raster_plot(spikes_gpe)
np.save("t_gpe_sim"+str(sim_id)+".npy",t_gpe)
np.save("n_gpe_sim"+str(sim_id)+".npy",n_gpe)
# STN spikes
spikes_stn = m_stn.get('spike')
t_stn, n_stn = m_stn.raster_plot(spikes_stn)
np.save("t_stn_sim"+str(sim_id)+".npy",t_stn)
np.save("n_stn_sim"+str(sim_id)+".npy",n_stn)

# Thal spikes
spikes_thal = m_thal.get('spike')
t_thal, n_thal = m_thal.raster_plot(spikes_thal)
np.save("t_thal_sim"+str(sim_id)+".npy",t_thal)
np.save("n_thal_sim"+str(sim_id)+".npy",n_thal)
# ThalI spikes
spikes_thalI = m_thalI.get('spike')
t_thalI, n_thalI = m_thalI.raster_plot(spikes_thalI)
np.save("t_thalI_sim"+str(sim_id)+".npy",t_thalI)
np.save("n_thalI_sim"+str(sim_id)+".npy",n_thalI)
#integretors
np.save("intg_sim"+str(sim_id)+".npy",m_int.get('g_ampa'))

#sd1 membran potential
sd1_v=m_sd1_v.get('v')
sd1_v=np.array([np.mean(sd1_v[:,0:100],1),np.mean(sd1_v[:,100:200],1),np.mean(sd1_v[:,200:300],1),np.mean(sd1_v[:,300:400],1),np.mean(sd1_v[:,400:500],1)])
np.save("sd1_v_sim"+str(sim_id)+".npy",sd1_v)


np.save("csd1_Apre_sim"+str(sim_id)+".npy",m1.get('Apre'))
np.save("csd1_Apost_sim"+str(sim_id)+".npy",m2.get('Apost'))
np.save("csd1_E_sim"+str(sim_id)+".npy",m3.get('E'))
np.save("csd1_dop_sim"+str(sim_id)+".npy",m4.get('dop'))
np.save("csd1_w_sim"+str(sim_id)+".npy",m5.get('w'))





#thalamus membran potential
thal_v=m_thal_v.get('v')
thal_v=np.array([np.mean(thal_v[:,0:100],1),np.mean(thal_v[:,100:200],1),np.mean(thal_v[:,200:300],1),np.mean(thal_v[:,300:400],1),np.mean(thal_v[:,400:500],1)])
np.save("thal_v_sim"+str(sim_id)+".npy",thal_v)







#STNGPe Details Monitore speichern#####################################################################################STNGPe Details Monitore speichern
neuron_nr=0
for monitor_neuron in monitor_neurons:
    np.save("rf_counter"+str(monitor_neuron)+"_sim"+str(sim_id)+".npy",rf_counter[neuron_nr,:])
    monitor_nr=0
    for monitor in monitors:
        if(monitor=='Apost' or monitor=='Apost2' or monitor=='z' or monitor=='dop' or monitor=='Rpos' or monitor=='LTP' or monitor=='postTr' or monitor=='LTP_postTr'):
            np.save(str(monitor)+"_STNGPe"+str(monitor_neuron)+"_sim"+str(sim_id)+".npy",m[neuron_nr,monitor_nr].get(monitor)[:,0])
        else:
            np.save(str(monitor)+"_STNGPe"+str(monitor_neuron)+"_sim"+str(sim_id)+".npy",m[neuron_nr,monitor_nr].get(monitor))
        monitor_nr+=1
    neuron_nr+=1
#STNGPe Details Monitore speichern#####################################################################################STNGPe Details Monitore speichern


#STNGPe Details Monitore stoppen#####################################################################################STNGPe Details Monitore stoppen
neuron_nr=0
for monitor_neuron in monitor_neurons:
    monitor_nr=0
    for monitor in monitors:
        m[neuron_nr,monitor_nr].stop()
        monitor_nr+=1
    neuron_nr+=1
#STNGPe Details Monitore stoppen#####################################################################################STNGPe Details Monitore stoppen
"""
















