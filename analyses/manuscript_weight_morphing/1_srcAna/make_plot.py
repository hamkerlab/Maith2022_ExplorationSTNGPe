from extra_functions import (
    get_output,
    initial_visualization,
    make_correlation_plot,
)
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

font = {"family": "Arial", "weight": "normal", "size": 8}
bold_font = {"family": "Arial", "weight": "bold", "size": 8}
large_bold_font = {"family": "Arial", "weight": "bold", "size": 10}
plt.rc("font", **font)

bold_font = {"family": "Arial", "weight": "bold", "size": 8}
folder = "../../../simulations/013_weight_morphing/2_dataRaw/data_hubel_6400sims"
num_sims = 6400
morph_selections = np.zeros((num_sims, 3))
for sim_id in tqdm(range(num_sims)):
    ### GET DATA OF TRIALS
    ### last decision = first exploration selection
    ### morph = percent change of weights
    _, _, decision, _, _, _, morph = get_output(folder, sim_id + 1)
    morph_selections[sim_id, 0] = morph[-1]
    morph_selections[sim_id, 1] = decision[-1]

    ### get for each morph value if selection is in or out cluster (1 or 0)
    morph_selections[sim_id, 2] = [None, 0, 1, 1, 1, 0][int(decision[-1])]

### initial visualization
initial_visualization(morph_selections[:, :2])


### get correlation for in vs out cluster over weight morph
### generate figure scatter plot like with fitted line
corr_coef_vps, corr_p_val_vps, df_vps, CI = make_correlation_plot(
    "../3_results/correlation_bars.svg",
    morph_selections,
    bold_font=bold_font,
    mode="bars",
)
corr_coef_vps, corr_p_val_vps, df_vps, CI = make_correlation_plot(
    "../3_results/correlation_inlcuster.svg",
    morph_selections,
    bold_font=bold_font,
    mode="incluster",
)
with open("../3_results/values.txt", "w") as f:
    print(
        "correlations results: r =",
        corr_coef_vps,
        " p =",
        corr_p_val_vps,
        " df =",
        df_vps,
        " 95%CI =",
        CI.round(2),
        file=f,
    )
