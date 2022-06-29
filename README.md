# Maith2022_ExplorationSTNGPe

Source code of simulations and analyses from Maith, O., Baladron, J., Einhäuser, W., & Hamker, F. H. (2022). Human exploration behaviour in a novel reversal learning task explained by a basal ganglia model with adaptive STN-GPe connections. Submitted to *iScience*.

## Authors:

* Oliver Maith (oliver.maith@informatik.tu-chemnitz.de)
* Javier Baladron (javier.baladron-pezoa@informatik.tu-chemnitz.de)

## Using the Scripts

### Files

- **analyses**/
  - folder for analyses of experimental and simulated data
  - analyses scripts can be found in **subfolder/1_srcAna/**
  - results are saved in **subfolder/3_results/**
  - for some analyses intermediate data are saved in **subfolder/2_dataEv/** for further evaluations
- **psychExp**/
  - placeholder directory structure for experimental data (available at https://doi.org/10.5281/zenodo.6546572)
- **simulations**/
  - includes the scripts to run the simulations with the model to generate all analyzed data
  - script to run a single simulation (*parallel.py*) can be found in **subfolder/1_srcSim/**
  - the shell scripts (*run.sh*) help to run multiple simulations in parallel
  - data are saved in **subfolder/2_dataRaw/** and **subfolder/4_dataEv/**, data generated for the study available at https://doi.org/10.5281/zenodo.6546572

### Results Pipelines

Description of the results generated in Python. For several figures, additional image processing software was used to create the final figures (to adjust the layout and/or colors).

Results | analysis | experimental data | simulated data | simulations | comment
-|-|-|-|-|-
Figure 3 | **manuscript_global_performance**/ and **manuscript_global_performance_vps**/ | yes | yes | 60 simulations from **001e...**/ | arguments for function `plot_column`: `mode='scatter'` and `classic_plot=1` (only in **manuscript_global_performance**/)
Figure 4 | **manuscript_SRtask_results**/ | yes | yes | 60 simulations from **001e...**/ and **002e...**/ each |
Figure 5 | **manuscript_vp_learning_details**/ | yes | no | - |
Figure 6 | **manuscript_Figure_activities_and_weightchanges**/ | no | yes | a single simulation from **001e...**/ and **007a...**/ | various simulation times are set manually in the analysis (taken from the output file of the simulation), a change of the simulation used requires an adjustment of these times
Figure S1 | **manuscript_global_performance**/ | no | yes | 60 simulations from **001e...**/ | arguments for function `plot_column`: `classic_plot=0`
Figure S2 | **manuscript_vp_learning_details**/ | yes | no | - |

All statistical tests of the study are performed in **analyses/manuscript_statistics**/ which requires the prior performance of the analyses: **manuscript_SRtask_results**/, **manuscript_global_performance**/ and **manuscript_global_performance_vps**/.

# Platforms

* GNU/Linux

# Dependencies (given versions used in study)

* Python >= 3.7.4
* ANNarchy >= 4.7.0
* matplotlib >= 3.4.1
* numpy >= 1.21.3
* pandas >= 1.3.5
* pingouin >= 0.5.1
* scikit-learn >= 0.22.2.post1
* scipy >= 1.7.1
* tqdm >= 4.36.1
