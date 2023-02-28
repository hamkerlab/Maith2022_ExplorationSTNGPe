### classic model predictions and "failed" rare experiment

### never-experiment learning model (classic dopa)
# regression format, learning_on, 001e, without reps, absolute exploration, stn_gpe_factor_idx=3
python make_manuscript_simulations_results.py 1 1 0 1 1 3

### rare-experiment learning model (classic dopa)
# regression format, learning_on, 001f, without reps, absolute exploration, stn_gpe_factor_idx=3
python make_manuscript_simulations_results.py 1 1 1 1 1 3

### never-experiment fixed model (classic dopa)
# regression format, learning_off, 001e(002e), without reps, absolute exploration, stn_gpe_factor_idx=3
python make_manuscript_simulations_results.py 1 0 0 1 1 3


### new model predictions for everything

### never-experiment learning model (new dopa)
# regression format, learning_on, 014a, without reps, absolute exploration, stn_gpe_factor_idx=3
python make_manuscript_simulations_results.py 1 1 3 1 1 3

### rare-experiment learning model (new dopa)
# regression format, learning_on, 014b, without reps, absolute exploration, stn_gpe_factor_idx=3
python make_manuscript_simulations_results.py 1 1 4 1 1 3

### never-experiment fixed model (new dopa)
# regression format, learning_off, 014a(014d), without reps, absolute exploration, stn_gpe_factor_idx=3
python make_manuscript_simulations_results.py 1 0 3 1 1 3

### all-experiment learning model (new dopa)
# regression format, learning_on, 014c, without reps, absolute exploration, stn_gpe_factor_idx=3
python make_manuscript_simulations_results.py 1 1 5 1 1 3
