import h5py
import numpy as np
import matplotlib.pyplot as plt

results_file = 'net/save_dir/results/model_results.h5'

with h5py.File(results_file, 'r') as f:
    # Load datasets
    sens = np.array(f['sens_preictal']) 
    fa = np.array(f['fa_per_hour'])  
    thresholds = np.array(f['thresholds'])

# Average over recordings (ignoring NaNs)
mean_sens = np.nanmean(sens, axis = 0)
mean_fa = np.nanmean(fa, axis = 0)

# Plot confusion matrix
plt.figure(figsize = (6,4))
plt.plot(mean_fa, mean_sens, marker = 'o')
plt.xlabel('False alarms / hour')
plt.ylabel('Pre-ictal sensitivity')
plt.title('Pre-ictal alarm trade-off')
plt.grid(True)
plt.tight_layout()