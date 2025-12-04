import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt


results_path = "net/save_dir/results/SZ2_eeg_hrv_results.h5"  
with h5py.File(results_path, "r") as f:
    cm = np.array(f["confusion_matrix_th02"]) 

# cm layout is [[TP_global, FP_global], [FN_global, TN_global]] in your code,
# but for a classical confusion matrix we usually display:
# [[TN, FP],
#  [FN, TP]]

TP = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TN = cm[1,1]

cm_display = np.array([[TN, FP],
                       [FN, TP]])

classes = ['Non-preictal', 'Preictal']

plt.figure(figsize=(6, 5))
plt.imshow(cm_display, interpolation = 'nearest', cmap = plt.cm.Blues)
plt.title('Confusion Matrix (th=0.2)')
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm_display.max() / 2.
for i in range(cm_display.shape[0]):
    for j in range(cm_display.shape[1]):
        plt.text(j, i, int(cm_display[i, j]),
                 horizontalalignment="center",
                 color="white" if cm_display[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()