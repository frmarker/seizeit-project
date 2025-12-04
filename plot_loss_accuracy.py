import os
import pandas as pd
import matplotlib.pyplot as plt

# Path to your training history
history_csv = "net/save_dir/models/SZ2_ChronoNet_frame-2_mode-eeg_hrv_eeg_hrv/History/SZ2_ChronoNet_frame-2_mode-eeg_hrv_eeg_hrv.csv"

df = pd.read_csv(history_csv)

# Plot Loss
plt.figure(figsize=(12,6))
plt.plot(df['loss'], label='Training Loss')
plt.plot(df['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot Accuracy
if 'accuracy' in df.columns:
    plt.figure(figsize = (12,6))
    plt.plot(df['accuracy'], label = 'Training Accuracy')
    plt.plot(df['val_accuracy'], label = 'Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot AUC
if 'auc' in df.columns:
    plt.figure(figsize = (12,6))
    plt.plot(df['auc'], label = 'Training AUC')
    plt.plot(df['val_auc'], label = 'Validation AUC')
    plt.title('AUC Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)
    plt.show()
