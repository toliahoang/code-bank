import os
import librosa
import numpy as np
import glob

import pandas as pd
import  matplotlib.pyplot as plt
path = "audio_data"
list_file = glob.iglob(path +"/"+"*.wav")

duration_list = []
for i in list_file:
    du = librosa.get_duration(filename=i)
    if du <= 2100:
        duration_list.append(du)
    else:
        continue

duration_dict = dict()
duration_dict["duration_audio"] = duration_list
df = pd.DataFrame(duration_dict)

fig, axes = plt.subplots()
print(df["duration_audio"].values)
axes.violinplot(dataset = df["duration_audio"].values)
print("max: ", df['duration_audio'].max())
print("min: ", df['duration_audio'].min())
print("avg: ", df['duration_audio'].mean())

print(df.shape[0])

"""
axes.set_title('Audio Dataset Duration')
axes.yaxis.grid(True)
axes.set_xlabel('Audio')
axes.set_ylabel('Duration')
#
plt.show()
"""




