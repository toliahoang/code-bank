import librosa
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import time

import pydub
import subprocess
from pydub import AudioSegment
from pydub import effects
from pydub.utils import make_chunks
import numpy as np
from scipy.signal import hilbert, chirp
import math
import scipy.integrate as integrate
import traceback
from datetime import timedelta

FRAME_SIZE = 2048
HOP_SIZE = 512
sr=16000



time_stamp = []
dict_score ={}

path = "/root/tung/laboratory/deepspeech/audio_data/"
file_path = "/root/tung/laboratory/deepspeech/audio_data"

def score_audio_df(my_audio, file_name):
    """Finding the Score of every segments"""
    chunk_length_ms = 3000  # pydub calculates in millisec
    count_timestamp = -chunk_length_ms / 1000
    mfcc_seg =[]
    mfcc_seg_1 = []
    mfcc_seg_2 = []
    print("Get the Score of every segment ...")
    myaudio = AudioSegment.from_file("{}".format(my_audio), "wav")
    # myaudio = my_audio
    chunks = make_chunks(myaudio, chunk_length_ms)
    for i_chunk, chunk in enumerate(chunks[:-1]):
        print("chunk: ", i_chunk)
        chunk_arr= chunk.get_array_of_samples()
        chunk_arr = np.asarray(chunk_arr,dtype=np.float32)
        mfccs = librosa.feature.mfcc(y=chunk_arr, n_mfcc=3, sr=sr)
        mfcc_seg.append(np.mean(mfccs[0]))
        mfcc_seg_1.append(np.mean(mfccs[1]))
        mfcc_seg_2.append(np.mean(mfccs[2]))
        count_timestamp += 3
        # time_stamp.append(time.strftime('%H:%M:%S', time.gmtime(count_timestamp)))
        time_stamp.append(str(timedelta(seconds=count_timestamp)))
    segment_data =[ti for ti in range(0,len(mfcc_seg))]
    dict_score_df=pd.DataFrame(list(zip(segment_data, time_stamp,mfcc_seg, mfcc_seg_1, mfcc_seg_2)), columns \
    =['number_of_segment','time_stamp','mfcc_score','mfcc_score_1', 'mfcc_score_2'])
    df_condition = dict_score_df['mfcc_score'].max()
    new_df = dict_score_df[dict_score_df['mfcc_score'] == df_condition]

    total_loudness_cond = dict_score_df['mfcc_score'].sum()
    print("sum_all_mfcc: ",total_loudness_cond)

    mfcc_val = new_df['mfcc_score'].tolist()
    if len(mfcc_val) > 0:
        mfcc_val_0 = mfcc_val[0]
        perceived_spread_sound = ((total_loudness_cond - mfcc_val_0)/total_loudness_cond)**2
        print("max mfcc: ", mfcc_val_0)
        print("perceived_spread_sound: ",perceived_spread_sound)
    else:
        mfcc_val_0 = 0
        perceived_spread_sound = ((total_loudness_cond - mfcc_val_0)/total_loudness_cond)**2


    time_stamp_val = new_df['time_stamp'].tolist()
    if len(time_stamp_val) > 0:
        time_val_0 = time_stamp_val[0]
        print("time stamp of max: ",time_val_0)
    else:
        time_val_0 = 0

    dict_score_df["relative_perceived_loudness"] = dict_score_df["mfcc_score"].div(total_loudness_cond)
    dict_score_df.to_csv('audio_csv/{}.csv'.format(file_name.split('.')[0]), index=0)

    return mfcc_val_0, time_val_0, perceived_spread_sound


###################################################################
###### Low level features
###################################################################

from test_audio import *
        
path = "/root/tung/laboratory/deepspeech/audio_data/"
file_path = "/root/tung/laboratory/deepspeech/audio_data"
list_file = [i for i in os.listdir(file_path) if i.endswith(".wav")]
max_mfcc_and_time = dict()
mfcc_list = []
audio_name = []
time_stamp_list = []
perceived_spread_sound_list = []
for au in list_file:
    _, _, duration = read_wave(path + au)
    print(au)
    if duration <= 2100:
        audio_name.append(au)
        mfcc_val, time_stamp_val, perceived_spread_sound = score_audio_df(path + au, au)
        mfcc_list.append(mfcc_val)
        time_stamp_list.append(time_stamp_val)
        perceived_spread_sound_list.append(perceived_spread_sound)
    else:
        continue
max_mfcc_and_time['audio_name'] = audio_name
max_mfcc_and_time['mfcc_0_max'] = mfcc_list
max_mfcc_and_time['perceived_spread_sound'] = perceived_spread_sound_list
max_mfcc_and_time['time_stamp_of_max'] = time_stamp_list
mfcc_df = pd.DataFrame(max_mfcc_and_time)
mfcc_df.to_csv("/root/tung/laboratory/deepspeech/mfcc_max.csv")





# list_file = [i for i in os.listdir(file_path) if i.endswith(".wav")]
# max_mfcc_and_time = dict()
# mfcc_list = []
# audio_name = []
# time_stamp_list = []
# for au in list_file:
#     _, _, duration = read_wave(path + au)
#     print(au)
#     if duration <= 2100:
#         audio_name.append(au)
#         score_audio_df(path + au, au)
#
#     else:
#         continue



        