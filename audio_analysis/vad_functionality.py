from scipy.io import wavfile
import struct
import numpy as np
from pydub import AudioSegment
from pydub import effects
from pydub.utils import make_chunks
import datetime
import time
import pandas as pd
# import mysql.connector as mysql
import pymysql
# from sqlalchemy import create_engine
# from mysql.connector import Error
# import mysql
# import mysql.connector
import json
# import sqlalchemy
import torch
from test_audio import *
import os
def get_hh_mm_ss(count_time):
    # return time.strftime('%H:%M:%S', time.gmtime(count_time))
    return datetime.datetime.fromtimestamp(count_time/1000).strftime('%H:%M:%S')


def merge_time(in_df):
    """Merge overlapped moments"""
    try:
        converted_df = in_df
        converted_df.sort_values("start", inplace=True)
        converted_df["group"]=(converted_df["start"]>converted_df["stop"].shift().cummax()).cumsum()
        # ## this returns min value of "START" column from a group and max value fro m "FINISH"
        result=converted_df.groupby(["group"]).agg({"start":"min", "stop": "max"})
        merged_inner = pd.merge(left=result, right=converted_df, left_on='group', right_on='group')
        merged_inner.drop_duplicates(subset = 'group',keep='first',inplace=True)
        new_merged_inner = merged_inner
        del new_merged_inner['start_y']
        del new_merged_inner['stop_y']
        new_merged_inner['start'] = new_merged_inner['start_x']
        new_merged_inner['stop'] = new_merged_inner['stop_x']
        del new_merged_inner['start_x']
        del new_merged_inner['stop_x']
        new_merged_inner.reset_index(drop=True,inplace=True)
        return new_merged_inner
    except:
        print("error with merge_time")


def get_video_id_from_url(url):
    # match youtube
    url = url.strip()
    match = re.match(r'https://(www)?.youtube.com/watch\?v=(?P<id>.+?)/?$', url)
    if match:
        video_id = match.group('id')
        return video_id

    if url[-1] == "/":
        video_id = url.split("/")[-2]
    else:
        video_id = url.split("/")[-1]
    return video_id


model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',model='silero_vad',force_reload=True)

def human_speech_func(train_audio_path,video_id_param, model, utils):

    """Silero vad model"""
    # try:
    torch.set_num_threads(1)
    # model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',model='silero_vad',force_reload=True)
    # (get_speech_ts, get_speech_ts_adaptive,save_audio,read_audio, state_generator, single_audio_stream, collect_chunks) = utils
    (get_speech_ts,
     _, read_audio,
     *_) = utils
    files_dir = torch.hub.get_dir() + '/snakers4_silero-vad_master/files'
    wav = read_audio('{}'.format(train_audio_path))
    # full audio
    # get speech timestamps from full audio file
    # speech_timestamps = get_speech_ts(wav, model, num_steps=4)
    speech_timestamps = get_speech_ts(wav, model)
    length_speech_chunk = [i['end'] - i['start'] for i in speech_timestamps]
    speech_sample_duration = sum(length_speech_chunk)
    duration_human_voice_sec = [speech_sample_duration / 16000]
    total_video_duration = len(wav)
    human_speech_score = [np.float32(speech_sample_duration / total_video_duration)]
    speech_timestamps_start = [i['start'] for i in speech_timestamps]
    list_start_in_sec = [start_time / 16000 for start_time in speech_timestamps_start]
    conversion_start_hmmss = [str(datetime.timedelta(seconds=sec_input)) for sec_input in list_start_in_sec]
    string_from_list_start_stop_new = '_'.join(conversion_start_hmmss)
    video_id = [video_id_param]
    audio_df = pd.DataFrame(list(zip(video_id, human_speech_score, [string_from_list_start_stop_new], duration_human_voice_sec)),columns=['video_id', 'human_speech_score', 'human_speech_timestamp', 'duration_human_voice_sec'])
    audio_df['human_speech_timestamp'] = audio_df['human_speech_timestamp'].astype(str)
    # audio_df.to_csv('df_human_speech_only0.csv', mode='a')
    duration_voice = audio_df['duration_human_voice_sec'].tolist()
    if len(duration_voice) > 0:
        return duration_voice[0]
    else:
        return 0
    # except:
    #     print("error with human_speech_func")

path = "audio_data/"
list_file = [i for i in os.listdir("audio_data") if i.endswith(".wav")]
vad_duration = dict()
vad_time = []
audio_name = []
for au in list_file:
    _, _, duration = read_wave(path + au)
    print(au)
    if duration <= 2100:
        audio_name.append(au)
        value = human_speech_func(path + au,au,model,utils)
        vad_time.append(value)
    else:
        continue
vad_duration['audio_name'] = audio_name
vad_duration['vad_duration'] = vad_time
vad_df = pd.DataFrame(vad_duration)
vad_df.to_csv("vad_duration.csv")
