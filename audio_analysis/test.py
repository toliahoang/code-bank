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


list_data = [{'end':10000,'start':5000},{'end':30000,'start':20000},{'end':70000,'start':60000}]
speech_timestamps = list_data
length_speech_chunk = [i['end'] - i['start'] for i in speech_timestamps]
# print(length)
speech_sample_duration = sum(length_speech_chunk)
duration_human_voice_sec = [speech_sample_duration / 16000]
# print("time_duration", duration_human_voice_sec)
total_video_duration = 20000000
human_speech_score = [np.float32(speech_sample_duration / total_video_duration)]
speech_timestamps_start = [i['start'] for i in speech_timestamps]
list_start_in_sec = [start_time / 16000 for start_time in speech_timestamps_start]
conversion_start_hmmss = [str(datetime.timedelta(seconds=sec_input)) for sec_input in list_start_in_sec]
# print(conversion_start_hmmss)
string_from_list_start_stop_new = '_'.join(conversion_start_hmmss)
# print(string_from_list_start_stop_new)

video_id = [100]

audio_df = pd.DataFrame(list(zip(video_id, human_speech_score, [string_from_list_start_stop_new], duration_human_voice_sec)), columns=['video_id', 'human_speech_score', 'human_speech_timestamp', 'duration_human_voice_sec'])
audio_df['human_speech_timestamp'] = audio_df['human_speech_timestamp'].astype(str)
# audio_df.to_csv('df_human_speech_only0.csv', mode='a')

print(audio_df.head())


