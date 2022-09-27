# instantiate pretrained speaker diarization pipeline
import os
from test_audio import *
from pyannote.audio import Pipeline
import pandas as pd
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

# apply pretrained pipeline
#diarization = pipeline("audio_data/1308393389_0.wav")
path = "audio_data"

list_file = [i for i in os.listdir("audio_data") if i.endswith(".wav")]
audio_name = []
speaker_each_audio = []
def speaker_count(au):
    diarization = pipeline("audio_data/{file_name}".format(file_name = au))
    # print the result
    list_speaker = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
        list_speaker.append(speaker)
    set_speaker = set(list_speaker)
    number_of_speaker = len(list(set_speaker))
    return number_of_speaker
dict_speaker = dict()
for au in list_file:
    _, _, duration = read_wave(path + '/' + au)
    print(au)
    if duration <= 2100:
        audio_name.append(au)
        number_of_speaker = speaker_count(au)
        speaker_each_audio.append(number_of_speaker)
    else:
        continue
dict_speaker['audio_name'] = audio_name
dict_speaker['speaker_count'] = speaker_each_audio
df = pd.DataFrame(dict_speaker)
df.to_csv("dict_speaker.csv")


