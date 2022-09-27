import torch
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import numpy as np
import pandas as pd
import os



audio_path = "/root/tung/laboratory/deepspeech/audio_data/"
file_wav_path =[wav_file for wav_file in os.listdir(audio_path) if wav_file.endswith('.wav')]
name_audio_files = [i.split(".")[0] for i in file_wav_path]


#path = "/root/tung/laboratory/deepspeech/audio_data/1308393389_0.wav"

# def map_to_array(example):
#     speech, _ = librosa.load(example["file"], sr=16000, mono=True)
#     example["speech"] = speech
#     return example
#
# # load a demo dataset and read audio files
# dataset = load_dataset("anton-l/superb_demo", "er", split="session1")
# dataset = dataset.map(map_to_array)
# print(len(dataset["speech"]))
model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")

for au,name in zip(file_wav_path,name_audio_files):
    print(au)
    duration = librosa.get_duration(filename=audio_path + au)
    print("duration:",duration)
    if duration <= 2100:

        speech, _ = librosa.load(audio_path + au, sr=16000, mono=True)
        speech_split = np.array_split(speech,6)

        emotion_list = []
        name_audio_list = []
        emotion_dict = dict()
        for segment in speech_split:

            # compute attention masks and normalize the waveform if needed
            #inputs = feature_extractor(dataset[:4]["speech"], sampling_rate=16000, padding=True, return_tensors="pt")
            inputs = feature_extractor(speech_split[0], sampling_rate=16000, padding=True, return_tensors="pt")
            logits = model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            labels = [model.config.id2label[_id] for _id in predicted_ids.tolist()]
            emotion_list.append(labels)
            name_audio_list.append(name)
        emotion_dict["name_audio"] = name_audio_list
        emotion_dict["emotion_recognition"] = emotion_list
        df = pd.DataFrame(emotion_dict)
        df.to_csv("emotion_csv/{}.csv".format(name))
    else:
        continue




