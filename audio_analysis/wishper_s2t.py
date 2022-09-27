import os
from pydub import AudioSegment
import whisper
import time
import glob
import pandas as pd
from load_audio_from_txt import get_list_audio

file_name = "file_name.txt"
file_path = "/root/data1/tung/docker_rep/audio_file"


class WishperObj:

    def __init__(self, file_path, file_name):
        self.model = None
        self.file_path = file_path
        self.file_name = file_name

    def load_model(self):
        if self.model is None:
            self.model = whisper.load_model("base")

    def inference_language(self, audio_file):
        audio = whisper.load_audio(self.file_path + "/" + audio_file)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        _, probs = self.model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")
        return mel

    def inference_text(self, mel):
        text_dict = dict()
        text_dict["text"] = []
        options = whisper.DecodingOptions()
        result = whisper.decode(self.model, mel, options)
        text_dict["text"].append(result.text)
        return text_dict

    def get_list_audio(self):
        lines = []
        with open(self.file_name) as file_in:
            for line in file_in:
                lines.append(line.split(".wav")[0] + ".wav")
        return lines


def get_language():

    wishper_obj = WishperObj(file_path,file_name)
    wishper_obj.load_model()
    audio_list = wishper_obj.get_list_audio()
    mel = wishper_obj.inference_language(audio_list[0])
    text_dict = wishper_obj.inference_text(mel)
    return text_dict


text_dict = get_language()
print(text_dict)

