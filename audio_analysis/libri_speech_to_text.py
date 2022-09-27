import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from datasets import load_dataset
import soundfile as sf

model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-medium-librispeech-asr")
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-medium-librispeech-asr")

#path_audio = "/root/tung/laboratory/deepspeech/audio_data/1308393389_0.wav"
#path_audio = "/root/tung/laboratory/deepspeech/audio_data/2022-06-03_1592627861_tildirudie_engswe_pro_noob_gaming.wav"
#path_audio = "/root/tung/laboratory/deepspeech/audio_data/2022-05-24_1582826996_smexiebeast_fortnite_with_the_ladies_twitchcon.wav"

path_audio = "/root/tung/laboratory/deepspeech/audio_data/2022-05-18_1576903322_perilous_outlaw_bonk.wav"
def map_to_array(path_audio):
    speech, _ = sf.read(path_audio)
    return speech


# def map_to_array(path):
#     speech, _ = sf.read(path)
#     return speech


ds = load_dataset(
    "patrickvonplaten/librispeech_asr_dummy",
    "clean",
    split="validation"
)
ds = map_to_array(path_audio)


input_features = processor(
    ds,
    sampling_rate=16_000,
    return_tensors="pt"
).input_features  # Batch size 1

print(input_features)





generated_ids = model.generate(input_features)
#
list_trans = []
transcription = processor.batch_decode(generated_ids)
list_trans.append(transcription)
print(transcription)
print(list_trans)