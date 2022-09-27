import speechbrain as sb
import soundfile as sf

path_au = "/root/tung/laboratory/deepspeech/audio_data/1308393389_0.wav"

speech, _ = sf.read(path_au)
spk = sb.processing.diarization(speech)
print(spk)
