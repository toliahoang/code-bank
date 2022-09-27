from deepspeech_speech_to_text import *
import os
import glob
import numpy as np
def main(input_file):
    # need audio, aggressive, and model
    # Point to a path containing the pre-trained models & resolve ~ if used
    model = './models'
    dirName = os.path.expanduser(model)


    audio = "audio_data/{}".format(input_file)
    aggressive = 0  # input("What level of non-voice filtering would you like? (0-3)")

    # Resolve all the paths of model files
    output_graph, scorer = resolve_models(dirName)

    # Load output_graph, alphabet and scorer
    model_retval = load_model(output_graph, scorer)

    title_names = ['Filename', 'Duration(s)', 'Inference Time(s)', 'Model Load Time(s)', 'Scorer Load Time(s)']
    print("\n%-30s %-20s %-20s %-20s %s" % (
    title_names[0], title_names[1], title_names[2], title_names[3], title_names[4]))

    inference_time = 0.0

    waveFile = audio
    segments, sample_rate, audio_length = vad_segment_generator(waveFile, aggressive)
    f = open(waveFile.rstrip(".wav") + ".txt", 'w')
    print("Saving Transcript @: %s" % waveFile.rstrip(".wav") + ".txt")
    for i, segment in enumerate(segments):
        # Run deepspeech on the chunk that just completed VAD
        print("Processing chunk %002d" % (i,))
        audio = np.frombuffer(segment, dtype=np.int16)
        output = stt(model_retval[0], audio, sample_rate)
        inference_time += output[1]
        print("Transcript: %s" % output[0])

        f.write(output[0] + " ")

    # Summary of the files processed
    f.close()

    # Extract filename from the full file path
    filename, ext = os.path.split(os.path.basename(waveFile))
    print(
        "************************************************************************************************************")
    print(
        "%-30s %-20s %-20s %-20s %s" % (title_names[0], title_names[1], title_names[2], title_names[3], title_names[4]))
    print("%-30s %-20.3f %-20.3f %-20.3f %-0.3f" % (
    filename + ext, audio_length, inference_time, model_retval[1], model_retval[2]))
    print(
        "************************************************************************************************************")
    print("%-30s %-20.3f %-20.3f %-20.3f %-0.3f" % (
    filename + ext, audio_length, inference_time, model_retval[1], model_retval[2]))


if __name__ == '__main__':
    path = 'audio_data'
    
    list_file = [i for i in os.listdir(path) if i.endswith('.wav')]
    for au in list_file:
        
        print(au)
        _,_,duration = read_wave(path + '/' + au)
        if duration <= 2100:
            main(au)
        else:
            continue    