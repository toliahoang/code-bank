import jedi.inference.compiled.value
import pandas as pd
import glob

path = "emotion_recognition_csv"
emotion_file_csv = glob.iglob(path + "/" + "*.csv")


# data = {
#     'labels': {'neu': 0, 'hap': 1, 'ang': 2, 'sad': 3},
#     'meta_data': meta_data
# }


# df = pd.read_csv("/root/tung/laboratory/deepspeech/emotion_recognition_csv/1308393389_0.csv")
file_name = []
ang_count_list = []
emotion_recog_dict = dict()
for file_emotion in emotion_file_csv:
    file_name.append(file_emotion)
    df = pd.read_csv(file_emotion)
    neu_count = df["emotion_recognition"].str.contains("neu").sum()
    hap_count = df["emotion_recognition"].str.contains("hap").sum()
    ang_count = df["emotion_recognition"].str.contains("ang").sum()
    sad_count = df["emotion_recognition"].str.contains("sad").sum()

    if ang_count > 0:
        ang_count_list.append(1)
    else:
        ang_count_list.append(0)

print(file_name)

file_name_ = [i.split(".csv")[0] for i in file_name]
file_name_new = [j.split("/")[-1] for j in file_name_]

emotion_recog_dict['audio_name'] = file_name_new
emotion_recog_dict['angry_emotion_distribution'] = ang_count_list
ang_df = pd.DataFrame(emotion_recog_dict)
ang_df.to_csv("emotion_recognition_transformer.csv")

