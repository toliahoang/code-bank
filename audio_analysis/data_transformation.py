import glob

import pandas as pd
import os
import glob

path = "speech_to_text"
list_file = [file for file in os.listdir(path) if file.endswith(".txt")]
def get_file_content(filename):
    content = ''
    with open(filename, 'r') as f:
        content = f.read()
    return content

contents = []
name_files = []
speech_to_text_dict = dict()
for txt_file in list_file:
    name_files.append(txt_file)
    contents.append(get_file_content(path + "/" + txt_file))

name_files_ = [i.split('.txt')[0] for i in name_files if i.endswith('.txt')]




speech_to_text_dict['audio_name'] = name_files_
speech_to_text_dict['speech_to_text'] = contents

speech_to_text_df = pd.DataFrame(speech_to_text_dict)
speech_to_text_df.to_csv("speech_to_text.csv")

