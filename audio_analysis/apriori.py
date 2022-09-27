import pandas as pd

import numpy as np

path = "C:/Users/Admin/Downloads/summary_audio.csv"
cols = ["laughter_count","screaming_distribution","angry_emotion_distribution"\
    ,"funny_keywords_distribution","bad_keywords_distribution","emotion_keywords_distribution"\
    ,"speaker_count","vad_duration"]
df = pd.read_csv(path)
df['speaker_count'] = np.where(df.speaker_count >= 2, 1,
                               0)
df['vad_duration'] = np.where(df.vad_duration > 1,1,0)

new_df = df[cols]





from itertools import combinations
def get_support(df):
    pp = []
    for cnum in range(1, len(df.columns)+1):
        for cols in combinations(df, cnum):
            s = df[list(cols)].all(axis=1).sum()
            pp.append([",".join(cols), s])
    sdf = pd.DataFrame(pp, columns=["Pattern", "Support"])
    return sdf

s = get_support(new_df)
s_df = s[s.Support >= 36]

print(s_df.to_string())