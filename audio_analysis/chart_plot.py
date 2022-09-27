import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



new_df_7 = pd.read_csv("summary_audio.csv")
count_laughter_count = new_df_7[new_df_7["laughter_count"] >0].count()
count_screaming_count = new_df_7[new_df_7["screaming_distribution"] >0].count()
count_emotion_recognition_transformer = new_df_7[new_df_7["angry_emotion_distribution"] >0].count()
count_funny_keywords_distribution = new_df_7[new_df_7["funny_keywords_distribution"] >0].count()
count_bad_keywords_distribution = new_df_7[new_df_7["bad_keywords_distribution"] >0].count()
count_emotion_keywords_distribution = new_df_7[new_df_7["emotion_keywords_distribution"] >0].count()
count_number_speakers = new_df_7[new_df_7["speaker_count"] >1].count()
count_vad = new_df_7[new_df_7["vad_duration"] > 0].count()
count_laughter_count_ = count_laughter_count["laughter_count"].tolist()
count_screaming_count_ = count_screaming_count["screaming_distribution"].tolist()
count_emotion_recognition_transformer_ = count_emotion_recognition_transformer["angry_emotion_distribution"].tolist()
count_funny_keywords_distribution_ = count_funny_keywords_distribution["funny_keywords_distribution"].tolist()
count_bad_keywords_distribution_ = count_bad_keywords_distribution["bad_keywords_distribution"].tolist()
count_emotion_keywords_distribution_ = count_emotion_keywords_distribution["emotion_keywords_distribution"].tolist()
count_number_speakers_ = count_number_speakers["speaker_count"].tolist()
count_vad_ = count_vad["vad_duration"].tolist()

features = ["laughter","screaming","angry_emotion","funny_keywords","bad_keywords","emotion_keywords","speaker_count","human_voice"]
count_feat = [count_laughter_count_ , count_screaming_count_ , count_emotion_recognition_transformer_ , count_funny_keywords_distribution_ , count_bad_keywords_distribution_ \
    , count_emotion_keywords_distribution_ ,  count_number_speakers_, count_vad_]

per_count_feat = [int((i/182)*100) for i in count_feat]

per_count_feat_, per_features_ = zip(*sorted(zip(per_count_feat,features)))
count_feat_ , features_ = zip(*sorted(zip(count_feat,features)))


x_labels = features_
frequencies = count_feat_


fig, ax = plt.subplots()

bar_x = [1,2,3,4,5,6,7, 8]
bar_height = frequencies
bar_tick_label = x_labels
bar_label = frequencies

bar_plot = plt.bar(bar_x,bar_height,tick_label=bar_tick_label)

def autolabel(rects):
    for idx,rect in enumerate(bar_plot):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05 + height,
                bar_label[idx],
                ha='center', va='bottom', rotation=0)

autolabel(bar_plot)

plt.ylim(0,200)

plt.title('Ranking high-level audio features (Absolute value)')
plt.xlabel("Features")
plt.ylabel("Frequency")

#plt.savefig("add_text_bar_matplotlib_01.png", bbox_inches='tight')
plt.show()
