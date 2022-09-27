import pandas as pd
import matplotlib.pyplot as plt

new_df_7 = pd.read_csv("summary_audio.csv")


# mfcc_max_df = pd.read_csv("mfcc_max.csv")
# mfcc_max_df["audio_name"] = mfcc_max_df["audio_name"] .str.replace('.wav','')
# laughter_count_df = pd.read_csv("laughter_count.csv")
# screaming_distribution_df = pd.read_csv("screaming_distribution.csv")
# emotion_recognition_transformer_df = pd.read_csv("emotion_recognition_transformer.csv")
# keywords_extraction_df = pd.read_csv("keywords_extraction.csv")
# number_of_speakers_df = pd.read_csv("number_of_speakers.csv")
# number_of_speakers_df["audio_name"] = number_of_speakers_df["audio_name"].str.replace('.wav','')
# vad_df = pd.read_csv("vad_duration.csv")
# vad_df["audio_name"] = vad_df["audio_name"].str.replace('.wav','')




# new_df_1 = mfcc_max_df.merge(laughter_count_df, on='audio_name',how='left')
# new_df_2 = new_df_1.merge(screaming_distribution_df, on='audio_name',how='left')
# new_df_3 = new_df_2.merge(emotion_recognition_transformer_df, on='audio_name',how='left')
# new_df_4 = new_df_3.merge(keywords_extraction_df, on='audio_name',how='left')
# new_df_5 = new_df_4.merge(number_of_speakers_df, on='audio_name',how='left')
# new_df_6 = new_df_5.merge(vad_df, on='audio_name', how='left')
#
# print(new_df_7.columns.tolist())


count_laughter_count = new_df_6[new_df_6["laughter_count"] >0].count()
count_screaming_count = new_df_6[new_df_6["screaming_distribution"] >0].count()
count_emotion_recognition_transformer = new_df_6[new_df_6["angry_emotion_distribution"] >0].count()
count_funny_keywords_distribution = new_df_6[new_df_6["funny_keywords_distribution"] >0].count()
count_bad_keywords_distribution = new_df_6[new_df_6["bad_keywords_distribution"] >0].count()
count_emotion_keywords_distribution = new_df_6[new_df_6["emotion_keywords_distribution"] >0].count()
count_number_speakers = new_df_6[new_df_6["speaker_count"] >1].count()
count_vad = new_df_6[new_df_6["vad_duration"] > 0].count()









count_laughter_count_ = count_laughter_count["laughter_count"].tolist()
count_screaming_count_ = count_screaming_count["screaming_distribution"].tolist()
count_emotion_recognition_transformer_ = count_emotion_recognition_transformer["angry_emotion_distribution"].tolist()
count_funny_keywords_distribution_ = count_funny_keywords_distribution["funny_keywords_distribution"].tolist()
count_bad_keywords_distribution_ = count_bad_keywords_distribution["bad_keywords_distribution"].tolist()
count_emotion_keywords_distribution_ = count_emotion_keywords_distribution["emotion_keywords_distribution"].tolist()
count_number_speakers_ = count_number_speakers["speaker_count"].tolist()
count_vad_ = count_vad["vad_duration"].tolist()
#
features = ["laughter","screaming","angry_emotion","funny_keywords","bad_keywords","emotion_keywords","speaker_count","human_voice"]
count_feat = [count_laughter_count_ , count_screaming_count_ , count_emotion_recognition_transformer_ , count_funny_keywords_distribution_ , count_bad_keywords_distribution_ \
    , count_emotion_keywords_distribution_ ,  count_number_speakers_, count_vad_]

per_count_feat = [(i/182)*100 for i in count_feat]

per_count_feat_, per_features_ = zip(*sorted(zip(per_count_feat,features)))
count_feat_ , features_ = zip(*sorted(zip(count_feat,features)))




















# new_df_6.to_csv("summary_audio.csv")


