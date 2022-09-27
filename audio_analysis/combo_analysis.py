import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("summary_audio.csv")

#cols = ["laughter_count","screaming_distribution","angry_emotion_distribution","funny_keywords_distribution","bad_keywords_distribution","emotion_keywords_distribution","speaker_count","vad_duration"]
cols = ["laughter_count","screaming_distribution","angry_emotion_distribution","funny_keywords_distribution","bad_keywords_distribution","emotion_keywords_distribution"]
cols1 = ["speaker_count"]
cols2 = ["vad_duration"]
df["count_combo"] = (df[cols] > 0).sum(axis=1) + (df[cols1] > 1).sum(axis=1) + (df[cols2] > 1).sum(axis=1)

print(df["count_combo"])


# new_df= df.sort_values("count_combo")
# new_df.to_csv("combo_ranking.csv")



new_df= df.groupby(["count_combo"])["count_combo"].count()
print(new_df.to_string())
ax =new_df.plot.bar(x='index', y='count_combo', rot=0)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x()+ p.get_width()/2., p.get_height()+ 1.05))
plt.title("Combo Distribution")
plt.xlabel("The Number of Features")
plt.ylabel("Frequency")
plt.show()