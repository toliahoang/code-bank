import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("combo_ranking.csv")
oh_count = df["emotion_keywords"].str.contains("oh").sum()
print(oh_count)
god_count = df["emotion_keywords"].str.contains("god").sum()
bad_count = df["emotion_keywords"].str.contains("bad").sum()
good_count = df["emotion_keywords"].str.contains("good").sum()
fuck_count = df["bad_keywords"].str.contains("fuck").sum()
damn_count = df["bad_keywords"].str.contains("damn").sum()
hell_count = df["bad_keywords"].str.contains("hell").sum()
hehe_count = df["funny_keywords"].str.contains("he he").sum()

count_features = [oh_count,god_count,bad_count,good_count,fuck_count,damn_count,hell_count,hehe_count]
name_features = ["oh","god","bad","good","fuck","damn","hell","hehe"]



x_labels = name_features
frequencies = count_features


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

plt.ylim(0,30)

plt.title('Ranking keywords (Absolute values)')
plt.xlabel("Keywords")
plt.ylabel("Frequency")

#plt.savefig("add_text_bar_matplotlib_01.png", bbox_inches='tight')
plt.show()