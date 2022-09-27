import re
import pandas as pd
import os


def filter_funny_word(list_word,filter = []):
    """
    :param list_word:
    :return: empty_list
    """
    words = list_word

    noise = ["www.","https.//"]
    empty_list = []
    for i_filter in filter:
        regex = re.compile(".*".join(i_filter), re.IGNORECASE)
        filtered_words = [word for word in words if regex.match(word)]
        # print(*filtered_words, sep="\n")
        if len(filtered_words) !=0:

            empty_list.append(filtered_words)
        else:
            empty_list = empty_list
    return empty_list


# def extract_funny(in_df):
#     """
#
#     :param in_df:
#     :return:
#     """
#     file_name = in_df.split("/")[-1]
#     new_file_name = in_df.split('\\')[-1]
#     chat_df = pd.read_csv(in_df,dtype=str)
#     row = chat_df['message']
#     length = row.size
#     list_out = []
#     for i in range(length):
#         list_word = row[i].split()
#         out = filter_funny_word(list_word)
#         list_out.append(out)
#     distribution_funny = [0 if len(i)==0 else 1 for i in list_out]
#     print(distribution_funny)
#     chat_df['funny_word'] = list_out
#     chat_df['funny_dist'] = distribution_funny
#     chat_df.to_csv("New_{file}".format(file=new_file_name))



def extract_funny(chat_df,funny_word,funny_dist,filter=[]):
    """

    :param in_df:
    :return:
    """


    row = chat_df['speech_to_text']
    length = row.size
    list_out = []
    for i in range(length):
        list_word = row[i].split()
        out = filter_funny_word(list_word, filter)
        list_out.append(out)
    distribution_funny = [0 if len(i)==0 else 1 for i in list_out]
    print(distribution_funny)
    chat_df['{}'.format(funny_word)] = list_out
    chat_df['{}'.format(funny_dist)] = distribution_funny




    chat_df.to_csv("keywords_extraction.csv")




# path = "www.twitch.tv/videos"
# out_path = "output_data"
def extract_funny_from_list(path):
    """

    :param path:
    :return: data with funny words
    """
    list_df = [i for i in os.listdir(path) if i.endswith(".csv")]
    for j in list_df:
        print(j)
        extract_funny(os.path.join(path,j))

# extract_funny_from_list(path)

def extract_bad_keywords(speech_to_text_csv):
    chat_df = pd.read_csv(speech_to_text_csv,dtype=str)
    filter1 = ["haha", "hehe", "wow", "lol", "lmao", "lul", "ha ha", "he he"," he ", " ha "]
    extract_funny(chat_df,"funny_keywords","funny_keywords_distribution", filter1)
    filter2 = ["fuck", "hell", "damn", "bloody hell", "bitch","shit", "badstard"]
    extract_funny(chat_df,"bad_keywords","bad_keywords_distribution", filter2)
    filter3 = ["my god","no no", "can't kill me", "bad", "hooray", "oh", "god"]
    extract_funny(chat_df,"emotion_keywords","emotion_keywords_distribution", filter3)




extract_bad_keywords("speech_to_text.csv")