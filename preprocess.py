import pickle
import re
from PIL import Image
import tensorflow as tf
import numpy as np
import collections
import requests
from io import BytesIO
import pandas as pd
import datetime

# get all text relevant to a video
def preprocess_words(titles, tags, max_window_size=50):
    # array of strings where each string is all hash tags and title of a video
    texts_lists = [None] * titles.shape[0]
    for i, (tag, title) in enumerate(zip(tags, titles)):
        # combining hash tags and video titles
        single_text = tag + " " + title
        # convert the caption to lowercase, and then remove all special characters from it
        text_nopunct = re.sub(r"[^a-zA-Z0-9]+", ' ', single_text.lower())
        # split the caption into separate words, and collect all words which are more than 
        # one character and which contain only alphabets (i.e. discard words with mixed alpha-numerics)
        clean_text = [word for word in text_nopunct.split() if ((len(word) > 1) and (word.isalpha()))]
        # join those words into a string
        words_new = ['<start>'] + clean_text[:max_window_size-1] + ['<end>']
        # adding text to array
        texts_lists[i] = words_new
    return texts_lists
    
# get thumbnails for each video    
def get_images_from_url(images):
    image_lists = [None] * images.shape[0]
    for i, url in enumerate(images):
        img_data = requests.get(url).content
        img = Image.open(BytesIO(img_data))
        image_lists[i] = np.asarray(img.getdata(), dtype = np.float32).reshape(120,90,3)
    return image_lists

# get dates published for each video
def get_num_from_date(dates):
    date_lists = [None] * dates.shape[0]
    for i, date in enumerate(dates):
        iso_date = datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S%z')
        utc_date = datetime.datetime.fromisoformat(str(iso_date)).replace(tzinfo=datetime.timezone.utc)
        unix_timestamp = int(utc_date.timestamp())
        date_lists[i] = unix_timestamp
    return date_lists

# 
def load_data(data_file):
    # read data from chosen csv file
    view_data = pd.read_csv(data_file)
    # make a copy, just in case
    view_data_copy = view_data.copy()

    # get various video attributes
    ids = view_data_copy['video_id']
    num_samples = int(ids.shape[0])
    thumbnails = get_images_from_url(view_data_copy['thumbnail_link'])
    dates = get_num_from_date(view_data_copy['publishedAt'])
    likes = view_data_copy['likes']
    numbers = np.concatenate([tf.expand_dims(dates, axis =1), tf.expand_dims(likes, axis =1)], axis = 1)
    views = view_data_copy['view_count']
    text = preprocess_words(view_data_copy['title'], view_data_copy['tags'], 50) # window of 50

    # replace words with unk
    train_num = int(0.7 * num_samples)
    train_text = text[:train_num]
    test_text = text[train_num:]
    word_count = collections.Counter()
    for words in train_text:
        word_count.update(words)
    def unk_text(texts, minimum_frequency):
        for text in texts:
            for index, word in enumerate(text):
                if word_count[word] <= minimum_frequency:
                    text[index] = '<unk>'
    unk_text(train_text, 10)
    unk_text(test_text, 10)

    # pad captions so they all have equal length
    def pad_captions(captions, max_window_size = 50):
        for caption in captions:
            caption += (max_window_size + 1 - len(caption)) * ['<pad>'] #used to be +1
    pad_captions(train_text)
    pad_captions(test_text)

    # assign unique ids to every word left in the vocabulary
    word2idx = {}
    vocab_size = 0
    for caption in train_text:
        for index, word in enumerate(caption):
            if word in word2idx:
                caption[index] = word2idx[word]
            else:
                word2idx[word] = vocab_size
                caption[index] = vocab_size
                vocab_size += 1
    for caption in test_text:
        for index, word in enumerate(caption):
            caption[index] = word2idx[word] 
    return dict(train_text = np.array(train_text),
                test_text = np.array(test_text),
                train_views = np.array(views[:train_num]),
                test_views = np.array(views[train_num:]),
                train_nums = np.array(numbers[:train_num]),
                test_nums = np.array(numbers[train_num:]),
                train_images = np.array(thumbnails[:train_num]),
                test_images = np.array(thumbnails[train_num:]),
                word2idx = word2idx,
                idx2word = {v:k for k,v in word2idx.items()})

def create_pickle(data_folder, data_file):
    with open(f'{data_folder}/useable_data.p', 'wb') as pickle_file:
        pickle.dump(load_data(data_file), pickle_file)
    print(f'Data has been dumped into {data_folder}/useable_data.p!')

if __name__ == '__main__':
    data_file = r'./kaggle_data/sample_trending_data_bigger.csv'
    data_folder = r'./kaggle_data'
    create_pickle(data_folder, data_file)
