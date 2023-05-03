import pickle
import random
import re
from PIL import Image
import tensorflow as tf
import numpy as np
import collections
from tqdm import tqdm
import requests
from io import BytesIO
import pandas as pd
import datetime
import time

def preprocess_words(titles, tags, max_window_size=50):
    # words_lists = np.concatenate([[tags], np.asarray(np.expand_dims(titles, axis = 1)).T], axis = 1)
    texts_lists = [None] * titles.shape[0]
    for i,  (tag, title) in enumerate(zip(tags, titles)):
        # Taken from:
        # https://towardsdatascience.com/image-captions-with-attention-in-tensorflow-step-by-step-927dad3569fa
        single_text = tag + " " + title
        # Convert the caption to lowercase, and then remove all special characters from it
        text_nopunct = re.sub(r"[^a-zA-Z0-9]+", ' ', single_text.lower())
        # Split the caption into separate words, and collect all words which are more than 
        # one character and which contain only alphabets (ie. discard words with mixed alpha-numerics)
        clean_text = [word for word in text_nopunct.split() if ((len(word) > 1) and (word.isalpha()))]
      
        # Join those words into a string
        words_new = ['<start>'] + clean_text[:max_window_size-1] + ['<end>']# used to be -1
      
        # Replace the old caption in the captions list with this new cleaned caption
        texts_lists[i] = words_new
    return texts_lists
    

def get_images_from_url(images):
    image_lists = [None] * images.shape[0]
    for i, url in enumerate(images):
        img_data = requests.get(url).content
        img = Image.open(BytesIO(img_data))
        image_lists[i] = np.asarray(img.getdata(), dtype = np.float32).reshape(120,90,3)
    return image_lists

def get_num_from_date(dates):
    date_lists = [None] * dates.shape[0]
    for i, date in enumerate(dates):
        iso_date = datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S%z')
        utc_date = datetime.datetime.fromisoformat(str(iso_date)).replace(tzinfo=datetime.timezone.utc)
        unix_timestamp = int(utc_date.timestamp())
        date_lists[i] = unix_timestamp
    return date_lists

def load_data(data_file):
    view_data = pd.read_csv(data_file)
    X = view_data.copy()
    ids = X['video_id']
    num_in = int(ids.shape[0])
    # indices = tf.random.shuffle(range(num_in))
    #ids = np.take(ids, indices)
    thumbnails = get_images_from_url(X['thumbnail_link'])
    #thumbnails = np.take(thumbnails, indices)
    dates = get_num_from_date(X['publishedAt'])
    # dates = np.take(dates, indices)
    likes = X['likes']
    numbers = np.concatenate([tf.expand_dims(dates, axis =1), tf.expand_dims(likes, axis =1)], axis = 1)
    # likes = np.take(likes], indices)
    views = X['view_count']
    # views = np.take(views , indices)
    text = preprocess_words(X['title'], X['tags'], 50) # window of 50
    # text = np.take(text, indices)
    train_num = int( 0.7 * num_in)
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

    unk_text(train_text, 50)
    unk_text(test_text, 50)

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
    return dict(
        train_text          = np.array(train_text),
        test_text           = np.array(test_text),
        train_views          = np.array(views[:train_num]),
        test_views           = np.array(views[train_num:]),
        # train_likes          = np.array(likes[:train_num]),
        # test_likes           = np.array(likes[train_num:]),
        # train_dates          = np.array(dates[:train_num]),
        # test_dates           = np.array(dates[train_num:]),
        train_nums             = np.array(numbers[:train_num]),
        test_nums             = np.array(numbers[train_num:]),
        train_images    = np.array(thumbnails[:train_num]),
        test_images     = np.array(thumbnails[train_num:]),
        word2idx                = word2idx,
        idx2word                = {v:k for k,v in word2idx.items()},
    )


def create_pickle(data_folder, data_file):
    with open(f'{data_folder}/useable_data.p', 'wb') as pickle_file:
        pickle.dump(load_data(data_file), pickle_file)
    print(f'Data has been dumped into {data_folder}/useable_data.p!')


if __name__ == '__main__':
    ## Download this and put the Images and captions.txt indo your ../data directory
    ## Flickr 8k Dataset: https://www.kaggle.com/datasets/adityajn105/flickr8k?resource=download
    data_file = r'C:\Users\matth\OneDrive\Desktop\DEEP_Learning\cs1470-Final-Project\kaggle_data\sample_trending_data.csv'
    data_folder = r'C:\Users\matth\OneDrive\Desktop\DEEP_Learning\cs1470-Final-Project\kaggle_data'
    create_pickle(data_folder, data_file)