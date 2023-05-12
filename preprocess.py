import re
import sys
import pickle
import requests
import datetime
import collections
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import tensorflow as tf


'''
preprocess_images() takes in a list of links to thumbnails of YouTube videos
and retrieves the thumbnails from the links at which they are stored.
'''
def preprocess_images(thumbnail_urls):
    preprocessed_images = [None] * thumbnail_urls.shape[0]
    for i, url in enumerate(thumbnail_urls):
        image_data = requests.get(url).content
        image = Image.open(BytesIO(image_data))
        preprocessed_images[i] = np.asarray(image.getdata(), dtype=np.float32).reshape(120, 90, 3) / 255
    return preprocessed_images

'''
preprocess_text() takes in the titles of videos as well as the hashtags associated
with the video, strips them of any formatting, and concatenates all the words into 
strings of length window_size. The decision to add the hashtags before video titles 
was arbitrary. This does imply that for videos with lots of hashtags or longer video
titles, the title will be truncated before hashtags are.
'''
def preprocess_text(titles, hashtags, window_size=50):
    preprocessed_text = [None] * titles.shape[0]
    for i, (hashtag, title) in enumerate(zip(hashtags, titles)):
        text_old_format = hashtag + " " + title
        text_unformatted = re.sub(r"[^a-zA-Z0-9]+", ' ', text_old_format.lower())
        text_unformatted = [word for word in text_unformatted.split() if ((len(word) > 1) and (word.isalpha()))]
        text_new_format = ['<start>'] + text_unformatted[:window_size-1] + ['<end>']
        preprocessed_text[i] = text_new_format
    return preprocessed_text

'''
preprocess_dates() takes in a list of the dates on which videos were published and
converts that information to a standardized format, albeit normalized.
'''
def preprocess_dates(dates):
    preprocessed_dates = [None] * dates.shape[0]
    for i, date in enumerate(dates):
        iso_date = datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S%z')
        utc_date = datetime.datetime.fromisoformat(str(iso_date)).replace(tzinfo=datetime.timezone.utc)
        unix_timestamp = int(utc_date.timestamp())
        preprocessed_dates[i] = unix_timestamp
    preprocessed_dates = (preprocessed_dates - np.mean(preprocessed_dates)) / np.var(preprocessed_dates)
    return preprocessed_dates

'''
preprocess_data() takes in the number of data points a user wants to use to train their 
model and their desired train-test data split. It then reads that data from a csv file
containing data on 200,000 YouTube videos, and then stores all the relevant data as a 
dictionary in a pickle file. The data is normalized before being stored as the dictionary.
'''
def preprocess_data(desired_range, desired_split):
    # data source: https://www.kaggle.com/datasets/rsrishav/youtube-trending-video-dataset/versions/1000
    data_file = r'./data/data.csv'
    specific_rows = [i for i in range(desired_range)]
    specific_data = pd.read_csv(data_file, skiprows=lambda x: x not in specific_rows)

    thumbnails = preprocess_images(specific_data['thumbnail_link'])
    text = preprocess_text(specific_data['title'], specific_data['tags'])
    dates = preprocess_dates(specific_data['publishedAt'])

    likes = specific_data['likes']
    views = specific_data['view_count']
    likes = (likes - np.mean(likes)) / np.var(likes)
    views = (views - np.mean(views)) / np.var(views)

    numbers = np.concatenate([tf.expand_dims(dates, axis=1), tf.expand_dims(likes, axis=1)], axis=1)

    ids = specific_data['video_id']
    num_samples = int(ids.shape[0])
    train_num = int(desired_split * num_samples)
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

    def pad_captions(captions, window_size=50):
        for caption in captions:
            caption += (window_size + 1 - len(caption)) * \
                ['<pad>']
    pad_captions(train_text)
    pad_captions(test_text)

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

    return dict(train_images=np.array(thumbnails[:train_num]),
                test_images=np.array(thumbnails[train_num:]),
                train_text=np.array(train_text),
                test_text=np.array(test_text),
                train_nums=np.array(numbers[:train_num]),
                test_nums=np.array(numbers[train_num:]),
                train_views=np.array(views[:train_num]),
                test_views=np.array(views[train_num:]),
                word2idx=word2idx,
                idx2word={v: k for k, v in word2idx.items()})


if __name__ == '__main__':

    n = len(sys.argv)

    # default run
    if n == 1:
        desired_range = 1000
        desired_split = 0.7
        with open(f'data/data.p', 'wb') as pickle_file:
            pickle.dump(preprocess_data(desired_range, desired_split), pickle_file)
        exit()
    
    if n != 3:
        print("usage: python preprocess.py [number of data points] [percentage of data for training]")
        print("example: python preprocess.py 1000 0.7")
        exit()

    try:
        desired_range = int(sys.argv[1])
    except:
        print("usage: python preprocess.py [number of data points] [percentage of data for training]")
        print("example: python preprocess.py 1000 0.7")
        print("[number of data points] must be an integer")
        exit()
    if desired_range < 1 or desired_range > 200000:
        print("usage: python preprocess.py [number of data points] [percentage of data for training]")
        print("example: python preprocess.py 1000 0.7")
        print("[number of data points] must be an integer greater than 0 and less than 200000")
        exit()

    try:
        desired_split = float(sys.argv[2])
    except:
        print("usage: python preprocess.py [number of data points] [percentage of data for training]")
        print("example: python preprocess.py 1000 0.7")
        print("[percentage of data for training] must be a float")
        exit()
    if desired_split < 0 or desired_split > 1:
        print("usage: python preprocess.py [number of data points] [percentage of data for training]")
        print("example: python preprocess.py 1000 0.7")
        print("[percentage of data for training] must be a float between 0 and 1")
        exit()

    with open(f'data/data.p', 'wb') as pickle_file:
        pickle.dump(preprocess_data(desired_range, desired_split), pickle_file)

    print(f'data has been dumped into data/data.p!')
