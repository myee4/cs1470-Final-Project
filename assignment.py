import argparse
import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from tqdm import tqdm
import pickle
from thumbnail_model import ThumbnailModel, accuracy_function, loss_function

def train(model, train_images, train_text, train_nums, train_views,  batch_size=4):
    """
    Runs through one epoch - all training examples.

    :param model: the initialized model to use for forward and backward pass
    :param train_captions: train data captions (all data for training) 
    :param train_images: train image features (all data for training) 
    :param padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
    :return: None
    """

    ## TODO: Implement similar to test below.
    ## NOTE: shuffle the training examples (perhaps using tf.random.shuffle on a
    ##       range of indices spanning # of training entries, then tf.gather) 
    ##       to make training smoother over multiple epochs.

    # indices = tf.random.shuffle(range(len(train_images)))
    # train_images = tf.gather(train_images, indices)
    # train_text = tf.gather(train_text, indices)
    # train_nums = tf.gather(train_nums, indices)
    # train_views = tf.gather(train_views, indices)

    ## NOTE: make sure you are calculating gradients and optimizing as appropriate
    ##       (similar to batch_step from HW2)
    num_batches = int(len(train_images) / batch_size)
    total_loss = 0
    total_acc = 0
    # DOesn't actually batch LMAO
    # for (b_image, b_text, b_nums, b_views) in zip(train_images, train_text, train_nums, train_views):
    for index, end in enumerate(range(batch_size, len(train_images)+1, batch_size)):
        start = end - batch_size
        batch_images = train_images[start:end, :]
        batch_text = train_text[start:end, :]
        batch_nums = train_nums[start:end, :]
        batch_views = train_views[start:end]
        with tf.GradientTape() as tape:
            preds = model(batch_images, batch_text, batch_nums)
            loss = loss_function(preds, batch_views)
            acc = accuracy_function(preds, batch_views)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        total_loss += loss
        total_acc += acc
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    return avg_loss.numpy(), avg_acc.numpy()


def test(model, test_images, test_text, test_nums, test_views):
    """
    DO NOT CHANGE; Use as inspiration

    Runs through one epoch - all testing examples.

    :param model: the initilized model to use for forward and backward pass
    :param test_captions: test caption data (all data for testing) of shape (num captions,20)
    :param test_image_features: test image feature data (all data for testing) of shape (num captions,1000)
    :param padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
    :returns: perplexity of the test set, per symbol accuracy on test set
    """
    preds = model(test_images, test_text, test_nums)
    loss = loss_function(preds, test_views)
    acc = accuracy_function(preds, test_views)
    return loss.numpy(), acc.numpy()


def main():
    file_path = './kaggle_data/useable_data.p'
    with open(file_path, 'rb') as data_file:
        data_dict = pickle.load(data_file)
    train_images  = np.array(data_dict['train_images'])
    test_images   = np.array(data_dict['test_images'])
    train_text  = np.array(data_dict['train_text'])
    test_text   = np.array(data_dict['test_text'])
    train_nums  = np.array(data_dict['train_nums'])
    test_nums   = np.array(data_dict['test_nums'])
    train_views  = np.array(data_dict['train_views'])
    test_views   = np.array(data_dict['test_views'])
    word2idx        = data_dict['word2idx']
    model = ThumbnailModel(128, 11, 32, len(word2idx), 50) #FROM PREPROCESS THIS IS BAD
    epochs = 30
    for epoch in range(epochs):
        print("-------------------------------------------------------")
        print(train(model, train_images, train_text, train_nums, train_views))
        print("-------------------------------------------------------")
    print("----------------TEST:---------------------")
    print(test(model, test_images, test_text, test_nums, test_views))


if __name__ == "__main__":
    main()