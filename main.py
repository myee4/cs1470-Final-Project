import pickle
import numpy as np
import tensorflow as tf
from model import ThumbnailModel, ImageNumModel, NumTextModel, ImageTextModel, ImageModel, TextModel, NumModel, SingleModel, accuracy_function, loss_function


def train(model, train_images, train_text, train_nums, train_views, batch_size=10):
    #total_loss = 0
    total_acc = 0
    num_batches = int(len(train_images) / batch_size)

    for _, end in enumerate(range(batch_size, len(train_images) + 1, batch_size)):
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
        model.optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))
        #total_loss += loss
        total_acc += acc

    # avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    # return avg_loss.numpy(), avg_acc.numpy()
    return avg_acc.numpy()


def test(model, test_images, test_text, test_nums, test_views):
    preds = model(test_images, test_text, test_nums)
    # print("text: ", test_text)
    print("prediction: ", preds)
    # print("real views: ", test_views)
    # loss = loss_function(preds, test_views)
    acc = accuracy_function(preds, test_views)
    # return loss.numpy(), acc.numpy()
    return acc.numpy()



def main():
    file_path = r'.\cs1470-Final-Project\data\data.p'
    with open(file_path, 'rb') as data_file:
        data_dict = pickle.load(data_file)
    train_images = np.array(data_dict['train_images'])
    test_images = np.array(data_dict['test_images'])
    train_text = np.array(data_dict['train_text'])
    test_text = np.array(data_dict['test_text'])
    train_nums = np.array(data_dict['train_nums'])
    test_nums = np.array(data_dict['test_nums'])
    train_views = np.array(data_dict['train_views'])
    test_views = np.array(data_dict['test_views'])
    word2idx = data_dict['word2idx']

    model = SingleModel(4096, 5, 128, len(word2idx), 50)

    epochs = 25

    print("---------------------------TRAIN-----------------------------")
    #accuracy array
    acc = [0] * epochs
    #count for a stable acccuracy
    stable_count = 0
    for i in range(epochs):
        acc[i] = train(model, train_images, train_text, train_nums, train_views)
        print(
            f"---------------------------EPOCH {i}---------------------------")
        print(acc[i])
        print("-------------------------------------------------------------")
        if (acc[i] < acc[i-1]+3) and (acc[i] > acc[i-1]-3):
            stable_count +=1
        else:
            stable_count = 0
        if stable_count >= 4:
            break

    print("---------------------------TEST------------------------------")
    print(test(model, test_images, test_text, test_nums, test_views))


if __name__ == "__main__":
    main()
