import sys
import pickle
import numpy as np
import tensorflow as tf
from models.enhanced_model import EnhancedModel, accuracy_function, loss_function
from models.combined_models import ImageNumModel, ImageTextModel, NumTextModel
from models.simple_models import ImageModel, TextModel, NumModel, SimpleModel, SemiSimpleModel


def train(model, train_images, train_text, train_nums, train_views, batch_size=10):
    # TODO: explain using inline comments why loss and accuracy are the same function
    # TODO: add this explanation in enhanced_model.py where accuracy and loss fucntion are defined
    # total_loss = 0
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
        # total_loss += loss
        total_acc += acc

    # avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    # return avg_loss.numpy(), avg_acc.numpy()
    return avg_acc.numpy()


def test(model, test_images, test_text, test_nums, test_views):
    preds = model(test_images, test_text, test_nums)
    acc = accuracy_function(preds, test_views)
    return acc.numpy()


def main(desired_model, desired_learning_rate, desired_batch_size, desired_epochs):

    file_path = './data/data.p'
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

    model = None

    match desired_model:
        case 'EnhancedModel':
            model = EnhancedModel(4096, 5, 128, len(word2idx), 50, desired_learning_rate)
        case 'ImageNumModel':
            model = ImageNumModel(4096, 5, 128, len(word2idx), 50, desired_learning_rate)
        case 'ImageTextModel':
            model = ImageTextModel(4096, 5, 128, len(word2idx), 50, desired_learning_rate)
        case 'NumTextModel':
            model = NumTextModel(4096, 5, 128, len(word2idx), 50, desired_learning_rate)
        case 'ImageModel':
            model = ImageModel(4096, 5, 128, len(word2idx), 50, desired_learning_rate)
        case 'NumModel':
            model = NumModel(32, 5, 128, len(word2idx), 50, desired_learning_rate)
        case 'TextModel':
            model = TextModel(4096, 5, 128, len(word2idx), 50, desired_learning_rate)
        case 'SimpleModel':
            model = SimpleModel(4096, 5, 128, len(word2idx), 50, desired_learning_rate)


    epochs = desired_epochs

    # In order to speed up training, we have implemented a cut-off function that ends training
    # so that we don't needlessly train our models wasting time or overfit the trainning data.
    # due to our previous experieces with how our models perform and learn, plus or minus 2.5%
    # accuracy for 4 epochs was deemed a platuea in minimizing loss

    print("---------------------------TRAIN-----------------------------")
    # accuracy array
    acc = [0] * epochs
    # count for a stable acccuracy
    stable_count = 0
    stable_accuracy = 0
    for i in range(epochs):
        acc[i] = train(model, train_images, train_text, train_nums, train_views, desired_batch_size)
        print(f"---------------------------EPOCH {i}---------------------------")
        print(acc[i])
        print("-------------------------------------------------------------")

        if(acc[i] < stable_accuracy + 2.5) and (acc[i] > stable_accuracy - 2.5):
            stable_count += 1
        else:
            stable_count = 0
            stable_accuracy = acc[i]

        if stable_count >= 4:
            break


    print("---------------------------TEST------------------------------")
    print(test(model, test_images, test_text, test_nums, test_views))


if __name__ == "__main__":

    n = len(sys.argv)

    # default run
    if n == 1:
        desired_model = 'EnhancedModel'
        desired_learning_rate = 0.1
        desired_batch_size = 10
        desired_epochs = 25
        main(desired_model, desired_learning_rate, desired_batch_size, desired_epochs)
        exit()
    
    if n != 5:
        print("usage: python main.py [model] [learning rate] [batch size] [epochs]")
        print("example: python main.py EnhancedModel 0.1 10 25")
        exit()

    models = ['EnhancedModel', 'ImageNumModel', 'ImageTextModel', 'NumTextModel', 'ImageModel', 'NumModel', 'TextModel', 'SimpleModel']
    desired_model = sys.argv[1]
    if desired_model not in models:
        print("usage: python main.py [model] [learning rate] [batch size] [epochs]")
        print("example: python main.py EnhancedModel 0.1 10 25")
        print("[model] must be one of the following: ")
        for model in models:
            print(model)
        exit()

    try:
        desired_learning_rate = float(sys.argv[2])
    except:
        print("usage: python main.py [model] [learning rate] [batch size] [epochs]")
        print("example: python main.py EnhancedModel 0.1 10 25")
        print("[learning rate] must be a float")
        exit()
    if desired_learning_rate < 0 or desired_learning_rate > 1:
        print("usage: python main.py [model] [learning rate] [batch size] [epochs]")
        print("example: python main.py EnhancedModel 0.1 10 25")
        print("[learning rate] must be a float between 0 and 1")
        exit()

    try:
        desired_batch_size = int(sys.argv[3])
        file_path = './data/data.p'
        with open(file_path, 'rb') as data_file:
            data_dict = pickle.load(data_file)
        max_batch_size = len(data_dict['train_images'])
    except:
        print("usage: python main.py [model] [learning rate] [batch size] [epochs]")
        print("example: python main.py EnhancedModel 0.1 10 25")
        print("[batch size] must be an integer")
        exit()
    if desired_batch_size < 1 or desired_batch_size > max_batch_size:
        print("usage: python main.py [model] [learning rate] [batch size] [epochs]")
        print("example: python main.py EnhancedModel 0.1 10 25")
        print(f"[batch size] must be an integer greater than 0 and less than {max_batch_size}")
        exit()

    try:
        desired_epochs = int(sys.argv[4])
    except:
        print("usage: python main.py [model] [learning rate] [batch size] [epochs]")
        print("example: python main.py EnhancedModel 0.1 10 25")
        print("[batch size] must be an integer")
        exit()
    if desired_epochs < 1 or desired_epochs > 100:
        print("usage: python main.py [model] [learning rate] [batch size] [epochs]")
        print("example: python main.py EnhancedModel 0.1 10 25")
        print("[epochs] must be an integer greater than 0 and less than 100")
        exit()

    main(desired_model, desired_learning_rate, desired_batch_size, desired_epochs)
