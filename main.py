import pandas as pd
import numpy as np
import keras
from keras.models import model_from_json
from keras.utils import np_utils as utils
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from model_1 import model_1
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from keras.utils.vis_utils import plot_model
from model_2 import model_2
from model_3 import model_3


width, height = 48, 48
lookup = ('anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral')

map = {
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'happiness',
    4: 'sadness',
    5: 'surprise',
    6: 'neutral'
}


# Method for predictions' visualization
def visualization(predictions, test_x, test_y):
    # Confusion matrix is displayed
    plt.figure(figsize=(10, 10))
    sns.heatmap(confusion_matrix(np.argmax(test_y, axis=1), np.argmax(predictions, axis=1)),
                xticklabels=lookup,
                yticklabels=lookup,
                annot=True,
                fmt='d'
                )
    plt.show()

    pred = np.argmax(predictions, axis=1)
    pred_y = np.argmax(test_y, axis=1)

    print(classification_report(pred, pred_y, digits=3))


# Method for filter visualization
def filter_visualization(model):
    for layer in model.layers:
        # check for convolutional layer
        if 'conv' not in layer.name:
            continue
        # get filter weights
        filters, biases = layer.get_weights()
        print(layer.name, filters.shape)

    fig_1 = plt.figure(figsize=(8, 12))
    c = 8
    r = 8
    n_filters = c * r

    # Filters are shown
    for i in range(1, n_filters + 1):
        f = filters[:, :, :, i - 1]
        fig_1 = plt.subplot(r, c, i)
        fig_1.set_xticks([])
        fig_1.set_yticks([])
        plt.imshow(f[:, :, 0], cmap='gray')

    plt.show()


# Predictions are plotted and compared with the actual label
def plot_predictions(model, test_x, test_y):
    np.random.seed(2)
    random_happy_imgs = np.random.choice(np.where(test_y[:, 3] == 1)[0], size=6)
    random_angry_imgs = np.random.choice(np.where(test_y[:, 0] == 1)[0], size=6)

    fig = plt.figure(1, (12, 4))

    for i, (happidx, angidx) in enumerate(zip(random_happy_imgs, random_angry_imgs)):
        ax = plt.subplot(2, 6, i + 1)
        sample_img = test_x[happidx, :, :, 0]
        ax.imshow(sample_img, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"true:happ., pred:{map[model.predict_classes(sample_img.reshape(1, 48, 48, 1))[0]]}")

        ax = plt.subplot(2, 6, i + 7)
        sample_img = test_x[angidx, :, :, 0]
        ax.imshow(sample_img, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t:anger, p:{map[model.predict_classes(sample_img.reshape(1, 48, 48, 1))[0]]}")

        plt.tight_layout()

    plt.show()


def main(input, model_no):
    df = pd.read_csv('data/fer2013.csv')
    print(df.head())

    train_x = []
    train_y = []
    test_x = []
    test_y = []

    # .csv file is read
    for idx, row in df.iterrows():
        pixel = row['pixels'].split(" ")
        try:
            if 'Training' in row['Usage']:
                train_x.append(np.array(pixel, 'float32'))
                train_y.append(row['emotion'])
            elif 'PublicTest' in row['Usage']:
                test_x.append(np.array(pixel, 'float32'))
                test_y.append(row['emotion'])
        except:
            print(f"error occured at index :{idx} and row:{row}")

    train_x = np.array(train_x, 'float32')
    train_y = np.array(train_y, 'float32')
    test_x = np.array(test_x, 'float32')
    test_y = np.array(test_y, 'float32')

    # Convert a class vector(integers) to binary class matrix,
    # considering 7 possible emotions.
    train_y = utils.to_categorical(train_y, num_classes=7)
    test_y = utils.to_categorical(test_y, num_classes=7)

    # We scale our data
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.fit_transform(test_x)

    # Dataset only contains one channel, so we reshape accordingly
    train_x = train_x.reshape(train_x.shape[0], 48, 48, 1)
    test_x = test_x.reshape(test_x.shape[0], 48, 48, 1)

    # Model is returned here, out of the three existing options
    if model_no == '1':
        model = model_1(train_x.shape[1:])
    elif model_no == '2':
        model = model_2(train_x.shape[1:])
    elif model_no == '3':
        model = model_3(train_x.shape[1:])

    # Model is compiled
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(),
                  metrics=['accuracy'])

    # Weights are loaded, in order to save time from training
    # If training is desired, comment this chunk and uncomment the one below
    if input == '0':
        json_file = open('weights/weights_model' + model_no + '/fer' + model_no + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights('weights/weights_model' + model_no + '/fer' + model_no + '.h5')

        # History gets also loaded
        history = np.load('weights/weights_model' + model_no + '/my_history.npy', allow_pickle='TRUE').item()

        # Plotting training process
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        plt.plot(history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        filter_visualization(model)

    elif input == '1':
        # Training the model, in case no weights are pre-loaded
        history = model.fit(train_x, train_y,
                            batch_size=64,
                            epochs=50,
                            verbose=1,
                            validation_data=(test_x, test_y))

        # Plotting training process
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        # Loss is plotted
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    # We make our prediction + visualizations based on them
    predictions = model.predict(test_x)
    visualization(predictions, test_x, test_y)

    # Model is plotted
    # plot_model(model, show_shapes=True, show_layer_names=True)

    plot_predictions(model, test_x, test_y)


if __name__ == '__main__':
    var = input("Enter 0 to load weights (recommended) or 1 to perform training: ")
    if var == '0':
        print("Weights will be loaded...")
    elif var == '1':
        print("Model will be trained...")
    else:
        print("Wrong number entered")
        exit()

    var2 = input("Enter 1, 2 or 3 as desired model: ")
    if var2 == '1':
        print("Model 1 selected")
    elif var2 == '2':
        print("Model 2 selected")
    elif var2 == '3':
        print("Model 3 selected")
    else:
        print("Wrong number entered")
        exit()

    main(var, var2)
