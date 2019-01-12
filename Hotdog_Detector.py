import tensorflow as tf
import numpy as np
from PIL import Image
import glob
from keras.models import Sequential
from keras.layers import Reshape, Flatten, Dense, Dropout, ELU
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras import backend as K
K.set_image_dim_ordering('th')
K.set_learning_phase(1)
from Freeze_Graph import freeze_graph



def load_and_format_images(dir_name):
    output_images = []
    for image_name in glob.glob(dir_name + '/*'):
        image = Image.open(image_name).resize((32, 32))
        output_image_data = np.asarray(image, dtype='float32') / 255.0
        output_image_data = np.transpose(output_image_data, (2, 0, 1))
        if output_image_data.shape == (3, 32, 32):
            output_images.append(output_image_data)
    return np.asarray(output_images)



def separate_data(hotdog_images, nothotdog_images):
    train_images = []
    train_labels = []
#    for i in range(35):
#        test_images.append(hotdog_images[i])
#        test_labels.append([1, 0])
#        test_images.append(nothotdog_images[i])
#        test_labels.append([0, 1])
    for i in range(len(hotdog_images)):
        train_images.append(hotdog_images[i])
        train_labels.append([1, 0])
        train_images.append(nothotdog_images[i])
        train_labels.append([0, 1])
    return np.asarray(train_images), np.asarray(train_labels)


#flatten the input
def reshape_images(input_array):
    output_array = []
    for image in input_array:
        output_array.append(image.reshape(-1))
    return np.asarray(output_array)

#create model
def create_model():
    model = Sequential()
    model.add(Reshape(target_shape=(3, 32, 32), input_shape=(3072, )))
    model.add(Conv2D(16 , (2, 2), input_shape=(3, 32, 32), padding='same', activation='relu',
                     kernel_constraint=maxnorm(3)))

    model.add(Conv2D(32,(2, 2), input_shape=(3, 32, 32), padding='same', activation='relu',
                     kernel_constraint=maxnorm(3)))
    model.add(Conv2D(64,(2, 2), input_shape=(3, 32, 32), padding='same', activation='relu',
                     kernel_constraint=maxnorm(3)))
    model.add(Flatten())
    model.add(Dropout(.2))

    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(Dense(2, activation='softmax'))
    return model


# Function to train and test the model
# Train the model with SGD optimizer and categorical crossentropy loss function using train images and labels
# Assess the model's accuracy using the test images and labels and print out the final score
def train_and_assess_model(model, train_images, train_labels, test_images, test_labels):
    epochs = 100
    learning_rate = 0.01
    decay = learning_rate / epochs
    sgd = SGD(lr=learning_rate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())

    model.fit(train_images, train_labels, validation_data=(train_images, train_labels), epochs=epochs, batch_size=12)

    print(model.input.op.name)
    print(model.output.op.name)
    frozen_graph = freeze_graph(K.get_session(), output_names=[model.output.op.name])
    tf.train.write_graph(frozen_graph, '.', 'hotdog_detector.pb', as_text=False)

    scores = model.evaluate(test_images, test_labels, verbose=0)
    print('Accuracy: %.2f%%' % scores[1] * 100)

    return model


# Function to get the results of our model prediction based on test images and compare to test labels
def predict_images(model, test_images, test_labels):
    for i in range(len(test_images)):
        test_image = np.expand_dims(test_images[i], axis=0)
#        print('Predicted label: {}, actual label: {}'.format(model.predict(test_image), test_labels[i]))


def run():
    hotdog_images = load_and_format_images('hot_dog_train')
    nothotdog_images = load_and_format_images('not_hot_dog_train')
    hotdog_images_test = load_and_format_images('hot_dog_test')
    nothotdog_images_test = load_and_format_images('not_hot_dog_test')
    
    train_images, train_labels = separate_data(hotdog_images, nothotdog_images)
    test_images, test_labels = separate_data(hotdog_images_test, nothotdog_images_test)
    train_images = reshape_images(train_images)
    test_images = reshape_images(test_images)

    model = create_model()
    model = train_and_assess_model(model, train_images, train_labels, test_images, test_labels)
    predict_images(model, test_images, test_labels)


run()
