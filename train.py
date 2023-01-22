# import the needed packages
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import numpy as np

TRAIN_PATH = "C:\\Users\\enes\Desktop\\githubb\\readymodel\\Keras-ConvolutionalNeuralNetwork-ImageClassificationWithPreTrainedModel\\dataset\\train"
TEST_PATH = "C:\\Users\\enes\Desktop\\githubb\\readymodel\\Keras-ConvolutionalNeuralNetwork-ImageClassificationWithPreTrainedModel\\dataset\\test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Load images for training

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_dataset = train_datagen.flow_from_directory(TRAIN_PATH, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                                                        shuffle=True)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_dataset = test_datagen.flow_from_directory(TEST_PATH, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                                                    shuffle=False)

# Pre-trained model
conv_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in conv_model.layers:
        layer.trainable=False

headModel = conv_model.output
headModel = Flatten()(headModel)
headModel = Dense(256, activation='relu', name='fc1')(headModel)
headModel = Dense(128, activation='relu', name='fc2')(headModel)
headModel = Dense(2, activation='softmax', name='fc3')(headModel)

full_model = Model(inputs=conv_model.input, outputs=headModel)
full_model.summary()

# Train
opt = optimizers.Adam(learning_rate=1e-4)
full_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc',])

full_model.fit(train_dataset, validation_data=test_dataset, epochs=1)

# Save model
full_model.save("model.h5")
