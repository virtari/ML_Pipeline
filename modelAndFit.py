from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import TensorBoard, EarlyStopping
from keras import backend as K
from keras import regularizers
from keras import layers
from keras import optimizers
import datetime
from time import time

# tensorboard --logdir=g:\BottleProject\logs


start_time = datetime.datetime.now()  # Log the start time
# dimensions of our images.
img_width, img_height = 128, 128
train_data_dir = 'output/Dataset2/TRAIN'
validation_data_dir = 'output/Dataset2/VALIDATE'
nb_train_samples = 53577
nb_validation_samples = 11205

epochs = 30
lamda = 5E-5
batch_size = 50

# Set up TensorBoard
callbacks = [TensorBoard(log_dir="logs/{}".format(time())),
             EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4, verbose=1, mode='auto', baseline=None,
                           restore_best_weights=False)]

print(K.image_data_format())
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)

else:
    input_shape = (img_width, img_height, 3)

#####DEFECT MODEL START#####
model = Sequential()
# Add a dropout layer for input layer
model.add(layers.Dropout(0.2, input_shape=input_shape))
# Convolution layer: 32 filters, kernal size 3 x 3, L2 regularization
model.add(Conv2D(32, (3, 3), input_shape=input_shape, strides=2, kernel_regularizer=regularizers.l2(lamda)))
model.add(Activation('relu'))
# Pooling layer: subsampling 2 x 2, stride 2
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
# Convolution layer: 64 filters, kernal size 3 x 3, L2 regularization
model.add(Conv2D(64, (3, 3), input_shape=input_shape, kernel_regularizer=regularizers.l2(lamda)))
model.add(Activation('relu'))
# Convolution layer: 64 filters, kernal size 3 x 3, L2 regularization
model.add(Conv2D(64, (3, 3), input_shape=input_shape, kernel_regularizer=regularizers.l2(lamda)))
model.add(Activation('relu'))
# Pooling layer: subsampling 2 x 2, stride 2
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
# Convolution layer: 128 filters, kernal size 3 x 3, L2 regularization
model.add(Conv2D(128, (3, 3), input_shape=input_shape, kernel_regularizer=regularizers.l2(lamda)))
model.add(Activation('relu'))
# Convolution layer: 128 filters, kernal size 3 x 3, L2 regularization
model.add(Conv2D(128, (3, 3), input_shape=input_shape, kernel_regularizer=regularizers.l2(lamda)))
model.add(Activation('relu'))
# Pooling layer: subsampling 2 x 2, stride 2
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
# Fully connected layer: 1024 Activation Units
model.add(layers.Dense(units=1024, activation='relu'))
# Dropout layer probability 0.5
model.add(layers.Dropout(0.5))
# Fully connected layer: 1024 Activation Units
model.add(layers.Dense(units=1024, activation='relu'))
# Dropout layer probability 0.5
model.add(layers.Dropout(0.5))
# Add fully connected layer with a sigmoid activation function
model.add(layers.Dense(units=1, activation='sigmoid'))  # org
print(model.summary())

######DEFECT MODEL END######
optimizer = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
#optimizer = optimizers.SGD(lr=0.01, momentum=0.9, decay=0.012, nesterov=False)

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])  # org

# Datagenerators
train_datagen = ImageDataGenerator(rotation_range=5, width_shift_range=0.1, height_shift_range=0.1, rescale=1. / 255,
                                   zoom_range=0.1, horizontal_flip=False, fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale=1. / 255)


train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height),
                                                    batch_size=batch_size, class_mode='binary', shuffle=True)

validation_generator = valid_datagen.flow_from_directory(validation_data_dir, target_size=(img_width, img_height),
                                                        batch_size=batch_size, class_mode='binary', shuffle=True)

model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs,
                    validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size,
                    verbose=2, callbacks=callbacks)

# serialize model to JSON
model_json = model.to_json()
with open("defect_cnn_box2.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights('defect_cnn_box2.h5')

#Finish, and print execution time
end_time = datetime.datetime.now()
print("execution time: " + str(end_time - start_time))
