import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    'C:\\Users\\user\\Desktop\\Workplace\\Machine Learning A-Z™ AI, Python & R\\dataset\\dataset\\training_set',
    target_size=(64, 64), batch_size=32, class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow_from_directory(
    'C:\\Users\\user\\Desktop\\Workplace\\Machine Learning A-Z™ AI, Python & R\\dataset\\dataset\\test_set',
    target_size=(64, 64), batch_size=32, class_mode='binary')

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(64, 64, 3)))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))