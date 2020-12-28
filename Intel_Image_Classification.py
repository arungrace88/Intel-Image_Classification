# Import library
import os
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Defining directory paths
original_dataset_dir = '/Users/fchollet/Downloads/kaggle_original_data'
base_dir = '/media/joann/HDD/01_WORK/02_PROJECTS/02_PYTHON/05_ConvNet/Intel_Image_Classification'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
# Training directory
train_buildings_dir = os.path.join(train_dir, 'buildings')
train_forest_dir = os.path.join(train_dir, 'forest')
train_glacier_dir = os.path.join(train_dir, 'glacier')
train_mountain_dir = os.path.join(train_dir, 'mountain')
train_sea_dir = os.path.join(train_dir, 'sea')
train_street_dir = os.path.join(train_dir, 'street')
# Testing directory
test_buildings_dir = os.path.join(test_dir, 'buildings')
test_forest_dir = os.path.join(test_dir, 'forest')
test_glacier_dir = os.path.join(test_dir, 'glacier')
test_mountain_dir = os.path.join(test_dir, 'mountain')
test_sea_dir = os.path.join(test_dir, 'sea')
test_street_dir = os.path.join(test_dir, 'street')

# ------------------------------------MODEL DEFINITION--------------------#
# Define a Sequential model
model = models.Sequential()
# Add Conv2D layer
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
# Flatten the layer
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(units=512, activation='relu'))
model.add(layers.Dense(units=6, activation='softmax'))
# View the model summary
print(model.summary())
# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

# ------------------------------------IMAGE DATA GENERATION--------------------#
train_datagen = image.ImageDataGenerator(rescale=1. / 255)

test_datagen = image.ImageDataGenerator(rescale=1. / 255)
# Training data generator
train_gen = train_datagen.flow_from_directory(directory=train_dir,
                                              target_size=(150, 150),
                                              batch_size=140, class_mode='categorical')
# Test data generator
test_gen = test_datagen.flow_from_directory(directory=test_dir,
                                            target_size=(150, 150),
                                            batch_size=30, class_mode='categorical')
# Fit the model
history = model.fit_generator(train_gen, steps_per_epoch=100, epochs=25,
                              validation_data=test_gen, validation_steps=100)

# Save the model
model.save('Intel_Image_Classification_epoch30_v2.0.h5')

# Plot the results
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
# Getting the number of epochs
epochs = range(1, len(acc) + 1)
#
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()
#
plt.figure()
#
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
