# %% Importing the necessary libraries:

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import cv2
import os
from sklearn.preprocessing import LabelEncoder
from keras.layers import Conv2D, BatchNormalization, Dense, RandomFlip, RandomTranslation, RandomRotation, GlobalAveragePooling2D, Input, RandomZoom, Rescaling, MaxPooling2D, Flatten
from keras.preprocessing import image_dataset_from_directory
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
# %% Getting all the dataset:

train = image_dataset_from_directory(directory = r"/kaggle/input/brain-tumor-mri-dataset/Training", batch_size=32, image_size=(256, 256))
validation = image_dataset_from_directory(directory = r"/kaggle/input/brain-tumor-mri-dataset/Training", batch_size=32, image_size=(256, 256))
# %% Visualizing the images:

def visual(image, class_name, number):
    for i in range(number):
        plt.subplot(int(np.sqrt(number)), int(np.sqrt(number)), i+1)
        plt.imshow(image[i].numpy().astype('uint8'))
        plt.title(class_name[label[i]])
        plt.axis('off')
    plt.show()
number = 9
class_name = train.class_names

for image, label in train.take(1):
    visual(image, class_name, number)
    img_shape = image.shape
    lab_shape = label.shape

print(img_shape)
print(lab_shape)

# %% Getting the distribution of labels of the dataset:

def counter(path):
    c = 0
    for p in os.scandir(path):
        if p.is_file():
            c += 1
    return c

print(class_name)

glioma_path = r'/kaggle/input/brain-tumor-mri-dataset/Training/glioma'
meningioma_path = r'/kaggle/input/brain-tumor-mri-dataset/Training/meningioma'
notumor_path = r'/kaggle/input/brain-tumor-mri-dataset/Training/notumor'
pituitary_path = r'/kaggle/input/brain-tumor-mri-dataset/Training/pituitary'

glioma_count = counter(glioma_path)
meningioma_count = counter(meningioma_path)
notumor_count = counter(notumor_path)
pituitary_count = counter(pituitary_path)

sn.barplot(x = class_name, y=[glioma_count, meningioma_count, notumor_count, pituitary_count], color='lightgreen', edgecolor='black')
plt.title("The distribution of the labels")
plt.show()
# %% Develop a CNN Architecture:

model = Sequential([
    Input(shape=(256, 256, 3), batch_size=32),
    
    Rescaling(1./255.),
    #RandomFlip('horizontal'),
    #RandomRotation(0.2),
    #RandomZoom(0.2),
    #RandomTranslation(height_factor=0.25,width_factor=0.25, fill_mode='reflect', interpolation='bilinear'),
    
    Conv2D(filters=16, kernel_size=(3,3), activation='relu', name='Conv2D_1'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Conv2D(filters=32, kernel_size=(3,3), activation='relu', name='Conv2D_2'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Conv2D(filters=64, kernel_size=(3,3), activation='relu', name='Conv2D_3'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Conv2D(filters=128, kernel_size=(3,3), activation='relu', name='Conv2D_4'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    
    #GlobalAveragePooling2D(),
    Flatten(),
    
    Dense(units=32, activation='relu'),
    BatchNormalization(),
    Dense(units=64, activation='relu'),
    BatchNormalization(),
    Dense(units=128, activation='relu'),
    BatchNormalization(),
    Dense(units=256, activation='relu'),
    BatchNormalization(),
    Dense(units=128, activation='relu'),
    BatchNormalization(),
    Dense(units=len(class_name), activation='softmax')
    
])
# %% Enhancing the model with improvements:

ES = EarlyStopping(monitor='val_accuracy', patience=10, verbose=2, restore_best_weights=True, mode='max', min_delta=0)
MP = ModelCheckpoint(filepath='Best_model.keras', monitor='val_accuracy', verbose=2, save_best_only=True, mode='max')
RP = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=2, min_lr=0.0001, factor=0.2)
# %% Complie the model before fitting:

model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

history = model.fit(train, validation_data=validation, epochs=20, callbacks=[ES, MP, RP])
# %% Getting the analysis of the performance:

sn.lineplot(x = np.arange(1, len(history.history['accuracy'])+1), y = history.history['accuracy'])
sn.lineplot(x = np.arange(1, len(history.history['val_accuracy'])+1), y = history.history['val_accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title("The Performance of the model")
plt.legend()
plt.show()

sn.lineplot(x = np.arange(1, len(history.history['loss'])+1), y = history.history['loss'])
sn.lineplot(x = np.arange(1, len(history.history['val_loss'])+1), y = history.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.title("The Loss of the model")
plt.legend()
plt.show()
# Prediction:

from tensorflow.keras.models import load_model

predictor = load_model(r'/kaggle/working/Best_model.keras')

image = r'/kaggle/input/brain-tumor-mri-dataset/Testing/pituitary/Te-piTr_0006.jpg'

image = cv2.imread(image)

image1 = cv2.resize(image, (256, 256))
batch_size = 32
batch = np.stack([image1] * batch_size, axis=0)

result = model.predict(batch)
result = (np.argmax(result, axis=1))[0]

label = ""

if result == 0:
    label += "Glioma tumor"
elif result == 1:
    label += "Meningioma tumor"
elif result == 2:
    label += "No Tumor"
elif result == 3:
    label += "Pituitary tumor"

plt.imshow(image)
plt.title(label)
plt.axis('off')
plt.show()
print(f'with a 100% of accuracy we assume that its {label}')
