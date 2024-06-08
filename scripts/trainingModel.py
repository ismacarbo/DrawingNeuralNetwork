import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# Percorso alla directory dei file .npy
percorso = 'C:/Users/INTEL/Desktop/neural network/datas'

# Carica i dati da file .npy
def load_quickdraw_data(categories, percorso, max_samples_per_category=4000):
    X, y = [], []
    for idx, category in enumerate(categories):
        percorsoFile = os.path.join(percorso, f'full_numpy_bitmap_{category}.npy')
        if os.path.exists(percorsoFile):
            data = np.load(percorsoFile)
            data = data[:max_samples_per_category]  # Limita il numero di campioni per categoria
            X.append(data)
            y.append(np.full(data.shape[0], idx))  # etichetta per ogni categoria
        else:
            print(f"File not found: {percorsoFile}")
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y

categories = ['airplane', 'apple', 'axe', 'cat', 'car']  # esempio di categorie

#carico i dati
X, y = load_quickdraw_data(categories, percorso)

X = X / 255.0
X = X.reshape(X.shape[0], 28, 28, 1)
num_classes = len(categories)
y = to_categorical(y, num_classes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

#definizione modello (aumento di neuroni, numero di epoche, e)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),  # + neuroni -> +pensatezza training / +accuracy
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compila il modello
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Checkpoint per salvare il modello
checkpoint = ModelCheckpoint('object_model.keras', save_best_only=True, monitor='val_loss', mode='min')

# Ridurre il learning rate se il modello non migliora
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# Allena il modello
model.fit(datagen.flow(X_train, y_train, batch_size=32), 
          validation_data=(X_test, y_test), 
          steps_per_epoch=len(X_train) // 32, 
          epochs=20,  # Aumentato il numero di epoche
          callbacks=[checkpoint, reduce_lr])  # Utilizzo delle callback
