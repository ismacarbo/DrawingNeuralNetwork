import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Dati di esempio (X: input, Y: output)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])  # XOR

# Suddividere i dati in training e test (per un esempio semplice useremo tutti come training)
X_train, Y_train = X, Y

# Definire il modello
model = Sequential()

# Aggiungere i livelli
#la relu è una modalità di attivazione comune nelle reti neurali, permette di far processare piu dati ai neuroni
model.add(Dense(4, input_dim=2, activation='relu'))  # Livello nascosto con 4 neuroni e attivazione ReLU
model.add(Dense(1, activation='sigmoid'))  # Livello di output con attivazione sigmoid

# Compilare il modello
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Allenare il modello
model.fit(X_train, Y_train, epochs=100, batch_size=1)

# Valutare il modello
loss, accuracy = model.evaluate(X_train, Y_train)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Fare previsioni
predictions = model.predict(X_train)
print(predictions)

