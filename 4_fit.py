import time
start_time = time.time()

import os
DIR = r'out_images'
CATEGORIES = os.listdir(DIR)

import pickle

X = pickle.load(open('X.pkl', 'rb'))
y = pickle.load(open('y.pkl', 'rb'))

X = X/255
X = X.reshape(-1, 256, 256, 1)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

from keras.callbacks import TensorBoard
import time

dense_layers = [3]
conv_layers = [3]
neurons = [48]

for dense_layer in dense_layers:
  for conv_layer in conv_layers:
    for neuron in neurons:

      NAME = '{}-denselayer-{}-convlayer-neuron-{}'.format(
          dense_layer,
          conv_layer,
          neuron,
          int(time.time())
      )

      tensorboard = TensorBoard(log_dir = 'logs\\{}'.format(NAME))

      model = Sequential()

      for i in range(conv_layer):
        model.add(Conv2D(neuron, (3,3), activation = 'relu'))
        model.add(MaxPooling2D((2,2)))

      model.add(Flatten())

      model.add(Dense(neuron, input_shape = X.shape[1:], activation = 'relu'))

      for l in range(dense_layer - 1):
        model.add(Dense(neuron, activation = 'relu'))

      model.add(Dense(len(CATEGORIES), activation = 'softmax'))

      model.compile(optimizer = 'adam',
                    loss = 'sparse_categorical_crossentropy',
                    metrics = ['accuracy'])

      model.fit(X, y, epochs = 16, batch_size = 64, validation_split = 0.01, callbacks = [tensorboard])

      model.save('NoJacket.model')

print("--- %s seconds ---" % (time.time() - start_time))
print("--- %s seconds ---" % (time.time() - start_time))
file = open("time.txt", "w")
file.write("--- %s seconds ---" % (time.time() - start_time))
file.close()

# os.system("shutdown /s /t 1") # Если долго будет работать, то можно выключить по завершении