import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import numpy as np


max_words = 500000
max_len = 50000


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(max_len,)),
    layers.ActivityRegularization(l1=0.01),
    layers.Dense(64, activation='relu'),
    layers.ActivityRegularization(l1=0.01),
    layers.Dense(32, activation='relu'),
    layers.ActivityRegularization(l1=0.01),
    layers.Dense(16, activation='relu'),
    layers.ActivityRegularization(l1=0.01),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=keras.optimizers.RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=15, batch_size=50, validation_split=0.2)
L = len(y_test)
correct = 0
YP = model.predict(x_test)
for i in range(L):
    y1 = np.argmax(y_test[i])
    ypred = np.argmax(YP[i])
    if ypred == y1:
        correct += 1
print(correct, ' ', L)
print(correct/L*100)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test Loss: {test_loss:.3f}')
print(f'Test Accuracy: {test_acc:.3f}')

random_index = np.random.randint(0, len(x_test))
random_review = x_test[random_index]
predicted_rating = model.predict(np.array([random_review]))
print("Случайный отзыв:", imdb.get_word_index().keys())
print("Оценка отзыва:", y_test[random_index])
print("Оценка модели:", predicted_rating[0][0])
