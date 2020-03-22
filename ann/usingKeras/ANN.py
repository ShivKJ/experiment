"""
author: Shiv
email: shivkj001@gmail.com
date: 3/22/20

MIT License

Copyright (c) [2018]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((len(X_train), -1)).astype(float)
X_test = X_test.reshape((len(X_test), -1)).astype(float)

input_dim = X_train.shape[-1]

X_train /= input_dim
X_test /= input_dim

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

model = Sequential()

model.add(Dense(32, input_dim=input_dim, activation='relu'))
model.add(Dense(len(lb.classes_), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, use_multiprocessing=True)
print(dict(zip(model.metrics_names,
               model.evaluate(X_test, y_test, use_multiprocessing=True))))
