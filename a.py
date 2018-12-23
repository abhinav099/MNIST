from mnist import MNIST
import numpy as np
import tensorflow as tf
#from tf.keras import Sequential
#from tf.keras.layers import Flatten,Dense,Dropout

mndata = MNIST("samples")
x_train,y_train = mndata.load_training()
x_test,y_test = mndata.load_testing()

x_train = np.array(x_train)
x_train = np.reshape(x_train,(60000,28,28))
y_train = np.array(y_train)
y_train = np.reshape(y_train,(60000,))
x_test = np.array(x_test)
x_test = np.reshape(x_test,(10000,28,28))
y_test = np.array(y_test)
y_test = np.reshape(y_test,(10000,))

x_train,x_test =x_train/255.0,x_test/255.0 

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape =(28,28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)