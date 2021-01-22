import keras
from keras.layers import Input, Dense, Concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Flatten, Conv2D, MaxPooling1D, MaxPooling2D
from keras.optimizers import SGD
from keras import initializers
from keras.regularizers import l2

import pandas as pd
import numpy as np
import board as b

data = pd.read_csv("training_data.csv")

X_str = data["init_state"].values
X_clean = np.array([b.Board.str2arr(x) for x in X_str])

Y_value = data["Value"].values
targets = (Y_value + 1).round()
Y_value_one_hot = np.eye(3)[targets.astype(int)]



Input_1 = Input(shape=(3, 3, 1))

x1 = Conv2D(filters=4, kernel_size=(1, 3), activation='relu',
            kernel_regularizer=l2(0.0005),
            kernel_initializer=initializers.RandomNormal(stddev=0.1, mean=0),
            input_shape=(3, 3, 1))(Input_1)

x2 = Conv2D(filters=4, kernel_size=(3, 1), activation='relu',
            kernel_regularizer=l2(0.0005),
            kernel_initializer=initializers.RandomNormal(stddev=0.1, mean=0),
            input_shape=(3, 3, 1))(Input_1)

x3 = Conv2D(filters=24, kernel_size=(3, 3), activation='relu',
            kernel_regularizer=l2(0.0005),
            kernel_initializer=initializers.RandomNormal(stddev=0.1, mean=0),
            input_shape=(3, 3, 1))(Input_1)

x1 = MaxPooling2D((3, 1))(x1)
x2 = MaxPooling2D((1, 3))(x2)
x3 = MaxPooling2D((1, 1))(x3)

x = Concatenate()([x1, x2, x3])
x = Flatten()(x)

value_head = Dense(100, activation='relu')(x)
value_head = Dense(10, activation='relu')(x)
value_head = Dense(3, activation='softmax', name="V")(x)

model = Model(inputs=Input_1, outputs=value_head)
model.compile(optimizer="adam",
                     loss= "categorical_crossentropy",
                     metrics=['acc'])

X_final = np.stack([X_clean], axis=-1)
model.fit(X_final, Y_value_one_hot, epochs=500, verbose=1)

model.save("./best_keras_model.tf")