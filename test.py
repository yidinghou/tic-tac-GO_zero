import keras
# from keras.layers import Input, Dense, Concatenate
# from keras.models import Sequential, Model
# from keras.layers import Dense, Conv1D, Flatten, Conv2D, MaxPooling1D, MaxPooling2D
# from keras.optimizers import SGD
# import keras.backend as K
# from keras import initializers
# from keras.regularizers import l2
# import neural_network

# Input_1 = Input(shape=(3, 3, 1))
#
# x1 = Conv2D(filters=3, kernel_size=(1, 3), activation='relu',
# 			kernel_regularizer=l2(0.0005),
# 			kernel_initializer=initializers.RandomNormal(stddev=0.1, mean=0),
# 			input_shape=(3, 3, 1))(Input_1)
#
# x2 = Conv2D(filters=3, kernel_size=(3, 1), activation='relu',
# 			kernel_regularizer=l2(0.0005),
# 			kernel_initializer=initializers.RandomNormal(stddev=0.1, mean=0),
# 			input_shape=(3, 3, 1))(Input_1)
#
# x3 = Conv2D(filters=8, kernel_size=(3, 3), activation='relu',
# 			kernel_regularizer=l2(0.0005),
# 			kernel_initializer=initializers.RandomNormal(stddev=0.1, mean=0),
# 			input_shape=(3, 3, 1))(Input_1)
#
# x1 = MaxPooling2D((3, 1))(x1)
# x2 = MaxPooling2D((1, 3))(x2)
# x3 = MaxPooling2D((1, 1))(x3)
#
# x = Concatenate()([x1, x2, x3])
# # x = MaxPooling2D((3,1))(x)
# x = Flatten()(x)
#
# value_head = Dense(10, activation='relu')(x)
# # value_head = Dense(10,  activation='relu')(x)
# # value_head = Dense(3,  activation='softmax', name = "V")(value_head)
# value_head = Dense(1, activation='relu', name="V")(value_head)
#
# model = Model(inputs=Input_1, outputs=value_head)
#
# opt = SGD(lr=0.1, momentum=0.09)
# # opt = keras.optimizers.Adam(learning_rate=0.01)
#
# # model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
# model.compile(optimizer=opt, loss='mse', metrics=['acc'])
#
# model.save('./asdf.tf')

keras.models.load_model('.//asdf.tf')
