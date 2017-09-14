from test_parsera import *
from keras.layers.core import Masking
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed

def network():
    X_train = representation_from_DM()
    Y_train = disorder_from_DM()
    X_test = representation_from_DM_test()
    Y_test = disorder_from_DM_test()

    model = Sequential()
    # masking values equal to 0
    model.add(Masking(mask_value=0., batch_input_shape=(None, 588, 100)))
    model.add(LSTM(588, return_sequences=True,input_shape=(588, 100)))

    # TimeDistributed layer for proper output shape
    model.add(TimeDistributed(Dense(3, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit([X_train], [Y_train],epochs=5)
    scores = model.evaluate(X_test, Y_test, batch_size=590, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    # print(model.predict(X_test))

if __name__ == '__main__':
    network()