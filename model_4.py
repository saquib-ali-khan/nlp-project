from keras.layers import Dense, Embedding, Input, concatenate, SpatialDropout1D
from keras.layers import Bidirectional, Dropout, CuDNNGRU, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Model
from keras.optimizers import RMSprop


def get_model(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size):
    input_layer = Input(shape=(sequence_length,))
    x = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNGRU(80, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(CuDNNGRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    output_layer = Dense(6, activation="sigmoid")(conc)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model