import tensorflow as tf

import numpy as np
import os
import time
import pickle
import json

tf.enable_eager_execution()

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))





def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text



def train_model():
    path_to_file = './musiclyrics.txt'
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    vocab = sorted(set(text))
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    text_as_int = np.array([char2idx[c] for c in text])
    seq_length = 100
    examples_per_epoch = len(text)//(seq_length+1)

    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

    dataset = sequences.map(split_input_target)

    BATCH_SIZE = 64
    BUFFER_SIZE = 10000

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    # Length of the vocabulary in chars
    vocab_size = len(vocab)

    # The embedding dimension
    embedding_dim = 256

    # Number of RNN units
    rnn_units = 1024
    model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

    data = {
        'vocab_size':len(vocab),
        'embedding_dim':embedding_dim,
        'rnn_units':rnn_units,
        'batch_size':BATCH_SIZE,
        'idx2char':idx2char,
        'char2idx':char2idx,
    }
    pickle_file = open('neuralnet_data.pickle', 'wb') 
    pickle.dump(data, pickle_file)
    pickle_file.close()

    model.compile(optimizer='adam', loss=loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    EPOCHS = 10
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])


    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    model.build(tf.TensorShape([1, None]))

    model.summary()


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model



def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)






def createModel():
    pickle_file = open('neuralnet_data.pickle', 'rb') 
    data = pickle.load(pickle_file)


    # model.compile(optimizer='adam', loss=loss)
    checkpoint_dir = './training_checkpoints'
    model = build_model(
        vocab_size=data['vocab_size'],
        embedding_dim=data['embedding_dim'],
        rnn_units=data['rnn_units'],
        batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))
    
    return model    

def generate_text(model, start_string, num_generate, temperature):
    pickle_file = open('neuralnet_data.pickle', 'rb') 
    data = pickle.load(pickle_file)
    idx2char = np.asarray(data['idx2char'])
    char2idx = data['char2idx']
    num_generate = 1000
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(
            predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


train_model()