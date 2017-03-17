import numpy as np
import tensorflow as tf
from train_lstm import *
import sys

checkpoint = './model_saves/lstm_acce_0.0'

state_size = 100
learning_rate = 0.001
number_of_layers = 3
num_steps = 80
batch_size = 32
num_chars = 1000
num_epochs = 20

arguments = sys.argv
if len(arguments) < 2:
    sys.exit('We need at least one file to accent!')

def accent_text(g,checkpoint,text_to_accent):
    accented_text = ''

    tf.get_variable_scope().reuse_variables()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        g['saver'].restore(sess,checkpoint)

        chars = ''
        state = None
        for char in text_to_accent:
            accented_text+=char
            #char_idx = vocab_to_idx[char]
            char_idx = char_to_idx(char)
            if state is not None:
                feed_dict={g['x']:[[char_idx]],g['init_state']:state}
            else:
                feed_dict={g['x']:[[char_idx]]}
            
            preds,state = sess.run([g['preds'],g['final_state']],feed_dict)

            #if preds[stress_symbol_idx] > 0.09:
                #accented_text+='`'

            #predicted_char = np.random.rand(vocab_size,1,p=np.squeeze(preds))[0]
            preds = np.squeeze(preds)
            predicted_char = preds.argmax()
            chars += idx_to_vocab[predicted_char] + ' '

            if predicted_char == stress_symbol_idx:
                accented_text+='`'

    print('Predicted chars:')
    print(chars)
    print('Accented text:')
    print(accented_text)

    return accented_text

lstm = LSTM_graph(state_size,learning_rate,number_of_layers)

g = lstm.rebuild_the_graph(num_steps = 1, batch_size = 1)

for file_ in arguments[1:]:
    with open(file_) as clean_data:
        text_to_accent = clean_data.read()
        accented_text = lstm.accent_text(g,checkpoint,text_to_accent)

        with open(file_+'.acc','w') as out_file:
            out_file.write(accented_text)
