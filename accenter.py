import numpy as np
import tensorflow as tf
from train_lstm import *

test_file = 'clean_file.txt'
accented_file_name = 'accented.txt'

with open(test_file) as clean_data:
    text_to_accent = clean_data.read()


def accent_text(g,text_to_accent):
    accent_text = ''

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        g['saver'].restore(sess,'./model_checkpoint')

        chars = ''
        state = None
        for char in text_to_accent:
            accent_text+=char
            #char_idx = vocab_to_idx[char]
            char_idx = char_to_idx(char)
            if state is not None:
                feed_dict={g['x']:[[char_idx]],g['init_state']:state}
            else:
                feed_dict={g['x']:[[char_idx]]}
            
            preds,state = sess.run([g['preds'],g['final_state']],feed_dict)

            preds = np.squeeze(preds)
            
            if preds[stress_symbol_idx] > 0.09:
                accent_text+='`'

            #predicted_char = np.random.rand(vocab_size,1,p=np.squeeze(preds))[0]
            preds = np.squeeze(preds)
            predicted_char = preds.argmax()
            chars += idx_to_vocab[predicted_char] + ' '

            if predicted_char == stress_symbol_idx:
                accent_text+='`'

    print('Predicted chars:')
    print(chars)
    print('Accented text:')
    print(accent_text)

    with open(accented_file_name,'w') as acent_file:
        acent_file.write(accent_text)

    return accent_text
        
g = build_lstm_graph(num_steps = 1, batch_size = 1)
accent_text(g,text_to_accent)
