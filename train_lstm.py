import tensorflow as tf
import numpy as np
import datetime as dt
import time
import pickle
from os import walk
from bs4 import BeautifulSoup as BS
from random import shuffle
from prepare_stihiru_data import prepare_stihiru_data
import sys

train_data_directory = '/home/gyroklim/documents/sstress/stihi_stressed_by_machine'

vocab_file = 'vocab_data'

class LSTM_graph:

    def __init__(self,state_size,learning_rate,number_of_layers,reduced_vocab=True):
        self.reduced_vocab = False
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.number_of_layers = number_of_layers
        self.checkpoint = './model_saves/lstm_'+dt.datetime.now().isoformat()
        self.reduced_vocab = reduced_vocab

    def char_to_idx(self,char):
        if self.reduced_vocab:
            if char not in self.vocab:
                idx = self.unknown_char_idx
            else:
                idx = self.vocab_to_idx[char]
        else:
            idx = self.vocab_to_idx[char]

        return idx

    def returnVocab(self):
        return self.vocab

    
    # will be more elaborate later (will take multipli files via tensor flow input functions)
    def prepare_data(self,file_names,stressed):

        raw_data_list = []
        self.data_list = []
        vocab = set()
        for file_name in file_names:
            prep_text = prepare_stihiru_data(file_name)
            vocab.upgrade(prep_text)
            raw_data_list.append(prep_text)

        if self.reduced_vocab:
            vocab = set('йцукенгшщзхъфывапролджэячсмитьбюёЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮЁ?!,.:; `~\n')

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.num_classes = self.vocab_size

        self.idx_to_vocab = dict(enumerate(vocab))
        self.vocab_to_idx = dict(zip(self.idx_to_vocab.values(), self.idx_to_vocab.keys()))

        if self.reduced_vocab:
            self.unknown_char_idx = self.vocab_to_idx['~']

        if stressed:
            self.stress_symbol_idx = self.vocab_to_idx['`']

        for raw_data in raw_data_list:
            self.data_list.append([self.char_to_idx(c) for c in raw_data])

        del raw_data_list

        # we need to remember this, cause we'll have to restart the model
        with open(vocab_file,'wb') as vfile:
            pickle.dump(self.vocab_to_idx,vfile)
            pickle.dump(self.idx_to_vocab,vfile)
            pickle.dump(self.vocab,vfile)
            pickle.dump(self.num_classes,vfile)
            pickle.dump(self.vocab_size,vfile)

        self.data_size = 0
        for data in self.data_list:
            self.data_size += len(data)

        print('data length',self.data_size)


    def shuffle_train_data(self):
        # avoiding optimiser bias
        t = time.time()
        shuffle(self.data_list)
        self.data = [var for data_ in self.data_list for var in data_]

    def gen_batch(self,batch_size,num_steps):
        self.shuffle_train_data()
        batch_partition_size = self.data_size // batch_size
        epoch_size = batch_partition_size // num_steps

        data_x = np.zeros([batch_size,batch_partition_size],dtype = np.int32)
        data_y = np.zeros([batch_size,batch_partition_size],dtype = np.int32)

        for i in range(batch_size):
            data_x[i] = self.data[batch_partition_size * i:batch_partition_size * (i + 1)]
            data_y[i] = self.data[batch_partition_size * i + 1:batch_partition_size * (i + 1) + 1]

        for i in range(epoch_size):
            x = data_x[:,i * num_steps: (i + 1) * num_steps]
            y = data_y[:,i * num_steps: (i + 1) * num_steps]
            yield (x,y)


    def gen_epochs(self,num_epochs,batch_size, num_steps):
        for i in range(num_epochs):
            yield self.gen_batch(batch_size,num_steps)


    def reset_graph(self):
        if 'sess' in globals() and sess:
            sess.close()
            tf.reset_default_graph()

    def rebuild_the_graph(self,num_steps = 10, batch_size = 200):

        with open(vocab_file,'rb') as vfile:
            self.vocab_to_idx = pickle.load(vfile)
            self.idx_to_vocab = pickle.load(vfile)
            self.vocab = pickle.load(vfile)
            self.num_classes = pickle.load(vfile)
            self.vocab_size = pickle.load(vfile)

        return self.build_the_graph(num_steps, batch_size)
        

    def build_the_graph(self,num_steps = 10, batch_size = 200):

        self.reset_graph()

        x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
        y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

        #embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size])

        #rnn_inputs = tf.nn.embedding_lookup(embeddings, x)
        rnn_inputs = tf.one_hot(x,self.num_classes)


        cell = tf.contrib.rnn.LSTMCell(self.state_size)
        cell = tf.contrib.rnn.MultiRNNCell([cell]*self.number_of_layers)
        init_state = cell.zero_state(batch_size,tf.float32)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [self.state_size, self.num_classes])
            b = tf.get_variable('b', [self.num_classes], initializer=tf.constant_initializer(0.0))

        rnn_outputs = tf.reshape(rnn_outputs, [-1, self.state_size])
        y_reshaped = tf.reshape(y, [-1])

        logits = tf.matmul(rnn_outputs, W) + b

        predictions = tf.nn.softmax(logits)

        #losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_reshaped,logits=logits)
        losses = tf.losses.sparse_softmax_cross_entropy(y_reshaped,logits)
        #losses = tf.nn.softmax_cross_entropy_with_logits(labels=y_reshaped,logits=logits)
        total_loss = tf.reduce_mean(losses)
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(total_loss)
        #train_step = tf.train.AdagradOptimizer(self.learning_rate).minimize(total_loss)

        return dict(
                x = x,
                y = y,
                init_state = init_state,
                total_loss = total_loss,
                final_state = final_state,
                train_step = train_step,
                preds = predictions,
                saver = tf.train.Saver()
        )

    def train_network(self,g,num_epochs, num_steps, batch_size , verbose=True):
        print('Training network!!!!!!!!!!')
        tf.set_random_seed(2345)
        tf.get_variable_scope().reuse_variables()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            training_losses = []
            for idx, epoch in enumerate(self.gen_epochs(num_epochs,batch_size, num_steps)):
                t = time.time()
                training_loss = 0
                steps = 0
                training_state = None
                if verbose:
                    print("\nEPOCH", idx)
                for X, Y in epoch:
                    steps += 1

                    if training_state is not None:
                        feed_dict[g['init_state']] = training_state
                    feed_dict = {g['x']:X,g['y']:Y}

                    training_loss_, training_state, train_stop = \
                        sess.run([g['total_loss'],
                                  g['final_state'],
                                  g['train_step']],
                                      feed_dict)
                    training_loss += training_loss_
                if verbose:
                    print("Average loss for epoch", idx, training_loss/steps)
                    print("epoch trained for ",(time.time()-t)/60.,"minutes")
                training_losses.append(training_loss/steps)

            g['saver'].save(sess,self.checkpoint)
            print('Saved to',self.checkpoint)

        return training_losses
    

    def generate_characters(self,g, checkpoint, num_chars, prompt='п', pick_top_chars=None):
        """ Accepts a current character, initial state"""

        print('Vocab to index',self.vocab_to_idx)

        #g = self.build_the_graph(num_steps = 1,batch_size = 1)
        tf.get_variable_scope().reuse_variables()
    
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            g['saver'].restore(sess, checkpoint)

            state = None
            current_char = self.vocab_to_idx[prompt]
            chars = [current_char]

            for i in range(num_chars):
                if state is not None:
                    feed_dict={g['x']: [[current_char]], g['init_state']: state}
                else:
                    feed_dict={g['x']: [[current_char]]}

                preds, state = sess.run([g['preds'],g['final_state']], feed_dict)

                if pick_top_chars is not None:
                    p = np.squeeze(preds)
                    p[np.argsort(p)[:-pick_top_chars]] = 0
                    p = p / np.sum(p)
                    current_char = np.random.choice(self.vocab_size, 1, p=p)[0]
                else:
                    current_char = np.random.choice(self.vocab_size, 1, p=np.squeeze(preds))[0]

                chars.append(current_char)

        chars = map(lambda x: self.idx_to_vocab[x], chars)
        print("".join(chars))
        return("".join(chars))

    def accent_text(self,g,checkpoint,text_to_accent,pick_top_chars = None):
        accent_text = ''

        tf.get_variable_scope().reuse_variables()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            g['saver'].restore(sess,checkpoint)
            self.unknown_char_idx = self.vocab_to_idx['~']

            state = None

            for char in text_to_accent:
                accent_text+=char
                #char_idx = vocab_to_idx[char]
                char_idx = self.char_to_idx(char)
                if state is not None:
                    feed_dict={g['x']:[[char_idx]],g['init_state']:state}
                else:
                    feed_dict={g['x']:[[char_idx]]}
                
                preds,state = sess.run([g['preds'],g['final_state']],feed_dict)

                if pick_top_chars is not None:
                    predicted_char = []
                    p = np.squeeze(preds)
                    p[np.argsort(p)[:-pick_top_chars]] = 0
                    predicted_chars_idx = np.nonzero(p)
                    predicted_char.extend(predicted_chars_idx[0])
                    predicted_char = list(map(lambda idx: self.idx_to_vocab[idx],predicted_char))
                    #p[np.argsort(p)[:-pick_top_chars]] = 0
                    #p = p / np.sum(p)
                    #predicted_char = np.random.choice(self.vocab_size, 1, p=p)[0]
                else:
                    predicted_char = np.random.choice(self.vocab_size, 1, p=np.squeeze(preds))[0]

                if pick_top_chars is not None:
                    if '`' in predicted_char:
                        accent_text+='`'

                else:
                    if predicted_char == self.vocab_to_idx['`']:
                        accent_text+='`'

        print('Accented text:')
        print(accent_text)

        return accent_text
