import tensorflow as tf
import numpy as np


file_name = 'final_file.txt'
test_file = 'clean_file.txt'

with open(test_file) as clean_data:
    text_to_accent = clean_data.read()

with open(file_name) as data_file:
    raw_data = data_file.read()
    print('Lenght of raw data is {}'.format(len(raw_data)))

vocab = set(raw_data)

print('Vocabulary:')
print(vocab)

vocab_size = len(vocab)
idx_to_vocab = dict(enumerate(vocab))
vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))

stress_symbol_idx = vocab_to_idx['`']

num_steps = 10
batch_size = 200
state_size = 10
learning_rate = 0.1

data = [vocab_to_idx[c] for c in raw_data]
data_size = len(data)
#data_one_hot = tf.one_hot(data,vocab_size)
del raw_data

def gen_batch(batch_size,num_steps):
    batch_partition_size = data_size // batch_size
    epoch_size = batch_partition_size // num_steps

    data_x = np.zeros([batch_size,batch_partition_size],dtype = np.int32)
    data_y = np.zeros([batch_size,batch_partition_size],dtype = np.int32)

    for i in range(batch_size):
        data_x[i] = data[batch_partition_size * i:batch_partition_size * (i + 1)]
        data_y[i] = data[batch_partition_size * i + 1:batch_partition_size * (i + 1) + 1]

    for i in range(epoch_size):
        x = data_x[:,i * num_steps: (i + 1) * num_steps]
        y = data_y[:,i * num_steps: (i + 1) * num_steps]
        yield (x,y)

def gen_epochs(num_epochs, num_steps):
    for i in range(num_epochs):
        yield gen_batch(batch_size,num_steps)

"""
Function to train the network
"""



def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
        tf.reset_default_graph()



def build_lstm_graph():

    reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

    x_one_hot = tf.one_hot(x, vocab_size)
    rnn_inputs = tf.unstack(x_one_hot, axis=1)

    #cell = tf.contrib.rnn.BasicRNNCell(state_size)
    cell = tf.contrib.rnn.BasicLSTMCell(state_size,state_is_tuple = False)
    #cell = tf.contrib.rnn.BasicLSTMCell(state_size)
    init_state = tf.zeros([batch_size, state_size*2])
    #init_state = cell.zero_state(batch_size,tf.float32)
    rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, initial_state=init_state)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, vocab_size])
        b = tf.get_variable('b', [vocab_size], initializer=tf.constant_initializer(0.0))
    logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
    predictions = [tf.nn.softmax(logit) for logit in logits]

    y_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(y, num_steps, 1)]

    to_tf = tf.convert_to_tensor

    loss_weights = [tf.ones([batch_size]) for i in range(num_steps)]
    losses = tf.contrib.seq2seq.sequence_loss(to_tf(logits), to_tf(y_as_list), to_tf(loss_weights))
    #losses = tf.contrib.seq2seq.sequence_loss(logits, y, loss_weights)
    total_loss = tf.reduce_mean(losses)
    train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

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


def train_network(g,num_epochs, num_steps = num_steps, batch_size = batch_size, verbose=True):
    print('Training network!!!!!!!!!!')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps)):
            training_loss = 0
            #training_state = np.zeros((batch_size, state_size*2))
            training_state = None
            if verbose:
                print("\nEPOCH", idx)
            for step, (X, Y) in enumerate(epoch):
                if training_state is not None:
                    feed_dict[g['init_state']] = training_state
                feed_dict = {g['x']:X,g['y']:Y}
                training_loss_, training_state, _ = \
                    sess.run([g['total_loss'],
                              g['final_state'],
                              g['train_step']],
                                  feed_dict)
                training_loss += training_loss_
                if step % 100 == 0 and step > 0:
                    if verbose:
                        print("Average loss at step", step,
                              "for last 250 steps:", training_loss/100)
                    training_losses.append(training_loss/100)
                    training_loss = 0

        g['saver'].save(sess,'./model_checkpoint')

    # now let's try accent something

    return training_losses

def accent_text(g,text_to_accent):
    accent_text = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        g['saver'].restore(sess,'./model_checkpoint')

        state = None
        for char in text_to_accent:
            accent_text.append(char)
            char_idx = vocab_to_idx[char]
            if state is None:
                feed_dict={g['x']:[[char_idx]],g['init_state']:state}
            else:
                feed_dict={g['x']:[[char_idx]]}
            
            pred,state = sess.run([g['preds'],g['final_state']],feed_dict)

            predicted_char = np.randon.rand(vocab_size,1,p=np.squeeze(preds))[0]

            if predicted_char == stress_symbol_idx:
                accent_text.append('`')

    print('Accented text:')
    print(accent_text)

    return accent_text
        
g = build_lstm_graph()
train_network(g,10)
accent_text(g,text_to_accent)
