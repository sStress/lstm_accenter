from train_lstm import *

state_size = 100
learning_rate = 0.001
number_of_layers = 3
num_steps = 80
batch_size = 32
num_chars = 1000
num_epochs = 20

lstm = LSTM_graph(state_size,learning_rate,number_of_layers)
#lstm.prepare_data('final_file.txt')
lstm.prepare_data('shake.txt',False)

g = lstm.build_the_graph(num_steps,batch_size)
lstm.train_network(g,num_epochs,num_steps,batch_size)
