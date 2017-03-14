from train_lstm import *
g = build_lstm_graph(num_steps = 100,batch_size = 32)
train_network(g,num_epochs = 20,num_steps = 100)
