import tensorflow as tf
from train_lstm import *
import numpy as np

state_size = 100
learning_rate = 0.001
number_of_layers = 3
num_steps = 80
batch_size = 21
num_chars = 1000
num_epochs = 1

lstm = LSTM_graph(state_size,learning_rate,number_of_layers)
#lstm.prepare_data('final_file.txt')
#lstm.prepare_data('shake.txt',False)

g = lstm.rebuild_the_graph(num_steps = 1, batch_size = 1)
lstm.generate_characters(g,'./model_saves/lstm_2017-03-16T17:56:25.271829',num_chars = 1000,pick_top_chars = 5,prompt = 'B')
