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

lstm = LSTM_graph(state_size,learning_rate,number_of_layers)

g = lstm.rebuild_the_graph(num_steps = 1, batch_size = 1)

for file_ in arguments[1:]:
    with open(file_) as clean_data:
        text_to_accent = clean_data.read()
        accented_text = lstm.accent_text(g,checkpoint,text_to_accent,None)

        with open(file_+'.acc','w') as out_file:
            out_file.write(accented_text)
