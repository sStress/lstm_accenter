from train_lstm import *

state_size = 100
learning_rate = 0.001
number_of_layers = 3
num_steps = 80
batch_size = 32
num_chars = 1000
num_epochs = 20

files = []
needed_files = []

directory = '/home/gyroklim/documents/sstress/stihi_stressed_by_machine'
for (_,_,filenames) in walk(directory):
    files.extend(filenames)

for file_ in files:
    if file_.find('z') == 0:
        needed_files.append(directory+'/'+file_)

lstm = LSTM_graph(state_size,learning_rate,number_of_layers)
lstm.prepare_data(needed_files,True)
#lstm.prepare_data('shake.txt',False)

g = lstm.build_the_graph(num_steps,batch_size)
lstm.train_network(g,num_epochs,num_steps,batch_size)
