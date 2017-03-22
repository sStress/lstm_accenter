from train_lstm import *

state_size = 200
learning_rate = 0.001
number_of_layers = 3
num_steps = 100
batch_size = 50
num_chars = 1000
num_epochs = 5

files = []
needed_files = []

directory = '/home/gyroklim/documents/sstress/stihi_stressed_by_machine'
for (_,_,filenames) in walk(directory):
    files.extend(filenames)

for file_ in files:
    if file_.find('z') == 0:
        needed_files.append(directory+'/'+file_)
    if file_.find('y') == 0:
        needed_files.append(directory+'/'+file_)
    if file_.find('x') == 0:
        needed_files.append(directory+'/'+file_)
    if file_.find('a') == 0:
        needed_files.append(directory+'/'+file_)

lstm = LSTM_graph(state_size,learning_rate,number_of_layers)
lstm.prepare_data(needed_files,True)
#lstm.prepare_data('shake.txt',False)

g = lstm.build_the_graph(num_steps,batch_size,with_dropout = True)
lstm.train_network(g,num_epochs,num_steps,batch_size)
