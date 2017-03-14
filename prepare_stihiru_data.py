from os import walk
from bs4 import BeautifulSoup as BS
import re

def prepare_stihiru_data(file_name):
    """
    Simple function turning data from stihi.ru corpus provided by Orehov
    into more easily handled format
    """
    soup = BS(open(file_name),'html.parser')
    for div in soup('div'):
        refined_text = ''
        text = div.text
        text = text.strip('\n')
        lines = text.split('\n')
        for line in lines[:-1]:
            refined_text = refined_text + line.split('\t')[0] + '\n'
        refined_text = refined_text.strip('\n')
        #print('refined text:')
        #print(refined_text)
        #print('end')

    return refined_text

files = []
needed_files = []

directory = '/home/gyroklim/documents/sstress/stihi_stressed_by_machine'
for (_,_,filenames) in walk(directory):
    files.extend(filenames)

#print('All files:')
#print(files)

def delete_stress(data):
    return data.replace('`','')

for file_ in files:
    if file_.find('mz') == 0:
        needed_files.append(file_)

print('Needed files')
print(needed_files)

final_file_name = 'test_file.txt'
with open(final_file_name,'w') as final_file:
    for file_ in needed_files:
        data = prepare_stihiru_data(directory+'/'+file_)
        data = delete_stress(data)
        final_file.write(data)
