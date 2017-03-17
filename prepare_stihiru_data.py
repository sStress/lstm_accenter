from os import walk
from bs4 import BeautifulSoup as BS
import re
import time

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
