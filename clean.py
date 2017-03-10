from bs4 import BeautifulSoup as BS

out_file_name = 'clean_file.txt'
file_name = 'mcloud.mc'

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
        print(refined_text)
        #print('end')

    return refined_text

refined = prepare_stihiru_data(file_name)
refined = refined.replace('`','')
with open(out_file_name,'w') as out_file:
    out_file.write(refined)
