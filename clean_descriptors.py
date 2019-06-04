import pandas as pd
descriptions = pd.read_csv('descriptions/descriptions.csv',  encoding='ISO-8859-1',sep=',', error_bad_lines=False, header=None, skiprows=1)
csv_file = open("descriptions/descriptions_cleaned.csv", "w")
csv_file.write('model_id,synset_id,description \n')

for index, row in descriptions.iterrows(): #for iterating
#    #check problematic row    
#    if row[0]=='20cc098043235921d0efcca115b32b84':
#        print('warning')
    
    char = "'"
    s = str(row[2]).replace(',', ' ')
    s = s.replace('/', ' ')
    s = s.replace('\n', ' ')
    s = s.replace('.', '')
    s = s.replace(';', '')
    s = s.replace('-', ' ')
    s = s.replace('+', ' ')
    s = s.replace('*', ' ')
    s = s.replace('[', ' ')
    s = s.replace(']', ' ')
    s = s.replace('$', ' ')
    s = s.replace('%', ' ')
    s = s.replace('(', ' ')
    s = s.replace(')', ' ')
    s = s.replace(char, '')
    s = s.replace('=', ' ')
    s = s.replace('<', ' ')
    s = s.replace('>', ' ')
    s = s.replace('&', ' ')
    s = str(str(row[0]) + ', '+  str(row[1]) +', ' + s + '\n').lower()
    csv_file.write(s)

csv_file.close()