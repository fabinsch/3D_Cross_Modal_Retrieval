import pandas as pd
import json


d = {}
d_length = {}

counter=0
threshold=20

def isNaN(num):
    return num != num

descriptions = pd.read_csv('descriptions/descriptions_cleaned.csv',  encoding='ISO-8859-1',sep=',', error_bad_lines=False, header=None, skiprows=1)
for index, row in descriptions.iterrows():
    for words in str(row[2]).split():
        if words in d.keys():
            d[words] = d[words] + 1
        else:
            d[words] = 1
        
    # count occurances of descriptions length
    if not isNaN(row[2]):
        l_sentence = len(row[2].split())
        if l_sentence <= threshold:
            counter=counter+1
    else:
        l_sentence = 'nan'
    if d_length.get(l_sentence, -1) == -1 :
        d_length[l_sentence]=1
    else:
        d_length[l_sentence]+=1
        
    
        
print(index, 'descriptions in total')   
print(counter, 'descriptions are below threshold of', threshold, '. That is', counter/index, 'of the whole descriptions')    
        
# get average of words
sum = 0
for k, v in d_length.items():
    if k!='nan':
        sum = sum + k*v
av = sum / index

print(len(d), 'words in total')
print(av, 'average length of descriptions')

with open('vocabulary.json', 'w') as fp:
    json.dump(d, fp)

d_clean = {}
for keys, value in d.items():
    if value > 2:
        d_clean[keys] = value

print('size of clean vocabulary:', len(d_clean))
with open('vocabulary_clean.json', 'w') as fp:
    json.dump(d_clean, fp)


''''
with open('data.json', 'r') as fp:
    data = json.load(fp)


max_len = 0
for keys, value in data.items():
    for i in range(1,len(value)):
        if max_len < len(value[i]):
            max_len = len(value[i])
print("Maximal lenght of Descriptions:", max_len)'''