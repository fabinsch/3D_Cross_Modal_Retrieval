import pandas as pd
from pylanguagetool import api

'''
run in command line the following to start local server

java -cp languagetool-server.jar org.languagetool.server.HTTPServer --port 8081

'''

# test language tool


def check_language(se='sentence'):
    
    
    counter[0]+=1
    

    
    LT_corr = api.check(se, api_url='http://localhost:8081/v2/', lang='en-US', disabled_rules='UPPERCASE_SENTENCE_START')
    #print(LT_corr['matches'])
    
    '''
    ignore typeName
    Hint -> corrects context, redundancy , ADJECTIVE_IN_ATTRIBUTE
    MORFOLOGIK_RULE_EN_US
    PHRASE_REPETITION
    
    '''

    if len(LT_corr['matches'])>0:
        
        e = []
        
        for i, m in enumerate(LT_corr['matches']):
            if m['rule']['issueType']=='misspelling':
                e.append(i)
            else:
                pass
            
        if len(e)>0:
            m = LT_corr['matches'][e[0]]
    
            #print(m['sentence'][m['offset']])
            #print(m['replacements']['value'])
            #print(se[:m['offset']]+se[m['offset']:m['offset']+m['length']].replace(m['sentence'][m['offset']:m['offset']+m['length']], m['replacements'][0]['value'])+se[m['offset']+m['length']:])
            if m['rule']['issueType']=='misspelling':
                
                print(se)
                print(m)
                if len(m['replacements'])>0:
                    c_sentence = se[:m['offset']]+se[m['offset']:m['offset']+m['length']].replace(m['sentence'][m['offset']:m['offset']+m['length']], m['replacements'][0]['value'])+se[m['offset']+m['length']:]
                else:
                    print('')
                    return se
            else: #unnecessary
                c_sentence = se
                
            if counter[0]<10:
                corrected = check_language(c_sentence)
            else:
                corrected = c_sentence
           
        else:
            corrected = se
            
          
        return corrected
    else:
        return se



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
    se = s.replace('&', ' ')
    
    #print(se)
    if index==432:
        print('')
        
    counter=[0]
    c_sentence = check_language(se)
    #print(c_sentence)
    #print('')
    print(index)
    
    s = str(str(row[0]) + ', '+  str(row[1]) +', ' + c_sentence + '\n').lower()
    csv_file.write(s)
    


csv_file.close()