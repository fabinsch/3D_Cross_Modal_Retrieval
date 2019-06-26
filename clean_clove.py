#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:33:56 2019

@author: fabischramm
"""
import os
import json
import numpy as np

                         
def clean_glove():
    with open(os.path.join(os.getcwd(), 'vocabulary_clean.json'), 'r') as fp:
        vocab = json.load(fp)
    
    glove_50=os.path.join(os.getcwd(), 'glove.6B/glove.6B.50d.txt') #TODO filter words we use
    with open(glove_50, "r+") as f:
        d = f.readlines()
        f.seek(0)
        liste=[]
        for line in d:
            word=line.split()[0]
            #print(line)
            if word in vocab.keys():
                f.write(line)
        f.truncate()

            
#clean_glove()

def create_dict_text_enc():
    embeddings_index = dict()
    embeddings_index_log_nat=dict()
    embeddings_index_log=dict()
    
    glove_50=os.path.join(os.getcwd(), 'glove.6B/glove.6B.50d_clean.txt')
    f = open(glove_50)
    embeddings_index_log_nat[0]='<START>'
    embeddings_index_log_nat[1]='<PAD>'
    embeddings_index_log_nat[2]='<END>'
    start_embedd=[0]+49*[0]
    pad_embedd=[1]+49*[0]
    end_embedd=[2]+49*[0]
    embeddings_index[str(start_embedd)]='<START>'
    embeddings_index[str(pad_embedd)]='<PAD>'
    embeddings_index[str(end_embedd)]='<END>'
    embeddings_index_log[str(start_embedd)]=0
    embeddings_index_log[str(pad_embedd)]=1
    embeddings_index_log[str(end_embedd)]=2
        
    for i, line in enumerate(f, start=3):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[str(coefs)] = word
        embeddings_index_log[str(coefs)]=i
        embeddings_index_log_nat[i]=word
                
    f.close()
    
    with open('dict_coef_ind.json', 'w') as fp:            
        json.dump(embeddings_index_log, fp)
    
    with open('dict_coef_word.json', 'w') as fp:            
        json.dump(embeddings_index, fp)
    
    with open('dict_ind_word.json', 'w') as fp:            
        json.dump(embeddings_index_log_nat, fp)
    
create_dict_text_enc()    