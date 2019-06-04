#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:33:56 2019

@author: fabischramm
"""
import os
import json

                         
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

            
clean_glove()