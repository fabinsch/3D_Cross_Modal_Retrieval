import json
import random
import numpy as np


#TODO convert to GloVe embeddings here

#with open('data.json', 'r') as fp:
#    data = json.load(fp)
#with open('triplets_train.json', 'r') as fp:
#    triplets_train = json.load(fp)
#with open('triplets_val.json', 'r') as fp:
#    triplets_val = json.load(fp)
#with open('triplets_test.json', 'r') as fp:
#    triplets_test = json.load(fp)

#def load_train_data(data, triplets_train):
#    triplets_shape_train = []
#    triplets_description_train = []
#    triplets_id_train = []
#    
#    for key, row in data.items():
#        if key in triplets_train.keys():
#            
#            triplets_shape_train.append(data[key][0])
#            
#            which_desc = random.randint(1, len(data[key])-1)
#            #which_desc = 1
#            
#            triplets_description_train.append(data[key][which_desc])
#            
#            triplets_id_train.append(key)
#            
#            triplets_shape_train.append(data[triplets_train[key]][0])
#            which_desc = random.randint(1, len(data[triplets_train[key]])-1)
#            #which_desc = 1
#            
#            triplets_description_train.append(data[triplets_train[key]][which_desc])
#            triplets_id_train.append(triplets_train[key])
#        
#    return triplets_shape_train, triplets_description_train, triplets_id_train

def load_train_data(data):
    triplets_shape_train = []
    triplets_description_train = []
    triplets_id_train = []
    
    for key, row in data.items():
        
        triplets_shape_train.append(data[key][0])
        
        which_desc = random.randint(1, len(data[key])-1)
        #which_desc = 1
        
        triplets_description_train.append(data[key][which_desc])
        
        triplets_id_train.append(key)
        
    return triplets_shape_train, triplets_description_train, triplets_id_train



#def load_val_data(data, triplets_val):
#    
#    triplets_shape_val = []
#    triplets_description_val = []
#    triplets_id_val = []
#    
#    for key, row in data.items():
#        if key in triplets_val.keys():
#            triplets_shape_val.append(data[key][0])
#            which_desc = random.randint(1, len(data[key])-1)
#            #which_desc = 1
#            triplets_description_val.append(data[key][which_desc])
#            triplets_id_val.append(key)
#            triplets_shape_val.append(data[triplets_val[key]][0])
#            which_desc = random.randint(1, len(data[triplets_val[key]])-1)
#            #which_desc = 1
#            triplets_description_val.append(data[triplets_val[key]][which_desc])
#            triplets_id_val.append(triplets_val[key])
#            
#    return triplets_shape_val, triplets_description_val, triplets_id_val
    
def load_val_data(data):
    
    triplets_shape_val = []
    triplets_description_val = []
    triplets_id_val = []
    
    for key, row in data.items():
        
        triplets_shape_val.append(data[key][0])
        which_desc = random.randint(1, len(data[key])-1)
        #which_desc = 1
        triplets_description_val.append(data[key][which_desc])
        triplets_id_val.append(key)

        
    return triplets_shape_val, triplets_description_val, triplets_id_val

def load_ret_data(data):
    
    triplets_shape_val = []
    triplets_description_val = []
    triplets_id_val = []
    
    for key, row in data.items():
        
        triplets_shape_val.append(data[key][0])
        #which_desc = random.randint(1, len(data[key])-1)
        which_desc = 1
        triplets_description_val.append(data[key][which_desc])
        triplets_id_val.append(key)

        
    return triplets_shape_val, triplets_description_val, triplets_id_val


def load_test_data(data, desc_ind):
    triplets_shape_val = []
    triplets_description_val = []
    triplets_id_val = []

    for key, row in data.items():
        triplets_shape_val.append(data[key][0])
        num_desc = len(data[key]) -1
        if (num_desc) > desc_ind:
            which_desc = desc_ind +1
        else:
            which_desc = num_desc
        triplets_description_val.append(data[key][which_desc])
        triplets_id_val.append(key)

    return triplets_shape_val, triplets_description_val, triplets_id_val


def load_val_samples(data, triplets_val):
    triplets_shape_val = []
    triplets_description_val = []
    triplets_id_val = []

    for key, row in data.items():
        if key in triplets_val.keys():
            triplets_shape_val.append(data[key][0])
            which_desc = random.randint(1, len(data[key]) - 1)
            which_desc = 1
            triplets_description_val.append(data[key][which_desc])
            triplets_id_val.append(key)

    return triplets_shape_val, triplets_description_val, triplets_id_val
        
    
#
#triplets_shape_test = []
#triplets_description_test = []
#triplets_id_test = []
#
#for key, row in data.items():
#    if key in triplets_train.keys():
#        triplets_shape_train.append(data[key][0])
#        which_desc = random.randint(1, len(data[key])-1)
#        which_desc = 1
#        triplets_description_train.append(data[key][which_desc])
#        triplets_id_train.append(key)
#        triplets_shape_train.append(data[triplets_train[key]][0])
#        which_desc = random.randint(1, len(data[triplets_train[key]])-1)
#        which_desc = 1
#        triplets_description_train.append(data[triplets_train[key]][which_desc])
#        triplets_id_train.append(triplets_train[key])
#    elif key in triplets_val.keys():
#        triplets_shape_val.append(data[key][0])
#        which_desc = random.randint(1, len(data[key])-1)
#        triplets_description_val.append(data[key][which_desc])
#        triplets_id_val.append(key)
#        triplets_shape_val.append(data[triplets_val[key]][0])
#        which_desc = random.randint(1, len(data[triplets_val[key]])-1)
#        triplets_description_val.append(data[triplets_val[key]][which_desc])
#        triplets_id_val.append(triplets_val[key])
#    else:
#        triplets_shape_test.append(data[key][0])
#        which_desc = random.randint(1, len(data[key])-1)
#        triplets_description_test.append(data[key][which_desc])
#        triplets_id_test.append(key)
#        triplets_shape_test.append(data[triplets_test[key]][0])
#        which_desc = random.randint(1, len(data[triplets_test[key]])-1)
#        triplets_description_test.append(data[triplets_test[key]][which_desc])
#        triplets_id_test.append(triplets_test[key])


#with open('triplets_shape_val.json', 'w') as fp:
#    json.dump(triplets_shape_val, fp)
#with open('triplets_description_val.json', 'w') as fp:
#    json.dump(triplets_description_val, fp)
#
#with open('triplets_shape_train.json', 'w') as fp:
#    json.dump(triplets_shape_train, fp)
#with open('triplets_description_train.json', 'w') as fp:
#    json.dump(triplets_description_train, fp)
#
#with open('triplets_shape_test.json', 'w') as fp:
#    json.dump(triplets_shape_test, fp)
#with open('triplets_description_test.json', 'w') as fp:
#    json.dump(triplets_description_test, fp)
#
#print(len(triplets_shape_train), "train list generated")
#print(len(triplets_shape_val), "val list generated")
#print(len(triplets_shape_test), "test list generated")