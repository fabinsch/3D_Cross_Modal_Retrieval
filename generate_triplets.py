import json
import random
import numpy as np

def load_train_data(train_name):
##loading json data
    with open(train_name, 'r') as fp:
        data_train = json.load(fp)
        
    return data_train

def load_val_data(val_name):
##loading json data
    with open(val_name, 'r') as fp:
        data_val = json.load(fp)
    
    return data_val

    
# function that generates a list of anchorID and other ID 
def generate_train_triplets(train_name):
    data_train = load_train_data(train_name)
    keys_train = list(data_train.keys())
    keys_train_perm = np.random.permutation(keys_train)
    
    triplets_train = {}
    for key, perm in zip(keys_train,keys_train_perm):
        perm2 = perm
        while key == perm2:
            perm2 = random.choice(keys_train_perm)
        triplets_train[key] = perm2
    print(len(triplets_train), "train triplets generated -> return dict")
    
    return triplets_train

#with open('triplets_train.json', 'w') as fp:
#    json.dump(triplets_train, fp)
def generate_val_triplets(val_name):
    data_val = load_val_data(val_name)
    keys_val = list(data_val.keys())
    keys_val_perm = np.random.permutation(keys_val)
    triplets_val = {}
    for key, perm in zip(keys_val,keys_val_perm):
        perm2 = perm
        while key == perm2:
            perm2 = random.choice(keys_val_perm)
        triplets_val[key] = perm2
    print(len(triplets_val), "val triplets generated")
    
    return triplets_val

#with open('triplets_val.json', 'w') as fp:
#    json.dump(triplets_val, fp)
#
#triplets_test = {}
#for key, perm in zip(keys_test, keys_test_perm):
#    perm2 = perm
#    while key == perm2:
#        perm2 = random.choice(keys_test_perm)
#    triplets_test[key] = perm2
#print(len(triplets_test), "test triplets generated")
#
#with open('triplets_test.json', 'w') as fp:
#    json.dump(triplets_test, fp)
    
#generate_train_triplets()