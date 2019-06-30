#import pymesh
from pyntcloud.io import read_ply
from pyntcloud.io import write_ply
import numpy as np
import pandas as pd
# from plyfile import PlyData, PlyElement
import pywavefront
import sys
import os
import json
import pandas as pd
import random
from operator import itemgetter

def isNaN(num):
    return num != num

def clean_obj(folder_name):
    working_dir = os.getcwd()
    data_dir = working_dir+folder_name
    for subdir in os.listdir(data_dir):
           if subdir=='.DS_Store':
               pass
           else:
               for object in os.listdir(data_dir+'/'+subdir):
                   if object=='.DS_Store':
                       pass
                   else:
                       file_path=data_dir+'/'+subdir+'/'+object+'/models/model_normalized.obj'
                       if os.path.isfile(file_path):
                           with open(file_path, "r+") as f:
                               d = f.readlines()
                               f.seek(0)
                               for i in d[0:1]:
                                   f.write(i)
                               for i in d[2:]:
                                   head, sep, tail = i.partition('/')
                                   if sep == '/':
                                       tail=tail.split(' ')
                                       head2, sep, tail2 = tail[1].partition('/')
                                       head3, sep, tail3 = tail[2].partition('/')
                                       i=head+' '+head2+' '+head3+'\n'
                                     
                                   if 'vt' not in i:
                                        f.write(i)
                               f.truncate()
                               print(object, 'cleaned')
                       else:
                           print(file_path, 'not found')
                               

# use this function once to clean to obj, get only one vertex format                
#clean_obj()
    

def triangle_area_multi(v1, v2, v3):
    return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)


def extract_color(scene):
    dict = {}
    for name, material in scene.materials.items():
        k = 0
        for i, v in enumerate(material.vertices):
            if i + k + 2 >= len(material.vertices):
                break
            dict[(
            material.vertices[i + k], material.vertices[i + k + 1], material.vertices[i + k + 2])] = material.diffuse
            k += 2

        # average over colors
#        for i, v in enumerate(material.vertices):
#            color = material.diffuse
#            if i + k + 2 >= len(material.vertices):
#                break
#            if dict.get((material.vertices[i + k], material.vertices[i + k + 1], material.vertices[i + k + 2])):
#                print('vertex already with color')
#                print(dict.get((material.vertices[i + k], material.vertices[i + k + 1], material.vertices[i + k + 2])))
#                # average over colors TODO think about good solution
#                color=((np.asarray(dict.get((material.vertices[i + k], material.vertices[i + k + 1], material.vertices[i + k + 2])))+np.asarray(material.diffuse))/2).tolist()
#                print('new color')
#                print(color)
#            dict[(
#            material.vertices[i + k], material.vertices[i + k + 1], material.vertices[i + k + 2])] = color
#            k += 2
            
        # define slightly different point - still non-sense, thoses point will never be selected
#        for i, v in enumerate(material.vertices):
#            
#            if i + k + 2 >= len(material.vertices):
#                break
#            if dict.get((material.vertices[i + k], material.vertices[i + k + 1], material.vertices[i + k + 2])):
#                print('vertex already with color')
#                print((material.vertices[i + k], material.vertices[i + k + 1], material.vertices[i + k + 2]))
#
#                dict[(
#                material.vertices[i + k]+0.000001, material.vertices[i + k + 1], material.vertices[i + k + 2])] = material.diffuse
#            else:
#                dict[(
#                material.vertices[i + k], material.vertices[i + k + 1], material.vertices[i + k + 2])] = material.diffuse
#            k += 2
    return dict



def convert_and_sample(path, n=1000, write=False, ret=True):
    # 1 - use pymesh
#    mesh = pymesh.load_mesh(path)
#    points_xyv = mesh.vertices
#    v1_xyz = points_xyv[mesh.faces[:, 0]]
#    v2_xyz = points_xyv[mesh.faces[:, 1]]
#    v3_xyz = points_xyv[mesh.faces[:, 2]]

    # 2 - use pywavefront to get vertices and faces
    # to additionally access color information
    mesh2 = pywavefront.Wavefront(path, collect_faces=True)
    points_xyv2 = np.asarray(mesh2.vertices)
    mesh_wavefront = list(mesh2.meshes.values())
    
    faces = mesh_wavefront[0].faces
    v1_xyz_index = [col[0] for col in faces]
    v1_xyz2 = points_xyv2[v1_xyz_index]
    v2_xyz_index = [col[1] for col in faces]
    v2_xyz2 = points_xyv2[v2_xyz_index]
    v3_xyz_index = [col[2] for col in faces]
    v3_xyz2 = points_xyv2[v3_xyz_index]

    # get color info
    mesh_colors = extract_color(mesh2)
    # v1_rgb = mesh_colors[tuple(v3_xyz2[0].tolist())]
#    v1_rgb = np.asarray([mesh_colors.get(tuple(x.tolist()), [100/255, 100/255, 100/255])[0:3] for x in v1_xyz2])
#    v2_rgb = np.asarray([mesh_colors.get(tuple(x.tolist()), [100/255, 100/255,100/255])[0:3] for x in v2_xyz2])
#    v3_rgb = np.asarray([mesh_colors.get(tuple(x.tolist()), [100/255, 100/255, 100/255])[0:3] for x in v3_xyz2])

    v1_rgb = np.asarray([mesh_colors.get(tuple(x.tolist()), [0/255, 0/255, 0/255])[0:3] for x in v1_xyz2])
    v2_rgb = np.asarray([mesh_colors.get(tuple(x.tolist()), [0/255, 0/255,0/255])[0:3] for x in v2_xyz2])
    v3_rgb = np.asarray([mesh_colors.get(tuple(x.tolist()), [0/255, 0/255, 0/255])[0:3] for x in v3_xyz2])


    # test if two different ways are similar
    # print(points_xyv==points_xyv2)
    # print(v1_xyz== v1_xyz2)
    # print(v2_xyz== v2_xyz2)
    # print(v3_xyz== v3_xyz2)

    # use pywavefront to sample, comment out if you want to use pymesh
    points_xyv = points_xyv2
    v1_xyz = v1_xyz2
    v2_xyz = v2_xyz2
    v3_xyz = v3_xyz2

    areas = triangle_area_multi(v1_xyz, v2_xyz, v3_xyz)
    prob = areas / areas.sum()
    weighted_ind = np.random.choice(range(len(areas)), size=n, p=prob)
    ind = weighted_ind
    v1_xyz = v1_xyz[ind]
    v2_xyz = v2_xyz[ind]
    v3_xyz = v3_xyz[ind]

    v1_rgb = v1_rgb[ind] * 255
    v2_rgb = v2_rgb[ind] * 255
    v3_rgb = v3_rgb[ind] * 255

    u = np.random.rand(n, 1)
    v = np.random.rand(n, 1)
    is_problem = u + v > 1
    u[is_problem] = 1 - u[is_problem]
    v[is_problem] = 1 - v[is_problem]
    w = 1 - (u + v)
    
    result_xyz = (v1_xyz * u) + (v2_xyz * v) + (v3_xyz * w)
    result_xyz = result_xyz.round(5)
    result_rgb = (v1_rgb * u) + (v2_rgb * v) + (v3_rgb * w)
    result_rgb = result_rgb.astype(np.uint8)
    
    result = np.hstack((result_xyz, result_rgb))

    path = path.replace('.obj', '.ply')
    if write:  # write file
        result = pd.DataFrame()
        result_xyz = (v1_xyz * u) + (v2_xyz * v) + (v3_xyz * w)
        result["x"] = result_xyz[:, 0]
        result["y"] = result_xyz[:, 1]
        result["z"] = result_xyz[:, 2]
    
        result_rgb = (v1_rgb * u) + (v2_rgb * v) + (v3_rgb * w)
        result_rgb = result_rgb.astype(np.uint8)
    
        result["red"] = result_rgb[:, 0]
        result["green"] = result_rgb[:, 1]
        result["blue"] = result_rgb[:, 2]

        write_ply(path, points=result, as_text=True)
    if ret:
        return result.tolist()
    
def objects_per_class(class_dict):
    class_occ = {}
    class_samples={}
    for c in class_dict.values():
        if c in class_occ.keys():
            class_occ[c]=class_occ[c]+1
        else:
            class_occ[c]=1
    print(class_occ)
    
        
    # create a dict with class as key and a list of all sample IDs as values
    for obj in class_dict.items():
        if obj[1] in class_samples.keys():
            class_samples[obj[1]].append(obj[0])
        else:
            class_samples[obj[1]]=[obj[0]]
    return (class_occ, class_samples)

def create_dictionary(input_folder, max_elements_per_class, suffix, points_per_object):
# function to convert the obj data to ply and save as json file
    d = {}
    working_dir = os.getcwd()
    data_dir = working_dir+input_folder
    #print(data_dir)
    i=0
    class_dict = {}

    for i, subdir in enumerate(os.listdir(data_dir)):

#       if i > 3: #just look at 4 classes
#           break

       if subdir=='.DS_Store':
           pass

       else:
           for counter, object in enumerate(os.listdir(data_dir+'/'+subdir)):
               if counter >= max_elements_per_class:
                   break
               if object=='.DS_Store':
                   pass
               else:
                   file_path=data_dir+'/'+subdir+'/'+object+'/models/model_normalized.obj'
                   
                   if os.path.isfile(file_path):
                       # Next line samples a nx6 matrix, the matrix gets reshaped to a vector, and is then saved as a first tuple in a list
                       d[object]=[tuple(np.reshape(convert_and_sample(file_path, points_per_object, write=False, ret=True), (points_per_object * 6,)))]
                       print('success:', object, subdir,'#', counter+1)
                       class_dict[object] = subdir
                   else:
                       print(file_path, 'not found')

    # get how many objects per class

    class_occ, class_samples = objects_per_class(class_dict)

    with open('class_dict'+ suffix +'.json', 'w') as fp:
        json.dump(class_dict, fp)

    with open('vocabulary_clean.json', 'r') as fp:
        vocabulary = json.load(fp)
    glove_50 = os.path.join(os.getcwd(), 'glove.6B/glove.6B.50d_clean.txt')  # TODO filter words we use
    f = open(glove_50)

    embeddings_index={}
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    descriptions = pd.read_csv('descriptions/descriptions_cleaned.csv',  encoding='ISO-8859-1',sep=',', error_bad_lines=False, header=None, skiprows=1)
    for index, row in descriptions.iterrows(): #for iterating
        if row[0] in d.keys():
            if not isNaN(row[2]):
                words = row[2].split()
            words_clean = []
            for word in words:
                # check if word is in GloVe
                if word in vocabulary.keys() and word in embeddings_index.keys():
                    words_clean.append(word)
            if len(words_clean) > 0:
                d[row[0]].append(words_clean)
            #d[row[0]].append(row[2].split())
            #print('Added description for: ', row[0])


    print("length dict:",len(d))


    # split into train, val, test with ratio (80,10,10)

    #np.random.seed(10)
    #random.seed(10)

    keys_all = np.random.permutation(list(d.keys()))
    keys_train = []
    keys_val = []
    keys_test = []

    ######### define split normal version ############################################
    for i in range(len(keys_all)):
        if i < len(keys_all) * 0.8:
            keys_train.append(keys_all[i])
        elif i < len(keys_all) * 0.9:
            keys_val.append(keys_all[i])
        else:
            keys_test.append(keys_all[i])
    #################################################################################

    ######## define split to hold out 100 val samples per class #####################
#    for c in class_occ.keys():
#        n = class_occ[c]
#        objects = list(np.random.permutation(class_samples[c])) # random sort of objects in class
#        if n>100: # check if at least 100 are available
#            train = n-100
#        else:
#            train = int(n*0.8)
#        for i in range(train):
#            keys_train.append(objects.pop(0)) # always take the first and pop it out
#            
#        for i in range(n-train):
#            keys_val.append(objects.pop(0))
#        print()
#        print(c, ':', train, 'train samples appended')
#        print(c, ':', n-train, 'val samples appended')
#        print()
     #################################################################################
        

    #keys_train_perm = np.random.permutation(keys_train)
    #keys_val_perm = np.random.permutation(keys_val)
    #keys_test_perm = np.random.permutation(keys_test)

    # create dicts for train,val,test
    d_train={}
    d_val={}
    d_test={}

    for key in keys_train:
        if len(d[key]) > 1: #checks if a at least one description exists for this object --> if not do not add it to the dict
            d_train[key]=d.get(key)

    for key in keys_val:
        if len(d[key]) > 1:
            d_val[key]=d.get(key)

    for key in keys_test:
        if len(d[key]) > 1:
            d_test[key]=d.get(key)

    # problem: c4a41bdc2246d79743b1666db9daca7c has no description

    print("Number of train samples:", len(d_train))
    print("Number of val samples:", len(d_val))
    print("Number of test samples:", len(d_test))
    # save 3 files for train, vak, test separately

    with open('data_train'+ suffix +'.json', 'w') as fp:
        json.dump(d_train, fp)

    with open('data_val'+ suffix +'.json', 'w') as fp:
        json.dump(d_val, fp)

    with open('data_test'+ suffix +'.json', 'w') as fp:
        json.dump(d_test, fp)

    ##
    #
    print()
    print("Created following files:")
    print('data_train'+ suffix +'.json')
    print('data_val'+ suffix +'.json')
    print('data_test'+ suffix +'.json')
    print('class_dict'+ suffix +'.json')
    ###loading json data
    ##with open('data.json', 'r') as fp:
    ##    data_loaded = json.load(fp)
    #
    #loaded_sample = np.reshape(data_loaded[key][0], (int(len(data_loaded[key][0])/6),6))
    ## loaded_sample[0] gives the first row

    # look at this file (FABIAN)
    #convert_and_sample('/Users/fabischramm/Documents/ADL4CV/adl4cv/data/02747177/1cb574d3f22f63ebd493bfe20f94b6ab/models/model_normalized.obj',n=5000, write=True, ret=False)

#create_dictionary(input_folder='/data', max_elements_per_class=10e9, suffix='_100VAL', points_per_object=2000)