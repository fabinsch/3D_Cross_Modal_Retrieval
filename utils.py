from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import json
from sklearn.neighbors import NearestNeighbors
import os
import torch
import SiameseNet

def retrieve_images(y_pred, ids, data_dir_val, class_dir, num_KNN, max_show, shuffle=True):

    with open(data_dir_val, 'r') as fp:
        data_val = json.load(fp)
    with open(class_dir, 'r') as fp:
        data_class = json.load(fp)
    import matplotlib.pyplot as plt

    y_true = [x for x in range(np.shape(y_pred)[0])]

    fig, axes = plt.subplots(nrows=max_show, ncols=num_KNN+1, figsize=(12, 8))

    if (shuffle == True):
        randomized = np.random.permutation(y_true)
    else:
        randomized = y_true

    for i in range(max_show):
        key = ids[randomized[i]]
        print( i, "ID:", key, " Descr.:", " ".join(data_val[key][1]))
        class_name = data_class[key]
        name = str('images/' + class_name + '/' + key + '/models/model_normalized.png')
        img = Image.open(name)
        axes[i,0].imshow(img)
        axes[i,0].set_title("Ground Truth")
        axes[i,0].set_xticks([])
        axes[i,0].set_yticks([])
        for j in range(num_KNN):
            ID = ids[y_pred[randomized[i]][j]]
            class_name = data_class[ID]
            name = str('images/' + class_name + '/' + ID + '/models/model_normalized.png')
            img = Image.open(name)
            axes[i,j+1].imshow(img)
            axes[i,j+1].set_title("Neighbour %s" %(j+1))
            axes[i,j+1].set_xticks([])
            axes[i,j+1].set_yticks([])

    for i, ax in enumerate(axes[:,0]):
        ax.set_ylabel(str(i), rotation=0, size='large')




def retrieve_one_sentence(net, data_dir_val, working_dir, sentence, class_dir, num_KNN):
    tokens = sentence.lower().split()
    batch_size = net.batch_size
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
    words_clean = []
    for token in tokens:
        # check if word is in GloVe
        if token in vocabulary.keys() and token in embeddings_index.keys():
            words_clean.append(token)

    d_vector = []
    i = 0

    clipping_length = 20 #TODO Increase
    #t0=time.time()

    for word in words_clean:
        if i < clipping_length:
            embedding_vector = embeddings_index.get(word)
            d_vector.append(embedding_vector)
            i += 1
        else:
            break

    if len(d_vector) == 0:
        d_vector = torch.zeros(clipping_length, 50)
    elif len(d_vector) < clipping_length:
        # define desired output dimension
        pad_vector = torch.zeros(clipping_length, 50)
        pad_vector[:len(d_vector), :] = torch.tensor(d_vector)

        d_vector=pad_vector
    else:
        pad_vector = torch.tensor(d_vector)
        d_vector = pad_vector

    points = np.zeros([batch_size, 6, 1000])
    d_vector2 = np.zeros([batch_size, 20, 50])
    d_vector2[0] = d_vector
    points = torch.from_numpy(points).type(torch.FloatTensor)
    d_vector2 = torch.from_numpy(d_vector2).type(torch.FloatTensor)
    with torch.no_grad():
        _, description = net([points, d_vector2],batch_size)
        _, y_pred , ids, shape, _ = SiameseNet.retrieval(net, data_dir_val, working_dir)

    k = 5  # define the rank of retrieval measure

    #description = description[batch_size:, :, :].reshape(len(description), np.shape(description[1])[0])
    ## get 10 nearest neighbor, could also be just k nearest but to experiment left at 10
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(shape)  # check that nbrs are sorted
    distances, indices = nbrs.kneighbors(np.asarray(description.cpu()).squeeze(2)) #description of sentence


    with open(data_dir_val, 'r') as fp:
        data_val = json.load(fp)
    with open(class_dir, 'r') as fp:
        data_class = json.load(fp)
    import matplotlib.pyplot as plt


    fig, axes = plt.subplots(nrows=1, ncols=num_KNN, figsize=(12, 8))

    print("Description:", sentence)

    for j in range(num_KNN):
        ID = ids[y_pred[0][j]]
        class_name = data_class[ID]
        name = str('images/' + class_name + '/' + ID + '/models/model_normalized.png')
        img = Image.open(name)
        axes[j].imshow(img)
        axes[j].set_title("Neighbour %s" %(j+1))
        axes[j].set_xticks([])
        axes[j].set_yticks([])