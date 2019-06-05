from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import json

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