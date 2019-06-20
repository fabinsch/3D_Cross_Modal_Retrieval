import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from transform_nets import InputTransformNet, FeatureTransformNet
import descriptionnet
import pointnet2
import os
import json
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tensorboard import Tensorboard
import time
from sklearn.metrics import precision_recall_fscore_support
from ndcg_scorer import ndcg_score
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D     

from tensorboardX.writer import SummaryWriter

from generate_triplets import generate_train_triplets, generate_val_triplets
from create_triplet_dataset import load_train_data, load_val_data, load_val_samples
from PIL import Image

import torch.optim as optim
import torchvision.transforms as transforms

#
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()

class SiameseNet(nn.Module):
    def __init__(self, batch_size):
        super(SiameseNet, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.torch.cuda.is_available() else "cpu")
        self.pointNet = pointnet2.PointNet2ClsSsg()
        self.hidden = (torch.randn(2, batch_size, 100).to(self.device), torch.randn(2, batch_size, 100).to(self.device))
        self.lstm = nn.LSTM(input_size=50, hidden_size=100, num_layers=2).to(self.device)
        self.linear = nn.Linear(100, 128).to(self.device)
        self.batch_size = batch_size
        self.fc_c = nn.Linear(128, 13)


    def forward(self, x, batch_size):
        t0 = time.time()
        x_shape = self.pointNet(x[0]).to(self.device)
        t_fp_shape = time.time() - t0
        description = x[1].to(self.device)
        out, hidden = self.lstm(description.permute(1,0,2), self.hidden)

        out = self.linear(out[-1])
        t_fp_desc = time.time() - t0
        #print('fp_s:', t_fp_shape)
        #print('fp_d:', t_fp_desc)

        desc_pred = self.fc_c(out).to(self.device)  # TODO make sure we use the same weights as for shape
        shape_pred = self.fc_c(x_shape.squeeze(2)).to(self.device)
        return x_shape, out.reshape(batch_size,128,1), shape_pred, desc_pred

    
class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, x_shape, x_desc, batch_size, margin):
        diff_pos_1 = x_shape[0:batch_size:2] - x_desc[0:batch_size:2]
        diff_neg_1 = x_shape[0:batch_size:2] - x_desc[1:batch_size:2]
        pos_red_1 = (diff_pos_1 ** 2).sum(1)
        neg_red_1 = (diff_neg_1 ** 2).sum(1)
        diff_pos_2 = x_desc[0:batch_size:2] - x_shape[0:batch_size:2]
        diff_neg_2 = x_desc[0:batch_size:2] - x_shape[1:batch_size:2]
        pos_red_2 = (diff_pos_2 ** 2).sum(1)
        neg_red_2 = (diff_neg_2 ** 2).sum(1)
        '''diff_pos_3 = x_shape[1:batch_size:2] - x_desc[1:batch_size:2]
        diff_neg_3 = x_shape[1:batch_size:2] - x_desc[0:batch_size:2]
        pos_red_3 = (diff_pos_3 ** 2).sum(1)
        neg_red_3 = (diff_neg_3 ** 2).sum(1)
        diff_pos_4 = x_desc[1:batch_size:2] - x_shape[1:batch_size:2]
        diff_neg_4 = x_desc[1:batch_size:2] - x_shape[0:batch_size:2]
        pos_red_4 = (diff_pos_4 ** 2).sum(1)
        neg_red_4 = (diff_neg_4 ** 2).sum(1)'''
        loss = F.relu(pos_red_1 - neg_red_1 + margin).sum(0) + F.relu(pos_red_2 - neg_red_2 + margin).sum(0)# +F.relu(pos_red_3 - neg_red_3 + margin).sum(0) +F.relu(pos_red_4 - neg_red_4 + margin).sum(0)
        return loss

    
class TripletLoss_hard_negative(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss_hard_negative, self).__init__()
        self.margin = margin

    def forward(self, x_shape, x_desc, batch_size, margin, hard_neg_ind):
        diff_pos_1 = x_shape - x_desc
        hard_neg_ind_shapes = (hard_neg_ind[:batch_size]-batch_size).squeeze() # because first 32 indices for the shapes, last 32 for desc
        diff_neg_1 = x_shape - x_desc[hard_neg_ind_shapes]
        pos_red_1 = (diff_pos_1 ** 2).sum(1)
        neg_red_1 = (diff_neg_1 ** 2).sum(1)
        diff_pos_2 = x_desc - x_shape
        hard_neg_ind_desc = hard_neg_ind[batch_size:].squeeze()
        diff_neg_2 = x_desc - x_shape[hard_neg_ind_desc]
        pos_red_2 = (diff_pos_2 ** 2).sum(1)
        neg_red_2 = (diff_neg_2 ** 2).sum(1)
        
#        print('show the hardest negatives')
#        print(hard_neg_ind)
#        print('show active pairs shape as anchor')
#        print(F.relu(pos_red_1 - neg_red_1 + margin)>0)
#        print('show active pairs description as anchor')
#        print(F.relu(pos_red_2 - neg_red_2 + margin)>0)
        
        loss = F.relu(pos_red_1 - neg_red_1 + margin).sum(0) + F.relu(pos_red_2 - neg_red_2 + margin).sum(0)# +F.relu(pos_red_3 - neg_red_3 + margin).sum(0) +F.relu(pos_red_4 - neg_red_4 + margin).sum(0)
        return loss


class pointcloudDataset(Dataset):
    """Point cloud dataset"""

    def __init__(self, json_data, root_dir, mode):
        with open(json_data, 'r') as fp:
            self.data = json.load(fp)
        
        if mode=='train':
            self.objects_shape, self.objects_description, self.train_IDs = load_train_data(self.data)
       
        if mode=='val':
#            self.objects_shape, self.objects_description, self.train_IDs = load_val_data(self.data, d_data)
            self.objects_shape, self.objects_description, self.train_IDs = load_val_data(self.data)

        #if mode=='ret':
            #self.objects_shape, self.objects_description, self.train_IDs = load_val_samples(self.data, d_data)


        self.root_dir = root_dir
        self.device = torch.device("cuda:0" if torch.cuda.torch.cuda.is_available() else "cpu")
        
        # define dict containing GloVe embeddings
        self.embeddings_index = dict()
        glove_50=os.path.join(os.getcwd(), 'glove.6B/glove.6B.50d_clean.txt')
        f = open(glove_50)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()
        print('Loaded %s word vectors.' % len(self.embeddings_index))

    def __len__(self):
        return len(self.objects_shape)

    def __getitem__(self, idx):
        points = torch.tensor(self.objects_shape[idx])
        points = points.view(int(points.shape[0] / 6), 6)

        # get just the positions and reshape
        #points = points[:, 0:3]
        points.transpose_(0, 1)
        # add description and convert to 50 dim vector GloVe
        d_vector = []
        i = 0
        clipping_length = 20 #TODO Increase
        #t0=time.time()

        for word in self.objects_description[idx]:
            if i < clipping_length:
                embedding_vector = self.embeddings_index.get(word)
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
        return [points.to(self.device), d_vector.to(self.device), self.train_IDs[idx]]
    
def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    device = torch.device("cuda:0" if torch.cuda.torch.cuda.is_available() else "cpu")
    dot_product = torch.matmul(embeddings, embeddings.t())

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = torch.diag(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances[distances < 0] = 0

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = distances==0
        distances = distances + mask.float() * 1e-16

        distances = torch.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask.float())

    return distances

def _get_anchor_negative_triplet_mask(pairwise_dist):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    #TODO integrate random suffling - all indices except one set to 1
    device = torch.device("cuda:0" if torch.cuda.torch.cuda.is_available() else "cpu")
    mask = torch.ones(pairwise_dist.shape).to(device)
    dim = list(pairwise_dist.size())
    if dim[0]%2==0:
        active = torch.zeros((int(dim[0]/2),int(dim[1]/2))).to(device)
        active = active + torch.eye((int(dim[0]/2))).to(device)

        mask[:int(dim[0]/2), int(dim[0]/2):] = active
        mask[int(dim[0]/2):, :int(dim[0]/2)] = active
    else:
        
        sys.exit('batchsize not even..aborting')
                  
        
    
    return mask

def batch_hard_triplet_loss(embeddings, margin, squared=False, rand=False):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    device = torch.device("cuda:0" if torch.cuda.torch.cuda.is_available() else "cpu")
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive ---- not applicable in our current implementation
    # First, we need to get a mask for every valid positive (they should have same label)
    #mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    #mask_anchor_positive = tf.to_float(mask_anchor_positive)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    #anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    #hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels) - 0 for valid negativ
    mask_anchor_negative = _get_anchor_negative_triplet_mask(pairwise_dist)
    

    # We add the maximum value in each row to the invalid negatives (to be able to apply min)
    max_anchor_negative_dist, _ = torch.max(pairwise_dist, dim=1, keepdim=True)
    
    if rand==False:
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * mask_anchor_negative
    else:
        anchor_negative_dist = mask_anchor_negative + torch.rand(mask_anchor_negative.shape).to(device)
        

    # shape (batch_size,)
    hardest_negative_dist, hardest_negative_ind  = torch.min(anchor_negative_dist, dim=1, keepdim=True)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    #triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

    # Get final mean triplet loss
    #triplet_loss = tf.reduce_mean(triplet_loss)

    return hardest_negative_ind


def classification_loss(x, shape_pred, desc_pred, class_dict, number_dict):
    device = torch.device("cuda:0" if torch.cuda.torch.cuda.is_available() else "cpu")
    ids = list(x[2])

    lables = torch.zeros(shape_pred.shape[0])
    for i, id in enumerate(ids):
        lables[i] = number_dict[class_dict[id]]
    loss = nn.functional.cross_entropy(shape_pred, lables.long().to(device)) + nn.functional.cross_entropy(desc_pred, lables.long().to(device))

    return loss


def train(net, num_epochs, margin, lr, print_batch, data_dir_train, data_dir_val, writer_suffix, path_to_params, working_dir, class_dir):
    
    writer = SummaryWriter(comment=writer_suffix)  # comment is a suffix, automatically names run with date+suffix
    optimizer = optim.Adam(net.parameters(), lr)
    #criterion = TripletLoss(margin=margin)
    criterion = TripletLoss_hard_negative(margin=margin)
    with open(class_dir, 'r') as fp:
        class_dict = json.load(fp)
    with open('number_dict.json', 'r') as fp:
        number_dict = json.load(fp)
    batch_size = net.batch_size
    
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        running_cl_loss = 0.0
        loss_epoch = 0.0
        val_loss_epoch = 0.0

        #d_train_triplets = generate_train_triplets(data_dir_train)

#        train_data = pointcloudDataset(d_data=d_train_triplets, json_data=data_dir_train, root_dir=working_dir,
#                                       mode='train')
        train_data = pointcloudDataset(json_data=data_dir_train, root_dir=working_dir,
                                       mode='train')
        trainloader = DataLoader(train_data, batch_size=batch_size,
                                 shuffle=True)
        print("Number of training triplets:", int(len(train_data) / 2))

        # fix train val test sets once in one scipt, but not triplets
        # Execute triplet generation for train here without random seed and load epoch data!
        for i_batch, sample_batched in enumerate(trainloader):

            # zero the parameter gradients
            optimizer.zero_grad()
            if len(sample_batched[0]) % batch_size != 0:
                break
            # forward + backward + optimize
            #t0 = time.time()
            x_shape, x_desc, shape_pred, desc_pred = net(sample_batched,batch_size)
            #t_elapsed_fp = time.time() - t0
            # print('forward :',t_elapsed_fp,'s')
            
            #t0 = time.time()
            embeddings = torch.cat((x_shape.squeeze(), x_desc.squeeze()))
            #m_distance = _pairwise_distances(embeddings)
            hard_neg_ind = batch_hard_triplet_loss(embeddings, margin, squared=False, rand=True)
            #t_elapsed_hard_neg = time.time() - t0
            #print(t_elapsed_hard_neg)

            #t0 = time.time()

            loss_cl = classification_loss(sample_batched, shape_pred, desc_pred, class_dict, number_dict)
            loss = criterion(x_shape, x_desc, batch_size, margin, hard_neg_ind) + loss_cl
            #t_elapsed_loss = time.time() - t0
            # print('loss    :',t_elapsed_loss,'s')

            #t0 = time.time()
            loss.backward()
            #t_elapsed_bp = time.time() - t0
            # print('backward:',t_elapsed_bp,'s')

            #if i_batch % print_batch ==0 and i_batch != 0:
             #   plot_grad_flow(net.named_parameters())

            optimizer.step()
            # print statistics
            running_loss += loss.detach().item()
            loss_epoch += loss.detach().item()
            running_cl_loss += loss_cl.detach().item()

            if i_batch % print_batch == 0 and i_batch != 0:  # print every print_batch mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i_batch + 1, running_loss / (print_batch * batch_size)))
                print('[%d, %5d] loss_cl: %.3f' %
                      (epoch + 1, i_batch + 1, running_cl_loss / (print_batch * batch_size)))
                running_loss = 0.0
                running_cl_loss = 0.0

        writer.add_scalar('Train loss per epoch', loss_epoch / (len(train_data) - (len(train_data) % batch_size)),
                          epoch)

        # track validation loss per epoch
        #d_val_triplets = generate_val_triplets(data_dir_val)

#        val_data = pointcloudDataset(d_data=d_val_triplets, json_data=data_dir_val, root_dir=working_dir, mode='val')
        val_data = pointcloudDataset(json_data=data_dir_val, root_dir=working_dir, mode='val')
        valloader = DataLoader(val_data, batch_size=batch_size,
                               shuffle=False)

        # print("Number of validation triplets:", int(len(val_data)/2))
        net.eval()
        print('Doing Evaluation with', len(val_data), 'validation triplets')

        with torch.no_grad():
            #            shape=np.zeros((batch_size,128,1))
            #            description=np.zeros((batch_size,128,1))
            for data in valloader:
                if len(data[0]) % batch_size != 0:
                    break
                output_shape, output_desc, shape_pred, desc_pred = net(data, batch_size)
                embeddings = torch.cat((output_shape.squeeze(), output_desc.squeeze()))
                hard_neg_ind = batch_hard_triplet_loss(embeddings, margin, squared=False, rand=True)
                loss_cl = classification_loss(sample_batched, shape_pred, desc_pred, class_dict, number_dict)
                loss_val = criterion(output_shape, output_desc, batch_size, margin, hard_neg_ind) + loss_cl
                val_loss_epoch += loss_val.item()
            #
            #                shape = np.vstack((shape, np.asarray(output_shape)))
            #                description = np.vstack((description, np.asarray(output_desc)))
            
            if (len(val_data) % batch_size) == len(val_data):
                den = 0
            else:
                den = (len(val_data) % batch_size)
            
            writer.add_scalar('Val loss per epoch', val_loss_epoch / (len(val_data) - den),
                              epoch)
            print("Validation Loss:", val_loss_epoch / (len(val_data) - den))
            if os.path.isfile(path_to_params) and num_epochs > 0:
                torch.save(net.state_dict(), path_to_params)  # Save model Parameters
    writer.close()
    print('Finished Training')
    return net

def val(net, margin, data_dir_val, writer_suffix, working_dir, class_dir, k,  images=False):
    #d_val_triplets = generate_val_triplets(data_dir_val)
    batch_size = net.batch_size
    writer = SummaryWriter(comment=writer_suffix)
    criterion = TripletLoss(margin=margin)
#    val_data = pointcloudDataset(d_data=d_val_triplets, json_data=data_dir_val, root_dir=working_dir, mode='val')
    val_data = pointcloudDataset(json_data=data_dir_val, root_dir=working_dir, mode='val')
    valloader = DataLoader(val_data, batch_size=batch_size,
                                shuffle=False) #TODO must be False
    #
    print("Number of validation triplets:", int(len(val_data)))
    net.eval()
    print('Doing Evaluation')
    #
    with torch.no_grad():
        shape = np.zeros((batch_size, 128, 1))
        description = np.zeros((batch_size, 128, 1))

        for data in valloader:
            if len(data[0]) % batch_size != 0:
                break
            output_shape, output_desc, shape_pred, desc_pred = net(data,batch_size)

            #loss = criterion(output_shape, output_desc, batch_size, margin)
            shape = np.vstack((shape, np.asarray(output_shape.cpu())))
            description = np.vstack((description, np.asarray(output_desc.cpu())))


    # reshape output predictions for kNN
    shape = shape[batch_size:, :, :].reshape(len(shape) - batch_size, np.shape(shape[1])[0])
    description = description[batch_size:, :, :].reshape(len(description) - batch_size, np.shape(shape[1])[0])

    # %%
    # create ground truth and prediction list
    # define the rank of retrieval measure

    # get 10 nearest neighbor, could also be just k nearest but to experiment left at 10
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(shape)  # check that nbrs are sorted
    distances, indices = nbrs.kneighbors(description)

    y_true = []
    y_pred = []
    y_pred2 = [-1] * len(indices)
    for i, ind in enumerate(indices):
        print(i, indices[i])
        y_true.append(i)
        y_pred.append(indices[i])

    # get recall and precision for top k retrievals
    # create y_pred2 by checking if true label is in top k retrievals
    if k != 1:
        for i, pred in enumerate(y_pred):
            for s in pred[:k]:
                if s == [y_true[i]]:
                    y_pred2[i] = s
                    break
    else:
        y_pred2 = y_pred
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred2,
                                                                         average='micro')  # verify that micro is correct, I think for now it's what we need  when just looking at objects from the same class
    print('precision:', precision)
    print('recall:', recall)
    #print('fscore:', fscore)

    ndcg = ndcg_score(y_true, y_pred, k=k)
    print('NDCG:', ndcg)

    # %%

    tags = []

    # visualize validaten data, note that label could still be passed to add_embedding metadata=
    shape = torch.from_numpy(shape).type(torch.FloatTensor)
    out = torch.cat((shape.data, torch.ones(len(shape), 1)), 1)
    with open(class_dir, 'r') as fp:
        class_dict = json.load(fp)
    keys = val_data.train_IDs
    trans = transforms.ToTensor()
    for i in range(int(len(out))):
        class_name = class_dict[keys[i]]
        tags.append(class_name)
        if images:
            name = str('images/' + class_name + '/' + keys[i] + '/models/model_normalized.png')
            img = Image.open(name)
            img.load()  # required for png.split()
            img = img.resize((64,64),Image.ANTIALIAS)
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
            if i ==0:
                label_img = trans(background).unsqueeze(0)
            else:
                label_img = torch.cat((label_img,trans(background).unsqueeze(0)),0)

    if images:
        writer.add_embedding(mat=out,label_img=label_img, tag='shape_val', metadata=tags)
    else:
        writer.add_embedding(mat=out, tag='shape_val', metadata=tags)
    description = torch.from_numpy(description).type(torch.FloatTensor)
    out2 = torch.cat((description.data, torch.ones(len(description), 1)), 1)
    writer.add_embedding(mat=out2, tag='description_val', metadata=tags)
    out3 = torch.cat((out, out2), 0)
    tags = []

    for i in range(int(len(out3)/2)):
        tags.append(str('shape'))  # + str(i)))
    for i in range(int(len(out3)/2)):
        tags.append(str('descr'))  # + str(i)))

    writer.add_embedding(mat=out3, tag='overall_embedding', metadata=tags)
    # close tensorboard writer
    writer.close()

def retrieval(net, data_dir_val, working_dir,print_nn=False):
    batch_size = net.batch_size
    #d_val_samples = generate_val_triplets(data_dir_val)
    val_data = pointcloudDataset(json_data=data_dir_val, root_dir=working_dir, mode='val')
    valloader = DataLoader(val_data, batch_size=batch_size,
                           shuffle=False)
    #
    print("Number of validation triplets:", int(len(val_data)))
    net.eval()
    #
    with torch.no_grad():
        shape = np.zeros((batch_size, 128, 1))
        description = np.zeros((batch_size, 128, 1))
        for data in valloader:
            if len(data[0]) % batch_size != 0:
                break
            output_shape, output_desc, shape_pred, desc_pred = net(data, batch_size)
            shape = np.vstack((shape, np.asarray(output_shape.cpu())))
            description = np.vstack((description, np.asarray(output_desc.cpu())))

    shape = shape[batch_size:, :, :].reshape(len(shape) - batch_size, np.shape(shape[1])[0])
    description = description[batch_size:, :, :].reshape(len(description) - batch_size, np.shape(shape[1])[0])

    # %%
    # create ground truth and prediction list
    k = 5  # define the rank of retrieval measure

    # get 10 nearest neighbor, could also be just k nearest but to experiment left at 10
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(shape)  # check that nbrs are sorted
    distances, indices = nbrs.kneighbors(description)

    y_true = []
    y_pred = []
    y_pred2 = [-1] * len(indices)

    for i, ind in enumerate(indices):
        if (print_nn):
            print(i, indices[i])
        y_true.append(i)
        y_pred.append(indices[i])
    return y_true, y_pred, list(val_data.train_IDs), shape, description

if __name__ == '__main__':
    
    #a = torch.rand(6,2)
    '''shape = torch.tensor([[1, 0, 0],[0, 1, 0],[0, 0, 1],[1, 1, 0],[1, 0, 1],[0,0,0]]).float()
    desc = torch.tensor([[0.5, 0, 1],[0, 1, 1],[1, 1, 1],[0.5, 1, 0],[0.5, 0.5, 1],[0.5,0,0]]).float()
    a=torch.cat((shape, desc))
    dist= _pairwise_distances(a)
    i = batch_hard_triplet_loss(a, 0.1)
    margin=0.5
    criterion = TripletLoss_hard_negative(margin=margin)
    batch_size=6
    loss = criterion(shape.sum(1).unsqueeze(1), desc.sum(1).unsqueeze(1), batch_size, margin, i)
    '''
    
    device = torch.device("cuda:0" if torch.cuda.torch.cuda.is_available() else "cpu")
    
    batch_size = 4
    net = SiameseNet(batch_size)
    suffix = '_test' # comment in if not coming from generating the dataset
    path_to_params = "models/_allClasses1000obj_5000points_100epochs.pt" # if file does not exist or is empty it starts from untrained and later saves to the file
    
    # shift to GPU if available
    
    net.to(device)
    for name, param in net.named_parameters():
        if param.requires_grad:
            print (name)#, param.data)
    working_dir = os.getcwd()
    data_dir_train = os.path.join(working_dir, 'data_train'+suffix+'.json')
    data_dir_val = os.path.join(working_dir, 'data_val'+suffix+'.json')
    class_dir = os.path.join(working_dir, 'class_dict'+suffix+'.json')
    
    if os.path.isfile(path_to_params):
        if os.stat(path_to_params).st_size != 0:
            net.load_state_dict(torch.load(path_to_params, map_location=device))  #Loads pretrained net if file exists and if not empty
    else:
        open(path_to_params, "x") #Creates parameter file if it does not exist
        
    #training parameters
    
    writer_suffix = 'understanding_HN1'
    margin = 0.5
    num_epochs = 1
    print_batch = 1
    lr = 1e-3
    
    net = train(net, num_epochs, margin, lr, print_batch, 
                           data_dir_train, data_dir_val, writer_suffix, path_to_params, working_dir, class_dir)
    
    # Validation
    margin = 0.5
    writer_suffix = 'understanding_HN_1_Val'
    val(net, margin, data_dir_val, writer_suffix, working_dir, class_dir, images=False)

