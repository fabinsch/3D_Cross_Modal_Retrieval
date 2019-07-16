obimport torch
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
from create_triplet_dataset import load_train_data, load_val_data, load_val_samples, load_ret_data, load_test_data
from PIL import Image

import torch.optim as optim
import torchvision.transforms as transforms


class SiameseNet(nn.Module):
    def __init__(self, batch_size, num_points):
        super(SiameseNet, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.torch.cuda.is_available() else "cpu")
        self.pointNet = pointnet2.PointNet2ClsSsg()
        self.hidden = (torch.randn(2, batch_size, 100).to(self.device), torch.randn(2, batch_size, 100).to(self.device))
        self.lstm = nn.LSTM(input_size=50, hidden_size=100, num_layers=2).to(self.device)
        self.linear = nn.Linear(100, 128).to(self.device)

        self.batch_size = batch_size
        self.num_points = num_points
        self.glove_size = 5105 + 2  #+2 for end and pad
        self.linear_text_dec = nn.Linear(128, self.glove_size).to(self.device)
        self.hidden_ct = (torch.randn(2, batch_size, 128).to(self.device))
        self.lstm_dec = nn.LSTM(input_size=50, hidden_size=128, num_layers=2).to(self.device)

#        self.seq1 = torch.nn.Sequential(
#        torch.nn.Linear(128, 512),
#        torch.nn.ReLU(),
#        torch.nn.Linear(512, 512),
#        torch.nn.ReLU(),
#        torch.nn.Linear(512, 3*self.num_points),
#        ).to(self.device)
        self.seq1 = torch.nn.Sequential(
        torch.nn.Linear(128, 512),
        nn.Dropout(p=0.5),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 1024)).to(self.device)

    def get_start_vector(self):
        vec = torch.zeros(self.batch_size,50)
        vec[np.arange(self.batch_size), 0] = 0.1
        vec[np.arange(self.batch_size), 1] = -0.1
        return vec.unsqueeze(1).to(self.device)

    def forward(self, x, batch_size):
        t0 = time.time()
        x_shape, x_intermediate = self.pointNet(x[0])
        x_shape = x_shape.to(self.device)
        x_intermediate = x_intermediate.to(self.device)
        t_fp_shape = time.time() - t0
        description = x[1].to(self.device)
        out, hidden = self.lstm(description.permute(1,0,2), self.hidden)
        #d = description.permute(1, 0, 2) # what is that for
        out = self.linear(out[-1])
        t_fp_desc = time.time() - t0

        # Decode embeddings to shape
        shape_dec_pc = self.seq1(x_shape.squeeze(2))
        #shape_dec_pc = shape_dec.view(batch_size, 3,  self.num_points)
        desc_dec_pc = self.seq1(out.reshape(batch_size,128))
        #desc_dec_pc = desc_dec.view(batch_size, 3, self.num_points)

        # Decode embeddings to text
        # fc to go from 128
        teacher = torch.cat((self.get_start_vector(), description), dim=1).permute(1, 0, 2)

        init_hidden_shape = (torch.cat((x_shape.permute(2,0,1),x_shape.permute(2,0,1)),dim=0), self.hidden_ct)
        shape_dec_txt, _ = self.lstm_dec(teacher, init_hidden_shape)

        init_hidden_txt = (torch.cat((out.unsqueeze(0), out.unsqueeze(0)), dim=0), self.hidden_ct)
        desc_dec_txt, _ = self.lstm_dec(teacher, init_hidden_txt)

        steps, bs, hidden_dim = desc_dec_txt.size()
        cats = torch.cat((shape_dec_txt.view(steps*bs,hidden_dim),desc_dec_txt.view(steps*bs,hidden_dim)),dim=0)
        cats = self.linear_text_dec(cats)

        res_txt, res_shape = torch.split(cats, steps*bs, dim=0)

        desc_dec_txt = res_txt.view(steps, bs, self.glove_size)
        shape_dec_txt = res_shape.view(steps, bs, self.glove_size)

        return x_shape, out.view(batch_size,128,1), shape_dec_pc, desc_dec_pc, shape_dec_txt, desc_dec_txt, x_intermediate
    
    '''def get_shape_loss(self, sample_batched, shape_dec_pc, desc_dec_pc):
        if torch.cuda.is_available():
            chamfer_dist = ChamferDistance() # CUDA required
            
            dist1, dist2 = chamfer_dist(sample_batched[0][:,0:3,:], shape_dec_pc)
            loss_s = (torch.mean(dist1)) + (torch.mean(dist2))
            #print('shape dec loss:', loss_s.item())
            
            dist1, dist2 = chamfer_dist(sample_batched[0][:,0:3,:], desc_dec_pc)
            loss_d = (torch.mean(dist1)) + (torch.mean(dist2))
            #print('text dec loss:', loss_d.item())
            
            loss = loss_s + loss_d
            
            
        else:
            loss = 0
        return loss'''
        
    def get_shape_loss(self, x_intermediate, shape_dec_pc, desc_dec_pc, L1=True):
        if L1:
            loss = nn.L1Loss()
        else:
            loss = nn.MSELoss()
        loss_shape_dec = loss(shape_dec_pc, x_intermediate)
        loss_text_dec = loss(desc_dec_pc, x_intermediate) 
        l1_regularization, l2_regularization = torch.tensor(0), torch.tensor(0) # TODO  regularization
#        for param in net.parameters():
#            print(param)
#        
        return (loss_shape_dec+loss_shape_dec)

    def get_txt_loss(self, sample_batched, shape_dec_txt, desc_dec_txt):
        gt = sample_batched[2]
        shape_dec_txt = shape_dec_txt.permute(1,0,2)
        
        desc_dec_txt = desc_dec_txt.permute(1,0,2)
        r,l = gt.shape
        gt = gt.view((r*l,))
        
        shape_dec_txt = shape_dec_txt.reshape((r*l, self.glove_size))
        desc_dec_txt = desc_dec_txt.reshape((r*l, self.glove_size))
        
        loss = nn.functional.cross_entropy(shape_dec_txt, gt.long()) + nn.functional.cross_entropy(desc_dec_txt, gt.long())
        

        return loss

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

    def __init__(self, json_data, root_dir, mode, desc_num=1):
        with open(json_data, 'r') as fp:
            self.data = json.load(fp)

        if mode == 'train':
            self.objects_shape, self.objects_description, self.train_IDs = load_train_data(self.data)

        if mode == 'val':
            self.objects_shape, self.objects_description, self.train_IDs = load_val_data(self.data)

        if mode == 'ret':
            self.objects_shape, self.objects_description, self.train_IDs = load_ret_data(self.data)

        if mode == 'test':
            self.objects_shape, self.objects_description, self.train_IDs = load_test_data(self.data, desc_num)


        self.root_dir = root_dir
        self.device = torch.device("cuda:0" if torch.cuda.torch.cuda.is_available() else "cpu")
        
        # define dict containing GloVe embeddings
        self.embeddings_index = dict()
        self.index_to_word = dict()
        self.index_to_word['0'] = '<pad>'
        #self.index_to_word['1'] = '<start>'
        self.index_to_word['1'] = '<end>'
        glove_50=os.path.join(os.getcwd(), 'glove.6B/glove.6B.50d_clean.txt')
        f = open(glove_50)
        for i, line in enumerate(f,start=2):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = (coefs, i)
            self.index_to_word[str(i)] = word
        f.close()
        #print('Loaded %s word vectors.' % len(self.embeddings_index))
        

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
        gt = torch.zeros(21)
        #gt[0] = 1
        gt[-1] = 1
        for k, word in enumerate(self.objects_description[idx]):
            if i < clipping_length:
                embedding_vector, gt[k] = self.embeddings_index.get(word)
                if embedding_vector is not None:
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
        return [points.to(self.device), d_vector.to(self.device), gt.to(self.device)]
    
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

    device = torch.device("cuda:0" if torch.cuda.torch.cuda.is_available() else "cpu")
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(pairwise_dist)
    
    # We add the maximum value in each row to the invalid negatives (to be able to apply min)
    max_anchor_negative_dist, _ = torch.max(pairwise_dist, dim=1, keepdim=True)
    
    if rand==False:
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * mask_anchor_negative
    else:
        anchor_negative_dist = mask_anchor_negative + torch.rand(mask_anchor_negative.shape).to(device)
        

    # shape (batch_size,)
    hardest_negative_dist, hardest_negative_ind  = torch.min(anchor_negative_dist, dim=1, keepdim=True)
    return (hardest_negative_ind)



def train(net, num_epochs, margin, lr, print_batch, data_dir_train, data_dir_val, writer_suffix, path_to_params, working_dir, class_dir):
    
    writer = SummaryWriter(comment=writer_suffix)  # comment is a suffix, automatically names run with date+suffix
    optimizer = optim.Adam(net.parameters(), lr)
    #criterion = TripletLoss(margin=margin)
    criterion = TripletLoss_hard_negative(margin=margin)
    batch_size = net.batch_size
    path_to_hidden = str(path_to_params[:-3] + '_hidden.pt')
    if (os.path.isfile(path_to_hidden) == False):
        torch.save(net.hidden, path_to_hidden)
    
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        running_txt_loss = 0.0
        running_shape_loss = 0.0
        loss_epoch = 0.0
        loss_epoch_txt = 0.0
        loss_epoch_shape = 0.0
        val_loss_epoch = 0.0
        val_loss_epoch_txt = 0.0
        val_loss_epoch_shape = 0.0
        if (epoch%50 == 0 and epoch >0):
            lr_adapted1 = lr/2
            optimizer = optim.Adam(net.parameters(), lr_adapted1)
            
        if (epoch%90 == 0 and epoch >0):
            lr_adapted2 = lr/4
            optimizer = optim.Adam(net.parameters(), lr_adapted2)
        
        #d_train_triplets = generate_train_triplets(data_dir_train)

#        train_data = pointcloudDataset(d_data=d_train_triplets, json_data=data_dir_train, root_dir=working_dir,
#                                       mode='train')
        train_data = pointcloudDataset(json_data=data_dir_train, root_dir=working_dir,
                                       mode='train')
        trainloader = DataLoader(train_data, batch_size=batch_size,
                                 shuffle=True)
        print("Number of training triplets:", int(len(train_data)))

        # fix train val test sets once in one scipt, but not triplets
        # Execute triplet generation for train here without random seed and load epoch data!
        for i_batch, sample_batched in enumerate(trainloader):

            # zero the parameter gradients
            optimizer.zero_grad()
            if len(sample_batched[0]) % batch_size != 0:
                break
            # forward + backward + optimize
            #t0 = time.time()
            x_shape, x_desc, shape_dec_pc, desc_dec_pc, shape_dec_txt, desc_dec_txt, x_intermediate = net(sample_batched,batch_size)
            #t_elapsed_fp = time.time() - t0
            # print('forward :',t_elapsed_fp,'s')

            #t0 = time.time()
            embeddings = torch.cat((x_shape.squeeze(), x_desc.squeeze()))
            #m_distance = _pairwise_distances(embeddings)
            hard_neg_ind = batch_hard_triplet_loss(embeddings, margin, squared=False, rand=True)
            #t_elapsed_hard_neg = time.time() - t0
            #print(t_elapsed_hard_neg)

            #t0 = time.time()
            
            #Losses:

            if epoch >= 0:
            
                loss_shape = net.get_shape_loss(x_intermediate, shape_dec_pc, desc_dec_pc)
                loss_txt = net.get_txt_loss(sample_batched, shape_dec_txt, desc_dec_txt)
                loss = criterion(x_shape, x_desc, batch_size, margin, hard_neg_ind) + loss_txt + loss_shape
         
            else:
                loss = criterion(x_shape, x_desc, batch_size, margin, hard_neg_ind)
                loss_txt = torch.zeros(1)
                loss_shape = torch.zeros(1)
                
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
            running_txt_loss += loss_txt.detach().item()
            loss_epoch_txt += loss_txt.detach().item()
            running_shape_loss += loss_shape.detach().item()
            loss_epoch_shape  += loss_shape.detach().item()

            if i_batch % print_batch == 0 and i_batch != 0:  # print every print_batch mini-batches
                #####################
                #alayse text decoder:

                gt = sample_batched[2][0].int()
                gt_sentence = [train_data.index_to_word[str(x)] for x in gt.tolist()]
                print("GT Sentence:")
                for token in gt_sentence:
                    print(token, end=" ")
                    
                _, ind = desc_dec_txt.permute(1,0,2).max(2)
                dec_sentence = [train_data.index_to_word[str(x)] for x in ind[0].tolist()]
                print("\n Decoded Sentence:")
                for token in dec_sentence:
                    print(token, end=" ")
                #####################
                
                print("\n")
                
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i_batch + 1, running_loss / (print_batch * batch_size)))
                running_loss = 0.0
                print('[%d, %5d] loss_txt: %.3f' %
                      (epoch + 1, i_batch + 1, running_txt_loss / (print_batch * batch_size)))
                running_txt_loss = 0.0
                print('[%d, %5d] loss_shape: %.3f' %
                      (epoch + 1, i_batch + 1, running_shape_loss / (print_batch * batch_size)))
                running_shape_loss = 0.0

        writer.add_scalar('Train loss per epoch', loss_epoch / (len(train_data) - (len(train_data) % batch_size)),
                          epoch)
        
        writer.add_scalar('Text loss per epoch', loss_epoch_txt / (len(train_data) - (len(train_data) % batch_size)),
                          epoch)
        writer.add_scalar('Shape loss per epoch', loss_epoch_shape / (len(train_data) - (len(train_data) % batch_size)),
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
        
        if os.path.isfile(path_to_params) and num_epochs > 0:
            torch.save(net.state_dict(), path_to_params)  # Save model Parameters
            
        with torch.no_grad():
            #            shape=np.zeros((batch_size,128,1))
            #            description=np.zeros((batch_size,128,1))
            for data in valloader:
                if len(data[0]) % batch_size != 0:
                    break
                output_shape, output_desc, shape_dec_pc, desc_dec_pc, shape_dec_txt, desc_dec_txt, x_intermediate = net(data, batch_size)
                embeddings = torch.cat((output_shape.squeeze(), output_desc.squeeze()))
                hard_neg_ind = batch_hard_triplet_loss(embeddings, margin, squared=False, rand=True)

                if epoch >= 0:
                    loss_shape = net.get_shape_loss(x_intermediate, shape_dec_pc, desc_dec_pc)
                    loss_txt = net.get_txt_loss(data, shape_dec_txt, desc_dec_txt)
                    #print('val_loss_txt:', loss_txt)
                    val_loss_epoch_shape += loss_shape.item()
                    val_loss_epoch_txt += loss_txt.item()
                    loss_val = criterion(output_shape, output_desc, batch_size, margin, hard_neg_ind)+loss_txt +1*loss_shape
                    
                else:
                    loss_val = criterion(output_shape, output_desc, batch_size, margin, hard_neg_ind)
                val_loss_epoch += loss_val.item()
            #
            #                shape = np.vstack((shape, np.asarray(output_shape)))
            #                description = np.vstack((description, np.asarray(output_desc)))
            
            if (len(val_data) % batch_size) == len(val_data):
                den = 0
            else:
                den = (len(val_data) % batch_size)
            print("Validation Loss:", val_loss_epoch / (len(val_data) - den))   
            writer.add_scalar('Shape Val loss per epoch', val_loss_epoch_shape / (len(val_data) - den),
                              epoch)
            writer.add_scalar('Text Val loss per epoch', val_loss_epoch_txt / (len(val_data) - den),
                              epoch)

            print("Text Validation Loss:", val_loss_epoch_txt / (len(val_data) - den))
            print("Shape Validation Loss:", val_loss_epoch_shape / (len(val_data) - den))
            writer.add_scalar('Val loss per epoch', val_loss_epoch / (len(val_data) - den),
                              epoch)



    writer.close()
    print('Finished Training')
    return net

def val(net, margin, data_dir_val, writer_suffix, working_dir, class_dir, k, images=False):
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
    #print('Doing Evaluation')
    #
    with torch.no_grad():
        shape = np.zeros((batch_size, 128, 1))
        description = np.zeros((batch_size, 128, 1))

        for data in valloader:
            if len(data[0]) % batch_size != 0:
                break
            output_shape, output_desc, _, _, _, _, _ = net(data, batch_size)

            #loss = criterion(output_shape, output_desc, batch_size, margin)
            shape = np.vstack((shape, np.asarray(output_shape.cpu())))
            description = np.vstack((description, np.asarray(output_desc.cpu())))


    # reshape output predictions for kNN
    shape = shape[batch_size:, :, :].reshape(len(shape) - batch_size, np.shape(shape[1])[0])
    description = description[batch_size:, :, :].reshape(len(description) - batch_size, np.shape(shape[1])[0])

    # %%
    # create ground truth and prediction list

    # get 10 nearest neighbor, could also be just k nearest but to experiment left at 10
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(shape)  # check that nbrs are sorted
    distances, indices = nbrs.kneighbors(description)

    hit_1 = 0
    hit_5 = 0
    for i, row in enumerate(indices):
        if (i==row[0]):
            hit_1 = hit_1 +1
        if i in row:
            hit_5 = hit_5 +1

    #print('RR@1: ', hit_1 / len(indices))

    #print('RR@5:  ', hit_5/len(indices))

    y_true = list(range(len(indices)))
    mat = np.zeros((len(y_true), len(y_true)))
    for i, row in enumerate(indices):
        fac = 0.9
        for el in row:
            mat[i][el] = fac
            fac = fac / 1.1

    ndcg_5 = ndcg_score(y_true, mat, k=5)


    nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(shape)  # check that nbrs are sorted
    distances, indices = nbrs.kneighbors(description)

    hit_10 = 0
    for i, row in enumerate(indices):
        if i in row:
            hit_10 = hit_10 + 1
    #print('RR@10:  ', hit_10 / len(indices))

    y_true = list(range(len(indices)))
    mat = np.zeros((len(y_true), len(y_true)))
    for i, row in enumerate(indices):
        fac = 0.9
        for el in row:
            mat[i][el] = fac
            fac = fac / 1.1

    ndcg_10 = ndcg_score(y_true, mat, k=10)
    #print('NDCG@5:', ndcg_5)
    #print('NDCG@10:', ndcg_10)

    scores = [hit_1 / len(indices),  hit_5 / len(indices),  hit_10 / len(indices), ndcg_5, ndcg_10]

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

    return scores

def retrieval(net, data_dir_val, working_dir,print_nn=False):
    batch_size = net.batch_size
    #d_val_samples = generate_val_triplets(data_dir_val)
    val_data = pointcloudDataset(json_data=data_dir_val, root_dir=working_dir, mode='ret')
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
            output_shape, output_desc, _, _, _, _, _ = net(data, batch_size)
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


def test(net, margin, data_dir_val, working_dir):
    # d_val_triplets = generate_val_triplets(data_dir_val)
    batch_size = net.batch_size
    criterion = TripletLoss(margin=margin)
    net.eval()
    with torch.no_grad():
        shape = np.zeros((batch_size, 128, 1))
        description = np.zeros((batch_size, 128, 1))
        for desc_num in range(5):
            val_data = pointcloudDataset(json_data=data_dir_val, root_dir=working_dir, mode='test', desc_num=desc_num)
            valloader = DataLoader(val_data, batch_size=batch_size,
                                   shuffle=False)  # TODO must be False
            for data in valloader:
                if len(data[0]) % batch_size != 0:
                    break
                output_shape, output_desc, _, _, _, _, _  = net(data, batch_size)

                # loss = criterion(output_shape, output_desc, batch_size, margin)
                if desc_num == 0:
                    shape = np.vstack((shape, np.asarray(output_shape.cpu())))

                description = np.vstack((description, np.asarray(output_desc.cpu())))


    # reshape output predictions for kNN
    shape = shape[batch_size:, :, :].reshape(len(shape) - batch_size, np.shape(shape[1])[0])
    description = description[batch_size:, :, :].reshape(len(description) - batch_size, np.shape(shape[1])[0])

    number_desc = len(description)
    number_shapes = len(shape)

    # create ground truth and prediction list

    # get 10 nearest neighbor, could also be just k nearest but to experiment left at 10
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(shape)  # check that nbrs are sorted
    distances, indices = nbrs.kneighbors(description)

    hit_1 = 0
    hit_5 = 0

    gt = [list(range(number_shapes)) for x in range(int(number_desc / number_shapes))]
    gt = [item for sublist in gt for item in sublist]

    for i, row in enumerate(indices):
        if (gt[i] == row[0]):
            hit_1 = hit_1 + 1
        if gt[i] in row:
            hit_5 = hit_5 + 1

    # print('RR@1: ', hit_1 / len(indices))

    # print('RR@5:  ', hit_5/len(indices))

    y_true = gt  # list(range(len(indices)))
    mat = np.zeros((len(y_true), len(y_true)))
    for i, row in enumerate(indices):
        fac = 0.9
        for el in row:
            mat[i][el] = fac
            fac = fac / 1.1

    ndcg_5 = ndcg_score(y_true, mat, k=5)

    nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(shape)  # check that nbrs are sorted
    distances, indices = nbrs.kneighbors(description)

    hit_10 = 0
    for i, row in enumerate(indices):
        if gt[i] in row:
            hit_10 = hit_10 + 1
    # print('RR@10:  ', hit_10 / len(indices))

    # y_true = list(range(len(indices)))
    mat = np.zeros((len(y_true), len(y_true)))
    for i, row in enumerate(indices):
        fac = 0.9
        for el in row:
            mat[i][el] = fac
            fac = fac / 1.1

    ndcg_10 = ndcg_score(y_true, mat, k=10)
    # print('NDCG@5:', ndcg_5)
    # print('NDCG@10:', ndcg_10)

    scores = [hit_1 / len(indices), hit_5 / len(indices), hit_10 / len(indices), ndcg_5, ndcg_10]
    return scores
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
    num_points = 1000
    net = SiameseNet(batch_size, num_points)
    suffix = '_test' # comment in if not coming from generating the dataset
    path_to_params = "models/testttt.pt" # if file does not exist or is empty it starts from untrained and later saves to the file
    
    # shift to GPU if available
    
    net.to(device)
    for name, param in net.named_parameters():
        if param.requires_grad:
            print (name)#, param.data)
    working_dir = os.getcwd()
    data_dir_train = os.path.join(working_dir, 'data_train'+suffix+'.json')
    data_dir_val = os.path.join(working_dir, 'data_val'+suffix+'.json')
    class_dir = os.path.join(working_dir, 'class_dict'+suffix+'.json')

    path_to_hidden = str(path_to_params[:-3] + '_hidden.pt')
    if os.path.isfile(path_to_hidden):
        net.hidden = torch.load(path_to_hidden)

    if os.path.isfile(path_to_params):
        if os.stat(path_to_params).st_size != 0:

            net.load_state_dict(torch.load(path_to_params, map_location=device))  #Loads pretrained net if file exists and if not empty
    else:
        open(path_to_params, "x") #Creates parameter file if it does not exist
        
    #training parameters
    
    writer_suffix = 'testing'
    margin = 0.5
    num_epochs = 1
    print_batch = 1
    lr = 1e-3
    k=5
    
    net = train(net, num_epochs, margin, lr, print_batch,
                           data_dir_train, data_dir_val, writer_suffix, path_to_params, working_dir, class_dir)
    
    # Validation
    margin = 0.5
    writer_suffix = 'understanding_HN_1_Val'
    #val(net, margin, data_dir_val, writer_suffix, working_dir, class_dir,k=k, images=False)
    test(net, margin, data_dir_val, working_dir)

