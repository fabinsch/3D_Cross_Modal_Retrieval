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

from tensorboardX.writer import SummaryWriter

from generate_triplets import generate_train_triplets, generate_val_triplets
from create_triplet_dataset import load_train_data, load_val_data, load_val_samples

import torch.optim as optim

#

class SiameseNet(nn.Module):
    def __init__(self, batch_size):
        super(SiameseNet, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.torch.cuda.is_available() else "cpu")
        self.pointNet = pointnet2.PointNet2ClsSsg()
        self.hidden = (torch.randn(2, batch_size, 100).to(self.device), torch.randn(2, batch_size, 100).to(self.device))
        self.lstm = nn.LSTM(input_size=50, hidden_size=100, num_layers=2).to(self.device)
        self.linear = nn.Linear(100, 128).to(self.device)

        self.batch_size = batch_size


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

        return x_shape, out.reshape(batch_size,128,1)
    
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


class pointcloudDataset(Dataset):
    """Point cloud dataset"""

    def __init__(self, d_data, json_data, root_dir, mode):
        with open(json_data, 'r') as fp:
            self.data = json.load(fp)
        
        if mode=='train':
            self.objects_shape, self.objects_description, self.train_IDs = load_train_data(self.data, d_data)
       
        if mode=='val':
            self.objects_shape, self.objects_description, self.train_IDs = load_val_data(self.data, d_data)

        if mode=='ret':
            self.objects_shape, self.objects_description, self.train_IDs = load_val_samples(self.data, d_data)


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
        return [points.to(self.device), d_vector.to(self.device)]

def train(net, num_epochs, margin, lr, print_batch, data_dir_train, data_dir_val, writer_suffix, path_to_params, working_dir):
    
    writer = SummaryWriter(comment=writer_suffix)  # comment is a suffix, automatically names run with date+suffix
    optimizer = optim.Adam(net.parameters(), lr)
    criterion = TripletLoss(margin=margin)

    batch_size = net.batch_size
    
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        loss_epoch = 0.0
        val_loss_epoch = 0.0

        d_train_triplets = generate_train_triplets(data_dir_train)

        train_data = pointcloudDataset(d_data=d_train_triplets, json_data=data_dir_train, root_dir=working_dir,
                                       mode='train')
        trainloader = DataLoader(train_data, batch_size=batch_size,
                                 shuffle=False)
        print("Number of training triplets:", int(len(train_data) / 2))

        # fix train val test sets once in one scipt, but not triplets
        # Execute triplet generation for train here without random seed and load epoch data!
        for i_batch, sample_batched in enumerate(trainloader):

            # zero the parameter gradients
            optimizer.zero_grad()
            if len(sample_batched[0]) % batch_size != 0:
                break
            # forward + backward + optimize
            t0 = time.time()
            x_shape, x_desc = net(sample_batched,batch_size)
            t_elapsed_fp = time.time() - t0
            # print('forward :',t_elapsed_fp,'s')

            t0 = time.time()
            loss = criterion(x_shape, x_desc, batch_size, margin)
            t_elapsed_loss = time.time() - t0
            # print('loss    :',t_elapsed_loss,'s')

            t0 = time.time()
            loss.backward()
            t_elapsed_bp = time.time() - t0
            # print('backward:',t_elapsed_bp,'s')

            optimizer.step()
            # print statistics
            running_loss += loss.item()
            loss_epoch += loss.item()

            if i_batch % print_batch == 0 and i_batch != 0:  # print every print_batch mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i_batch + 1, running_loss / (print_batch * batch_size)))
                running_loss = 0.0

        writer.add_scalar('Train loss per epoch', loss_epoch / (len(train_data) - (len(train_data) % batch_size)),
                          epoch)

        # track validation loss per epoch
        d_val_triplets = generate_val_triplets(data_dir_val)

        val_data = pointcloudDataset(d_data=d_val_triplets, json_data=data_dir_val, root_dir=working_dir, mode='val')
        valloader = DataLoader(val_data, batch_size=batch_size,
                               shuffle=False)

        # print("Number of validation triplets:", int(len(val_data)/2))
        net.eval()
        print('Doing Evaluation with', len(val_data) / 2, 'validation triplets')

        with torch.no_grad():
            #            shape=np.zeros((batch_size,128,1))
            #            description=np.zeros((batch_size,128,1))
            for data in valloader:
                if len(data[0]) % batch_size != 0:
                    break
                output_shape, output_desc = net(data, batch_size)
                loss_val = criterion(output_shape, output_desc, batch_size, margin)
                val_loss_epoch += loss_val.item()
            #
            #                shape = np.vstack((shape, np.asarray(output_shape)))
            #                description = np.vstack((description, np.asarray(output_desc)))
            writer.add_scalar('Val loss per epoch', val_loss_epoch / (len(val_data) - (len(val_data) % batch_size)),
                              epoch)
            print("Validation Loss:", val_loss_epoch / (len(val_data) - (len(val_data) % batch_size)))
            if os.path.isfile(path_to_params) and num_epochs > 0:
                torch.save(net.state_dict(), path_to_params)  # Save model Parameters
    writer.close()
    print('Finished Training')
    return net

def val(net, margin, data_dir_val, writer_suffix, working_dir, class_dir):
    d_val_triplets = generate_val_triplets(data_dir_val)
    batch_size = net.batch_size
    writer = SummaryWriter(comment=writer_suffix)
    criterion = TripletLoss(margin=margin)
    val_data = pointcloudDataset(d_data=d_val_triplets, json_data=data_dir_val, root_dir=working_dir, mode='val')
    valloader = DataLoader(val_data, batch_size=batch_size,
                                shuffle=False)
    #
    print("Number of validation triplets:", int(len(val_data)/2))
    net.eval()
    print('Doing Evaluation')
    #
    with torch.no_grad():
        shape = np.zeros((batch_size, 128, 1))
        description = np.zeros((batch_size, 128, 1))

        for data in valloader:
            if len(data[0]) % batch_size != 0:
                break
            output_shape, output_desc = net(data,batch_size)
            #loss = criterion(output_shape, output_desc, batch_size, margin)
            shape = np.vstack((shape, np.asarray(output_shape)))
            description = np.vstack((description, np.asarray(output_desc)))

    # reshape output predictions for kNN
    shape = shape[batch_size:, :, :].reshape(len(shape) - batch_size, np.shape(shape[1])[0])
    description = description[batch_size:, :, :].reshape(len(description) - batch_size, np.shape(shape[1])[0])

    # %%
    # create ground truth and prediction list
    k = 5  # define the rank of retrieval measure

    # get 10 nearest neighbor, could also be just k nearest but to experiment left at 10
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(shape[0::2])  # check that nbrs are sorted
    distances, indices = nbrs.kneighbors(description[0::2])

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

    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred2,
                                                                         average='micro')  # verify that micro is correct, I think for now it's what we need  when just looking at objects from the same class
    print('precision:', precision)
    print('recall:', recall)
    print('fscore:', fscore)
    # print(recall)
    # print(fscore)
    # print(support)

    ndcg = ndcg_score(y_true, y_pred, k=k)
    print('NDCG:', ndcg)

    # %%

    tags = []

    # visualize validaten data, note that label could still be passed to add_embedding metadata=
    shape = torch.from_numpy(shape).type(torch.FloatTensor)
    out = torch.cat((shape.data, torch.ones(len(shape), 1)), 1)
    with open(class_dir, 'r') as fp:
        class_dict = json.load(fp)
    keys = list(d_val_triplets.keys())
    for i in range(int(len(out) / 2)):
        tags.append(class_dict[keys[i]])
    writer.add_embedding(mat=out[0::2], tag='shape_val', metadata=tags)
    description = torch.from_numpy(description).type(torch.FloatTensor)
    out2 = torch.cat((description.data, torch.ones(len(description), 1)), 1)
    writer.add_embedding(mat=out2[0::2], tag='description_val', metadata=tags)
    out3 = torch.cat((out[0::2], out2[0::2]), 0)
    tags = []

    for i in range(int(len(out3) / 2)):
        tags.append(str('shape'))  # + str(i)))
    for i in range(int(len(out3) / 2)):
        tags.append(str('descr'))  # + str(i)))

    writer.add_embedding(mat=out3, tag='overall_embedding', metadata=tags)
    # close tensorboard writer
    writer.close()
def retrieval(net, data_dir_val, working_dir):
    batch_size = net.batch_size
    d_val_samples = generate_val_triplets(data_dir_val)
    val_data = pointcloudDataset(d_data=d_val_samples, json_data=data_dir_val, root_dir=working_dir, mode='ret')
    valloader = DataLoader(val_data, batch_size=batch_size,
                           shuffle=False)
    #
    print("Number of validation triplets:", int(len(val_data) / 2))
    net.eval()
    #
    with torch.no_grad():
        shape = np.zeros((batch_size, 128, 1))
        description = np.zeros((batch_size, 128, 1))
        for data in valloader:
            if len(data[0]) % batch_size != 0:
                break
            output_shape, output_desc = net(data, batch_size)
            shape = np.vstack((shape, np.asarray(output_shape.cpu())))
            description = np.vstack((description, np.asarray(output_desc)))

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
        print(i, indices[i])
        y_true.append(i)
        y_pred.append(indices[i])
    return y_true, y_pred, list(d_val_samples.keys()), shape, description

if __name__ == '__main__':
    ###############################################
    ###
    batch_size = 10 #TODO normalization of data
    margin = 1  #TODO finetune
    num_epochs = 2
    print_batch = 11

    net = SiameseNet(batch_size)

    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    path_to_params = "models/all_classes_BS20_m5.pt" # if file does not exist or is empty it starts from untrained and later saves to the file
    writer = SummaryWriter(comment='_1epochs_allclasses_0001_BS32') # comment is a suffix, automatically names run with date+suffix

    ###
    ###############################################
    # execute in bash: tensorboard --logdir runs

    if os.path.isfile(path_to_params):
        if os.stat(path_to_params).st_size != 0:
            net.load_state_dict(torch.load(path_to_params))  #Loads pretrained net if file exists and if not empty
    else:
        open(path_to_params, "x") #Creates parameter file if it does not exist

    criterion = TripletLoss(margin=margin)
    working_dir = os.getcwd()
    data_dir_train = os.path.join(working_dir, 'data_train.json')
    data_dir_val = os.path.join(working_dir, 'data_val.json')
    ## TRAINING
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        loss_epoch = 0.0
        val_loss_epoch = 0.0

        d_train_triplets = generate_train_triplets(data_dir_train)

        train_data = pointcloudDataset(d_data=d_train_triplets, json_data=data_dir_train, root_dir=working_dir, mode='train')
        trainloader = DataLoader(train_data, batch_size=batch_size,
                            shuffle=False)
        print("Number of training triplets:", int(len(train_data)/2))

        #fix train val test sets once in one scipt, but not triplets
        #Execute triplet generation for train here without random seed and load epoch data!
        for i_batch, sample_batched in enumerate(trainloader):

            # zero the parameter gradients
            optimizer.zero_grad()
            if len(sample_batched[0]) % batch_size != 0:
                break
            # forward + backward + optimize
            t0 = time.time()
            x_shape, x_desc = net(sample_batched, batch_size)
            t_elapsed_fp = time.time() - t0
            #print('forward :',t_elapsed_fp,'s')

            t0 = time.time()
            loss = criterion(x_shape, x_desc, batch_size, margin)
            t_elapsed_loss = time.time() - t0
            #print('loss    :',t_elapsed_loss,'s')

            t0 = time.time()
            loss.backward()
            t_elapsed_bp = time.time() - t0
            #print('backward:',t_elapsed_bp,'s')


            optimizer.step()
            # print statistics
            running_loss += loss.item()
            loss_epoch += loss.item()

            if i_batch % print_batch == 0 and i_batch != 0:    # print every print_batch mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i_batch + 1, running_loss / (print_batch*batch_size)))
                running_loss = 0.0

        writer.add_scalar('Train loss per epoch', loss_epoch/(len(train_data)- (len(train_data) % batch_size)), epoch)

        # track validation loss per epoch
        d_val_triplets = generate_val_triplets()

        val_data = pointcloudDataset(d_data=d_val_triplets, json_data=data_dir_val, root_dir=working_dir, mode='val')
        valloader = DataLoader(val_data, batch_size=batch_size,
                                shuffle=False)

        #print("Number of validation triplets:", int(len(val_data)/2))
        net.eval()
        print('Doing Evaluation with',len(val_data)/2,'validation tripltes')

        with torch.no_grad():
#            shape=np.zeros((batch_size,128,1))
#            description=np.zeros((batch_size,128,1))
            for data in valloader:
                if len(data[0]) % batch_size != 0:
                    break
                output_shape, output_desc = net(data, batch_size)
                loss_val = criterion(output_shape, output_desc, batch_size, margin)
                val_loss_epoch += loss_val.item()
#
#                shape = np.vstack((shape, np.asarray(output_shape)))
#                description = np.vstack((description, np.asarray(output_desc)))
            writer.add_scalar('Val loss per epoch', val_loss_epoch/(len(val_data)- (len(val_data) % batch_size)), epoch)
            print("Validation Loss:", val_loss_epoch/(len(val_data) - (len(val_data) % batch_size)))
        

    print('Finished Training')
    if os.path.isfile(path_to_params) and num_epochs > 0:
        torch.save(net.state_dict(), path_to_params) #Save model Parameters
    
#    # visualize embedding of training batch
#    x_shape=x_shape.reshape(batch_size, 128)
#    out = torch.cat((x_shape.data, torch.ones(len(x_shape), 1)), 1)
#    
#    # metadata is label, put name of object
#    writer.add_embedding(mat=out,tag='shape_train', metadata=['eins','zwei','drei','vier'])
#    
#    x_desc=x_desc.reshape(batch_size, 128)
#    out = torch.cat((x_desc.data, torch.ones(len(x_desc), 1)), 1)
#    writer.add_embedding(mat=out,tag='description_train', metadata=['eins','zwei','drei','vier'])
#    
#    writer.close()
    
    #%%
    # EVALUATION --- shifted to epoch

 #   d_val_triplets = generate_val_triplets()

#    val_data = pointcloudDataset(d_data=d_val_triplets, json_data=data_dir_val, root_dir=working_dir, mode='val')
#    valloader = DataLoader(val_data, batch_size=batch_size,
#                            shuffle=False)
#
#    print("Number of validation triplets:", int(len(val_data)/2))
    net.eval()
#    print('Doing Evaluation')
#
    with torch.no_grad():
        shape=np.zeros((batch_size,128,1))
        description=np.zeros((batch_size,128,1))
        
        for data in valloader:
            if len(data[0]) % batch_size != 0:
                break
            output_shape, output_desc = net(data, batch_size)
            loss = criterion(output_shape, output_desc, batch_size, margin)
            shape = np.vstack((shape, np.asarray(output_shape)))
            description = np.vstack((description, np.asarray(output_desc)))
      
    # reshape output predictions for kNN     
    shape=shape[batch_size:,:,:].reshape(len(shape)-batch_size, np.shape(shape[1])[0])
    description=description[batch_size:,:,:].reshape(len(description)-batch_size, np.shape(shape[1])[0])      
    
    #%%
    # create ground truth and prediction list
    k = 5 # define the rank of retrieval measure
    
    # get 10 nearest neighbor, could also be just k nearest but to experiment left at 10
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(shape[0::2]) # check that nbrs are sorted
    distances, indices = nbrs.kneighbors(description[0::2])

    y_true=[]
    y_pred=[]
    y_pred2=[-1]*len(indices)
    for i, ind in enumerate(indices):
        print(i, indices[i])
        y_true.append(i)
        y_pred.append(indices[i])
        
    # get recall and precision for top k retrievals
    # create y_pred2 by checking if true label is in top k retrievals    
    if k != 1:
        for i, pred in enumerate(y_pred):
            for s in pred[:k]:
                if s==[y_true[i]]:
                    y_pred2[i]=s
                    break
    
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred2, average='micro') # verify that micro is correct, I think for now it's what we need  when just looking at objects from the same class
    print('precision:', precision)
    print('recall:', recall)
    print('fscore:',fscore)
    #print(recall)
    #print(fscore)
    #print(support)
    
    ndcg = ndcg_score(y_true, y_pred, k=k)
    print('NDCG:', ndcg)
    
    #%%

    tags = []

    # visualize validaten data, note that label could still be passed to add_embedding metadata=
    shape = torch.from_numpy(shape).type(torch.FloatTensor)
    out = torch.cat((shape.data, torch.ones(len(shape), 1)), 1)
    with open('class_dict.json', 'r') as fp:
        class_dict = json.load(fp)
    keys = list(d_val_triplets.keys())
    for i in range(int(len(out)/2)):
        tags.append(class_dict[keys[i]])
    writer.add_embedding(mat=out[0::2],tag='shape_val',metadata=tags)
    description = torch.from_numpy(description).type(torch.FloatTensor)
    out2 = torch.cat((description.data, torch.ones(len(description), 1)), 1)
    writer.add_embedding(mat=out2[0::2], tag='description_val',metadata=tags)
    out3 = torch.cat((out[0::2], out2[0::2]), 0)
    tags = []

    for i in range(int(len(out3)/2)):
        tags.append(str('shape' ))#+ str(i)))
    for i in range(int(len(out3)/2)):
        tags.append(str('descr' ))#+ str(i)))

    writer.add_embedding(mat=out3, tag='overall_embedding', metadata=tags)
    # close tensorboard writer
    writer.close()

