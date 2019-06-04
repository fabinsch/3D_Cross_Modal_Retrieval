import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from transform_nets import InputTransformNet, FeatureTransformNet

import json
import os

class PointNet(nn.Module):
    def __init__(self, global_feature=True):
        super(PointNet, self).__init__()
        self.global_feature = global_feature
        self.input_transform = InputTransformNet()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.feature_transform = FeatureTransformNet()
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, 1) # 128
        self.bn4 = nn.BatchNorm1d(128) #128
        #self.conv5 = nn.Conv1d(128, 1024, 1)
        #self.bn5 = nn.BatchNorm1d(1024)


    
    def forward(self, x):
        """
        x: [B, 3+3, N]
        Take as input a B x K x N matrix of B batches of N points with K 
        dimensions
        """
        #split xyz and rgb
        x = x[:, 0:3]
        #print(x.shape)
        B, N = x.shape[0], x.shape[2]
        input_transform = self.input_transform(x) #[B, 3+3, 3]
        x = torch.matmul(x.permute(0, 2, 1), input_transform.permute(0, 2, 1)).permute(0, 2, 1) #[B, 3+3, N]
        # add rgb
        x = F.relu(self.bn1(self.conv1(x))) #[B, 64, N]
        x = F.relu(self.bn2(self.conv2(x))) #[B, 64, N]
        feature_transform = self.feature_transform(x) #[B, 64, 64]
        x = torch.matmul(x.permute(0, 2, 1), feature_transform.permute(0, 2, 1)).permute(0, 2, 1) #[B, 64, N]
        point_feature = x
        x = F.relu(self.bn3(self.conv3(x))) #[B, 64, N]
        x = F.relu(self.bn4(self.conv4(x))) #[B, 128, N]
        #x = F.relu(self.bn5(self.conv5(x))) #[B, 1024, N]
        x = nn.MaxPool1d(N)(x) #[B, 1024, 1]
        if not self.global_feature:
            x = x.repeat([1, 1, N]) #[B, 1024, N]
            x = torch.cat([point_feature, x], 1) #[B, 1088, N]
        return x


'''if __name__ == '__main__':
    net = PointNet()
    working_dir = os.getcwd()
    data_dir = os.path.join(working_dir, 'data.json')        
    #root_dir = working_dir.replace('pointnet3', 'data')       
    points_data = pointcloudDataset(json_file=data_dir, root_dir=working_dir)  
    #print(points_data[2])
    dataloader = DataLoader(points_data, batch_size=4,
                            shuffle=True, num_workers=4)
    for i_batch, sample_batched in enumerate(dataloader):
        #print(i_batch, sample_batched)
        sample_batched2=sample_batched['points']
        x=net(sample_batched2)
        print(x.shape)
        break'''
    
        
    