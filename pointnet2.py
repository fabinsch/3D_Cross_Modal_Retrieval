from pointnet import PointNet
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from time import time

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def square_distance(src, dst):
    """
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, D1, D2, ..., Dn]
    Return:
        new_points:, indexed points data, [B, D1, D2, ..., Dn, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud data, [B, npoint, C]
    """
    device = xyz.device
    B, N, C = xyz.shape
    S = npoint
    centroids = torch.zeros(B, S, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(S):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid)**2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    K = nsample
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:,:,:K]
    group_first = group_idx[:,:,0].view(B, S, 1).repeat([1, 1, K])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = npoint

    new_xyz = index_points(xyz, farthest_point_sample(xyz, npoint))
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz -= new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all
    
    def forward(self, xyz, points):
        """
        Input: 
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
            #new_points = F.relu(conv(new_points))
        
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

#class PointNetSetAbstractionMsg(nn.Module):
#    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
#        super(PointNetSetAbstractionMsg, self).__init__()
#        self.npoint = npoint
#        self.radius_list = radius_list
#        self.nsample_list = nsample_list
#        self.conv_blocks = nn.ModuleList()
#        self.bn_blocks = nn.ModuleList()
#        for i in range(len(mlp_list)):
#            convs = nn.ModuleList()
#            bns = nn.ModuleList()
#            last_channel = in_channel + 3
#            for out_channel in mlp_list[i]:
#                convs.append(nn.Conv2d(last_channel, out_channel, 1))
#                bns.append(nn.BatchNorm2d(out_channel))
#                last_channel = out_channel
#            self.conv_blocks.append(convs)
#            self.bn_blocks.append(bns)
#    
#    def forward(self, xyz, points):
#        """
#        Input:
#            xyz: input points position data, [B, C, N]
#            points: input points data, [B, D, N]
#        Return:
#            new_xyz: sampled points position data, [B, C, S]
#            new_points_concat: sample points feature data, [B, D', S]
#        """
#        xyz = xyz.permute(0, 2, 1)
#        if points is not None:
#            points = points.permute(0, 2, 1)
#
#        B, N, C = xyz.shape
#        S = self.npoint
#        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
#        new_points_list = []
#        for i, radius in enumerate(self.radius_list):
#            K = self.nsample_list[i]
#            group_idx = query_ball_point(radius, K, xyz, new_xyz)
#            grouped_xyz = index_points(xyz, group_idx)
#            grouped_xyz -= new_xyz.view(B, S, 1, C)
#            if points is not None:
#                grouped_points = index_points(points, group_idx)
#                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
#            else:
#                grouped_points = grouped_xyz
#
#            grouped_points = grouped_points.permute(0, 3, 2, 1) #[B, D, K, S]
#            for j in range(len(self.conv_blocks[i])):
#                conv = self.conv_blocks[i][j]
#                bn = self.bn_blocks[i][j]
#                grouped_points = F.relu(bn(conv(grouped_points)))
#            new_points = torch.max(grouped_points, 2)[0] #[B, D', S]
#            new_points_list.append(new_points)
#
#        new_xyz = new_xyz.permute(0, 2, 1)
#        new_points_concat = torch.cat(new_points_list, dim=1)
#        return new_xyz, new_points_concat
#
        
#class PointNetFeaturePropagation(nn.Module):
#    def __init__(self, in_channel, mlp):
#        super(PointNetFeaturePropagation, self).__init__()
#        self.mlp_convs = nn.ModuleList()
#        self.mlp_bns = nn.ModuleList()
#        last_channel = in_channel
#        for out_channel in mlp:
#            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
#            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
#            last_channel = out_channel
#    
#    def forward(self, xyz1, xyz2, points1, points2):
#        """
#        Input:
#            xyz1: input points position data, [B, C, N]
#            xyz2: sampled input points position data, [B, C, S]
#            points1: input points data, [B, D, N]
#            points2: input points data, [B, D, S]
#        Return:
#            new_points: upsampled points data, [B, D', N]
#        """
#        xyz1 = xyz1.permute(0, 2, 1)
#        xyz2 = xyz2.permute(0, 2, 1)
#        
#        points2 = points2.permute(0, 2, 1)
#        B, N, C = xyz1.shape
#        _, S, _ = xyz2.shape
#
#        if S == 1:
#            interpolated_points = points2.repeat(1, N, 1)
#        else:
#            dists = square_distance(xyz1, xyz2)
#            dists, idx = dists.sort(dim=-1)
#            dists, idx = dists[:,:,:3], idx[:,:,:3] #[B, N, 3]
#            dists[dists < 1e-10] = 1e-10
#            weight = 1.0 / dists #[B, N, 3]
#            weight = weight / torch.sum(weight, dim=-1).view(B, N, 1) #[B, N, 3]
#            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim = 2)
#
#        if points1 is not None:
#            points1 = points1.permute(0, 2, 1)
#            new_points = torch.cat([points1, interpolated_points], dim=-1)
#        else:
#            new_points = interpolated_points
#
#        new_points = new_points.permute(0, 2, 1)
#        for i, conv in enumerate(self.mlp_convs):
#            bn = self.mlp_bns[i]
#            new_points = F.relu(bn(conv(new_points)))
#        return new_points

#
#class PointNet2ClsMsg(nn.Module):
#    def __init__(self):
#        super(PointNet2ClsMsg, self).__init__()
#        self.sa1 = PointNetSetAbstractionMsg(512, [0.1,0.2,0.4], [16,32,128], 0, [[32,32,64], [64,64,128], [64,96,128]])
#        self.sa2 = PointNetSetAbstractionMsg(128, [0.2,0.4,0.8], [32,64,128], 320, [[64,64,128], [128,128,256], [128,128,256]])
#        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
#        self.fc1 = nn.Linear(1024, 512)
#        self.bn1 = nn.BatchNorm1d(512)
#        self.drop1 = nn.Dropout(0.4)
#        self.fc2 = nn.Linear(512, 256)
#        self.bn2 = nn.BatchNorm1d(256)
#        self.drop2 = nn.Dropout(0.4)
#        self.fc3 = nn.Linear(256, 40)
#
#    def forward(self, xyz):
#        B, _, _ = xyz.shape
#        l1_xyz, l1_points = self.sa1(xyz, None)
#        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#        x = l3_points.view(B, 1024)
#        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
#        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
#        x = self.fc3(x)
#        x = F.log_softmax(x, -1)
#        return x

class PointNet2ClsSsg(nn.Module):
    def __init__(self):
        super(PointNet2ClsSsg, self).__init__()
        # init with (npoint, radius, nsample, in_channel, mlp, group_all)
        # modify in_channels of first abstraction layer to 6
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, 3+3, [64, 64, 128], False)
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128 + 3, [128,128,256], False)
        self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0)
        self.fc3 = nn.Linear(256, 128)
    
    def forward(self, xyz):
        B, _, _ = xyz.shape
        
        #split up coordinates and colors
        rgb=xyz[:,3:]/255 - 0.5
        xyz=xyz[:,:3]
        #mean_rgb = rgb.mean()
        #std_rgb = rgb.std()
        #mean_xyz = xyz.mean()
        #std_xyz = xyz.std()
        l1_xyz, l1_points = self.sa1(xyz, rgb)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        
        #x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        #x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        
        x = self.fc3(x)
        #x = F.log_softmax(x, -1)
        return x.unsqueeze(2)


if __name__ == '__main__':
    
    pointNet = PointNet2ClsSsg()
    points = torch.rand(4, 6, 1000)
    out = pointNet(points)
    
#     for i in range(10):
#         t = time()
#         xyz = torch.rand(16, 3, 2048)#.cuda()
#         net = PointNet2SemSeg(2048)
#         #net.cuda()
#         x = net(xyz)
#         timeit('it', t)

    # xyz1 = torch.rand(4, 3, 2048).cuda()
    # xyz2 = torch.rand(4, 3, 512).cuda()
    # points2 = torch.rand(4, 64, 2048).cuda()
    # net = PointNetFeaturePropagation(64, [128, 256])
    # net.cuda()
    # new_points = net(xyz1, xyz2, None, points2)
    # print(new_points.shape)

    # xyz = torch.rand(8, 3, 2048).cuda()
    # net = PointNet2SemSeg(2048)
    # net.cuda()
    # x = net(xyz)
    # print(x.shape)
