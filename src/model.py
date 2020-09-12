import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = np.finfo(np.float32).eps


def knn(x, k):
    batch_size = x.shape[0]
    indices = np.arange(0, k)
    with torch.no_grad():
        distances = []
        for b in range(batch_size):
            inner = -2 * torch.matmul(x[b:b + 1].transpose(2, 1), x[b:b + 1])
            xx = torch.sum(x[b:b + 1] ** 2, dim=1, keepdim=True)
            pairwise_distance = -xx - inner - xx.transpose(2, 1)
            distances.append(pairwise_distance)
        distances = torch.stack(distances, 0)
        distances = distances.squeeze(1)
        idx = distances.topk(k=k, dim=-1)[1][:, :, indices]
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.contiguous()
    x = x.view(batch_size, -1, num_points).contiguous()
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    # device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points
    idx_base = idx_base.cuda(torch.get_device(x))
    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    try:
        feature = x.view(batch_size * num_points, -1)[idx, :]
    except:
        import ipdb;
        ipdb.set_trace()
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature


class DGCNNControlPoints(nn.Module):
    def __init__(self, num_control_points, num_points=40, mode=0):
        """
        Control points prediction network. Takes points as input
        and outputs control points grid.
        :param num_control_points: size of the control points grid.
        :param num_points: number of nearest neighbors used in DGCNN.
        :param mode: different modes are used that decides different number of layers.
        """
        super(DGCNNControlPoints, self).__init__()
        self.k = num_points
        self.mode = mode
        if self.mode == 0:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
            self.bn4 = nn.BatchNorm2d(256)
            self.bn5 = nn.BatchNorm1d(1024)
            self.drop = 0.0
            self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                       self.bn1,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                       self.bn2,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                       self.bn3,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                       self.bn4,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                       self.bn5,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.controlpoints = num_control_points
            self.conv6 = torch.nn.Conv1d(1024, 1024, 1)
            self.conv7 = torch.nn.Conv1d(1024, 1024, 1)

            # Predicts the entire control points grid.
            self.conv8 = torch.nn.Conv1d(1024, 3 * (self.controlpoints ** 2), 1)

            self.bn6 = nn.BatchNorm1d(1024)
            self.bn7 = nn.BatchNorm1d(1024)

        if self.mode == 1:
            self.bn1 = nn.BatchNorm2d(128)
            self.bn2 = nn.BatchNorm2d(256)
            self.bn3 = nn.BatchNorm2d(256)
            self.bn4 = nn.BatchNorm2d(512)
            self.bn5 = nn.BatchNorm1d(1024)
            self.drop = 0.0

            self.conv1 = nn.Sequential(nn.Conv2d(6, 128, kernel_size=1, bias=False),
                                       self.bn1,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.conv2 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                       self.bn2,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.conv3 = nn.Sequential(nn.Conv2d(256 * 2, 256, kernel_size=1, bias=False),
                                       self.bn3,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.conv4 = nn.Sequential(nn.Conv2d(256 * 2, 512, kernel_size=1, bias=False),
                                       self.bn4,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.conv5 = nn.Sequential(nn.Conv1d(1024 + 128, 1024, kernel_size=1, bias=False),
                                       self.bn5,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.controlpoints = num_control_points
            self.conv6 = torch.nn.Conv1d(1024, 1024, 1)
            self.conv7 = torch.nn.Conv1d(1024, 1024, 1)

            # Predicts the entire control points grid.
            self.conv8 = torch.nn.Conv1d(1024, 3 * (self.controlpoints ** 2), 1)
            self.bn6 = nn.BatchNorm1d(1024)
            self.bn7 = nn.BatchNorm1d(1024)

        self.tanh = nn.Tanh()

    def forward(self, x, weights=None):
        """
        :param weights: weights of size B x N
        """
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)

        if isinstance(weights, torch.Tensor):
            weights = weights.reshape((1, 1, -1))
            x = x * weights

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)

        x1 = torch.unsqueeze(x1, 2)

        x = F.dropout(F.relu(self.bn6(self.conv6(x1))), self.drop)

        x = F.dropout(F.relu(self.bn7(self.conv7(x))), self.drop)
        x = self.conv8(x)
        x = self.tanh(x[:, :, 0])

        x = x.view(batch_size, self.controlpoints * self.controlpoints, 3)
        return x
