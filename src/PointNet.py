import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data


def knn(x, k1, k2):
    batch_size = x.shape[0]
    indices = np.arange(0, k2, k2 // k1)
    with torch.no_grad():
        distances = []
        for b in range(batch_size):
            inner = -2 * torch.matmul(x[b:b + 1].transpose(2, 1), x[b:b + 1])
            xx = torch.sum(x[b:b + 1] ** 2, dim=1, keepdim=True)
            pairwise_distance = -xx - inner - xx.transpose(2, 1)
            distances.append(pairwise_distance)
        distances = torch.stack(distances, 0)
        distances = distances.squeeze(1)
        try:
            idx = distances.topk(k=k2, dim=-1)[1][:, :, indices]
        except:
            import ipdb;
            ipdb.set_trace()
    return idx


def knn_points_normals(x, k1, k2):
    """
    The idea is to design the distance metric for computing 
    nearest neighbors such that the normals are not given
    too much importance while computing the distances.
    Note that this is only used in the first layer.
    """
    batch_size = x.shape[0]
    indices = np.arange(0, k2, k2 // k1)
    with torch.no_grad():
        distances = []
        for b in range(batch_size):
            p = x[b: b + 1, 0:3]
            n = x[b: b + 1, 3:6]

            inner = 2 * torch.matmul(p.transpose(2, 1), p)
            xx = torch.sum(p ** 2, dim=1, keepdim=True)
            p_pairwise_distance = xx - inner + xx.transpose(2, 1)

            inner = 2 * torch.matmul(n.transpose(2, 1), n)
            n_pairwise_distance = 2 - inner

            # This pays less attention to normals
            pairwise_distance = p_pairwise_distance * (1 + n_pairwise_distance)

            # This pays more attention to normals
            # pairwise_distance = p_pairwise_distance * torch.exp(n_pairwise_distance)

            # pays too much attention to normals
            # pairwise_distance = p_pairwise_distance + n_pairwise_distance

            distances.append(-pairwise_distance)

        distances = torch.stack(distances, 0)
        distances = distances.squeeze(1)
        try:
            idx = distances.topk(k=k2, dim=-1)[1][:, :, indices]
        except:
            import ipdb;
            ipdb.set_trace()
    return idx


def get_graph_feature(x, k1=20, k2=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        idx = knn(x, k1=k1, k2=k2)

    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    try:
        feature = x.view(batch_size * num_points, -1)[idx, :]
    except:
        import ipdb;
        ipdb.set_trace()
        print(feature.shape)

    feature = feature.view(batch_size, num_points, k1, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k1, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    return feature


def get_graph_feature_with_normals(x, k1=20, k2=20, idx=None):
    """
    normals are treated separtely for computing the nearest neighbor
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        idx = knn_points_normals(x, k1=k1, k2=k2)

    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    try:
        feature = x.view(batch_size * num_points, -1)[idx, :]
    except:
        import ipdb;
        ipdb.set_trace()
        print(feature.shape)

    feature = feature.view(batch_size, num_points, k1, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k1, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    return feature


class DGCNNEncoderGn(nn.Module):
    def __init__(self, mode=0, input_channels=3, nn_nb=80):
        super(DGCNNEncoderGn, self).__init__()
        self.k = nn_nb
        self.dilation_factor = 1
        self.mode = mode
        self.drop = 0.0
        if self.mode == 0 or self.mode == 5:
            self.bn1 = nn.GroupNorm(2, 64)
            self.bn2 = nn.GroupNorm(2, 64)
            self.bn3 = nn.GroupNorm(2, 128)
            self.bn4 = nn.GroupNorm(4, 256)
            self.bn5 = nn.GroupNorm(8, 1024)

            self.conv1 = nn.Sequential(nn.Conv2d(input_channels * 2, 64, kernel_size=1, bias=False),
                                       self.bn1,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                       self.bn2,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                       self.bn3,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.mlp1 = nn.Conv1d(256, 1024, 1)
            self.bnmlp1 = nn.GroupNorm(8, 1024)
            self.mlp1 = nn.Conv1d(256, 1024, 1)
            self.bnmlp1 = nn.GroupNorm(8, 1024)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.shape[2]

        if self.mode == 0 or self.mode == 1:
            # First edge conv
            x = get_graph_feature(x, k1=self.k, k2=self.k)

            x = self.conv1(x)
            x1 = x.max(dim=-1, keepdim=False)[0]

            # Second edge conv
            x = get_graph_feature(x1, k1=self.k, k2=self.k)
            x = self.conv2(x)
            x2 = x.max(dim=-1, keepdim=False)[0]

            # Third edge conv
            x = get_graph_feature(x2, k1=self.k, k2=self.k)
            x = self.conv3(x)
            x3 = x.max(dim=-1, keepdim=False)[0]

            x_features = torch.cat((x1, x2, x3), dim=1)
            x = F.relu(self.bnmlp1(self.mlp1(x_features)))

            x4 = x.max(dim=2)[0]

            return x4, x_features

        if self.mode == 5:
            # First edge conv
            x = get_graph_feature_with_normals(x, k1=self.k, k2=self.k)
            x = self.conv1(x)
            x1 = x.max(dim=-1, keepdim=False)[0]

            # Second edge conv
            x = get_graph_feature(x1, k1=self.k, k2=self.k)
            x = self.conv2(x)
            x2 = x.max(dim=-1, keepdim=False)[0]

            # Third edge conv
            x = get_graph_feature(x2, k1=self.k, k2=self.k)
            x = self.conv3(x)
            x3 = x.max(dim=-1, keepdim=False)[0]

            x_features = torch.cat((x1, x2, x3), dim=1)
            x = F.relu(self.bnmlp1(self.mlp1(x_features)))
            x4 = x.max(dim=2)[0]

            return x4, x_features


class PrimitivesEmbeddingDGCNGn(nn.Module):
    """
    Segmentation model that takes point cloud as input and returns per
    point embedding or membership function. This defines the membership loss
    inside the forward function so that data distributed loss can be made faster.
    """

    def __init__(self, emb_size=50, num_primitives=8, primitives=False, embedding=False, mode=0, num_channels=3,
                 loss_function=None, nn_nb=80):
        super(PrimitivesEmbeddingDGCNGn, self).__init__()
        self.mode = mode
        self.encoder = DGCNNEncoderGn(mode=mode, input_channels=num_channels, nn_nb=nn_nb)
        self.drop = 0.0
        self.loss_function = loss_function

        if self.mode == 0 or self.mode == 3 or self.mode == 4 or self.mode == 5 or self.mode == 6:
            self.conv1 = torch.nn.Conv1d(1024 + 256, 512, 1)
        elif self.mode == 1 or self.mode == 2:
            self.conv1 = torch.nn.Conv1d(1024 + 512, 512, 1)

        self.bn1 = nn.GroupNorm(8, 512)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)

        self.bn2 = nn.GroupNorm(4, 256)

        self.softmax = torch.nn.Softmax(dim=1)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.tanh = torch.nn.Tanh()
        self.emb_size = emb_size
        self.primitives = primitives
        self.embedding = embedding

        if self.embedding:
            self.mlp_seg_prob1 = torch.nn.Conv1d(256, 256, 1)
            self.mlp_seg_prob2 = torch.nn.Conv1d(256, self.emb_size, 1)
            self.bn_seg_prob1 = nn.GroupNorm(4, 256)

        if primitives:
            self.mlp_prim_prob1 = torch.nn.Conv1d(256, 256, 1)
            self.mlp_prim_prob2 = torch.nn.Conv1d(256, num_primitives, 1)
            self.bn_prim_prob1 = nn.GroupNorm(4, 256)

    def forward(self, points, labels, compute_loss=True):
        batch_size = points.shape[0]
        num_points = points.shape[2]
        x, first_layer_features = self.encoder(points)

        # first_layer_features = first_layer_features[:, :, self.l_permute]
        x = x.view(batch_size, 1024, 1).repeat(1, 1, num_points)
        x = torch.cat([x, first_layer_features], 1)

        x = F.dropout(F.relu(self.bn1(self.conv1(x))), self.drop)
        x_all = F.dropout(F.relu(self.bn2(self.conv2(x))), self.drop)

        if self.embedding:
            x = F.dropout(F.relu(self.bn_seg_prob1(self.mlp_seg_prob1(x_all))), self.drop)
            embedding = self.mlp_seg_prob2(x)

        if self.primitives:
            x = F.dropout(F.relu(self.bn_prim_prob1(self.mlp_prim_prob1(x_all))), self.drop)
            x = self.mlp_prim_prob2(x)
            primitives_log_prob = self.logsoftmax(x)
        if compute_loss:
            embed_loss = self.loss_function(embedding, labels.data.cpu().numpy())
        else:
            embed_loss = torch.zeros(1).cuda()
        return embedding, primitives_log_prob, embed_loss


class PrimitivesEmbeddingDGCNGne2e(nn.Module):
    """
    Segmentation model that takes point cloud as input and returns per
    point embedding or membership function. This defines the membership loss
    inside the forward function so that data distributed loss can be made faster.
    """

    def __init__(self, emb_size=50, num_primitives=8, primitives=False, embedding=False, mode=0, num_channels=3,
                 loss_function=None, nn_nb=80):
        super(PrimitivesEmbeddingDGCNGne2e, self).__init__()
        self.mode = mode
        self.encoder = DGCNNEncoderGn(mode=mode, input_channels=num_channels, nn_nb=nn_nb)
        self.drop = 0.0
        self.loss_function = loss_function

        if self.mode == 0 or self.mode == 3 or self.mode == 4 or self.mode == 5 or self.mode == 6:
            self.conv1 = torch.nn.Conv1d(1024 + 256, 512, 1)
        elif self.mode == 1 or self.mode == 2:
            self.conv1 = torch.nn.Conv1d(1024 + 512, 512, 1)

        self.bn1 = nn.GroupNorm(8, 512)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)

        self.bn2 = nn.GroupNorm(4, 256)

        self.softmax = torch.nn.Softmax(dim=1)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.tanh = torch.nn.Tanh()
        self.emb_size = emb_size
        self.primitives = primitives
        self.embedding = embedding

        if self.embedding:
            self.mlp_seg_prob1 = torch.nn.Conv1d(256, 256, 1)
            self.mlp_seg_prob2 = torch.nn.Conv1d(256, self.emb_size, 1)
            self.bn_seg_prob1 = nn.GroupNorm(4, 256)

        if primitives:
            self.mlp_prim_prob1 = torch.nn.Conv1d(256, 256, 1)
            self.mlp_prim_prob2 = torch.nn.Conv1d(256, num_primitives, 1)
            self.bn_prim_prob1 = nn.GroupNorm(4, 256)

    def forward(self, points, labels, primitives, quantile, debug, compute_loss=True):
        batch_size = points.shape[0]
        num_points = points.shape[2]
        normals = points[:, 3:, :].permute(0, 2, 1)

        x, first_layer_features = self.encoder(points)

        # first_layer_features = first_layer_features[:, :, self.l_permute]
        x = x.view(batch_size, 1024, 1).repeat(1, 1, num_points)
        x = torch.cat([x, first_layer_features], 1)

        x = F.dropout(F.relu(self.bn1(self.conv1(x))), self.drop)
        x_all = F.dropout(F.relu(self.bn2(self.conv2(x))), self.drop)

        if self.embedding:
            x = F.dropout(F.relu(self.bn_seg_prob1(self.mlp_seg_prob1(x_all))), self.drop)
            embedding = self.mlp_seg_prob2(x)

        if self.primitives:
            x = F.dropout(F.relu(self.bn_prim_prob1(self.mlp_prim_prob1(x_all))), self.drop)
            x = self.mlp_prim_prob2(x)
            primitives_log_prob = self.logsoftmax(x)

        if compute_loss:
            embed_loss = self.loss_function(embedding, points.permute(0, 2, 1), labels.data.cpu().numpy())
        else:
            embed_loss = torch.zeros(1).cuda()

        res_loss = self.evaluation.fitting_loss(
            embedding.permute(0, 2, 1),
            points.permute(0, 2, 1)[:, :, 0:3],
            normals,
            labels,
            primitives,
            primitives_log_prob,
            quantile=0.025,
            debug=False,
        )
        return res_loss, embedding, primitives_log_prob, embed_loss
