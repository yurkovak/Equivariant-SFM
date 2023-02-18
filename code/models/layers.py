import torch
from torch.nn import Linear, ReLU, BatchNorm1d, Sequential, Module, Identity
from utils.sparse_utils import SparseMat
from utils.pos_enc_utils import get_embedder


def get_linear_layers(feats, final_layer=False, batchnorm=True):
    layers = []

    # Add layers
    for i in range(len(feats) - 2):
        layers.append(Linear(feats[i], feats[i + 1]))

        if batchnorm:
            layers.append(BatchNorm1d(feats[i + 1], track_running_stats=False))

        layers.append(ReLU())

    # Add final layer
    layers.append(Linear(feats[-2], feats[-1]))
    if not final_layer:
        if batchnorm:
            layers.append(BatchNorm1d(feats[-1], track_running_stats=False))

        layers.append(ReLU())

    return Sequential(*layers)


class Parameter3DPts(torch.nn.Module):
    def __init__(self, n_pts):
        super().__init__()

        # Init points randomly
        pts_3d = torch.normal(mean=0, std=0.1, size=(3, n_pts), requires_grad=True)

        self.pts_3d = torch.nn.Parameter(pts_3d)

    def forward(self):
        return self.pts_3d


class SetOfSetLayer(Module):
    def __init__(self, d_in, d_out):
        super(SetOfSetLayer, self).__init__()
        # n is the number of points and m is the number of cameras
        self.lin_all = Linear(d_in, d_out)
        self.lin_n = Linear(d_in, d_out)
        self.lin_m = Linear(d_in, d_out)
        self.lin_both = Linear(d_in, d_out)

    def forward(self, x):
        # x is [m,n,d] sparse matrix
        out_all = self.lin_all(x.values)  # [all_points_everywhere, d_in] -> [all_points_everywhere, d_out]

        mean_rows = x.mean(dim=0) # [m,n,d_in] -> [n,d_in]
        out_rows = self.lin_n(mean_rows)  # [n,d_in] -> [n,d_out]  # each track's mean representation gets weighted

        mean_cols = x.mean(dim=1) # [m,n,d_in] -> [m,d_in]
        out_cols = self.lin_m(mean_cols)  # [m,d_in] -> [m,d_out]  # each camera's mean representation gets weighted

        out_both = self.lin_both(x.values.mean(dim=0, keepdim=True))  # [1,d_in] -> [1,d_out]

        new_features = (out_all + out_rows[x.indices[1], :] + out_cols[x.indices[0], :] + out_both) / 4  # [nnz,d_out]
        new_shape = (x.shape[0], x.shape[1], new_features.shape[1])

        return SparseMat(new_features, x.indices, x.cam_per_pts, x.pts_per_cam, new_shape)


def multi_head_attention(q, k, v):
  """A differentiable multi head attention function.

  Args:
    q (torch.Tensor): The query embedding.
      Has shape `(num_heads, sequence_size, head_emb_dim)`.
    k (torch.Tensor): The key embedding.
      Has shape `(num_heads, sequence_size, head_emb_dim)`.
    v (torch.Tensor): The value embedding.
      Has shape `(num_heads, sequence_size, head_emb_dim)`.

  Returns:
    y (torch.Tensor): The multi head attention output.
      Has shape `(sequence_size, num_heads * head_emb_dim)`.
  """
  num_heads, sequence_size, head_emb_dim = q.shape
  all_head_emb_dim = head_emb_dim * num_heads
  attention_scores = torch.matmul(q, k.transpose(1, 2))
  attention_scores = attention_scores / (head_emb_dim ** 0.5)
  attention_probs = torch.softmax(attention_scores, dim=-1)

  context_layer = torch.matmul(attention_probs, v)
  context_layer = context_layer.transpose(0, 1)
  return context_layer.reshape(sequence_size, all_head_emb_dim)


class AttentionLayer(Module):
    def __init__(self, d_in, d_out, num_heads):
        super(
        ).__init__()
        self.lin_qkv = Linear(d_in, d_out * 3)
        self.lin = Linear(d_out, d_out)
        self.num_heads = num_heads

    def forward(self, x: SparseMat):
        # x is [m,n,d] sparse matrix
        new_features = self.lin_qkv(x.values)  # [all_points, d_in] -> [all_points, d_out]
        new_features = new_features.reshape((new_features.shape[0], 3, self.num_heads, -1))  # [num_all_points, 3, num_heads, d_out/num_heads]
        features_by_track, inds_by_track = [], []
        for track_i in range(x.shape[1]):
            track_mask = x.indices[1] == track_i
            track_features = new_features[x.indices[1] == track_i, :]  # [num_track_points, num_heads, 3, d_out/num_heads]
            track_features = track_features.permute((2, 1, 0, 3))  # [num_heads, 3, num_track_points, d_out/num_heads]
            new_track_features = multi_head_attention(*track_features.unbind(dim=1))
            features_by_track.append(new_track_features)
            inds_by_track.append(x.indices[:, track_mask])

        new_values = self.lin(torch.cat(features_by_track))
        new_shape = (x.shape[0], x.shape[1], new_values.shape[1])
        return SparseMat(new_values, torch.cat(inds_by_track, dim=1), x.cam_per_pts, x.pts_per_cam, new_shape)


class ProjLayer(Module):
    def __init__(self, d_in, d_out):
        super(ProjLayer, self).__init__()
        # n is the number of points and m is the number of cameras
        self.lin_all = Linear(d_in, d_out)

    def forward(self, x):
        # x is [m,n,d] sparse matrix
        new_features = self.lin_all(x.values)  # [nnz,d_in] -> [nnz,d_out]
        new_shape = (x.shape[0], x.shape[1], new_features.shape[1])
        return SparseMat(new_features, x.indices, x.cam_per_pts, x.pts_per_cam, new_shape)


class NormalizationLayer(Module):
    def forward(self, x):
        features = x.values
        norm_features = features - features.mean(dim=0, keepdim=True)
        # norm_features = norm_features / norm_features.std(dim=0, keepdim=True)
        return SparseMat(norm_features, x.indices, x.cam_per_pts, x.pts_per_cam, x.shape)


class ActivationLayer(Module):
    def __init__(self):
        super(ActivationLayer, self).__init__()
        self.relu = ReLU()

    def forward(self, x):
        new_features = self.relu(x.values)
        return SparseMat(new_features, x.indices, x.cam_per_pts, x.pts_per_cam, x.shape)


class IdentityLayer(Module):
    def forward(self, x):
        return x


class EmbeddingLayer(Module):
    def __init__(self, multires, in_dim):
        super(EmbeddingLayer, self).__init__()
        if multires > 0:
            self.embed, self.d_out = get_embedder(multires, in_dim)
        else:
            self.embed, self.d_out = (Identity(), in_dim)

    def forward(self, x):
        embeded_features = self.embed(x.values)
        new_shape = (x.shape[0], x.shape[1], embeded_features.shape[1])
        return SparseMat(embeded_features, x.indices, x.cam_per_pts, x.pts_per_cam, new_shape)
