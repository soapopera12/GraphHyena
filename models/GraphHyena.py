import warnings
import networkx as nx
from collections import Counter

from typing import Dict, Tuple, Union, List, Optional, Set

import math
import numpy as np

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import NeighborSampler
from models.modules import TimeEncoder

class GraphHyena(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, patch_size: int , max_input_sequence_length: int,
                 channel_embedding_dim: int, hyena_dim: int=256, hyena_depth: int=3, hyena_max_seq_len: int=1024, 
                 num_channels: int = 4, dropout: float = 0.1, device: str = 'cpu'):

        super(GraphHyena, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.device = device
        self.num_channels = num_channels

        self.dropout = dropout
        self.patch_size = patch_size
        self.channel_embedding_dim = channel_embedding_dim
        self.max_input_sequence_length = max_input_sequence_length

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        self.resource_alloc_feat_dim = self.channel_embedding_dim
        self.local_path_encoder = LocalPathEncoderRobustAdvancedTemporal(path_feat_dim=self.resource_alloc_feat_dim,
                                            neighbor_sampler=self.neighbor_sampler,
                                            device=self.device)
        
        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.patch_size * self.node_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'edge': nn.Linear(in_features=self.patch_size * self.edge_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'time': nn.Linear(in_features=self.patch_size * self.time_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'neighbor_co_occurrence': nn.Linear(in_features=self.patch_size * self.resource_alloc_feat_dim, out_features=self.channel_embedding_dim, bias=True)
        })

        self.hyena_dim = hyena_dim
        self.hyena_depth = hyena_depth
        self.hyena_max_seq_len = hyena_max_seq_len
        
        self.ssm = nn.ModuleList([
            HyenaEncoderV2(feat_size=self.num_channels * self.channel_embedding_dim, dim=self.hyena_dim, depth=self.hyena_depth, dropout=self.dropout, max_seq_len=self.hyena_max_seq_len)
            for _ in range(2)
        ])
        
        self.output_layer = nn.Linear(in_features=self.num_channels * self.channel_embedding_dim, out_features=self.node_feat_dim, bias=True)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, num_neighbors: int = 20, time_gap: int = 2000):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param num_neighbors: int, number of neighbors to sample for each node
        :param time_gap: int, time gap for neighbors to compute node features
        :return:
        """
        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings, dst_node_embeddings  = self.compute_both_node_temporal_embeddings(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times,
                                                                    num_neighbors=num_neighbors, time_gap=time_gap)

        # print(src_node_embeddings.shape, dst_node_embeddings.shape)
        return src_node_embeddings, dst_node_embeddings

    def compute_both_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray,
                                         num_neighbors: int = 10, time_gap: int = 2000):
        """
        given node ids node_ids, and the corresponding time node_interact_times, return the temporal embeddings of nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :param time_gap: int, time gap for neighbors to compute node features
        :return:
        """
        
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
           self.neighbor_sampler.get_historical_neighbors(node_ids=src_node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=self.max_input_sequence_length)

       
        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_historical_neighbors(node_ids=dst_node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=self.max_input_sequence_length)

        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times = \
            self.new_pad_sequences(node_ids=src_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=src_nodes_edge_ids_list, nodes_neighbor_times_list=src_nodes_neighbor_times_list,
                               patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length)

        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times = \
            self.new_pad_sequences(node_ids=dst_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=dst_nodes_edge_ids_list, nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
                               patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length)
        
        src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features = self.local_path_encoder(
            src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
            dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
            src_node_ids=src_node_ids,
            dst_node_ids=dst_node_ids,
            node_interact_times=node_interact_times,
            src_padded_nodes_neighbor_times=src_padded_nodes_neighbor_times,
            dst_padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times,
        )
                
        src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=src_padded_nodes_edge_ids, padded_nodes_neighbor_times=src_padded_nodes_neighbor_times, time_encoder=self.time_encoder)

        dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=dst_padded_nodes_edge_ids, padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times, time_encoder=self.time_encoder)

        src_patches_nodes_neighbor_node_raw_features, src_patches_nodes_edge_raw_features, \
        src_patches_nodes_neighbor_time_features, src_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(padded_nodes_neighbor_node_raw_features=src_padded_nodes_neighbor_node_raw_features,
                             padded_nodes_edge_raw_features=src_padded_nodes_edge_raw_features,
                             padded_nodes_neighbor_time_features=src_padded_nodes_neighbor_time_features,
                             padded_nodes_neighbor_co_occurrence_features=src_padded_nodes_neighbor_co_occurrence_features,
                             patch_size=self.patch_size)

        dst_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_edge_raw_features, \
        dst_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(padded_nodes_neighbor_node_raw_features=dst_padded_nodes_neighbor_node_raw_features,
                             padded_nodes_edge_raw_features=dst_padded_nodes_edge_raw_features,
                             padded_nodes_neighbor_time_features=dst_padded_nodes_neighbor_time_features,
                             padded_nodes_neighbor_co_occurrence_features=dst_padded_nodes_neighbor_co_occurrence_features,
                             patch_size=self.patch_size)

        # align the patch encoding dimension
        # Tensor, shape (batch_size, src_num_patches, channel_embedding_dim)
        src_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](src_patches_nodes_neighbor_node_raw_features)
        src_patches_nodes_edge_raw_features = self.projection_layer['edge'](src_patches_nodes_edge_raw_features)
        src_patches_nodes_neighbor_time_features = self.projection_layer['time'](src_patches_nodes_neighbor_time_features)
        src_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](src_patches_nodes_neighbor_co_occurrence_features)
        
        # Tensor, shape (batch_size, dst_num_patches, channel_embedding_dim)
        dst_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](dst_patches_nodes_neighbor_node_raw_features)
        dst_patches_nodes_edge_raw_features = self.projection_layer['edge'](dst_patches_nodes_edge_raw_features)
        dst_patches_nodes_neighbor_time_features = self.projection_layer['time'](dst_patches_nodes_neighbor_time_features)
        dst_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](dst_patches_nodes_neighbor_co_occurrence_features)
        
        batch_size = len(src_patches_nodes_neighbor_node_raw_features)
        src_num_patches = src_patches_nodes_neighbor_node_raw_features.shape[1]
        dst_num_patches = dst_patches_nodes_neighbor_node_raw_features.shape[1]

        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, channel_embedding_dim)
        patches_nodes_neighbor_node_raw_features = torch.cat([src_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_neighbor_node_raw_features], dim=1)
        patches_nodes_edge_raw_features = torch.cat([src_patches_nodes_edge_raw_features, dst_patches_nodes_edge_raw_features], dim=1)
        patches_nodes_neighbor_time_features = torch.cat([src_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_time_features], dim=1)
        patches_nodes_neighbor_co_occurrence_features = torch.cat([src_patches_nodes_neighbor_co_occurrence_features, dst_patches_nodes_neighbor_co_occurrence_features], dim=1)

        # patching data
        patches_data = [patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features,
                        patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features]
        patches_data = torch.stack(patches_data, dim=2)
        patches_data = patches_data.reshape(batch_size, src_num_patches + dst_num_patches, self.num_channels * self.channel_embedding_dim)
        
        for hyenaLayer in self.ssm:
            patches_data = hyenaLayer(patches_data)
        
        # src_patches_data, Tensor, shape (batch_size, src_num_patches, num_channels * channel_embedding_dim)
        src_patches_data = patches_data[:, : src_num_patches, :]
        # dst_patches_data, Tensor, shape (batch_size, dst_num_patches, num_channels * channel_embedding_dim)
        dst_patches_data = patches_data[:, src_num_patches: src_num_patches + dst_num_patches, :]
        # src_patches_data, Tensor, shape (batch_size, num_channels * channel_embedding_dim)
        src_patches_data = torch.mean(src_patches_data, dim=1)
        # dst_patches_data, Tensor, shape (batch_size, num_channels * channel_embedding_dim)
        dst_patches_data = torch.mean(dst_patches_data, dim=1)

        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings = self.output_layer(src_patches_data)
        # Tensor, shape (batch_size, node_feat_dim)
        dst_node_embeddings = self.output_layer(dst_patches_data)

        return src_node_embeddings, dst_node_embeddings
    
    def new_pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray, nodes_neighbor_ids_list: list, nodes_edge_ids_list: list,
                      nodes_neighbor_times_list: list, patch_size: int = 1, max_input_sequence_length: int = 256):
        """
        pad the sequences for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param nodes_neighbor_ids_list: list of ndarrays, each ndarray contains neighbor ids for nodes in node_ids
        :param nodes_edge_ids_list: list of ndarrays, each ndarray contains edge ids for nodes in node_ids
        :param nodes_neighbor_times_list: list of ndarrays, each ndarray contains neighbor interaction timestamp for nodes in node_ids
        :param patch_size: int, patch size
        :param max_input_sequence_length: int, maximal number of neighbors for each node
        :return:
        """
        assert max_input_sequence_length - 1 > 0, 'Maximal number of neighbors for each node should be greater than 1!'

        # Temporary lists to store the (potentially truncated) neighbor sequences
        truncated_neighbor_ids = []
        truncated_edge_ids = []
        truncated_neighbor_times = []
        
        # Initialize max_seq_length based on the truncated sequences
        current_max_seq_length_after_truncation = 0

        # First pass: Truncate sequences and determine the true maximum length
        for idx in range(len(nodes_neighbor_ids_list)):
            assert len(nodes_neighbor_ids_list[idx]) == len(nodes_edge_ids_list[idx]) == len(nodes_neighbor_times_list[idx]), \
                f"Mismatched lengths for neighbors, edges, and times at index {idx}"

            # Get current sequence (making a copy if necessary to avoid modifying original input lists directly)
            current_neighbor_ids = nodes_neighbor_ids_list[idx]
            current_edge_ids = nodes_edge_ids_list[idx]
            current_neighbor_times = nodes_neighbor_times_list[idx]

            # Truncate the sequences if they exceed the allowed length
            # We reserve 1 slot for the node itself, so neighbors can take up to max_input_sequence_length - 1
            if len(current_neighbor_ids) > max_input_sequence_length - 1:
                current_neighbor_ids = current_neighbor_ids[-(max_input_sequence_length - 1):]
                current_edge_ids = current_edge_ids[-(max_input_sequence_length - 1):]
                current_neighbor_times = current_neighbor_times[-(max_input_sequence_length - 1):]
            
            # Store the (possibly truncated) sequences
            truncated_neighbor_ids.append(current_neighbor_ids)
            truncated_edge_ids.append(current_edge_ids)
            truncated_neighbor_times.append(current_neighbor_times)

            # Update the maximum length found so far
            if len(current_neighbor_ids) > current_max_seq_length_after_truncation:
                current_max_seq_length_after_truncation = len(current_neighbor_ids)

        # Calculate the final `max_seq_length` for the padded arrays
        # This includes 1 extra slot for the node itself at position 0
        final_max_seq_length = current_max_seq_length_after_truncation + 1
        
        # Adjust `final_max_seq_length` to be a multiple of `patch_size`
        if final_max_seq_length % patch_size != 0:
            final_max_seq_length += (patch_size - final_max_seq_length % patch_size)
        assert final_max_seq_length % patch_size == 0, "Final sequence length must be a multiple of patch_size"

        # Initialize the padded arrays with zeros, using the determined final_max_seq_length
        padded_nodes_neighbor_ids = np.zeros((len(node_ids), final_max_seq_length)).astype(np.longlong)
        padded_nodes_edge_ids = np.zeros((len(node_ids), final_max_seq_length)).astype(np.longlong)
        padded_nodes_neighbor_times = np.zeros((len(node_ids), final_max_seq_length)).astype(np.float32)

        # Second pass: Populate the padded arrays
        for idx in range(len(node_ids)):
            # Place the current node's ID, a placeholder edge ID (0), and its interaction time at the first position
            padded_nodes_neighbor_ids[idx, 0] = node_ids[idx]
            padded_nodes_edge_ids[idx, 0] = 0
            padded_nodes_neighbor_times[idx, 0] = node_interact_times[idx]

            # Place the (possibly truncated) neighbor sequences starting from the second position
            current_neighbors_len = len(truncated_neighbor_ids[idx])
            if current_neighbors_len > 0:
                padded_nodes_neighbor_ids[idx, 1 : current_neighbors_len + 1] = truncated_neighbor_ids[idx]
                padded_nodes_edge_ids[idx, 1 : current_neighbors_len + 1] = truncated_edge_ids[idx]
                padded_nodes_neighbor_times[idx, 1 : current_neighbors_len + 1] = truncated_neighbor_times[idx]

        # Return the three padded NumPy arrays
        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times

    def get_features(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray, padded_nodes_edge_ids: np.ndarray,
                     padded_nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder):
        """
        get node, edge and time features
        :param node_interact_times: ndarray, shape (batch_size, )
        :param padded_nodes_neighbor_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_edge_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_neighbor_times: ndarray, shape (batch_size, max_seq_length)
        :param time_encoder: TimeEncoder, time encoder
        :return:
        """
        # Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        padded_nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(padded_nodes_neighbor_ids)]
        # Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        padded_nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(padded_nodes_edge_ids)]
        # Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        padded_nodes_neighbor_time_features = time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times).float().to(self.device))

        # ndarray, set the time features to all zeros for the padded timestamp
        padded_nodes_neighbor_time_features[torch.from_numpy(padded_nodes_neighbor_ids == 0)] = 0.0

        return padded_nodes_neighbor_node_raw_features, padded_nodes_edge_raw_features, padded_nodes_neighbor_time_features

    def get_patches(self, padded_nodes_neighbor_node_raw_features: torch.Tensor, padded_nodes_edge_raw_features: torch.Tensor,
                    padded_nodes_neighbor_time_features: torch.Tensor, padded_nodes_neighbor_co_occurrence_features: torch.Tensor = None, patch_size: int = 1):

        assert padded_nodes_neighbor_node_raw_features.shape[1] % patch_size == 0
        num_patches = padded_nodes_neighbor_node_raw_features.shape[1] // patch_size

        # list of Tensors with shape (num_patches, ), each Tensor with shape (batch_size, patch_size, node_feat_dim)
        patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features, \
        patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features = [], [], [], []

        for patch_id in range(num_patches):
            start_idx = patch_id * patch_size
            end_idx = patch_id * patch_size + patch_size
            patches_nodes_neighbor_node_raw_features.append(padded_nodes_neighbor_node_raw_features[:, start_idx: end_idx, :])
            patches_nodes_edge_raw_features.append(padded_nodes_edge_raw_features[:, start_idx: end_idx, :])
            patches_nodes_neighbor_time_features.append(padded_nodes_neighbor_time_features[:, start_idx: end_idx, :])
            patches_nodes_neighbor_co_occurrence_features.append(padded_nodes_neighbor_co_occurrence_features[:, start_idx: end_idx, :])

        batch_size = len(padded_nodes_neighbor_node_raw_features)
        # Tensor, shape (batch_size, num_patches, patch_size * node_feat_dim)
        patches_nodes_neighbor_node_raw_features = torch.stack(patches_nodes_neighbor_node_raw_features, dim=1).reshape(batch_size, num_patches, patch_size * self.node_feat_dim)
        # Tensor, shape (batch_size, num_patches, patch_size * edge_feat_dim)
        patches_nodes_edge_raw_features = torch.stack(patches_nodes_edge_raw_features, dim=1).reshape(batch_size, num_patches, patch_size * self.edge_feat_dim)
        # Tensor, shape (batch_size, num_patches, patch_size * time_feat_dim)
        patches_nodes_neighbor_time_features = torch.stack(patches_nodes_neighbor_time_features, dim=1).reshape(batch_size, num_patches, patch_size * self.time_feat_dim)

        patches_nodes_neighbor_co_occurrence_features = torch.stack(patches_nodes_neighbor_co_occurrence_features, dim=1).reshape(batch_size, num_patches, patch_size * self.resource_alloc_feat_dim)

        return patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features, patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()

class HyenaOperatorV2(nn.Module):
    """
    The core Hyena operator, combining a short local convolution
    with a data-controlled long-range convolution.
    """
    def __init__(self, dim, max_seq_len=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Short, local, depthwise convolution
        self.short_conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=dim
        )

        # Projections for the long convolution
        self.proj_v = nn.Linear(dim, dim)
        self.proj_z = nn.Linear(dim, dim)

        # Parameterizing the long convolution kernel
        self.kernel_params = nn.Parameter(torch.randn(dim, max_seq_len, 1))
        self.kernel_proj = nn.Linear(max_seq_len, max_seq_len)

    def forward(self, x):
        B, T, C = x.shape
        
        # Short Convolution Branch
        x_short = x.transpose(1, 2)  # (B, C, T) for Conv1d
        x_short = self.short_conv(x_short)[:, :, :T]
        x_short = x_short.transpose(1, 2)  # (B, T, C)

        # Long Convolution Branch
        v = F.gelu(self.proj_v(x))
        z = self.proj_z(x)

        # Generate the long convolution kernel `h`
        h_unproj = self.kernel_params.squeeze(-1)
        h_proj = self.kernel_proj(h_unproj) 
        h = h_proj.unsqueeze(0).transpose(1,2)[:,:T,:]

        # Perform FFT-based convolution
        h_fft = torch.fft.rfft(h, n=2 * T, dim=1)
        v_fft = torch.fft.rfft(v, n=2 * T, dim=1)
        y_fft = h_fft * v_fft
        y_long = torch.fft.irfft(y_fft, n=2 * T, dim=1)[:, :T, :]

        # Gating and Combination
        return x_short + (y_long * z)

class GatedMLP(nn.Module):
    """ Standard Gated MLP for channel mixing. """
    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()
        inner_dim = int(dim * mult)
        self.proj_in = nn.Linear(dim, inner_dim * 2)
        self.proj_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.proj_in(x)
        x, gate = x.chunk(2, dim=-1)
        x = F.gelu(gate) * x
        x = self.dropout(x)
        x = self.proj_out(x)
        return x

class HyenaBlock(nn.Module):
    """ A full Hyena block with sequence and channel mixing. """
    def __init__(self, dim, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.hyena = HyenaOperatorV2(dim, max_seq_len=max_seq_len)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = GatedMLP(dim, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Sequence mixing block
        x = x + self.dropout1(self.hyena(self.norm1(x)))
        # Channel mixing block
        x = x + self.dropout2(self.mlp(self.norm2(x)))
        return x

class HyenaEncoderV2(nn.Module):
    """
    A more powerful Hyena-based encoder using full Hyena blocks.
    """
    def __init__(self, feat_size=444, dim=256, depth=3, dropout=0.1, max_seq_len=1024):
        super().__init__()
        self.input_proj = nn.Linear(feat_size, dim)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(HyenaBlock(dim, max_seq_len=max_seq_len, dropout=dropout))
        self.output_norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, feat_size)

    def forward(self, x):
        # x: B, T, F
        if x.shape[1] > self.layers[0].hyena.max_seq_len:
             x = x[:, :self.layers[0].hyena.max_seq_len, :]

        z = self.input_proj(x)
        for layer in self.layers:
            z = layer(z)
        z = self.output_norm(z)
        return self.output_proj(z)

class LocalPathEncoderRobustAdvancedTemporal(nn.Module):

    def __init__(self, path_feat_dim: int, neighbor_sampler: NeighborSampler, device: str = 'cpu'):
        """
        Local Path and Co-occurrence Encoder.
        This class computes rich local features based on neighbor co-occurrence, common neighbors,
        and direct/2-hop paths between the source and destination nodes.

        :param path_feat_dim: int, dimension of path features (encodings)
        :param neighbor_sampler: NeighborSampler, neighbor sampler (kept for API compatibility)
        :param device: str, device
        """
        super(LocalPathEncoderRobustAdvancedTemporal, self).__init__()
        self.path_feat_dim = path_feat_dim
        self.neighbor_sampler = neighbor_sampler
        self.device = device

        self.path_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.path_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.path_feat_dim, out_features=self.path_feat_dim))

    def forward(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray,
                src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray,
                src_padded_nodes_neighbor_times: np.ndarray, dst_padded_nodes_neighbor_times: np.ndarray):
        """
        Compute the local path and co-occurrence features of nodes.
        (Function signature remains the same)
        """
        src_padded_nodes_paths = []
        dst_padded_nodes_paths = []
        epsilon = 1e-6

        for i in range(src_padded_nodes_neighbor_ids.shape[0]):
            src_neighbors, dst_neighbors = src_padded_nodes_neighbor_ids[i], dst_padded_nodes_neighbor_ids[i]
            src_id, dst_id = src_node_ids[i], dst_node_ids[i]
            src_neighbor_times, dst_neighbor_times = src_padded_nodes_neighbor_times[i], dst_padded_nodes_neighbor_times[i]
            current_time = node_interact_times[i]

            src_unique_keys, src_inverse_indices, src_counts = np.unique(src_neighbors, return_inverse=True, return_counts=True)
            src_mapping = dict(zip(src_unique_keys, src_counts))
            dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(dst_neighbors, return_inverse=True, return_counts=True)
            dst_mapping = dict(zip(dst_unique_keys, dst_counts))
            
            src_time_mapping = {nid: ts for nid, ts in zip(src_neighbors, src_neighbor_times) if nid != 0}
            dst_time_mapping = {nid: ts for nid, ts in zip(dst_neighbors, dst_neighbor_times) if nid != 0}

            # OVERALL AVERAGE INTER-ARRIVAL TIMES (IAT)
            # src avg IAT
            src_avg_iat = {}
            for neighbor_id in src_unique_keys:
                if neighbor_id == 0: continue
                timestamps = np.sort(src_neighbor_times[src_neighbors == neighbor_id])
                src_avg_iat[neighbor_id] = np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 0.0
            # dst avg IAT
            dst_avg_iat = {}
            for neighbor_id in dst_unique_keys:
                if neighbor_id == 0: continue
                timestamps = np.sort(dst_neighbor_times[dst_neighbors == neighbor_id])
                dst_avg_iat[neighbor_id] = np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 0.0

            # SPLIT-HISTORY (RECENT) CADENCE
            # src recent iat (second-half)
            src_recent_iat = {}
            for neighbor_id in src_unique_keys:
                if neighbor_id == 0: continue
                timestamps = np.sort(src_neighbor_times[src_neighbors == neighbor_id])
                # Need at least 4 timestamps for a meaningful split (2 in each half)
                if len(timestamps) >= 4:
                    split_point = len(timestamps) // 2
                    recent_timestamps = timestamps[split_point:]
                    src_recent_iat[neighbor_id] = np.mean(np.diff(recent_timestamps))
                else:
                    src_recent_iat[neighbor_id] = 0.0
            # dst recent iat (second-half)
            dst_recent_iat = {}
            for neighbor_id in dst_unique_keys:
                if neighbor_id == 0: continue
                timestamps = np.sort(dst_neighbor_times[dst_neighbors == neighbor_id])
                if len(timestamps) >= 4:
                    split_point = len(timestamps) // 2
                    recent_timestamps = timestamps[split_point:]
                    dst_recent_iat[neighbor_id] = np.mean(np.diff(recent_timestamps))
                else:
                    dst_recent_iat[neighbor_id] = 0.0

            # src side feats
            src_counts_in_src = src_counts[src_inverse_indices].astype(np.float32)
            src_counts_in_dst = np.array([dst_mapping.get(nid, 0) for nid in src_neighbors], dtype=np.float32)
            src_is_dst = np.array([1.0 if nid == dst_id else 0.0 for nid in src_neighbors], dtype=np.float32)
            src_neighbor_connects_to_dst = (src_counts_in_dst > 0).astype(np.float32)
            # replacing freq_diff with freq_diff log for stabalization
            src_freq_asymmetry = np.where(src_counts_in_dst > 0, src_counts_in_src / (src_counts_in_dst + epsilon), 0).astype(np.float32)
            # src_log_freq_diff = np.log1p(src_counts_in_src) - np.log1p(src_counts_in_dst)
            recency_in_src = current_time - src_neighbor_times
            recency_in_dst_on_src_side = np.array([current_time - dst_time_mapping.get(nid, 0.0) for nid in src_neighbors], dtype=np.float32)
            src_temporal_asymmetry = np.where(recency_in_src > epsilon, recency_in_dst_on_src_side / (recency_in_src + epsilon), 0).astype(np.float32)
            src_iat_with_src = np.array([src_avg_iat.get(nid, 0.0) for nid in src_neighbors], dtype=np.float32)
            src_iat_with_dst = np.array([dst_avg_iat.get(nid, 0.0) for nid in src_neighbors], dtype=np.float32)
            src_iat_asymmetry = np.where(src_iat_with_dst > epsilon, src_iat_with_src / (src_iat_with_dst + epsilon), 0).astype(np.float32)
            src_recent_iat_with_src = np.array([src_recent_iat.get(nid, 0.0) for nid in src_neighbors], dtype=np.float32)
            src_recent_iat_with_dst = np.array([dst_recent_iat.get(nid, 0.0) for nid in src_neighbors], dtype=np.float32)
            src_recent_iat_asymmetry = np.where(src_recent_iat_with_dst > epsilon, src_recent_iat_with_src / (src_recent_iat_with_dst + epsilon), 0).astype(np.float32)

            # stacking src feats
            src_features = np.stack([src_counts_in_src, src_counts_in_dst, src_is_dst, src_neighbor_connects_to_dst,
                                     src_freq_asymmetry, src_temporal_asymmetry, src_iat_asymmetry, src_recent_iat_asymmetry], axis=1)
            src_padded_nodes_paths.append(torch.from_numpy(src_features).float())

            # dst side feats
            dst_counts_in_src = np.array([src_mapping.get(nid, 0) for nid in dst_neighbors], dtype=np.float32)
            dst_counts_in_dst = dst_counts[dst_inverse_indices].astype(np.float32)
            dst_is_src = np.array([1.0 if nid == src_id else 0.0 for nid in dst_neighbors], dtype=np.float32)
            dst_neighbor_connects_to_src = (dst_counts_in_src > 0).astype(np.float32)
            # replacing freq_diff with freq_diff log for stabalization
            dst_freq_asymmetry = np.where(dst_counts_in_src > 0, dst_counts_in_dst / (dst_counts_in_src + epsilon), 0).astype(np.float32)
            # dst_log_freq_diff = np.log1p(dst_counts_in_src) - np.log1p(dst_counts_in_dst)
            recency_in_dst = current_time - dst_neighbor_times
            recency_in_src_on_dst_side = np.array([current_time - src_time_mapping.get(nid, 0.0) for nid in dst_neighbors], dtype=np.float32)
            dst_temporal_asymmetry = np.where(recency_in_dst > epsilon, recency_in_src_on_dst_side / (recency_in_dst + epsilon), 0).astype(np.float32)
            dst_iat_with_src = np.array([src_avg_iat.get(nid, 0.0) for nid in dst_neighbors], dtype=np.float32)
            dst_iat_with_dst = np.array([dst_avg_iat.get(nid, 0.0) for nid in dst_neighbors], dtype=np.float32)
            dst_iat_asymmetry = np.where(dst_iat_with_src > epsilon, dst_iat_with_dst / (dst_iat_with_src + epsilon), 0).astype(np.float32)
            dst_recent_iat_with_src = np.array([src_recent_iat.get(nid, 0.0) for nid in dst_neighbors], dtype=np.float32)
            dst_recent_iat_with_dst = np.array([dst_recent_iat.get(nid, 0.0) for nid in dst_neighbors], dtype=np.float32)
            dst_recent_iat_asymmetry = np.where(dst_recent_iat_with_src > epsilon, dst_recent_iat_with_dst / (dst_recent_iat_with_src + epsilon), 0).astype(np.float32)
            
            # stacking dst feats
            dst_features = np.stack([dst_counts_in_src, dst_counts_in_dst, dst_is_src, dst_neighbor_connects_to_src,
                                     dst_freq_asymmetry, dst_temporal_asymmetry, dst_iat_asymmetry, dst_recent_iat_asymmetry], axis=1)
            dst_padded_nodes_paths.append(torch.from_numpy(dst_features).float())

        # Stack batch results
        src_paths = torch.stack(src_padded_nodes_paths, dim=0).to(self.device)
        dst_paths = torch.stack(dst_padded_nodes_paths, dim=0).to(self.device)

        # Apply padding mask
        src_padding_mask = torch.from_numpy(src_padded_nodes_neighbor_ids == 0).to(self.device)
        src_paths[src_padding_mask] = 0.0
        dst_padding_mask = torch.from_numpy(dst_padded_nodes_neighbor_ids == 0).to(self.device)
        dst_paths[dst_padding_mask] = 0.0

        # Encode and aggregate features
        # add flag for aggr using sum, mean, var
        src_path_features = self.path_encode_layer(src_paths.unsqueeze(dim=-1)).sum(dim=2)
        dst_path_features = self.path_encode_layer(dst_paths.unsqueeze(dim=-1)).sum(dim=2)

        return src_path_features, dst_path_features