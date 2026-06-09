import sys
import networkx as nx
from collections import Counter
import warnings
import time # <--- Added for CPU timing

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

from typing import Dict, Tuple, Union, List, Optional, Set

import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
import torch.nn.functional as F
import torch.fft

from utils.utils import NeighborSampler
from models.modules import TimeEncoder

class GraphLSTM(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, patch_size: int , max_input_sequence_length: int,
                 channel_embedding_dim: int, hyena_dim: int=256, hyena_depth: int=3, hyena_max_seq_len: int=1024, 
                 num_channels: int = 4, dropout: float = 0.1, device: str = 'cpu'):

        super(GraphLSTM, self).__init__()

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

        # config = {
        #     # Source features
        #     'src_counts_in_src': True, 'src_counts_in_dst': True, 'src_is_dst': False, 'src_neighbor_connects_to_dst': False,
        #     'src_freq_asymmetry': False, 'src_temporal_asymmetry': False, 'src_iat_asymmetry': False, 'src_recent_iat_asymmetry': False,
            
        #     'dst_counts_in_src': True, 'dst_counts_in_dst': True, 'dst_is_dst': False, 'dst_neighbor_connects_to_dst': False,
        #     'dst_freq_asymmetry': False, 'dst_temporal_asymmetry': False, 'dst_iat_asymmetry': False, 'dst_recent_iat_asymmetry': False
        # }

        config = {
            # Source features
            'src_counts_in_src': True, 'src_counts_in_dst': True, 'src_is_dst': True, 'src_neighbor_connects_to_dst': True,
            'src_freq_asymmetry': True, 'src_temporal_asymmetry': True, 'src_iat_asymmetry': True, 'src_recent_iat_asymmetry': True,
            
            'dst_counts_in_src': True, 'dst_counts_in_dst': True, 'dst_is_dst': True, 'dst_neighbor_connects_to_dst': True,
            'dst_freq_asymmetry': True, 'dst_temporal_asymmetry': True, 'dst_iat_asymmetry': True, 'dst_recent_iat_asymmetry': True
        }

        self.local_path_encoder = LocalPathEncoderRobustAdvancedTemporalSelective(path_feat_dim=self.resource_alloc_feat_dim,
                                            neighbor_sampler=self.neighbor_sampler,
                                            feature_config=config,
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

        self.total_cpu_prep_ms = 0.0
        self.total_gpu_compute_ms = 0.0
        self.num_inference_batches = 0


    def print_timing_report(self):
        """Call this after an evaluation epoch to get the exact breakdown."""
        if self.num_inference_batches == 0:
            print("No inference batches measured yet.")
            return
        
        avg_cpu = self.total_cpu_prep_ms / self.num_inference_batches
        avg_gpu = self.total_gpu_compute_ms / self.num_inference_batches
        
        print(f"\n=======================================================")
        print(f"      GraphHyena Internal Inference Timing Report      ")
        print(f"=======================================================")
        print(f" Average CPU Data Prep Time   : {avg_cpu:.2f} ms/batch")
        print(f" Average True GPU Latency     : {avg_gpu:.2f} ms/batch")
        print(f"=======================================================\n")
        
        # Reset counters
        self.total_cpu_prep_ms = 0.0
        self.total_gpu_compute_ms = 0.0
        self.num_inference_batches = 0.0

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
        '''
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=src_node_ids, node_interact_times=node_interact_times)

        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=dst_node_ids, node_interact_times=node_interact_times)

        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=src_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=src_nodes_edge_ids_list, nodes_neighbor_times_list=src_nodes_neighbor_times_list,
                               patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length)

        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=dst_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=dst_nodes_edge_ids_list, nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
                               patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length)
        '''

        # =========================================================
        # 1. CPU PHASE: Numpy loops, neighbor fetching, and padding
        # =========================================================
        cpu_start_time = time.time()

        
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

        cpu_time_ms = (time.time() - cpu_start_time) * 1000.0
        
        src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features, lp_cpu_ms, lp_gpu_ms = self.local_path_encoder(
            src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
            dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
            src_node_ids=src_node_ids,
            dst_node_ids=dst_node_ids,
            node_interact_times=node_interact_times,  # <<< ADD THIS ARGUMENT
            src_padded_nodes_neighbor_times=src_padded_nodes_neighbor_times,
            dst_padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times,
        )

        cpu_time_ms += lp_cpu_ms

        # =========================================================
        # 2. GPU PHASE: PyTorch Neural Network Execution
        # =========================================================
        gpu_start_event = torch.cuda.Event(enable_timing=True)
        gpu_end_event = torch.cuda.Event(enable_timing=True)
        gpu_start_event.record()
                
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

        patches_data = [patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features,
                        patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features]
        patches_data = torch.stack(patches_data, dim=2)
        patches_data = patches_data.reshape(batch_size, src_num_patches + dst_num_patches, self.num_channels * self.channel_embedding_dim)

        # patches_data = self.ssm(patches_data)
        
        for transformer in self.ssm:
            patches_data = transformer(patches_data)
        
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

        gpu_end_event.record()
        torch.cuda.synchronize()
        gpu_time_ms = gpu_start_event.elapsed_time(gpu_end_event) + lp_gpu_ms

        # Accumulate metrics ONLY if we are in eval mode (so we don't slow down training)
        if not self.training:
            self.total_cpu_prep_ms += cpu_time_ms
            self.total_gpu_compute_ms += gpu_time_ms
            self.num_inference_batches += 1

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
    
    def pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray, nodes_neighbor_ids_list: list, nodes_edge_ids_list: list,
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
        max_seq_length = 0
        # first cut the sequence of nodes whose number of neighbors is more than max_input_sequence_length - 1 (we need to include the target node in the sequence)
        for idx in range(len(nodes_neighbor_ids_list)):
            assert len(nodes_neighbor_ids_list[idx]) == len(nodes_edge_ids_list[idx]) == len(nodes_neighbor_times_list[idx])
            if len(nodes_neighbor_ids_list[idx]) > max_input_sequence_length - 1:
                # cut the sequence by taking the most recent max_input_sequence_length interactions
                nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][-(max_input_sequence_length - 1):]
            if len(nodes_neighbor_ids_list[idx]) > max_seq_length:
                max_seq_length = len(nodes_neighbor_ids_list[idx])

        # include the target node itself
        max_seq_length += 1
        if max_seq_length % patch_size != 0:
            max_seq_length += (patch_size - max_seq_length % patch_size)
        assert max_seq_length % patch_size == 0

        # pad the sequences
        # three ndarrays with shape (batch_size, max_seq_length)
        padded_nodes_neighbor_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_edge_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_neighbor_times = np.zeros((len(node_ids), max_seq_length)).astype(np.float32)

        for idx in range(len(node_ids)):
            padded_nodes_neighbor_ids[idx, 0] = node_ids[idx]
            padded_nodes_edge_ids[idx, 0] = 0
            padded_nodes_neighbor_times[idx, 0] = node_interact_times[idx]

            if len(nodes_neighbor_ids_list[idx]) > 0:
                padded_nodes_neighbor_ids[idx, 1: len(nodes_neighbor_ids_list[idx]) + 1] = nodes_neighbor_ids_list[idx]
                padded_nodes_edge_ids[idx, 1: len(nodes_edge_ids_list[idx]) + 1] = nodes_edge_ids_list[idx]
                padded_nodes_neighbor_times[idx, 1: len(nodes_neighbor_times_list[idx]) + 1] = nodes_neighbor_times_list[idx]

        # three ndarrays with shape (batch_size, max_seq_length)
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
    # max_seq_len=1024, kernel_size=3
    def __init__(self, dim, max_seq_len=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # 1. Short, local, depthwise convolution
        self.short_conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=dim  # This makes it a depthwise convolution
        )

        # 2. Projections for the long convolution
        self.proj_v = nn.Linear(dim, dim)
        self.proj_z = nn.Linear(dim, dim)

        # 3. Parameterizing the long convolution kernel
        # Instead of making the kernel data-dependent (on x), we make it
        # position-dependent, which is a common practice in S4/Hyena.
        self.kernel_params = nn.Parameter(torch.randn(dim, max_seq_len, 1))
        self.kernel_proj = nn.Linear(max_seq_len, max_seq_len)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        
        # --- 1. Short Convolution Branch ---
        x_short = x.transpose(1, 2)  # (B, C, T) for Conv1d
        x_short = self.short_conv(x_short)[:, :, :T]
        x_short = x_short.transpose(1, 2)  # (B, T, C)

        # --- 2. Long Convolution Branch ---
        # Project inputs for the long convolution
        v = F.gelu(self.proj_v(x))
        z = self.proj_z(x)

        # Generate the long convolution kernel `h`
        # (C, T_max, 1) -> (C, T_max)
        h_unproj = self.kernel_params.squeeze(-1)
        h_proj = self.kernel_proj(h_unproj) # (C, T_max)
        h = h_proj.unsqueeze(0).transpose(1,2)[:,:T,:] # -> (1, T, C)

        # Perform FFT-based convolution
        h_fft = torch.fft.rfft(h, n=2 * T, dim=1)
        v_fft = torch.fft.rfft(v, n=2 * T, dim=1)
        y_fft = h_fft * v_fft
        y_long = torch.fft.irfft(y_fft, n=2 * T, dim=1)[:, :T, :]

        # --- 3. Gating and Combination ---
        # The output is a modulation of the long convolution by the input `z`
        # combined with the short convolution result.
        return x_short + (y_long * z)

class GatedMLP(nn.Module):
    """ Standard Gated MLP for channel mixing. """
    # mult=4
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
        # You may need to pad/truncate x to max_seq_len if it's dynamic
        if x.shape[1] > self.layers[0].hyena.max_seq_len:
            x = x[:, :self.layers[0].hyena.max_seq_len, :]
            
        z = self.input_proj(x)
        for layer in self.layers:
            z = layer(z)
        z = self.output_norm(z)
        return self.output_proj(z)
        
class LocalPathEncoderRobustAdvancedTemporalSelective(nn.Module):

    def __init__(self, path_feat_dim: int, neighbor_sampler: NeighborSampler,
                 feature_config: Optional[Dict[str, bool]] = None, device: str = 'cpu'):
        """
        Local Path and Co-occurrence Encoder with fine-grained, per-feature flags.

        :param path_feat_dim: int, dimension of path features (encodings)
        :param neighbor_sampler: NeighborSampler, neighbor sampler (kept for API compatibility)
        :param feature_config: dict, specifies which of the 16 features to use.
                               Keys must match self.feature_names. If None, all are used.
        :param device: str, device
        """
        super(LocalPathEncoderRobustAdvancedTemporalSelective, self).__init__()
        self.path_feat_dim = path_feat_dim
        self.neighbor_sampler = neighbor_sampler
        self.device = device

        # Define the names of all 16 available features
        self.feature_names = [
            # Source-side features
            'src_counts_in_src', 'src_counts_in_dst', 'src_is_dst', 'src_neighbor_connects_to_dst',
            'src_freq_asymmetry', 'src_temporal_asymmetry', 'src_iat_asymmetry', 'src_recent_iat_asymmetry',
            # Destination-side features
            'dst_counts_in_dst', 'dst_counts_in_src', 'dst_is_src', 'dst_neighbor_connects_to_src',
            'dst_freq_asymmetry', 'dst_temporal_asymmetry', 'dst_iat_asymmetry', 'dst_recent_iat_asymmetry'
        ]
        
        # Set up boolean flags for each feature
        self._setup_feature_flags(feature_config)
        
        # The encoder layer remains the same, as it processes each feature value independently
        self.path_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.path_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.path_feat_dim, out_features=self.path_feat_dim))

    def _setup_feature_flags(self, feature_config: Optional[Dict[str, bool]]):
        """Helper to set boolean flags and count enabled features for each side."""
        if feature_config is None:
            # Default to using all features if no config is provided
            feature_config = {name: True for name in self.feature_names}

        for name in self.feature_names:
            # Set an attribute like self.use_src_counts_in_src = True/False
            setattr(self, f"use_{name}", feature_config.get(name, False))

        self.num_src_features = sum(1 for name in self.feature_names if name.startswith('src_') and getattr(self, f"use_{name}"))
        self.num_dst_features = sum(1 for name in self.feature_names if name.startswith('dst_') and getattr(self, f"use_{name}"))
        
        if self.num_src_features + self.num_dst_features == 0:
            raise ValueError("At least one feature must be enabled in feature_config.")

    def forward(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray,
                src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray,
                src_padded_nodes_neighbor_times: np.ndarray, dst_padded_nodes_neighbor_times: np.ndarray):
        """
        Compute local path features based on the enabled flags for each of the 16 features.
        """

        # --- CPU PHASE TIMING ---
        cpu_start_time = time.time()

        batch_size = src_padded_nodes_neighbor_ids.shape[0]
        num_neighbors = src_padded_nodes_neighbor_ids.shape[1]
        epsilon = 1e-6

        src_padded_nodes_paths, dst_padded_nodes_paths = [], []

        for i in range(batch_size):
            # Per-interaction data
            src_neighbors, dst_neighbors = src_padded_nodes_neighbor_ids[i], dst_padded_nodes_neighbor_ids[i]
            src_neighbor_times, dst_neighbor_times = src_padded_nodes_neighbor_times[i], dst_padded_nodes_neighbor_times[i]
            src_id, dst_id = src_node_ids[i], dst_node_ids[i]
            current_time = node_interact_times[i]

            # Common pre-computations for frequency
            src_unique_keys, src_inverse_indices, src_counts = np.unique(src_neighbors, return_inverse=True, return_counts=True)
            src_mapping = dict(zip(src_unique_keys, src_counts))
            dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(dst_neighbors, return_inverse=True, return_counts=True)
            dst_mapping = dict(zip(dst_unique_keys, dst_counts))

            # --- Conditionally Pre-compute Expensive IATs ---
            src_avg_iat, dst_avg_iat, src_recent_iat, dst_recent_iat = {}, {}, {}, {}
            # Check if any IAT-based feature is enabled
            if self.use_src_iat_asymmetry or self.use_dst_iat_asymmetry:
                for n_id in src_unique_keys:
                    if n_id == 0: continue
                    ts = np.sort(src_neighbor_times[src_neighbors == n_id])
                    src_avg_iat[n_id] = np.sum(np.diff(ts)) if len(ts) > 1 else 0.0
                for n_id in dst_unique_keys:
                    if n_id == 0: continue
                    ts = np.sort(dst_neighbor_times[dst_neighbors == n_id])
                    dst_avg_iat[n_id] = np.sum(np.diff(ts)) if len(ts) > 1 else 0.0
            
            if self.use_src_recent_iat_asymmetry or self.use_dst_recent_iat_asymmetry:
                for n_id in src_unique_keys:
                    if n_id == 0: continue
                    ts = np.sort(src_neighbor_times[src_neighbors == n_id])
                    src_recent_iat[n_id] = np.sum(np.diff(ts[len(ts)//2:])) if len(ts) >= 4 else 0.0
                for n_id in dst_unique_keys:
                    if n_id == 0: continue
                    ts = np.sort(dst_neighbor_times[dst_neighbors == n_id])
                    dst_recent_iat[n_id] = np.sum(np.diff(ts[len(ts)//2:])) if len(ts) >= 4 else 0.0
            
            # ======================= SOURCE SIDE FEATURES =======================
            src_feature_list = []
            if self.num_src_features > 0:
                # --- Calculate intermediate values needed for source features ---
                src_counts_in_src_val = src_counts[src_inverse_indices].astype(np.float32)
                src_counts_in_dst_val = np.array([dst_mapping.get(nid, 0) for nid in src_neighbors], dtype=np.float32)

                if self.use_src_counts_in_src: src_feature_list.append(src_counts_in_src_val)
                if self.use_src_counts_in_dst: src_feature_list.append(src_counts_in_dst_val)
                if self.use_src_is_dst: src_feature_list.append((src_neighbors == dst_id).astype(np.float32))
                if self.use_src_neighbor_connects_to_dst: src_feature_list.append((src_counts_in_dst_val > 0).astype(np.float32))
                if self.use_src_freq_asymmetry: src_feature_list.append(np.where(src_counts_in_dst_val > 0, src_counts_in_src_val / (src_counts_in_dst_val + epsilon), 0).astype(np.float32))
                if self.use_src_temporal_asymmetry:
                    dst_time_mapping = {nid: ts for nid, ts in zip(dst_neighbors, dst_neighbor_times) if nid != 0}
                    recency_in_src = current_time - src_neighbor_times
                    recency_in_dst = np.array([current_time - dst_time_mapping.get(nid, 0.0) for nid in src_neighbors], dtype=np.float32)
                    src_feature_list.append(np.where(recency_in_src > epsilon, recency_in_dst / (recency_in_src + epsilon), 0).astype(np.float32))
                if self.use_src_iat_asymmetry:
                    iat_with_src = np.array([src_avg_iat.get(nid, 0.0) for nid in src_neighbors], dtype=np.float32)
                    iat_with_dst = np.array([dst_avg_iat.get(nid, 0.0) for nid in src_neighbors], dtype=np.float32)
                    src_feature_list.append(np.where(iat_with_dst > epsilon, iat_with_src / (iat_with_dst + epsilon), 0).astype(np.float32))
                if self.use_src_recent_iat_asymmetry:
                    recent_iat_with_src = np.array([src_recent_iat.get(nid, 0.0) for nid in src_neighbors], dtype=np.float32)
                    recent_iat_with_dst = np.array([dst_recent_iat.get(nid, 0.0) for nid in src_neighbors], dtype=np.float32)
                    src_feature_list.append(np.where(recent_iat_with_dst > epsilon, recent_iat_with_src / (recent_iat_with_dst + epsilon), 0).astype(np.float32))

                src_padded_nodes_paths.append(torch.from_numpy(np.stack(src_feature_list, axis=1)).float())
            
            # ======================= DESTINATION SIDE FEATURES =======================
            dst_feature_list = []
            if self.num_dst_features > 0:
                # --- Calculate intermediate values needed for destination features ---
                dst_counts_in_dst_val = dst_counts[dst_inverse_indices].astype(np.float32)
                dst_counts_in_src_val = np.array([src_mapping.get(nid, 0) for nid in dst_neighbors], dtype=np.float32)
                
                if self.use_dst_counts_in_dst: dst_feature_list.append(dst_counts_in_dst_val)
                if self.use_dst_counts_in_src: dst_feature_list.append(dst_counts_in_src_val)
                if self.use_dst_is_src: dst_feature_list.append((dst_neighbors == src_id).astype(np.float32))
                if self.use_dst_neighbor_connects_to_src: dst_feature_list.append((dst_counts_in_src_val > 0).astype(np.float32))
                if self.use_dst_freq_asymmetry: dst_feature_list.append(np.where(dst_counts_in_src_val > 0, dst_counts_in_dst_val / (dst_counts_in_src_val + epsilon), 0).astype(np.float32))
                if self.use_dst_temporal_asymmetry:
                    src_time_mapping = {nid: ts for nid, ts in zip(src_neighbors, src_neighbor_times) if nid != 0}
                    recency_in_dst = current_time - dst_neighbor_times
                    recency_in_src = np.array([current_time - src_time_mapping.get(nid, 0.0) for nid in dst_neighbors], dtype=np.float32)
                    dst_feature_list.append(np.where(recency_in_dst > epsilon, recency_in_src / (recency_in_dst + epsilon), 0).astype(np.float32))
                if self.use_dst_iat_asymmetry:
                    iat_with_dst = np.array([dst_avg_iat.get(nid, 0.0) for nid in dst_neighbors], dtype=np.float32)
                    iat_with_src = np.array([src_avg_iat.get(nid, 0.0) for nid in dst_neighbors], dtype=np.float32)
                    dst_feature_list.append(np.where(iat_with_src > epsilon, iat_with_dst / (iat_with_src + epsilon), 0).astype(np.float32))
                if self.use_dst_recent_iat_asymmetry:
                    recent_iat_with_dst = np.array([dst_recent_iat.get(nid, 0.0) for nid in dst_neighbors], dtype=np.float32)
                    recent_iat_with_src = np.array([src_recent_iat.get(nid, 0.0) for nid in dst_neighbors], dtype=np.float32)
                    dst_feature_list.append(np.where(recent_iat_with_src > epsilon, recent_iat_with_dst / (recent_iat_with_src + epsilon), 0).astype(np.float32))
                
                dst_padded_nodes_paths.append(torch.from_numpy(np.stack(dst_feature_list, axis=1)).float())

        cpu_time_ms = (time.time() - cpu_start_time) * 1000.0
        
        # --- GPU PHASE TIMING ---
        gpu_start_event = torch.cuda.Event(enable_timing=True)
        gpu_end_event = torch.cuda.Event(enable_timing=True)
        gpu_start_event.record()

        
        # --- Final Processing, Encoding, and Aggregation ---
        # Process source side
        if self.num_src_features > 0:
            src_paths = torch.stack(src_padded_nodes_paths, dim=0).to(self.device)
            src_padding_mask = torch.from_numpy(src_padded_nodes_neighbor_ids == 0).to(self.device)
            src_paths[src_padding_mask] = 0.0
            src_path_features = self.path_encode_layer(src_paths.unsqueeze(dim=-1)).sum(dim=2)
        else:
            # Return a zero tensor if no source features were enabled
            src_path_features = torch.zeros((batch_size, num_neighbors, self.path_feat_dim), device=self.device)

        # Process destination side
        if self.num_dst_features > 0:
            dst_paths = torch.stack(dst_padded_nodes_paths, dim=0).to(self.device)
            dst_padding_mask = torch.from_numpy(dst_padded_nodes_neighbor_ids == 0).to(self.device)
            dst_paths[dst_padding_mask] = 0.0
            dst_path_features = self.path_encode_layer(dst_paths.unsqueeze(dim=-1)).sum(dim=2) 
        else:
            # Return a zero tensor if no destination features were enabled
            dst_path_features = torch.zeros((batch_size, num_neighbors, self.path_feat_dim), device=self.device)

        gpu_end_event.record()
        torch.cuda.synchronize()
        gpu_time_ms = gpu_start_event.elapsed_time(gpu_end_event)
        
        return src_path_features, dst_path_features, cpu_time_ms, gpu_time_ms


# class LocalPathEncoderRobustAdvancedTemporalSelective(nn.Module):

#     def __init__(self, path_feat_dim: int, neighbor_sampler: NeighborSampler,
#                  feature_config: Optional[Dict[str, bool]] = None, device: str = 'cpu'):
#         """
#         Local Path and Co-occurrence Encoder with fine-grained, per-feature flags.
#         """
#         super(LocalPathEncoderRobustAdvancedTemporalSelective, self).__init__()
#         self.path_feat_dim = path_feat_dim
#         self.neighbor_sampler = neighbor_sampler
#         self.device = device

#         self.feature_names = [
#             'src_counts_in_src', 'src_counts_in_dst', 'src_is_dst', 'src_neighbor_connects_to_dst',
#             'src_freq_asymmetry', 'src_temporal_asymmetry', 'src_iat_asymmetry', 'src_recent_iat_asymmetry',
#             'dst_counts_in_dst', 'dst_counts_in_src', 'dst_is_src', 'dst_neighbor_connects_to_src',
#             'dst_freq_asymmetry', 'dst_temporal_asymmetry', 'dst_iat_asymmetry', 'dst_recent_iat_asymmetry'
#         ]
        
#         self._setup_feature_flags(feature_config)
        
#         # IMPROVEMENT: Use distinct linear layers to process the concatenated feature vectors
#         # instead of incorrectly sharing 1D weights across completely different feature semantics.
#         if self.num_src_features > 0:
#             self.src_path_encode_layer = nn.Sequential(
#                 nn.Linear(in_features=self.num_src_features, out_features=self.path_feat_dim),
#                 nn.LayerNorm(self.path_feat_dim),
#                 nn.GELU(),
#                 nn.Linear(in_features=self.path_feat_dim, out_features=self.path_feat_dim)
#             )

#         if self.num_dst_features > 0:
#             self.dst_path_encode_layer = nn.Sequential(
#                 nn.Linear(in_features=self.num_dst_features, out_features=self.path_feat_dim),
#                 nn.LayerNorm(self.path_feat_dim),
#                 nn.GELU(),
#                 nn.Linear(in_features=self.path_feat_dim, out_features=self.path_feat_dim)
#             )

#     def _setup_feature_flags(self, feature_config: Optional[Dict[str, bool]]):
#         if feature_config is None:
#             feature_config = {name: True for name in self.feature_names}

#         for name in self.feature_names:
#             setattr(self, f"use_{name}", feature_config.get(name, False))

#         self.num_src_features = sum(1 for name in self.feature_names if name.startswith('src_') and getattr(self, f"use_{name}"))
#         self.num_dst_features = sum(1 for name in self.feature_names if name.startswith('dst_') and getattr(self, f"use_{name}"))
        
#         if self.num_src_features + self.num_dst_features == 0:
#             raise ValueError("At least one feature must be enabled in feature_config.")

#     def forward(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray,
#                 src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray,
#                 src_padded_nodes_neighbor_times: np.ndarray, dst_padded_nodes_neighbor_times: np.ndarray):

#         # --- CPU PHASE TIMING ---
#         cpu_start_time = time.time()
        
#         batch_size = src_padded_nodes_neighbor_ids.shape[0]
#         num_neighbors = src_padded_nodes_neighbor_ids.shape[1]
#         epsilon = 1e-6

#         src_padded_nodes_paths, dst_padded_nodes_paths = [], []

#         for i in range(batch_size):
#             src_neighbors, dst_neighbors = src_padded_nodes_neighbor_ids[i], dst_padded_nodes_neighbor_ids[i]
#             src_neighbor_times, dst_neighbor_times = src_padded_nodes_neighbor_times[i], dst_padded_nodes_neighbor_times[i]
#             src_id, dst_id = src_node_ids[i], dst_node_ids[i]
#             current_time = node_interact_times[i]

#             src_unique_keys, src_inverse_indices, src_counts = np.unique(src_neighbors, return_inverse=True, return_counts=True)
#             src_mapping = dict(zip(src_unique_keys, src_counts))
#             dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(dst_neighbors, return_inverse=True, return_counts=True)
#             dst_mapping = dict(zip(dst_unique_keys, dst_counts))

#             src_avg_iat, dst_avg_iat, src_recent_iat, dst_recent_iat = {}, {}, {}, {}
#             if self.use_src_iat_asymmetry or self.use_dst_iat_asymmetry:
#                 for n_id in src_unique_keys:
#                     if n_id == 0: continue
#                     ts = np.sort(src_neighbor_times[src_neighbors == n_id])
#                     src_avg_iat[n_id] = np.mean(np.diff(ts)) if len(ts) > 1 else 0.0
#                 for n_id in dst_unique_keys:
#                     if n_id == 0: continue
#                     ts = np.sort(dst_neighbor_times[dst_neighbors == n_id])
#                     dst_avg_iat[n_id] = np.mean(np.diff(ts)) if len(ts) > 1 else 0.0
            
#             if self.use_src_recent_iat_asymmetry or self.use_dst_recent_iat_asymmetry:
#                 for n_id in src_unique_keys:
#                     if n_id == 0: continue
#                     ts = np.sort(src_neighbor_times[src_neighbors == n_id])
#                     src_recent_iat[n_id] = np.mean(np.diff(ts[len(ts)//2:])) if len(ts) >= 4 else 0.0
#                 for n_id in dst_unique_keys:
#                     if n_id == 0: continue
#                     ts = np.sort(dst_neighbor_times[dst_neighbors == n_id])
#                     dst_recent_iat[n_id] = np.mean(np.diff(ts[len(ts)//2:])) if len(ts) >= 4 else 0.0
            
#             # ======================= SOURCE SIDE FEATURES =======================
#             src_feature_list = []
#             if self.num_src_features > 0:
#                 src_counts_in_src_val = src_counts[src_inverse_indices].astype(np.float32)
#                 src_counts_in_dst_val = np.array([dst_mapping.get(nid, 0) for nid in src_neighbors], dtype=np.float32)

#                 # IMPROVEMENT: Applying log1p protects from explosion on highly active nodes
#                 if self.use_src_counts_in_src: src_feature_list.append(np.log1p(src_counts_in_src_val))
#                 if self.use_src_counts_in_dst: src_feature_list.append(np.log1p(src_counts_in_dst_val))
#                 if self.use_src_is_dst: src_feature_list.append((src_neighbors == dst_id).astype(np.float32))
#                 if self.use_src_neighbor_connects_to_dst: src_feature_list.append((src_counts_in_dst_val > 0).astype(np.float32))
                
#                 if self.use_src_freq_asymmetry: 
#                     ratio = np.where(src_counts_in_dst_val > 0, src_counts_in_src_val / (src_counts_in_dst_val + epsilon), 0)
#                     src_feature_list.append(np.log1p(ratio).astype(np.float32))
#                 if self.use_src_temporal_asymmetry:
#                     dst_time_mapping = {nid: ts for nid, ts in zip(dst_neighbors, dst_neighbor_times) if nid != 0}
#                     recency_in_src = np.maximum(0.0, current_time - src_neighbor_times)
#                     # IMPROVEMENT: Use `current_time` fallback so disjoint nodes evaluate to a 0 time gap, dodging epoch overflow
#                     recency_in_dst = np.maximum(0.0, np.array([current_time - dst_time_mapping.get(nid, current_time) for nid in src_neighbors], dtype=np.float32))
#                     ratio = np.where(recency_in_src > epsilon, recency_in_dst / (recency_in_src + epsilon), 0)
#                     src_feature_list.append(np.log1p(ratio).astype(np.float32))
#                 if self.use_src_iat_asymmetry:
#                     iat_with_src = np.array([src_avg_iat.get(nid, 0.0) for nid in src_neighbors], dtype=np.float32)
#                     iat_with_dst = np.array([dst_avg_iat.get(nid, 0.0) for nid in src_neighbors], dtype=np.float32)
#                     ratio = np.where(iat_with_dst > epsilon, iat_with_src / (iat_with_dst + epsilon), 0)
#                     src_feature_list.append(np.log1p(ratio).astype(np.float32))
#                 if self.use_src_recent_iat_asymmetry:
#                     recent_iat_with_src = np.array([src_recent_iat.get(nid, 0.0) for nid in src_neighbors], dtype=np.float32)
#                     recent_iat_with_dst = np.array([dst_recent_iat.get(nid, 0.0) for nid in src_neighbors], dtype=np.float32)
#                     ratio = np.where(recent_iat_with_dst > epsilon, recent_iat_with_src / (recent_iat_with_dst + epsilon), 0)
#                     src_feature_list.append(np.log1p(ratio).astype(np.float32))

#                 src_padded_nodes_paths.append(torch.from_numpy(np.stack(src_feature_list, axis=1)).float())
            
#             # ======================= DESTINATION SIDE FEATURES =======================
#             dst_feature_list = []
#             if self.num_dst_features > 0:
#                 dst_counts_in_dst_val = dst_counts[dst_inverse_indices].astype(np.float32)
#                 dst_counts_in_src_val = np.array([src_mapping.get(nid, 0) for nid in dst_neighbors], dtype=np.float32)
                
#                 if self.use_dst_counts_in_dst: dst_feature_list.append(np.log1p(dst_counts_in_dst_val))
#                 if self.use_dst_counts_in_src: dst_feature_list.append(np.log1p(dst_counts_in_src_val))
#                 if self.use_dst_is_src: dst_feature_list.append((dst_neighbors == src_id).astype(np.float32))
#                 if self.use_dst_neighbor_connects_to_src: dst_feature_list.append((dst_counts_in_src_val > 0).astype(np.float32))
                
#                 if self.use_dst_freq_asymmetry:
#                     ratio = np.where(dst_counts_in_src_val > 0, dst_counts_in_dst_val / (dst_counts_in_src_val + epsilon), 0)
#                     dst_feature_list.append(np.log1p(ratio).astype(np.float32))
#                 if self.use_dst_temporal_asymmetry:
#                     src_time_mapping = {nid: ts for nid, ts in zip(src_neighbors, src_neighbor_times) if nid != 0}
#                     recency_in_dst = np.maximum(0.0, current_time - dst_neighbor_times)
#                     recency_in_src = np.maximum(0.0, np.array([current_time - src_time_mapping.get(nid, current_time) for nid in dst_neighbors], dtype=np.float32))
#                     ratio = np.where(recency_in_dst > epsilon, recency_in_src / (recency_in_dst + epsilon), 0)
#                     dst_feature_list.append(np.log1p(ratio).astype(np.float32))
#                 if self.use_dst_iat_asymmetry:
#                     iat_with_dst = np.array([dst_avg_iat.get(nid, 0.0) for nid in dst_neighbors], dtype=np.float32)
#                     iat_with_src = np.array([src_avg_iat.get(nid, 0.0) for nid in dst_neighbors], dtype=np.float32)
#                     ratio = np.where(iat_with_src > epsilon, iat_with_dst / (iat_with_src + epsilon), 0)
#                     dst_feature_list.append(np.log1p(ratio).astype(np.float32))
#                 if self.use_dst_recent_iat_asymmetry:
#                     recent_iat_with_dst = np.array([dst_recent_iat.get(nid, 0.0) for nid in dst_neighbors], dtype=np.float32)
#                     recent_iat_with_src = np.array([src_recent_iat.get(nid, 0.0) for nid in dst_neighbors], dtype=np.float32)
#                     ratio = np.where(recent_iat_with_src > epsilon, recent_iat_with_dst / (recent_iat_with_src + epsilon), 0)
#                     dst_feature_list.append(np.log1p(ratio).astype(np.float32))
                
#                 dst_padded_nodes_paths.append(torch.from_numpy(np.stack(dst_feature_list, axis=1)).float())

        
#         cpu_time_ms = (time.time() - cpu_start_time) * 1000.0
        
#         # --- GPU PHASE TIMING ---
#         gpu_start_event = torch.cuda.Event(enable_timing=True)
#         gpu_end_event = torch.cuda.Event(enable_timing=True)
#         gpu_start_event.record()
        
#         # --- Final Processing, Encoding, and Aggregation ---
        
#         # Process source side
#         if self.num_src_features > 0:
#             src_paths = torch.stack(src_padded_nodes_paths, dim=0).to(self.device)
#             # Pass correctly through the dedicated input vector dimension -> path_feat_dim
#             src_path_features = self.src_path_encode_layer(src_paths)
            
#             src_padding_mask = torch.from_numpy(src_padded_nodes_neighbor_ids == 0).to(self.device)
#             src_path_features[src_padding_mask] = 0.0
#         else:
#             src_path_features = torch.zeros((batch_size, num_neighbors, self.path_feat_dim), device=self.device)

#         # Process destination side
#         if self.num_dst_features > 0:
#             dst_paths = torch.stack(dst_padded_nodes_paths, dim=0).to(self.device)
#             dst_path_features = self.dst_path_encode_layer(dst_paths)
            
#             dst_padding_mask = torch.from_numpy(dst_padded_nodes_neighbor_ids == 0).to(self.device)
#             dst_path_features[dst_padding_mask] = 0.0
#         else:
#             dst_path_features = torch.zeros((batch_size, num_neighbors, self.path_feat_dim), device=self.device)

#         gpu_end_event.record()
#         torch.cuda.synchronize()
#         gpu_time_ms = gpu_start_event.elapsed_time(gpu_end_event)
        
#         return src_path_features, dst_path_features, cpu_time_ms, gpu_time_ms