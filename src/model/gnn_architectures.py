#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This file contains the GNN architecture, as the GNN class.
One of the fundamental steps in the GNN's update rule is the
use of an appropriate convolution. We define 2 convolutions,
one for coloured edges, and one for colourless edges.

@author: ----
"""
from os import write

import torch

from torch_geometric.nn import MessagePassing

import torch.nn.functional as F
from torch.nn import Parameter

# Define a convolution step (will be used in each layer of the model)
class EC_GCNConv(MessagePassing):

    # in_channels (int) - Size of each input sample
    # out_channels (int) - Size of each output sample
    def __init__(self, in_channels, out_channels, edge_colours, aggregation):

        self.aggr = aggregation
        aggr_name = getattr(aggregation, "value", aggregation)
        if aggr_name not in {"sum","max"}:
            raise ValueError(f"Unsupported aggregation mode: {aggr_name!r}")
        super(EC_GCNConv, self).__init__(aggr=aggr_name)
        self.weights = Parameter(torch.Tensor(edge_colours, out_channels, in_channels))
        self.weights.data.normal_(0, 0.001)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_colours = edge_colours
        
    def forward(self, x, edge_index, edge_colour):
        out = torch.zeros(x.size(0), self.out_channels, device=x.device)
        for i in range(self.edge_colours):
            edge_mask = edge_colour == i
            temp_edges = edge_index[:, edge_mask]
            out += F.linear(self.propagate(temp_edges, x=x, size=(x.size(0), x.size(0))), self.weights[i], bias=None)
        return out

class GNN(torch.nn.Module):

    def __init__(self, feature_dimension, num_edge_colours, num_layers, aggregations):
        super(GNN, self).__init__()

        self.num_layers = num_layers

        self.num_colours = num_edge_colours
        # From layer 0 (left) to layer L (right)
        self.dimensions = [feature_dimension]
        for _ in range(num_layers - 1):
            self.dimensions.append(2 * feature_dimension)
        self.dimensions.append(feature_dimension)

        self.agg_functions = aggregations

        self.convs = torch.nn.ModuleList()
        self.lin_selfs = torch.nn.ModuleList()
        for l in range(num_layers):
            self.convs.append(EC_GCNConv(self.dimensions[l], self.dimensions[l + 1], num_edge_colours, self.agg_functions[l]))
            self.lin_selfs.append(torch.nn.Linear(self.dimensions[l], self.dimensions[l + 1]))
                
        self.output = torch.nn.Sigmoid()

    # One thing to keep in mind is that since this is a torch.nn.Module, you can call a GNN by writing model([yourdata])
    # and this essentially calls this forward. So think of this as a __call__ method

    # Note also that unlike most "forward" implementations, this returns all feature vectors of intermediate layers.

    def forward(self, data):
        x, edge_index, edge_colour = data.x, data.edge_index, data.edge_type
        activations = [x.detach().clone()] # Not needed for training

        for l in range(self.num_layers):
            x = self.lin_selfs[l](x) + self.convs[l](x, edge_index, edge_colour)

            if l < self.num_layers - 1:
                x = torch.relu(x)
                activations.append(x.detach().clone()) 
            else:
                # Note: this translation is irrelevant since the bias vectors are not
                # constrained to the positive reals, therefore it isn't mentioned in
                # the report. However, I've left it here for completeness since the
                # models were trained with it.
                x = self.output(x - 10)
                activations.append(x) # Final layer's activations are kept as its the output

        return activations

    def layer_dimension(self, layer):
        return self.dimensions[layer]

    def matrix_A(self, layer):
        if 1 <= layer <= self.num_layers:
            return self.lin_selfs[layer - 1].weight.detach()
        else:
            return None 

    def matrix_B(self, layer, colour):
        if 1 <= layer <= self.num_layers:
            return self.lin_selfs[layer - 1].weights[colour].detach()
        else:
            return None 

    def bias(self, layer):
        if 1 <= layer <= self.num_layers:
            return self.lin_selfs[layer - 1].bias.detach()
        elif layer == self.num_layers:
            return self.lin_selfs[layer - 1].bias.detach() - 10
        else:
            return None

    def activation(self, layer):
        if 1 <= layer < self.num_layers:
            return torch.relu
        elif layer == self.num_layers:
            return torch.nn.Sigmoid()
        else:
            return None

    def aggregation_function(self, layer):
        if 1 <= layer <= self.num_layers:
            return self.aggs[layer - 1]
        else:
            return None
