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

    def __init__(self, feature_dimension, num_edge_colours, aggregation_1, aggregation_2, num_layers=2):
        super(GNN, self).__init__()

        #self.num_layers = 2 # Currently hardcoded!
        self.num_layers = num_layers
        self.num_colours = num_edge_colours

        # From layer 0 (left) to layer L (right)
        #self.dimensions = [feature_dimension, 2*feature_dimension, feature_dimension]
        
        # Expand and keep shape of dimensions for varible layers
        self.dimensions = [feature_dimension] + [2*feature_dimension]*(self.num_layers-1) + [feature_dimension]

        ## self.agg_1 = aggregation_1
        ## self.agg_2 = aggregation_2

        # apply agg_2 only to the final layer, and agg_1 to all other layers
        self._aggregations = [aggregation_2 if i == num_layers - 1 else aggregation_1 for i in range(num_layers)]

        ## self.conv1 = EC_GCNConv(self.dimensions[0], self.dimensions[1], num_edge_colours, self.agg_1)
        ## self.conv2 = EC_GCNConv(self.dimensions[1], self.dimensions[2], num_edge_colours, self.agg_2)

        ## self.lin_self_1 = torch.nn.Linear(self.dimensions[0], self.dimensions[1])
        ## self.lin_self_2 = torch.nn.Linear(self.dimensions[1], self.dimensions[2])
        # Build conv and lin layers
        self.convs = torch.nn.ModuleList([
            EC_GCNConv(self.dimensions[i], self.dimensions[i + 1],
                       num_edge_colours, self._aggregations[i])
            for i in range(num_layers)
        ])
        self.lin_selfs = torch.nn.ModuleList([
            torch.nn.Linear(self.dimensions[i], self.dimensions[i + 1])
            for i in range(num_layers)
        ])
        self.output = torch.nn.Sigmoid()

    # One thing to keep in mind is that since this is a torch.nn.Module, you can call a GNN by writing model([yourdata])
    # and this essentially calls this forward. So think of this as a __call__ method

    # Note also that unlike most "forward" implementations, this returns all feature vectors of intermediate layers.

    def forward(self, data):
        x, edge_index, edge_colour = data.x, data.edge_index, data.edge_type

        intermediates = [x.detach().clone()] 

        for i in range(self.num_layers):
            x = self.lin_selfs[i](x) + self.convs[i](x, edge_index, edge_colour)
            if i < self.num_layers - 1:
                # Hidden layer: ReLU activation; detach and save as fl_{i+1}.
                x = torch.relu(x)
                intermediates.append(x.detach().clone())
            else:
                # Note: this translation is irrelevant since the bias vectors are not
                # constrained to the positive reals, therefore it isn't mentioned in
                # the report. However, I've left it here for completeness since the
                # models were trained with it.
                x = self.output(x - 10)

        return x, intermediates  # (flL with grad, [fl0, ..., fl_{L-1}] detached)

    def layer_dimension(self, layer):
        return self.dimensions[layer]

    def matrix_A(self, layer):
        if layer == 1:
            return self.lin_self_1.weight.detach()
        elif layer == 2:
            return self.lin_self_2.weight.detach()
        else:
            return None

    def matrix_B(self, layer, colour):
        if layer == 1:
            return self.conv1.weights[colour].detach()
        elif layer == 2:
            return self.conv2.weights[colour].detach()
        else:
            return None

    def bias(self, layer):
        if layer == 1:
            return self.lin_self_1.bias.detach()
        elif layer == 2:
            return self.lin_self_2.bias.detach() - 10
        else:
            return None

    def activation(self, layer):
        if layer == 1:
            return torch.relu
        elif layer == 2:
            m = torch.nn.Sigmoid()
            return m
        else:
            return None

    def aggregation_function(self, layer):
        if layer == 1:
            return self.agg_1
        elif layer == 2:
            return self.agg_2
        else:
            return None
#
