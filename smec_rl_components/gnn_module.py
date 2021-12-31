import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
        """

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
        self.hidden_dim = hidden_dim

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                linear_res = self.linears[layer](h)
                preserve_shape = linear_res.shape
                linear_res = self.batch_norms[layer](linear_res.reshape(-1, self.hidden_dim))
                linear_res = linear_res.reshape(preserve_shape)
                h = F.relu(linear_res)
            return self.linears[self.num_layers - 1](h)


class GraphCNN(nn.Module):
    def __init__(self,
                 num_layers,
                 num_mlp_layers,
                 input_dim,
                 hidden_dim,
                 # final_dropout,
                 learn_eps,
                 neighbor_pooling_type):
        """
        num_layers: number of layers in the neural networks (INCLUDING the input layer)
        num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
        input_dim: dimensionality of input features
        hidden_dim: dimensionality of hidden units at ALL layers
        output_dim: number of classes for prediction
        final_dropout: dropout ratio on the final linear layer
        learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
        """

        super(GraphCNN, self).__init__()

        # self.final_dropout = final_dropout
        self.num_layers = num_layers
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        # common out the eps if you do not need to use it, otherwise the it will cause
        # error "not in the computational graph"
        # if self.learn_eps:
        #     self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))

        # List of MLPs
        self.mlps = torch.nn.ModuleList()

        # List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.hidden_dim = hidden_dim

    def next_layer_eps(self, h, layer, padded_neighbor_list=None, Adj_block=None):
        # pooling neighboring nodes and center nodes separately by epsilon reweighting.

        if self.neighbor_pooling_type == "max":
            # If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            pooled = torch.mm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.mm(Adj_block, torch.ones((Adj_block.shape[0], 1)))
                pooled = pooled / degree

        # Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer]) * h
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    def next_layer(self, h, layer, padded_neighbor_list=None, Adj_block=None):

        # pooling neighboring nodes and center nodes altogether
        if self.neighbor_pooling_type == "max":
            # If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            pooled = torch.bmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.bmm(Adj_block, torch.ones((Adj_block.shape[0], 1)))
                pooled = pooled / degree
        # representation of neighboring and center nodes
        pooled_rep = self.mlps[layer](pooled)
        before_norm_s = pooled_rep.shape
        h = self.batch_norms[layer](pooled_rep.reshape(-1, self.hidden_dim)).reshape(before_norm_s)

        # non-linearity
        h = F.relu(h)
        return h

    def forward(self, x, padded_nei, adj):
        x_concat = x
        padded_neighbor_list, adj_block = None, None
        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = padded_nei
        else:
            adj_block = adj

        # list of hidden representation at each layer (including input)
        h = x_concat

        for layer in range(self.num_layers - 1):
            if self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, Adj_block=adj_block)
            elif self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, Adj_block=adj_block)

        h_nodes = h.clone()
        return h_nodes


if __name__ == '__main__':
    ''' Test attention block
    attention = Attention()
    g = torch.tensor([[1., 2.]], requires_grad=True)
    candidates = torch.tensor([[3., 3.],
                               [2., 2.]], requires_grad=True)

    ret = attention(g, candidates)
    print(ret)
    loss = ret.sum()
    print(loss)

    grad = torch.autograd.grad(loss, g)

    print(grad)
    '''
    default_gnn = GraphCNN(num_layers=3, num_mlp_layers=2, input_dim=2, hidden_dim=64, learn_eps=False,
                           neighbor_pooling_type='sum')
