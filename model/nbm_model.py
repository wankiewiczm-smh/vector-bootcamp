"""
This is a simplified re-implementation of the Neural Basis Model.
available at https://github.com/facebookresearch/nbm-spam

"""


import math
from collections import OrderedDict
from itertools import combinations
import torch
import torch.nn as nn

# -----------------------------------------------
# Basic building block: MLP to learn n-ary bases.
# -----------------------------------------------
class ConceptNNBasesNary(nn.Module):
    """
    Neural Network module to learn basis functions from n-ary feature interactions.
    """
    def __init__(self, order, num_bases, hidden_dims, dropout=0.0, batchnorm=False):
        """
        Args:
            order (int): Order of the n-ary interactions.
            num_bases (int): Number of bases to learn.
            hidden_dims (list or tuple): List of hidden layer dimensions.
            dropout (float): Dropout probability.
            batchnorm (bool): If True, use batch normalization.
        """
        super(ConceptNNBasesNary, self).__init__()

        if order <= 0:
            raise ValueError("Order of n-ary interactions must be > 0.")

        layers = []
        input_dim = order
        for dim in hidden_dims:
            layers.append(nn.Linear(in_features=input_dim, out_features=dim))
            if batchnorm:
                layers.append(nn.BatchNorm1d(dim))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.ReLU())
            input_dim = dim

        # Final layer to produce the basis outputs
        layers.append(nn.Linear(in_features=input_dim, out_features=num_bases))
        if batchnorm:
            layers.append(nn.BatchNorm1d(num_bases))
        layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ------------------------------------------------------------------
# Main model: Neural Basis Model for n-ary concept interactions.
# ------------------------------------------------------------------
class ConceptNBMNary(nn.Module):
    """
    Neural Basis Model (NBM) learns a set of global basis functions (via an MLP)
    that are applied to each n-ary combination of input features. Supports arbitrary
    n-ary orders via dynamic combination generation.

    Note: The "polynomial" mode (which uses a SPAM module) is not supported in this pure
    PyTorch implementation.
    """
    def __init__(
        self,
        num_concepts,
        num_classes,
        nary=None,
        num_bases=100,
        hidden_dims=(256, 128, 128),
        num_subnets=1,
        dropout=0.0,
        bases_dropout=0.0,
        batchnorm=True,
        output_penalty=0.0,
        polynomial=None,
    ):
        """
        Args:
            num_concepts (int): Number of input concepts.
            num_classes (int): Number of output classes.
            nary: None (default unary), a list of orders (e.g. [1, 2, 3]), or a dict mapping orders to indices.
            num_bases (int): Number of bases learned.
            hidden_dims (tuple): Hidden dimensions for the MLP bases.
            num_subnets (int): Number of subnetworks per n-ary order.
            dropout (float): Dropout probability within the MLP.
            bases_dropout (float): Dropout probability applied to basis outputs.
            batchnorm (bool): Whether to use batch normalization.
            output_penalty (float): (Unused in this snippet) output penalty coefficient.
            polynomial: If not None, triggers a polynomial mode (not supported here).
        """
        super(ConceptNBMNary, self).__init__()

        self._num_concepts = num_concepts
        self._num_classes = num_classes
        self._num_bases = num_bases
        self._num_subnets = num_subnets
        self._batchnorm = batchnorm
        self._output_penalty = output_penalty

        if polynomial is not None:
            raise NotImplementedError("Polynomial mode is not supported in this pure PyTorch implementation.")

        # Generate n-ary combinations
        if nary is None:
            # If not specified, use all unary interactions.
            self._nary_indices = {"1": list(combinations(range(self._num_concepts), 1))}
        elif isinstance(nary, list):
            self._nary_indices = {
                str(order): list(combinations(range(self._num_concepts), order))
                for order in nary
            }
        elif isinstance(nary, dict):
            self._nary_indices = nary
        else:
            raise TypeError("'nary' must be None, a list, or a dict.")

        # Create a basis network for each order and subnet.
        self.bases_nary_models = nn.ModuleDict()
        for order in self._nary_indices.keys():
            for subnet in range(self._num_subnets):
                key = self.get_key(order, subnet)
                self.bases_nary_models[key] = ConceptNNBasesNary(
                    order=int(order),
                    num_bases=self._num_bases,
                    hidden_dims=hidden_dims,
                    dropout=dropout,
                    batchnorm=batchnorm,
                )

        self.bases_dropout = nn.Dropout(p=bases_dropout)

        # Calculate total number of basis outputs across orders and subnets.
        num_out_features = (
            sum(len(self._nary_indices[order]) for order in self._nary_indices.keys())
            * self._num_subnets
        )

        # Featurizer: uses a grouped 1D convolution to combine basis outputs.
        self.featurizer = nn.Conv1d(
            in_channels=num_out_features * self._num_bases,
            out_channels=num_out_features,
            kernel_size=1,
            groups=num_out_features,
        )

        self._use_spam = False
        self.classifier = nn.Linear(
            in_features=num_out_features,
            out_features=self._num_classes,
            bias=True,
        )

    def get_key(self, order, subnet):
        return f"ord{order}_net{subnet}"

    def forward(self, input):
        bases = []
        # Loop over each n-ary order and corresponding subnet.
        for order in self._nary_indices.keys():
            for subnet in range(self._num_subnets):
                key = self.get_key(order, subnet)
                # Select the features corresponding to the current combination.
                input_order = input[:, self._nary_indices[order]]
                # Reshape to feed into the MLP basis network.
                out = self.bases_nary_models[key](
                    input_order.reshape(-1, input_order.shape[-1])
                )
                # Reshape back to [batch, n_tuples, num_bases].
                out = out.reshape(input_order.shape[0], input_order.shape[1], -1)
                bases.append(self.bases_dropout(out))

        # Concatenate outputs from all orders and subnets.
        bases = torch.cat(bases, dim=-2)
        # Apply featurization via a grouped convolution.
        out_feats = self.featurizer(bases.reshape(input_order.shape[0], -1, 1)).squeeze(-1)
        out = self.classifier(out_feats)
        if self.training:
            return out, out_feats
        else:
            return out


# --------------------------------------------------------------------
# Sparse variant: Only compute basis outputs for “non-ignored” feature tuples.
# --------------------------------------------------------------------
class ConceptNBMNarySparse(nn.Module):
    """
    ConceptNBMNarySparse
    --------------------
    A Sparse Neural Basis Model (NBM) that computes basis functions only for feature tuples 
    not equal to a given ignore value. This model supports n-ary combinations of input concepts 
    and allows for sparse computation by ignoring specific input values.

    Attributes:
        _num_concepts (int): Number of input concepts.
        _num_classes (int): Number of output classes.
        _num_bases (int): Number of basis functions learned.
        _batchnorm (bool): Whether batch normalization is applied.
        _output_penalty (float): (Unused) Output penalty coefficient.
        _nary_indices (dict): Dictionary mapping n-ary orders to their respective indices.
        _nary_ignore_input (OrderedDict): Values to ignore for sparse computation, per n-ary order.
        bases_nary_models (nn.ModuleDict): Dictionary of n-ary basis models.
        bases_dropout (nn.Dropout): Dropout layer applied to basis outputs.
        featurizer_params (nn.ModuleDict): Dictionary of featurizer parameters (weights and biases).
        classifier (nn.Linear): Linear layer for classification.

    Methods:
        __init__(self, num_concepts, num_classes, nary=None, num_bases=100, hidden_dims=(256, 128, 128), 
                 dropout=0.0, bases_dropout=0.0, batchnorm=False, output_penalty=0.0, nary_ignore_input=0.0):
            Initializes the ConceptNBMNarySparse model.

                nary (None, list, or dict): Specifies n-ary orders. Default is None (unary).
                num_bases (int): Number of basis functions learned. Default is 100.
                hidden_dims (tuple): Hidden dimensions for the MLP bases. Default is (256, 128, 128).
                dropout (float): Dropout probability in the MLP. Default is 0.0.
                bases_dropout (float): Dropout probability applied to basis outputs. Default is 0.0.
                batchnorm (bool): Whether to use batch normalization. Default is False.
                output_penalty (float): (Unused) Output penalty coefficient. Default is 0.0.
                nary_ignore_input (float or dict): Value(s) to ignore for sparse computation. Default is 0.0.

        reset_parameters(self):
            Resets the parameters of the model using Kaiming initialization.

            This method initializes the weights and biases of the featurizer parameters 
            for each n-ary order using Kaiming uniform initialization.

        forward(self, input):
            Performs a forward pass through the model.

                input (torch.Tensor): Input tensor of shape (batch_size, num_concepts).

            Returns:
                torch.Tensor: Output logits of shape (batch_size, num_classes).
                torch.Tensor: (If training) Intermediate features of shape (batch_size, num_out_features).
    """
    def __init__(
        self,
        num_concepts,
        num_classes,
        nary=None,
        num_bases=100,
        hidden_dims=(256, 128, 128),
        dropout=0.0,
        bases_dropout=0.0,
        batchnorm=False,
        output_penalty=0.0,
        nary_ignore_input=0.0,
    ):
        """
        Args:
            num_concepts (int): Number of input concepts.
            num_classes (int): Number of output classes.
            nary: None (default unary), a list of orders, or a dict mapping orders to indices.
            num_bases (int): Number of bases learned.
            hidden_dims (tuple): Hidden dimensions for the MLP bases.
            dropout (float): Dropout probability in the MLP.
            bases_dropout (float): Dropout probability applied to basis outputs.
            batchnorm (bool): Whether to use batch normalization.
            output_penalty (float): (Unused) output penalty coefficient.
            nary_ignore_input (float or dict): Value(s) to ignore for sparse computation.
        """
        super(ConceptNBMNarySparse, self).__init__()
        self._num_concepts = num_concepts
        self._num_classes = num_classes
        self._num_bases = num_bases
        self._batchnorm = batchnorm
        self._output_penalty = output_penalty

        if nary is None:
            self._nary_indices = {"1": list(combinations(range(self._num_concepts), 1))}
        elif isinstance(nary, list):
            self._nary_indices = {
                str(order): list(combinations(range(self._num_concepts), order))
                for order in nary
            }
        elif isinstance(nary, dict):
            self._nary_indices = nary
        else:
            raise TypeError("'nary' must be None, a list, or a dict.")

        # Set the ignore value for sparse computation.
        if isinstance(nary_ignore_input, float):
            self._nary_ignore_input = OrderedDict(
                {order: nary_ignore_input for order in self._nary_indices.keys()}
            )
        elif isinstance(nary_ignore_input, dict):
            self._nary_ignore_input = OrderedDict(sorted(nary_ignore_input.items()))
        else:
            raise TypeError("'nary_ignore_input' must be a float or dict.")

        self.bases_nary_models = nn.ModuleDict({
            order: ConceptNNBasesNary(
                order=int(order),
                num_bases=self._num_bases,
                hidden_dims=hidden_dims,
                dropout=dropout,
                batchnorm=batchnorm,
            )
            for order in self._nary_indices.keys()
        })

        self.bases_dropout = nn.Dropout(p=bases_dropout)

        self.featurizer_params = nn.ModuleDict({
            order: nn.ParameterDict({
                "weight": nn.Parameter(
                    torch.empty((len(self._nary_indices[order]), self._num_bases))
                ),
                "bias": nn.Parameter(torch.empty(len(self._nary_indices[order])))
            })
            for order in self._nary_indices.keys()
        })

        num_out_features = sum(len(self._nary_indices[order]) for order in self._nary_indices.keys())
        self.classifier = nn.Linear(in_features=num_out_features, out_features=self._num_classes, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        # Use Kaiming initialization for the featurizer parameters.
        for order in self._nary_indices.keys():
            nn.init.kaiming_uniform_(self.featurizer_params[order]["weight"], a=math.sqrt(5))
            if self.featurizer_params[order]["bias"] is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    self.featurizer_params[order]["weight"]
                )
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.featurizer_params[order]["bias"], -bound, bound)

    def forward(self, input):
        out_feats = []
        for order in self._nary_indices.keys():
            input_order = input[:, self._nary_indices[order]]
            # Create a mask where input values differ from the ignore value.
            ignore_input = self._nary_ignore_input[order]
            sparse_indices = torch.any(input_order != ignore_input, dim=-1)

            bases_sparse = self.bases_dropout(
                self.bases_nary_models[order](input_order[sparse_indices, :])
            )
            # Get corresponding parameters for the non-ignored indices.
            indices = sparse_indices.nonzero()[:, -1]
            weight = self.featurizer_params[order]["weight"][indices, :]
            bias = self.featurizer_params[order]["bias"][indices]

            out_feats_sparse = torch.mul(weight, bases_sparse).sum(dim=-1) + bias

            out_feats_dense = torch.zeros(
                (input_order.shape[0], input_order.shape[1]), device=input.device
            )
            out_feats_dense[sparse_indices] = out_feats_sparse

            out_feats.append(out_feats_dense)

        out_feats = torch.cat(out_feats, dim=-1)
        out = self.classifier(out_feats)
        if self.training:
            return out, out_feats
        else:
            return out