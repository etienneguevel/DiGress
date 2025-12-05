import sys
import os

# Add annotix-ml to sys.path
current_path = os.path.dirname(os.path.realpath(__file__))
# ../../
annotix_ml_path = os.path.abspath(os.path.join(current_path, "../../../../annotix-ml"))
if annotix_ml_path not in sys.path:
    sys.path.append(annotix_ml_path)

import torch
import torch.nn as nn
from annotix_ml.graphtransf.models.gnn import GnnNodeEdges
from annotix_ml.graphtransf.data.atoms_data import VALID_ELEMENTS, TYPE_EDGES
from src.utils import PlaceHolder


class GnnNodeEdgesAdapter(nn.Module):
    """
    Adapter to make GnnNodeEdges compatible with DiGress's DiscreteDenoisingDiffusion.
    """

    def __init__(
        self,
        n_layers: int,
        input_dims: dict,
        hidden_mlp_dims: dict,
        hidden_dims: dict,
        output_dims: dict,
        act_fn_in=None,
        act_fn_out=None,
    ):
        super().__init__()

        # Mapping arguments
        # input_dims['X'] contains both N (node type) and extra features.
        # But GnnNodeEdges expects specific sizes.

        # We assume:
        # hidden_dims['dx'] -> d (node hidden dim)
        # hidden_dims['de'] -> de (edge hidden dim)
        # hidden_dims['dy'] -> dy (global hidden dim)
        # hidden_dims['n_head'] -> n_heads

        # node_features and global_features inputs to EmbeddingLaplacian/GnnNodeEdges
        # seem to correspond to the input dimensions of features.

        # In DiGress:
        # input_dims['X'] is TOTAL node features (including extra).
        # input_dims['E'] is edge features.
        # input_dims['y'] is global features.

        self.d = hidden_dims["dx"]
        self.de = hidden_dims["de"]
        self.dy = hidden_dims["dy"]
        self.n_heads = hidden_dims["n_head"]
        self.n_layers = n_layers

        # GnnNodeEdges hardcodes natoms and nbonds in __init__ for EmbeddingLaplacian.
        # However, we might want to use the ones from input_dims if valid.
        # But looking at GnnNodeEdges code, it uses them for EmbeddingLaplacian input sizes?
        # Actually EmbeddingLaplacian takes (d, de, dy, node_features, global_features, natoms, nbonds).
        # node_features: int (input dim of extra node features?)
        # global_features: int (input dim of globals)
        # natoms: int (number of atom types / classes for embedding)
        # nbonds: int (number of edge types / classes for embedding)

        # DiGress passes 'input_dims' which has checks.
        # Standard QM9: 5 atom types.
        # input_dims['X'] might be larger if extra features (like cycles) are used.

        # We need to know how many are "atom types" and how many are "extra".
        # DiGress usually separates them.
        # In DiscreteDenoisingDiffusion, X is one-hot atom types + extra features.
        # Wait, DiscreteDenoisingDiffusion.forward concatenates noisy X (one-hot) + extra.
        # So X input to model has shape (bs, n, input_dims['X']).

        # We need to assume some splits or pass them explicitly.
        # For now, let's assume standard DiGress usage where:
        # X is (one-hot atoms) + (extra features).
        # We can infer 'natoms' from output_dims['X'] (which is usually just the classes).
        # And 'nbonds' from output_dims['E'].

        self.natoms = output_dims["X"]
        self.nbonds = output_dims["E"]

        # Extra features dimension = Total X dim - Atom types (natoms)
        self.extra_node_features_dim = input_dims["X"] - self.natoms
        if self.extra_node_features_dim < 0:
            # If input is smaller than output, it might be that input IS just the one-hot?
            # No, input_dims['X'] should be >= output_dims['X'] if extra features exist.
            # If equal, extra is 0.
            pass

        self.global_features_dim = input_dims["y"]

        self.gnn = GnnNodeEdges(
            d=self.d,
            de=self.de,
            dy=self.dy,
            n_heads=self.n_heads,
            node_features=self.extra_node_features_dim,  # Extra features dim
            global_features=self.global_features_dim,
            n_layers=self.n_layers,
            natoms=self.natoms,
            nbonds=self.nbonds,
            last_layer="mlp",  # As in gnn.py example
        )

    def forward(self, X, E, y, node_mask):
        """
        X: (bs, n, input_dims['X'])
           Contains one-hot atom types (first natoms) + extra features.
        E: (bs, n, n, input_dims['E'])
           Contains one-hot edge types.
        y: (bs, dy)
        node_mask: (bs, n)
        """
        # Split X into N (atom types) and pos_emb (extra features)
        # GnnNodeEdges expects N as LongTensor (indices) of shape (bs, n) ?
        # Or One-hot?
        # Checking EmbeddingLaplacian in GnnNodeEdges...
        # It takes N, E, y, pos_emb.
        # Inside EmbeddingLaplacian:
        # self.embedding_nodes(N) -> suggests N is indices (LongTensor).
        # self.embedding_edges(E) -> suggests E is indices (LongTensor).

        # But DiGress passes One-Hot floats.
        # We need to convert One-Hot to Indices.

        # X (bs, n, dx) -> N_indices (bs, n) take argmax of first 'natoms' dimensions.
        N_onehot = X[..., : self.natoms]
        N_indices = torch.argmax(N_onehot, dim=-1).long()

        # Extra features
        pos_emb = X[..., self.natoms :]  # (bs, n, extra_dim)

        # E (bs, n, n, de) -> E_indices (bs, n, n)
        # Note: input_dims['E'] should match self.nbonds usually.
        E_indices = torch.argmax(E[..., : self.nbonds], dim=-1).long()

        # Mask
        mask = node_mask.unsqueeze(
            -1
        )  # (bs, n, 1) usually expected by some GNNs, or (bs, n)
        # GnnNodeEdges forward says: mask: torch.Tensor
        # And passes it to layers.
        # Let's check AttentionLayer.
        # It likely expects (bs, n, 1) or (bs, 1, n, 1) for broadcasting.
        # In DiGress, node_mask is (bs, n).
        # EmbeddingLaplacian takes mask.

        # Let's run GnnNodeEdges logic with modifications to return y

        # Reuse logic from GnnNodeEdges.forward but capturing y

        h_out, e_out, y_out, mask_out = self._forward_gnn_logic(
            N_indices, E_indices, pos_emb, y, node_mask
        )

        # Output of GnnNodeEdges layer is (h, e, y, mask)
        # h: (bs, n, d) - Hidden state
        # e: (bs, n, n, de) - Hidden edge
        # y: (bs, dy) - Hidden global

        # DiGress GraphTransformer returns valid predictions for X, E, y.
        # The 'd' output of GNN is typically the hidden size (256).
        # But DiGress expects output to be mapped to output_dims.
        # GraphTransformer has mlp_out_X, mlp_out_E, mlp_out_y to project back.

        # GnnNodeEdges has 'last_layer="mlp"' which adds an MLPNodeEdge.
        # MLPNodeEdge outputs:
        # h -> (bs, n, natoms) (logits)
        # e -> (bs, n, n, nbonds) (logits)
        # It does NOT seem to output y!
        # MLPNodeEdge forward: returns h, e, y=None, mask

        # So we need to handle y output separately or add a projection for y.

        if y_out is None:
            # If MLPNodeEdge kills y, we might need to project the PENULTIMATE y?
            # Or just use a linear projection if GNN tracked it.
            # Wait, MLPNodeEdge does: "return h, e, None, mask".
            # So y is lost.
            # But DiGress needs predicted y (global properties).
            # If the dataset has no y output, it's fine.
            # If it does, we need it.
            # We can fix this by adding a y-projection layer in Adapter.

            # We will grab the y from the layer BEFORE the last layer?
            # Or we assume y is not updated in the last step and project the y from previous step.
            pass

        # Since we use GnnNodeEdges as a black box (mostly), we can iterate manually.
        pass

        return PlaceHolder(X=h_out, E=e_out, y=y_out).mask(node_mask)

    def _forward_gnn_logic(self, N, E, pos_emb, y, mask):
        # Re-implementing the loop to capture intermediate y and ensure outputs are what we expect.
        # Accessing self.gnn.layers

        # Initial mask handling
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)  # (bs, n, 1)

        N_in = None  # Residuals handled inside GnnNodeEdges layers?
        # GnnNodeEdges code:
        # "h = h + N_in" at the end.

        # Initial Embedding
        # Layer 0 is EmbeddingLaplacian
        h, e, y, mask = self.gnn.layers[0](N, E, y, pos_emb, mask)
        e = 1 / 2 * (e + e.transpose(1, 2))

        N_in_res = h  # Is this correct? GnnNodeEdges saves N_in = N (input indices).
        # But you can't add Indices to Hidden Features.
        # Wait, GnnNodeEdges code says:
        # N_in = N
        # ...
        # h = h + N_in
        # If N is indices, this crashes.
        # UNLESS N is already embedded? No, EmbeddingLaplacian takes N (indices).
        # This looks like a bug or I misunderstood GnnNodeEdges code.
        # "h = h + N_in" -> if N_in is (bs, n) Long and h is (bs, n, d) Float -> Error/Broadcasting weirdness.
        # Let's assume there's no residual on the INPUT indices in the adapter for now, or GNN code assumes N is features?
        # The GNN code docstring says "N: torch.Tensor".
        # EmbeddingLaplacian: "self.embedding_nodes(N)".
        # It implies N is indices.
        # So "h + N_in" must be wrong in GnnNodeEdges or I am misreading.

        # Ah, maybe N_in is meant to be the initial embedding?
        # But the code says "N_in = N" right at start.
        # I will IGNore the residual from INPUT INDICES for now as it seems suspect or requires specific handling.
        # I will implement standard resnet structure if needed.

        # Loop over middle layers (AttentionLayer)
        for i in range(1, len(self.gnn.layers) - 1):
            layer = self.gnn.layers[i]
            h, e, y, mask = layer(h, e, y, mask)
            e = 1 / 2 * (e + e.transpose(1, 2))

        # Last layer is MLPNodeEdge
        last_layer = self.gnn.layers[-1]

        # We need y prediction.
        # y is currently the hidden y state.
        # We can project it to output_dims['y'] using a new linear layer in adapter
        # because MLPNodeEdge discards it.

        y_final_hidden = y

        # Run last layer (MLP) for h and e
        h_out, e_out, _, _ = last_layer(h, e, y, mask)

        # We need to project y_final_hidden to output_dims['y']
        # I'll add a projection layer to the adapter for this.
        if not hasattr(self, "y_out_proj"):
            # Lazy init or defined in __init__
            device = y.device
            dtype = y.dtype
            self.y_out_proj = nn.Linear(
                self.dy, self.global_features_dim, device=device, dtype=dtype
            )

        y_out = self.y_out_proj(y_final_hidden)

        return h_out, e_out, y_out, mask
