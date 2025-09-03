import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# Optional but useful utilities -------------------------------------------------
class Timer:
    """Context-manager for wall-clock timing."""

    def __enter__(self):
        import time
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        import time
        self._t1 = time.perf_counter()
        self.seconds = self._t1 - self._t0

# ---------------- HARD-CONCRETE GATE ------------------------------------------
class HardConcrete(nn.Module):
    """Stochastic gate in (0,1) with differentiable L0 penalty."""

    def __init__(self, shape: Tuple[int, ...], bias: float = -3.0, tau: float = 0.1):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.full(shape, bias))
        self.tau = tau
        self.register_buffer("eps", torch.tensor(1e-6))

    def forward(self):
        if self.training:
            u = torch.rand_like(self.log_alpha).clamp_(self.eps, 1 - self.eps)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha) / self.tau)
        else:
            s = torch.sigmoid(self.log_alpha)
        # Stretch & hard-sigmoid
        s_bar = s * (1.1 - 0.1) + 0.1  # (0.1, 1.1)
        return torch.clamp(s_bar, 0.0, 1.0)

    def l0_penalty(self):
        # Expected L0 (Louizos et al.)
        return torch.sigmoid(self.log_alpha - self.tau * math.log(-0.1 / 1.1)).mean()


# ---------------- AP-LIB LAYER -------------------------------------------------
from torch_geometric import utils as tg_utils
import torch_geometric as tg
import scipy.sparse as sp


def _normalized_adj(edge_index, num_nodes, device):
    """Return symmetric normalized adjacency (with self-loops) as edge_index."""
    ei, _ = tg_utils.add_self_loops(edge_index, num_nodes=num_nodes)
    adj = tg_utils.to_scipy_sparse_matrix(ei, num_nodes=num_nodes).astype('float32')
    deg = adj.sum(1).A1
    deg_inv_sqrt = 1.0 / (deg + 1e-12) ** 0.5
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
    ei_norm, _ = tg_utils.from_scipy_sparse_matrix(adj_norm)
    return ei_norm.to(device)


class AdaptiveProp(nn.Module):
    """Single AP-LIB message-passing layer supporting up to K hop powers."""

    def __init__(self, dim: int, k_max: int = 8, tau: float = 0.5):
        super().__init__()
        self.dim = dim
        self.k_max = k_max
        self.W0 = nn.Linear(dim, dim, bias=False)
        self.W = nn.Linear(dim, dim, bias=False)
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3 * dim + 1, dim // 2), nn.ReLU(),
                nn.Linear(dim // 2, 1)
            ) for _ in range(k_max)
        ])
        self.hardconcrete = HardConcrete((k_max,), tau=tau)

    def forward(self, x: torch.Tensor, edge_index):
        out = self.W0(x)
        num_nodes = x.size(0)
        device = x.device
        norm_ei = _normalized_adj(edge_index, num_nodes, device)

        # --- precompute k-hop features (simple power iteration) -----------------
        k_feats: List[torch.Tensor] = []
        h = x
        for _ in range(self.k_max):
            h = tg_utils.scatter_mean(h[norm_ei[0]], norm_ei[1], dim=0, dim_size=num_nodes)
            k_feats.append(h)

        # Node degrees for gate net
        degree = tg_utils.degree(edge_index[0], num_nodes=num_nodes).view(-1, 1)

        hc_sample = self.hardconcrete()
        gates_values: List[torch.Tensor] = []
        for k, (gate_net, k_feat) in enumerate(zip(self.gates, k_feats)):
            z = torch.cat([x, k_feat, x - k_feat, degree], dim=1)
            g_logits = gate_net(z).squeeze(-1)
            g = torch.sigmoid(g_logits) * hc_sample[k]  # node-wise × global
            gates_values.append(g.view(-1, 1))
            out = out + g.view(-1, 1) * self.W(k_feat)
        return out, gates_values


# ---------------- FULL MODEL ---------------------------------------------------
class AP_LIB_GCN(nn.Module):
    def __init__(self, in_dim: int, hid: int, out_dim: int, depth: int, k_max: int, tau: float):
        super().__init__()
        self.in_lin = nn.Linear(in_dim, hid)
        self.layers = nn.ModuleList([
            AdaptiveProp(hid, k_max=k_max, tau=tau) for _ in range(depth)
        ])
        self.out_lin = nn.Linear(hid, out_dim)

    def forward(self, data):
        x = F.relu(self.in_lin(data.x))
        all_gates = []
        for layer in self.layers:
            x, gates = layer(x, data.edge_index)
            x = F.relu(x)
            all_gates.extend(gates)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.out_lin(x), all_gates


# ---------------- TRAIN STEP ---------------------------------------------------
from sklearn.metrics import accuracy_score

def train_step(model, data, optimizer, criterion, lambdas, device):
    model.train()
    optimizer.zero_grad()
    out, gates = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])

    # Regularisers (place-holders for future work)
    l_mix = torch.tensor(0.0, device=device)
    l_denoise = torch.tensor(0.0, device=device)
    l0_pen = torch.stack([g.mean() for g in gates]).mean()

    loss = loss + lambdas[0] * l_mix + lambdas[1] * l_denoise + lambdas[2] * l0_pen
    loss.backward()
    optimizer.step()
    return loss.item()
