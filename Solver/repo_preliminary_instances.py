from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import torch
from torch import Tensor

# 1. Field-only (no coupling). Trivial: s_i = sign(h_i), E* = -sum |h_i|.
def field_only(N: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(N, generator=g)
    J = torch.zeros(N, N)
    return J, h, -h.abs().sum().item()

# 2. Ferromagnet (J_ij = +1). All spins aligned. E* = -N(N-1)/2.
def ferromagnet(N: int = 10):
    J = torch.ones(N, N) - torch.eye(N)
    h = torch.zeros(N)
    return J, h, -N * (N - 1) / 2.0

# 3. Bipartite antiferromagnet on a cycle C_{2N} (even). Alternating ±1.
#    Each of the 2N edges contributes -1 to E (since J=-1, s_i s_{i+1}=-1).
#    E* = -2N.
def antiferro_even_cycle(N: int = 6):
    n = 2 * N
    J = torch.zeros(n, n)
    for i in range(n):
        J[i, (i + 1) % n] = -1.0
        J[(i + 1) % n, i] = -1.0
    h = torch.zeros(n)
    return J, h, -float(n)

# 4. Frustrated triangle: K_3 with J_ij = -1 (antiferro). Cannot satisfy all.
#    Best is 2 satisfied + 1 frustrated. E* = -1.
def frustrated_triangle():
    J = -(torch.ones(3, 3) - torch.eye(3))
    h = torch.zeros(3)
    return J, h, -1.0

# 5. Max-Cut on K_4 with unit weights. Best cut has 4 of 6 edges. E* = -2.
#    (Convention: Max-Cut -> J_ij = -w_ij.)
def maxcut_K4():
    J = -(torch.ones(4, 4) - torch.eye(4))
    h = torch.zeros(4)
    return J, h, -2.0   # cut value = 4 ; H = (#same) - (#cut) = 2 - 4

# 6. Max-Cut on the 5-cycle C_5 (odd, frustrated). Max cut = 4. E* = -3.
#    H = (#same edges) - (#cut edges) = 1 - 4.
def maxcut_C5():
    n = 5
    J = torch.zeros(n, n)
    for i in range(n):
        J[i, (i + 1) % n] = -1.0
        J[(i + 1) % n, i] = -1.0
    h = torch.zeros(n)
    return J, h, -3.0

# 7. Number partitioning of {3,1,1,2,2,1}. Perfect split exists, diff = 0.
#    With J_ij = -a_i a_j (zero diag) and h = 0:
#       H(s) = (1/2)(sum a_i s_i)^2  -  (1/2) sum a_i^2.
#    Optimum: H* = -(1/2) sum a_i^2 = -10.
def partition_small():
    a = torch.tensor([3., 1., 1., 2., 2., 1.])
    J = -torch.outer(a, a); J.fill_diagonal_(0.)
    h = torch.zeros_like(a)
    return J, h, -0.5 * (a * a).sum().item()


# 8. Sherrington-Kirkpatrick spin glass (small). No closed-form optimum;
def sk_glass(N: int = 12, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    J = torch.randn(N, N, generator=g) / (N ** 0.5)
    J = 0.5 * (J + J.T); J.fill_diagonal_(0.)
    h = torch.zeros(N)
    return J, h, None


# --------------------------------------------------------------------------- #

def brute_force_optimum(J: torch.Tensor, h: torch.Tensor) -> tuple[float, torch.Tensor]:
    """Exhaustive search; returns (min_energy, best_configuration)."""
    N = J.shape[0]
    assert N <= 22, "too large for brute force"
    
    s = torch.tensor([[1 - 2 * ((k >> i) & 1) for i in range(N)]
                      for k in range(2 ** N)], dtype=J.dtype, device=J.device)
  
    energies = -0.5 * (s * (s @ J)).sum(-1) - s @ h
    min_energy, min_idx = torch.min(energies, dim=0)
    
    return float(min_energy), s[min_idx]