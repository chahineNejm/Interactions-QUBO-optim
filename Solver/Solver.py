from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import torch
from torch import Tensor

PumpSchedule = Callable[[int, int], float]  # (step, n_steps) -> p(t)


def linear_pump(p_max: float) -> PumpSchedule:
    """p(t) ramps linearly from 0 to p_max over the run."""
    return lambda t, T: p_max * t / max(T - 1, 1)

@dataclass
class SBConfig:
    """ current issues : 1. I need the pumping to go over the delta at the end (or to issue a warning at the end)
      2.
    """
    # Physical hyperparameters (eq. 5)
    K: float = 1.0          # Kerr nonlinearity

    delta: float = 1.0      # detuning Delta
    xi0: float = 0.5        # coupling scale xi_0
    p_max: float = 1.0      # final pump value (past bifurcation threshold)

    # Integration
    dt: float = 0.25
    n_steps: int = 1000

    # Ensemble
    n_parallel: int = 128
    init_std: float = 1e-2  # x_i(0) ~ N(0, init_std^2), y_i(0) = 0

    # Backend
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = field(default=torch.float32)
    seed: Optional[int] = None


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
class SBSolver:
    """Batched simulated bifurcation solver for Ising ground states."""

    def __init__(
        self,
        J: Tensor,
        h: Optional[Tensor] = None,
        config: SBConfig = SBConfig(),
        schedule: Optional[PumpSchedule] = None,
    ) -> None:
        if J.ndim != 2 or J.shape[0] != J.shape[1]:
            raise ValueError(f"J must be square, got shape {tuple(J.shape)}")
        if not torch.allclose(J, J.T, atol=1e-6):
            raise ValueError("J must be symmetric.")

        self.cfg = config
        self.N = J.shape[0]
        self.J = J.to(device=config.device, dtype=config.dtype)
        # zero out diagonal defensively (self-couplings are an additive const)
        self.J.fill_diagonal_(0.0) 
        self.h = (
            torch.zeros(self.N) if h is None else h
        ).to(device=config.device, dtype=config.dtype)
        self.schedule = schedule or linear_pump(config.p_max)

    # --------------------------------------------------------------------- #
    def ising_energy(self, s: Tensor) -> Tensor:
        """H(s) for s of shape (..., N) with entries in {-1, +1}."""
        return -0.5 * (s * (s @ self.J)).sum(-1) - s @ self.h

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def run(self) -> "SBResult":
        cfg = self.cfg
        gen = (
            torch.Generator(device=cfg.device).manual_seed(cfg.seed)
            if cfg.seed is not None else None
        )
        x = cfg.init_std * torch.randn(
            cfg.n_parallel, self.N,
            device=cfg.device, dtype=cfg.dtype, generator=gen,
        )
        y = torch.zeros_like(x)

        for t in range(cfg.n_steps):
            p = self.schedule(t, cfg.n_steps)
            # symplectic Euler: kick y with the force at current x, then drift x
            force = (
                -(cfg.K * x.pow(2) - p + cfg.delta) * x
                + cfg.xi0 * (x @ self.J)
                + cfg.xi0 * self.h
            )
            y.add_(force, alpha=cfg.dt)
            x.add_(y, alpha=cfg.dt * cfg.delta)

        s = torch.sign(x)
        s[s == 0] = 1.0
        energies = self.ising_energy(s)
        best = int(torch.argmin(energies))
        return SBResult(
            spins=s[best].to("cpu"),
            energy=float(energies[best]),
            all_spins=s.to("cpu"),
            all_energies=energies.to("cpu"),
        )

@dataclass
class SBResult:
    spins: Tensor       # (N,) in {-1, +1}
    energy: float
    all_spins: Tensor   # (B, N)
    all_energies: Tensor  # (B,)

def solve(
    J: Tensor,
    h: Optional[Tensor] = None,
    config: SBConfig = SBConfig(),
    schedule: Optional[PumpSchedule] = None,
) -> SBResult:
    """One-shot entry point."""
    return SBSolver(J, h, config, schedule).run()
