"""System B2 on a network that is not a single series line.

The machine and controls are those of `dfig_full.py`. The network becomes

    terminals -- plant impedance (r_t, x_t) -- plant bus --+-- line (r_e, x_e, x_c) ----+-- bus
                                                           +-- parallel path (r_p, x_p) +

with the series capacitor in the line and both branches ending at the infinite bus.

Two states are added: the flux of the loop formed by the two parallel branches,
mu = x_p * i_p - x_e * i_c, where i_c is the current in the compensated line and i_p the
current in the parallel path. The currents now come from a 4 x 4 linear solve instead of
a closed form. See docs/EQUATIONS.md section 17.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .dfig_full import STATE_NAMES as _BASE_NAMES
from .dfig_full import Controls, DFIGFull
from .induction import IGData, _t

N_STATE = 25
STATE_NAMES = (*_BASE_NAMES, "mu_D", "mu_Q")
_NET = ("r_t", "x_t", "r_p", "x_p")


@dataclass
class DFIGMeshed(DFIGFull):
    data: IGData = field(default_factory=IGData)
    controls: Controls = field(default_factory=Controls)
    r_t: float = 0.010  # plant transformers and collector, pu on the plant base
    x_t: float = 0.15
    par_ratio: float = 1.0  # parallel path impedance as a multiple of the line's

    # ------------------------------------------------------------------ parameters
    def params(self, x_c, **overrides) -> dict:
        net = {
            "r_t": self.r_t,
            "x_t": self.x_t,
            "r_p": self.par_ratio * self.data.r_e,
            "x_p": self.par_ratio * self.data.x_e,
        }
        for key in _NET:
            if overrides.get(key) is not None:
                net[key] = overrides[key]
        base = super().params(x_c, **{k: v for k, v in overrides.items() if k not in _NET})
        base.update({k: _t(v) for k, v in net.items()})
        return base

    # ------------------------------------------------------------------ algebra
    def _matrix(self, p: dict) -> torch.Tensor:
        """Fluxes [lam1, lam2, psi_r, mu] = M @ currents [i, i_g, i_r, i_p]."""
        x_s = p["x_ls"] + p["x_m"]
        x_r = p["x_lr"] + p["x_m"]
        x_tl = p["x_t"] + p["x_e"]
        zero = torch.zeros_like(x_s + x_r + x_tl + p["x_f"] + p["x_p"])
        rows = [
            [-(x_s + x_tl), -x_tl, p["x_m"], p["x_e"]],
            [x_tl, p["x_f"] + x_tl, zero, -p["x_e"]],
            [-p["x_m"], zero, x_r, zero],
            [-p["x_e"], -p["x_e"], zero, p["x_p"] + p["x_e"]],
        ]
        return torch.stack([torch.stack([v + zero for v in row], dim=-1) for row in rows], dim=-2)

    def _solve(self, p: dict, lam1, lam2, psi_r, mu) -> torch.Tensor:
        """Currents from fluxes. Each argument is a (D, Q) pair; returns (..., 4, 2)."""
        rhs = torch.stack([torch.stack(pair, dim=-1) for pair in (lam1, lam2, psi_r, mu)], dim=-2)
        return torch.linalg.solve(self._matrix(p), rhs)

    def currents(self, x: torch.Tensor, p: dict) -> dict:
        """Stator, grid-side, rotor and parallel-path currents, plus i_l and ic (line totals)."""
        sol = self._solve(
            p,
            (x[..., 0], x[..., 1]),
            (x[..., 6], x[..., 7]),
            (x[..., 2], x[..., 3]),
            (x[..., 23], x[..., 24]),
        )
        out = {}
        for n, ax in enumerate("DQ"):
            i_s, i_g, i_r, i_p = (sol[..., m, n] for m in range(4))
            out[f"i_{ax}"], out[f"ig_{ax}"], out[f"ir_{ax}"], out[f"ip_{ax}"] = i_s, i_g, i_r, i_p
            out[f"il_{ax}"] = i_s + i_g
            out[f"ic_{ax}"] = i_s + i_g - i_p
        return out

    def _network(self, x: torch.Tensor, p: dict, c: dict, vg_d, vg_q) -> dict:
        wb = self.data.wb
        i_d, i_q, ig_d, ig_q = c["i_D"], c["i_Q"], c["ig_D"], c["ig_Q"]
        il_d, il_q, ic_d, ic_q, ip_d, ip_q = (
            c["il_D"], c["il_Q"], c["ic_D"], c["ic_Q"], c["ip_D"], c["ip_Q"]
        )  # fmt: skip
        l1_d, l1_q, vc_d, vc_q = x[..., 0], x[..., 1], x[..., 4], x[..., 5]
        l2_d, l2_q, mu_d, mu_q = x[..., 6], x[..., 7], x[..., 23], x[..., 24]
        drop_d = p["r_t"] * il_d + p["r_e"] * ic_d  # resistive drop from terminals to the bus
        drop_q = p["r_t"] * il_q + p["r_e"] * ic_q
        dmu = (
            wb * (vc_d + p["r_e"] * ic_d - p["r_p"] * ip_d + mu_q),
            wb * (vc_q + p["r_e"] * ic_q - p["r_p"] * ip_q - mu_d),
        )
        return {
            "dl1": (
                wb * (p["v_b"] + vc_d + p["r_s"] * i_d + drop_d + l1_q),
                wb * (vc_q + p["r_s"] * i_q + drop_q - l1_d),
            ),
            "dvc": (wb * (p["x_c"] * ic_d + vc_q), wb * (p["x_c"] * ic_q - vc_d)),
            "dl2": (
                wb * (vg_d - p["v_b"] - vc_d - p["r_f"] * ig_d - drop_d + l2_q),
                wb * (vg_q - vc_q - p["r_f"] * ig_q - drop_q - l2_d),
            ),
            "psi_s": (
                l1_d + p["x_t"] * il_d + p["x_e"] * ic_d,
                l1_q + p["x_t"] * il_q + p["x_e"] * ic_q,
            ),
            "dmu": dmu,
            "extra": list(dmu),
        }

    def _gsc_current_rate(self, p: dict, net: dict, dpr_d, dpr_q) -> tuple:
        sol = self._solve(p, net["dl1"], net["dl2"], (dpr_d, dpr_q), net["dmu"])
        return sol[..., 1, 0], sol[..., 1, 1]

    # ------------------------------------------------------------------ steady state
    def _branches(self, q: dict, x_c: float) -> tuple[complex, complex]:
        return q["r_e"] + 1j * (q["x_e"] - x_c), q["r_p"] + 1j * q["x_p"]

    def _network_impedance(self, q: dict, x_c: float) -> complex:
        z_c, z_p = self._branches(q, x_c)
        return q["r_t"] + 1j * q["x_t"] + z_c * z_p / (z_c + z_p)

    def _network_steady(self, q: dict, x_c: float, i_l: complex, psi_s: complex, i_g: complex):
        z_c, z_p = self._branches(q, x_c)
        i_c = i_l * z_p / (z_c + z_p)
        i_p = i_l - i_c
        lam1 = psi_s - q["x_t"] * i_l - q["x_e"] * i_c
        lam2 = q["x_f"] * i_g + q["x_t"] * i_l + q["x_e"] * i_c
        mu = q["x_p"] * i_p - q["x_e"] * i_c
        return lam1, lam2, -1j * x_c * i_c, [mu.real, mu.imag]
