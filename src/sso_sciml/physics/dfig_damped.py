"""System B2 on the meshed network with a sub-synchronous damping controller.

The rotor current loop of `dfig_full.py` has proportional gain kp. Seen from the stator
at a sub-synchronous frequency the loop looks like a resistance kp in the rotor circuit,
divided by a negative slip: the larger kp, the more negative the machine's resistance at
the network resonance. The damping controller takes part of that gain away in a band
around the frequency of the oscillation as the controller sees it, and leaves it alone
at low frequency, where the loop tracks its reference:

    command = kp * (reference - i_r) + ki * integral + gain * kp * B(s) i_r

    B(s) = (s w0 / Q) / (s^2 + s w0 / Q + w0^2)      band-pass, 1 at w0, 0 at s = 0

so the proportional gain at w0 is kp * (1 - gain). The frequency is in the controller's
frame, where a network resonance at f_er appears at 60 - f_er.

Four states are added after the 25 of `dfig_meshed.py`:

    25, 26   z1_d, z2_d    filter states of the d axis; z2 is the filter output
    27, 28   z1_q, z2_q    the same for the q axis

The filter output is zero in steady state, so the operating point is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .dfig_full import THETA, Controls, _unrot
from .dfig_meshed import STATE_NAMES as _BASE_NAMES
from .dfig_meshed import DFIGMeshed
from .induction import IGData, _t

N_STATE = 29
STATE_NAMES = (*_BASE_NAMES, "z1_d", "z2_d", "z1_q", "z2_q")
_DAMP = ("damp_gain", "damp_w", "damp_q")


@dataclass
class DFIGDamped(DFIGMeshed):
    data: IGData = field(default_factory=IGData)
    controls: Controls = field(default_factory=Controls)
    damp_gain: float = 0.0  # share of kp taken away at the centre frequency
    damp_hz: float = 30.0  # centre frequency in the controller's frame
    damp_quality: float = 1.0

    def params(self, x_c, **overrides) -> dict:
        own = {"damp_gain": self.damp_gain, "damp_w": 2 * np.pi * self.damp_hz,
               "damp_q": self.damp_quality}  # fmt: skip
        for key in _DAMP:
            if overrides.get(key) is not None:
                own[key] = overrides[key]
        base = super().params(x_c, **{k: v for k, v in overrides.items() if k not in _DAMP})
        base.update({k: _t(v) for k, v in own.items()})
        return base

    def _supplementary(self, x: torch.Tensor, p: dict, irc_d, irc_q) -> tuple:
        w, band = p["damp_w"], p["damp_w"] / p["damp_q"]
        z1_d, z2_d, z1_q, z2_q = (x[..., j] for j in range(25, 29))
        gain = p["damp_gain"] * p["kp"]
        rates = [z2_d, -w * w * z1_d - band * z2_d + band * irc_d,
                 z2_q, -w * w * z1_q - band * z2_q + band * irc_q]  # fmt: skip
        return gain * z2_d, gain * z2_q, rates

    def equilibrium(self, x_c: float, **kwargs) -> tuple[torch.Tensor, dict]:
        x0, p = super().equilibrium(x_c, **kwargs)
        c = self.currents(x0, p)
        irc_d, irc_q = _unrot(c["ir_D"], c["ir_Q"], torch.cos(x0[THETA]), torch.sin(x0[THETA]))
        rest = 1.0 / (p["damp_q"] * p["damp_w"])  # z1 at rest is the input times this
        z = torch.stack([irc_d * rest, torch.zeros_like(irc_d),
                         irc_q * rest, torch.zeros_like(irc_q)])  # fmt: skip
        return torch.cat([x0, z.to(x0.dtype)]), p
