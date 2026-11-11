"""Impedance-based formulation: the modes of machine plus network as roots of Z(s) = 0.

This is a second, independent way to get the modes of the current-loop model of
`dfig.py`. The state-space model integrates fluxes; this one builds the loop impedance
as a ratio of polynomials in the per-unit stator-frame frequency s and finds its zeros.
It has no states and no Jacobian, so agreement between the two is a real check.

Because the network enters only as an impedance, any network that can be written as
series and parallel combinations of R, L and C is covered: a plant transformer, a
parallel uncompensated path, a different number of units online.

Frequencies: s is per unit in the stator frame. A root s corresponds to the eigenvalue
lambda = wb * (s - j) of the synchronous-frame model, in 1/s and rad/s.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import polynomial as npoly


@dataclass(frozen=True)
class Rational:
    """A ratio of polynomials in s with complex coefficients, lowest power first."""

    num: np.ndarray
    den: np.ndarray

    @staticmethod
    def of(value) -> Rational:
        if isinstance(value, Rational):
            return value
        return Rational(np.array([value], dtype=complex), np.array([1.0], dtype=complex))

    def __add__(self, other) -> Rational:
        o = Rational.of(other)
        return Rational(
            npoly.polyadd(npoly.polymul(self.num, o.den), npoly.polymul(o.num, self.den)),
            npoly.polymul(self.den, o.den),
        )

    __radd__ = __add__

    def __mul__(self, other) -> Rational:
        o = Rational.of(other)
        return Rational(npoly.polymul(self.num, o.num), npoly.polymul(self.den, o.den))

    __rmul__ = __mul__

    def __truediv__(self, other) -> Rational:
        o = Rational.of(other)
        return Rational(npoly.polymul(self.num, o.den), npoly.polymul(self.den, o.num))

    def __call__(self, s: complex) -> complex:
        return complex(npoly.polyval(s, self.num) / npoly.polyval(s, self.den))


S = Rational(np.array([0.0, 1.0], dtype=complex), np.array([1.0], dtype=complex))


def poly(*coefficients) -> Rational:
    """Polynomial in s, lowest power first."""
    return Rational(np.array(coefficients, dtype=complex), np.array([1.0], dtype=complex))


def parallel(a: Rational, b: Rational) -> Rational:
    return (a * b) / (a + b)


def branch(r: float, x: float, x_c: float = 0.0) -> Rational:
    """Series R, L and optional C: r + s x + x_c / s."""
    z = poly(r, x)
    return z + Rational.of(x_c) / S if x_c else z


def dfig_impedance(
    r_s: float,
    x_ls: float,
    x_m: float,
    r_r: float,
    x_lr: float,
    w_r: float = 1.0,
    kp: float = 0.0,
    ki: float = 0.0,
    f0: float = 60.0,
) -> Rational:
    """Stator impedance of the machine with a PI rotor current controller.

    The same expression as `slip.dfig_impedance`, kept as polynomials so that all of its
    zeros and poles are available.
    """
    wb = 2 * np.pi * f0
    r_eff = Rational.of(r_r + kp) + Rational.of(ki / wb) / poly(-1j, 1.0)
    z_rotor = r_eff * S / poly(-1j * w_r, 1.0) + poly(0.0, x_lr)
    z_mag = poly(0.0, x_m)
    return poly(r_s, x_ls) + parallel(z_mag, z_rotor)


def roots(z: Rational, tol: float = 1e-5) -> np.ndarray:
    """Zeros of Z(s).

    Building Z by cross-multiplication leaves common factors in numerator and denominator.
    Each denominator root removes the numerator root that coincides with it.
    """
    zeros = list(npoly.polyroots(np.trim_zeros(z.num, "b")))
    for pole in npoly.polyroots(np.trim_zeros(z.den, "b")):
        if not zeros:
            break
        dist = [abs(zero - pole) for zero in zeros]
        k = int(np.argmin(dist))
        if dist[k] < tol * max(1.0, abs(pole)):
            zeros.pop(k)
    return np.array(zeros, dtype=complex)


def modes(z: Rational, f0: float = 60.0) -> np.ndarray:
    """Eigenvalues (1/s, rad/s) in the synchronous frame: lambda = wb (s - j)."""
    return 2 * np.pi * f0 * (roots(z) - 1j)


def sso_growth(lam: np.ndarray, f_lo: float = 2.0, f_hi: float = 58.0) -> float:
    """Largest growth rate among modes between f_lo and f_hi Hz in the rotating frame."""
    f = np.abs(lam.imag) / (2 * np.pi)
    band = lam.real[(f > f_lo) & (f < f_hi)]
    return float(band.max()) if band.size else -1e3
