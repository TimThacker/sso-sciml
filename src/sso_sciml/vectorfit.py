"""Least-squares fit of a rational model to a frequency scan (vector fitting), and the
modes of the fitted plant on a known network.

An interpolating fit (`admittance.fit_growth`) passes through its points and so fits the
noise in them. Here the order is fixed and small, the fit is least squares, and all four
entries of the admittance share one set of poles:

    Y_ij(s) = sum_m c_ij,m phi_m(s) + d_ij

where phi_m are partial fractions on the poles. The poles are found by the iteration of
Gustavsen and Semlyen (1999): fit with trial poles and a common denominator, take the
zeros of that denominator as the next trial poles, repeat.

The fitted plant is then closed on the network exactly. The network is known (R, X, C),
so its equations are written out, and the modes are the finite eigenvalues of the pair
of matrices that hold plant and network together.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eig

from .physics.converter import WB

J = np.array([[0.0, -1.0], [1.0, 0.0]])
MODE_LIMIT = 1e7  # 1/s; see `closed_loop_modes`


def _basis(s: np.ndarray, poles: np.ndarray) -> np.ndarray:
    """Partial fractions with real coefficients. `poles` holds real poles and one of
    each complex pair (the one with positive imaginary part)."""
    cols = []
    for a in poles:
        if a.imag == 0:
            cols.append(1 / (s - a.real))
        else:
            cols.append(1 / (s - a) + 1 / (s - np.conj(a)))
            cols.append(1j / (s - a) - 1j / (s - np.conj(a)))
    return np.stack(cols, -1)


def _realisation(poles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A and b with c^T (sI - A)^-1 b = sum_m c_m phi_m(s)."""
    n = sum(1 if a.imag == 0 else 2 for a in poles)
    a_mat, b = np.zeros((n, n)), np.zeros(n)
    k = 0
    for a in poles:
        if a.imag == 0:
            a_mat[k, k], b[k] = a.real, 1.0
            k += 1
        else:
            a_mat[k : k + 2, k : k + 2] = [[a.real, a.imag], [-a.imag, a.real]]
            b[k] = 2.0
            k += 2
    return a_mat, b


def _tidy(lam: np.ndarray) -> np.ndarray:
    """Eigenvalues as a pole list: unstable ones mirrored into the left half-plane, then
    the real ones and one of each complex pair."""
    lam = -np.abs(lam.real) + 1j * lam.imag
    real = np.abs(lam.imag) < 1e-9 * np.maximum(1.0, np.abs(lam))
    return np.concatenate([lam[real].real.astype(complex), lam[~real & (lam.imag > 0)]])


def _stack(m: np.ndarray) -> np.ndarray:
    return np.concatenate([m.real, m.imag])


def _lstsq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    scale = np.linalg.norm(a, axis=0)
    scale[scale == 0] = 1.0
    return np.linalg.lstsq(a / scale, b, rcond=None)[0] / scale


def vector_fit(s: np.ndarray, y: np.ndarray, pairs: int, iterations: int = 8) -> dict:
    """Fit y, shape (points, q), sampled at s = j w with w >= 0, by `pairs` complex pole
    pairs shared by the q columns. Errors are weighted by 1 / |y|, so the residual is
    relative. Returns the realisation (a, b), the coefficients c (q, n) and d (q), and
    the root-mean-square relative residual.
    """
    q = y.shape[1]
    w = 1.0 / np.maximum(np.abs(y), 1e-300)
    beta = np.geomspace(np.abs(s[s != 0]).min(), np.abs(s).max(), pairs)
    poles = -beta / 100 + 1j * beta
    ones = np.ones((s.size, 1))
    for _ in range(iterations):
        phi = _basis(s, poles)
        m = phi.shape[1]
        own = np.concatenate([phi, ones], 1)
        rows, rhs = [], []
        for e in range(q):
            # Each column has its own numerator; only the denominator is shared. QR
            # removes the numerator unknowns and leaves equations for the denominator.
            left = _stack(own * w[:, e : e + 1])
            right = _stack(-(y[:, e] * w[:, e])[:, None] * phi)
            qm, r = np.linalg.qr(np.concatenate([left, right], 1))
            rows.append(r[m + 1 :, m + 1 :])
            rhs.append(qm[:, m + 1 :].T @ _stack(y[:, e] * w[:, e]))
        den = _lstsq(np.concatenate(rows), np.concatenate(rhs))
        a_mat, b = _realisation(poles)
        poles = _tidy(np.linalg.eigvals(a_mat - np.outer(b, den)))
    phi = np.concatenate([_basis(s, poles), ones], 1)
    coef = np.array(
        [_lstsq(_stack(phi * w[:, e : e + 1]), _stack(y[:, e] * w[:, e])) for e in range(q)]
    )
    a_mat, b = _realisation(poles)
    resid = float(np.sqrt(np.mean(np.abs((phi @ coef.T - y) * w) ** 2)))
    return {"poles": poles, "a": a_mat, "b": b, "c": coef[:, :-1], "d": coef[:, -1],
            "residual": resid}  # fmt: skip


def fit_admittance(
    f_hz: np.ndarray,
    y: np.ndarray,
    noise: float,
    orders: tuple[int, ...] = (4, 8, 12),
    support: float = 3.0,
) -> dict:
    """State-space model of a 2 x 2 admittance from its scan at f_hz >= 0.

    The smallest order whose residual is within 1.5 times the noise level is taken; if
    none is, the one with the smallest residual. The model is then cut to what the scan
    supports. Each pole has a 2 x 2 residue, and each singular value of that residue is
    one state. A state is kept only if, at some point of the scan, it supplies a share
    of |Y| above `support` times the noise level. Without this, surplus poles stay in
    the model as modes that no terminal can see, and they turn up among the modes of the
    closed loop.

    Returns a (n, n) diagonal and complex, b (n, 2), c (2, n), d (2, 2), the number of
    pole pairs fitted, the number of states kept and the residual.
    """
    s = 2j * np.pi * f_hz
    best = None
    for pairs in orders:
        fit = vector_fit(s, y.reshape(-1, 4), pairs)
        fit["pairs"] = pairs
        if best is None or fit["residual"] < best["residual"]:
            best = fit
        if fit["residual"] <= 1.5 * max(noise, 1e-9):
            break
    poles, res = [], []
    k = 0
    for pole in best["poles"]:
        if pole.imag == 0:
            poles.append(pole)
            res.append(best["c"][:, k].astype(complex))
            k += 1
        else:
            r = best["c"][:, k] + 1j * best["c"][:, k + 1]
            poles += [pole, np.conj(pole)]
            res += [r, np.conj(r)]
            k += 2
    size = np.linalg.norm(y, ord=2, axis=(-2, -1))
    floor = max(support * noise, 1e-7)
    a_diag, b_rows, c_cols = [], [], []
    for pole, r in zip(poles, res, strict=True):
        u, sv, vh = np.linalg.svd(r.reshape(2, 2))
        # Largest share of |Y| this pole supplies at any point of the scan.
        share = (1.0 / (np.abs(s - (pole.real + 1j * abs(pole.imag))) * size)).max()
        for j in range(2):
            if sv[j] * share > floor:
                a_diag.append(pole)
                b_rows.append(sv[j] * vh[j])
                c_cols.append(u[:, j])
    n = len(a_diag)
    return {
        "a": np.diag(np.array(a_diag, dtype=complex)),
        "b": np.array(b_rows, dtype=complex).reshape(n, 2),
        "c": np.array(c_cols, dtype=complex).reshape(n, 2).T,
        "d": best["d"].reshape(2, 2),
        "pairs": best["pairs"],
        "states": n,
        "residual": best["residual"],
    }


def closed_loop_modes(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, net: dict
) -> np.ndarray:
    """Modes of a plant dx/dt = A x + B v, current drawn = C x + D v, on the network of
    `admittance.network_impedance` (keys r_e, x_e, x_c, r_p, x_p, r_t, x_t).

    Unknowns: plant state, line current, path current, capacitor voltage, plant-bus
    voltage, terminal voltage. Synchronous frame, so an inductor is
    v = (X / wb) di/dt + X J i and a capacitor is dv/dt = wb (X_c i - J v).
    """
    n = a.shape[0]
    cap = net["x_c"] != 0
    ic, ip = slice(n, n + 2), slice(n + 2, n + 4)
    vc = slice(n + 4, n + 6)
    vm = slice(n + 4 + 2 * cap, n + 6 + 2 * cap)
    vt = slice(n + 6 + 2 * cap, n + 8 + 2 * cap)
    size = n + 8 + 2 * cap
    e, f = np.zeros((size, size), dtype=complex), np.zeros((size, size), dtype=complex)
    eye = np.eye(2)
    # Plant.
    e[:n, :n] = np.eye(n)
    f[:n, :n], f[:n, vt] = a, b
    # Compensated line, from the plant bus to the infinite bus.
    e[ic, ic] = net["x_e"] / WB * eye
    f[ic, ic], f[ic, vm] = -(net["r_e"] * eye + net["x_e"] * J), eye
    # Parallel path.
    e[ip, ip] = net["x_p"] / WB * eye
    f[ip, ip], f[ip, vm] = -(net["r_p"] * eye + net["x_p"] * J), eye
    if cap:
        f[ic, vc] = -eye
        e[vc, vc] = eye
        f[vc, ic], f[vc, vc] = WB * net["x_c"] * eye, -WB * J
    # The current out of the plant is the sum of the two branch currents.
    f[vm, ic], f[vm, ip], f[vm, :n], f[vm, vt] = eye, eye, c, d
    # Plant-side impedance between terminal and plant bus.
    e[vt, ic] = e[vt, ip] = net["x_t"] / WB * eye
    f[vt, vt], f[vt, vm] = eye, -eye
    f[vt, ic] = f[vt, ip] = -(net["r_t"] * eye + net["x_t"] * J)
    # The set has algebraic equations, so some of its generalised eigenvalues are at
    # infinity. In floating point those come back as alpha / beta with beta near zero:
    # sometimes inf, sometimes a finite number around 1e18 with either sign. A mode of
    # this model cannot be faster than MODE_LIMIT, and anything beyond it is dropped.
    alpha, beta = eig(f, e, right=False, homogeneous_eigvals=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = alpha / beta
    return lam[np.isfinite(lam) & (np.abs(lam) < MODE_LIMIT)]
