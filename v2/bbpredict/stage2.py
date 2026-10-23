"""Stage 2: a conditional-logit (Plackett-Luce) layer over base rating + features, Benter style.

u_i = x_i . beta;  P(i wins) = softmax(u) within the field.
Fitted on walk-forward rows only, so no row ever saw its own result.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .plfit import encode, neg_loglik, stack


class Stage2:
    def __init__(self, k_terms: int | None = 1, l2: float = 1.0):
        self.k_terms, self.l2 = k_terms, l2   # k_terms=1: fit the winner only (conditional logit)
        self.beta = None
        self.mu = self.sd = None

    def fit(self, items):
        """items: [(X, places)], places nan for athletes without a placing (left out of the fit)."""
        rows, parts, off = [], [], 0
        for X, pl in items:
            mask = ~np.isnan(pl)
            m = int(mask.sum())
            if m < 2:
                continue
            rows.append(X[mask])
            parts.append((encode(np.arange(off, off + m), pl[mask], False, self.k_terms), 1.0))
            off += m
        X = np.vstack(rows)
        s = stack(parts)
        self.mu = X.mean(0)
        self.sd = X.std(0)
        self.sd[self.sd == 0] = 1.0
        Z = (X - self.mu) / self.sd
        zero = np.zeros(len(Z))

        def f(beta):
            val, g = neg_loglik(Z @ beta, s, 0.0, zero)
            return val + 0.5 * self.l2 * beta @ beta, Z.T @ g + self.l2 * beta

        self.beta = minimize(f, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B", options=dict(maxiter=2000)).x
        return self

    def utility(self, X):
        return ((X - self.mu) / self.sd) @ self.beta

    def coef_table(self, names):
        """(name, coefficient per raw unit, standardised coefficient), largest effect first."""
        return sorted(zip(names, self.beta / self.sd, self.beta), key=lambda t: -abs(t[2]))
