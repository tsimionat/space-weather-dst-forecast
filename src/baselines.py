"""
Reference forecasts that every machine-learning model must beat.

1. Persistence          : "Dst in h hours = Dst now". Surprisingly hard to beat
                          at short horizons because Dst changes slowly.
2. Physics (O'Brien & McPherron 2000): Burton-type ring-current model,
       dDst*/dt = Q(VBs) - Dst*/tau(VBs)
                          integrated forward with the solar wind "frozen"
                          at its current value (we do not know the future wind).
3. Ridge regression     : a linear model on the whole flattened input window.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

# ---------------------------------------------------------------------------
# 1. Persistence
# ---------------------------------------------------------------------------
def persistence(dst_now: np.ndarray, horizons: list[int]) -> np.ndarray:
    return np.repeat(dst_now[:, None], len(horizons), axis=1).astype(np.float64)


# ---------------------------------------------------------------------------
# 2. O'Brien & McPherron (2000) empirical physics model
# ---------------------------------------------------------------------------
OM_B = 7.26    # nT / sqrt(nPa)   magnetopause-current (pressure) coefficient
OM_C = 11.0    # nT               quiet-time offset
OM_EC = 0.49   # mV/m             threshold electric field for injection


def om_injection(vbs: np.ndarray) -> np.ndarray:
    """Ring-current injection rate Q [nT/h]; zero below the threshold field."""
    return np.where(vbs > OM_EC, -4.4 * (vbs - OM_EC), 0.0)


def om_decay_time(vbs: np.ndarray) -> np.ndarray:
    """Ring-current decay time tau [h]; shorter when driving is strong."""
    return 2.4 * np.exp(9.74 / (4.69 + vbs))


def obrien_mcpherron(dst_now: np.ndarray, p_now: np.ndarray, vbs_now: np.ndarray,
                     horizons: list[int]) -> np.ndarray:
    """
    Forecast Dst(t+h) with the O'Brien-McPherron model.

    With V*Bs and P held constant over the forecast interval, the model
    dDst*/dt = Q - Dst*/tau is a linear ODE with constant coefficients and the
    exact solution is
        Dst*(t+h) = Q*tau + (Dst*(t) - Q*tau) * exp(-h/tau).
    """
    vbs = np.clip(vbs_now.astype(np.float64), 0.0, None)
    p = np.clip(p_now.astype(np.float64), 0.0, None)

    pressure_term = OM_B * np.sqrt(p) - OM_C          # Dst = Dst* + b*sqrt(P) - c
    dst_star_now = dst_now - pressure_term

    q = om_injection(vbs)
    tau = om_decay_time(vbs)
    steady_state = q * tau

    out = np.empty((len(dst_now), len(horizons)))
    for j, h in enumerate(horizons):
        dst_star = steady_state + (dst_star_now - steady_state) * np.exp(-h / tau)
        out[:, j] = dst_star + pressure_term
    return out


# ---------------------------------------------------------------------------
# 3. Ridge regression on the flattened (already standardised) window
# ---------------------------------------------------------------------------
def fit_ridge(X_scaled: np.ndarray, y_scaled: np.ndarray, alpha: float) -> Ridge:
    model = Ridge(alpha=alpha)
    model.fit(X_scaled.reshape(len(X_scaled), -1), y_scaled)
    return model


def predict_ridge(model: Ridge, X_scaled: np.ndarray) -> np.ndarray:
    return model.predict(X_scaled.reshape(len(X_scaled), -1))
