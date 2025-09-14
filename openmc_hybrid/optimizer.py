"""Simple grid-search optimizer for minimal net-positive reactor design.

This module provides a helper function to scan over geometry parameters
for the toy fixed-source model and identify the smallest configuration
that produces net-positive thermal power when driven by an external
neutron source.  It relies on the existing model builders and fusor
physics utilities in the package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional OpenMC import
    import openmc
except Exception:  # pragma: no cover
    openmc = None

from .model import build_fixed_source_model, run_model
from .fusion_physics import neutron_yield_rate, power_balance

EV_TO_J = 1.602176634e-19


@dataclass
class OptimizationResult:
    """Container for optimization results."""

    geometry_kind: str
    fuel_radius_cm: float
    moderator_thickness_cm: float
    add_reflector: bool
    reflector_thickness_cm: float
    height_cm: float
    net_power_W: float
    volume_cm3: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "geometry_kind": self.geometry_kind,
            "fuel_radius_cm": self.fuel_radius_cm,
            "moderator_thickness_cm": self.moderator_thickness_cm,
            "add_reflector": self.add_reflector,
            "reflector_thickness_cm": self.reflector_thickness_cm,
            "height_cm": self.height_cm,
            "net_power_W": self.net_power_W,
            "volume_cm3": self.volume_cm3,
        }


def _estimate_fission_power(sp: "openmc.StatePoint", source_rate_n_per_s: float) -> float:
    """Estimate thermal power from fuel fission using kappa-fission tally."""

    try:
        tally = sp.get_tally(name="kappa_fission_fuel")
    except Exception:
        return 0.0

    df = tally.get_pandas_dataframe()
    if df.empty:
        return 0.0

    kappa_eV = float(df["mean"].sum())
    return kappa_eV * source_rate_n_per_s * EV_TO_J


def _configuration_volume_cm3(
    geom: str, fuel_r: float, mod_t: float, height: float, add_ref: bool, ref_t: float
) -> float:
    """Calculate total outer volume in cm^3 for the simple geometry."""

    radius = fuel_r + mod_t + (ref_t if add_ref else 0.0)
    if geom == "sphere":
        return 4.0 / 3.0 * math.pi * radius ** 3
    else:  # cylinder
        return math.pi * radius ** 2 * height


def optimize_min_reactor(
    source_type: str,
    voltage_kV: float,
    current_mA: float,
    pressure_mTorr: float,
    source_rate_n_per_s: float,
    geometry_kind: str = "sphere",
    fuel_enrichment_wt_pct: float = 5.0,
    fuel_density_gcc: float = 10.5,
    moderator: Optional[str] = "water",
    particles: int = 5000,
    batches: int = 20,
    fuel_radius_range: Iterable[float] = (3.0, 5.0, 7.0),
    moderator_thickness_range: Iterable[float] = (0.0, 2.0, 5.0),
    reflector_thickness_range: Iterable[float] = (0.0, 2.0, 5.0),
    height_range: Iterable[float] = (5.0, 10.0, 15.0),
) -> Optional[OptimizationResult]:
    """Search for the smallest net-positive reactor configuration.

    Returns the best :class:`OptimizationResult` or ``None`` if no
    configuration in the search space achieves net-positive power.
    """

    if openmc is None:
        raise RuntimeError("OpenMC is not installed in this environment.")

    best: Optional[OptimizationResult] = None

    for fuel_r in fuel_radius_range:
        for mod_t in moderator_thickness_range:
            for ref_t in reflector_thickness_range:
                for height in height_range:
                    add_ref = ref_t > 0.0
                    model = build_fixed_source_model(
                        geometry_kind=geometry_kind,
                        fuel_enrichment_wt_pct=fuel_enrichment_wt_pct,
                        fuel_density_gcc=fuel_density_gcc,
                        moderator=moderator,
                        fuel_radius_cm=fuel_r,
                        moderator_thickness_cm=mod_t,
                        add_reflector=add_ref,
                        reflector_thickness_cm=ref_t,
                        height_cm=height,
                        source_type=source_type,
                    )

                    sp = run_model(model, particles=particles, batches=batches)
                    fission_power = _estimate_fission_power(sp, source_rate_n_per_s)

                    elec_W, fus_W, _ = power_balance(
                        source_type, voltage_kV, current_mA, source_rate_n_per_s
                    )
                    net_W = fission_power + fus_W - elec_W
                    volume_cm3 = _configuration_volume_cm3(
                        geometry_kind, fuel_r, mod_t, height, add_ref, ref_t
                    )

                    if net_W > 0:
                        if best is None or volume_cm3 < best.volume_cm3:
                            best = OptimizationResult(
                                geometry_kind=geometry_kind,
                                fuel_radius_cm=fuel_r,
                                moderator_thickness_cm=mod_t,
                                add_reflector=add_ref,
                                reflector_thickness_cm=ref_t,
                                height_cm=height,
                                net_power_W=net_W,
                                volume_cm3=volume_cm3,
                            )
    return best
