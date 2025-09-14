#!/usr/bin/env python3
"""Coarse grid search for a net-positive hybrid reactor design.

The script varies basic cylindrical fusor-assembly parameters and evaluates
net thermal power minus electrical input.  It requires OpenMC with nuclear
data and may take several minutes to run.
"""
import itertools
import numpy as np
from dataclasses import dataclass

from openmc_hybrid.model import build_fusor_cylindrical_model, run_model
from openmc_hybrid.fusion_physics import neutron_yield_rate, power_balance

EV_TO_J = 1.602176634e-19


def evaluate(params):
    """Build and run a model, returning (net_power_W, volume_cm3)."""
    model = build_fusor_cylindrical_model(
        height_cm=params["height_cm"],
        vacuum_radius_cm=params["vacuum_radius_cm"],
        moderator_thickness_cm=params["moderator_thickness_cm"],
        fuel_thickness_cm=params["fuel_thickness_cm"],
        reflector_thickness_cm=params["reflector_thickness_cm"],
        shielding_thickness_cm=params["shielding_thickness_cm"],
        moderator_material=params["moderator_material"],
        fuel_material="uo2",
        fuel_enrichment_wt_pct=10.0,
        fuel_density_gcc=10.5,
        reflector_material=params["reflector_material"],
        shielding_material=params["shielding_material"],
        source_type="dt",
        n_sources_azimuthal=1,
        n_sources_axial=1,
        run_mode="fixed source",
    )
    sp = run_model(model, particles=1000, batches=20)
    total_eV = 0.0
    for rn in ["vacuum", "moderator", "fuel", "reflector", "shield"]:
        try:
            t = sp.get_tally(name=f"heating_{rn}")
            total_eV += float(t.get_pandas_dataframe()["mean"].sum())
        except Exception:
            pass
    thermal_W = total_eV * params["source_rate_nps"] * EV_TO_J
    elec_W, fusion_W, _ = power_balance(
        "dt", params["voltage_kV"], params["current_mA"], params["source_rate_nps"]
    )
    net_W = thermal_W + fusion_W - elec_W
    outer_r = (
        params["vacuum_radius_cm"]
        + params["moderator_thickness_cm"]
        + params["fuel_thickness_cm"]
        + params["reflector_thickness_cm"]
        + params["shielding_thickness_cm"]
    )
    volume_cm3 = np.pi * outer_r ** 2 * params["height_cm"]
    return net_W, volume_cm3


def main():
    base = {
        "vacuum_radius_cm": 5.0,
        "shielding_thickness_cm": 10.0,
        "moderator_material": "heavy_water",
        "reflector_material": "beryllium",
        "shielding_material": "lead",
        "voltage_kV": 80.0,
        "current_mA": 30.0,
        "pressure_mTorr": 1.0,
    }
    base["source_rate_nps"] = neutron_yield_rate(
        "dt", base["voltage_kV"], base["current_mA"], base["pressure_mTorr"]
    )

    search_space = {
        "height_cm": [40.0, 50.0, 60.0],
        "moderator_thickness_cm": [5.0, 10.0],
        "fuel_thickness_cm": [15.0, 20.0, 25.0],
        "reflector_thickness_cm": [10.0, 15.0],
    }

    best = None
    for combo in itertools.product(*search_space.values()):
        params = base.copy()
        for key, val in zip(search_space.keys(), combo):
            params[key] = val
        try:
            net, vol = evaluate(params)
        except Exception as exc:
            print(f"Configuration {params} failed: {exc}")
            continue
        print(
            f"Checked {params}: net={net:.2e} W, volume={vol:.1f} cm^3"
        )
        if net > 0 and (best is None or vol < best[1]):
            best = (params, vol, net)
    if best:
        params, vol, net = best
        print("\nBest configuration:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        print(f"Volume: {vol:.1f} cm^3, Net power: {net:.2e} W")
    else:
        print("No net-positive configuration found")


if __name__ == "__main__":
    main()
