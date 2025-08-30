"""Fusion physics calculations for fusor modeling."""

import numpy as np
from typing import Tuple


def fusion_cross_section(reaction: str, energy_keV: float) -> float:
    """
    Calculate fusion cross section in barns for given reaction and ion energy.
    
    Args:
        reaction: "pp", "dd", or "dt"
        energy_keV: Center-of-mass energy in keV
        
    Returns:
        Cross section in barns (1 barn = 10^-24 cm^2)
    """
    if reaction == "pp":
        # p-p fusion: extremely low cross section
        # Approximation for stellar conditions, not accurate for fusors
        if energy_keV < 1:
            return 1e-30
        return 1e-28 * np.exp(-44.0 / np.sqrt(energy_keV))
    
    elif reaction == "dd":
        # D-D fusion: Bosch-Hale parameterization
        # Valid for 0.5 - 100 keV
        if energy_keV < 0.5:
            return 0.0
        
        # Gamow factor
        bg = 31.3970  # keV^0.5
        gamow = bg / np.sqrt(energy_keV)
        
        # Astrophysical S-factor coefficients for D-D
        a1 = 5.3701e4
        a2 = 3.3027e2
        a3 = -1.2706e-1
        a4 = 2.9327e-5
        a5 = -2.5151e-9
        
        # Calculate S-factor
        s_factor = a1 + a2*energy_keV + a3*energy_keV**2 + a4*energy_keV**3 + a5*energy_keV**4
        
        # Cross section in barns
        sigma = (s_factor / energy_keV) * np.exp(-gamow) * 1e-27
        return sigma
    
    elif reaction == "dt":
        # D-T fusion: simplified fit for fusor energies
        # Peak cross section around 64 keV (~5 barns)
        if energy_keV < 1.0:
            return 1e-10  # Very small but non-zero (barns)
        elif energy_keV < 20:
            # Low energy: exponential rise (barns)
            return 1e-5 * np.exp(energy_keV / 5.0)
        elif energy_keV < 100:
            # Peak region: use simplified Gaussian around 64 keV
            sigma_max = 5.0  # ~5 barns at peak
            return sigma_max * np.exp(-((energy_keV - 64) / 30)**2)
        else:
            # High energy: falling off
            return 5.0 * np.exp(-(energy_keV - 100) / 50)
    
    else:
        raise ValueError(f"Unknown reaction: {reaction}")


def neutron_yield_rate(
    reaction: str,
    voltage_kV: float,
    current_mA: float,
    pressure_mTorr: float = 1.0
) -> float:
    """
    Estimate neutron yield rate for a fusor.
    
    Uses empirical fits to real fusor data rather than first-principles calculation.
    
    Args:
        reaction: "pp", "dd", or "dt"
        voltage_kV: Fusor operating voltage in kV
        current_mA: Ion current in mA
        pressure_mTorr: Operating pressure in milliTorr
        
    Returns:
        Neutron production rate in neutrons/second
    """
    # Empirical formula based on fusor.net data and literature
    # Typical amateur D-D fusor: 10^6-10^8 n/s at 30-50 kV, 10-30 mA
    # Professional D-D fusor: 10^8-10^10 n/s at 50-100 kV, 50-200 mA
    
    if reaction == "pp":
        # p-p fusion is extremely rare in fusors
        # Rough estimate: 10^-4 times D-D rate
        base_rate = 1e2
        voltage_exp = 2.5
        current_exp = 1.0
        pressure_exp = -0.5
        
    elif reaction == "dd":
        # D-D fusion: calibrated to real fusor data
        # ~10^7 n/s at 40 kV, 20 mA, 0.5 mTorr
        base_rate = 1e7
        voltage_exp = 2.2  # Strong voltage dependence
        current_exp = 0.9  # Nearly linear with current
        pressure_exp = -0.3  # Optimal around 0.1-1 mTorr
        
    elif reaction == "dt":
        # D-T fusion: ~100x more productive than D-D
        # Due to much higher cross section
        base_rate = 1e9
        voltage_exp = 2.0
        current_exp = 0.9
        pressure_exp = -0.3
        
    else:
        return 0.0
    
    # Reference conditions for normalization
    ref_voltage = 40.0  # kV
    ref_current = 20.0  # mA
    ref_pressure = 0.5  # mTorr
    
    # Calculate neutron rate with empirical scaling
    neutron_rate = base_rate * \
                   (voltage_kV / ref_voltage) ** voltage_exp * \
                   (current_mA / ref_current) ** current_exp * \
                   (ref_pressure / pressure_mTorr) ** pressure_exp
    
    # Saturation effects at very high voltage/current
    if voltage_kV > 100:
        neutron_rate *= np.exp(-(voltage_kV - 100) / 50)
    if current_mA > 200:
        neutron_rate *= np.exp(-(current_mA - 200) / 100)
    
    # Minimum rate (noise floor)
    neutron_rate = max(neutron_rate, 1.0)
    
    return neutron_rate


def power_balance(
    reaction: str,
    voltage_kV: float,
    current_mA: float,
    neutron_rate: float
) -> Tuple[float, float, float]:
    """
    Calculate power balance for fusor.
    
    Returns:
        (electrical_power_W, fusion_power_W, Q_value)
    """
    # Electrical input power
    electrical_power_W = voltage_kV * 1000 * current_mA * 0.001
    
    # Energy per fusion reaction
    if reaction == "pp":
        energy_per_fusion_MeV = 0.42  # Beta+ decay energy
    elif reaction == "dd":
        energy_per_fusion_MeV = 3.27  # Average of two branches
    elif reaction == "dt":
        energy_per_fusion_MeV = 17.6  # 14.1 MeV neutron + 3.5 MeV alpha
    else:
        energy_per_fusion_MeV = 0.0
    
    # Fusion power output
    energy_per_fusion_J = energy_per_fusion_MeV * 1.602e-13
    
    # Account for branching ratio
    if reaction == "dd":
        reactions_per_neutron = 2.0  # Only 50% of reactions produce neutrons
    else:
        reactions_per_neutron = 1.0
    
    fusion_power_W = neutron_rate * reactions_per_neutron * energy_per_fusion_J
    
    # Q value (fusion power / input power)
    Q = fusion_power_W / electrical_power_W if electrical_power_W > 0 else 0.0
    
    return electrical_power_W, fusion_power_W, Q
