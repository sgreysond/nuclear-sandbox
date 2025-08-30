import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import openmc
    from openmc.stats import Point, Isotropic, Tabular, Watt, Maxwell, Box, PolarAzimuthal
except Exception as exc:  # pragma: no cover
    openmc = None


@dataclass
class GeometrySpec:
    geometry_kind: str  # "sphere" or "cylinder"
    fuel_radius_cm: float = 5.0
    moderator_thickness_cm: float = 5.0
    height_cm: float = 10.0  # used for cylinder
    add_reflector: bool = False
    reflector_thickness_cm: float = 5.0
    reflector_material: str = "graphite"


@dataclass
class MaterialSpec:
    fuel_form: str = "uo2"  # or "metal"
    fuel_enrichment_wt_pct: float = 5.0
    fuel_density_gcc: float = 10.5
    moderator: Optional[str] = "water"  # "water", "graphite", "heavy_water", None


@dataclass
class SourceSpec:
    source_type: str = "dd"  # "dd", "dt", or "custom"
    position_cm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction_isotropic: bool = True
    custom_energy_eV: Optional[np.ndarray] = None
    custom_pdf: Optional[np.ndarray] = None
    # Maxwellian temperature (kT) in eV (if source_type == "maxwell")
    maxwell_kT_eV: Optional[float] = None
    # Custom Watt parameters (if source_type == "watt_custom")
    watt_a_eV: Optional[float] = None
    watt_b_inv_eV: Optional[float] = None


def _build_materials(spec: MaterialSpec) -> Dict[str, "openmc.Material"]:
    if openmc is None:
        raise RuntimeError("OpenMC is not installed in this environment.")

    mats = {}

    if spec.fuel_form == "uo2":
        uo2 = openmc.Material(name="UO2")
        # Convert wt% enrichment to atom fractions approximately (quick-start)
        # For more fidelity, use proper isotopic vectors and densities.
        u235_wt = spec.fuel_enrichment_wt_pct / 100.0
        u238_wt = 1.0 - u235_wt
        # Build U mixture
        u = openmc.Material(name="U metal (virtual)")
        u.add_nuclide("U235", 1.0 * u235_wt)
        u.add_nuclide("U238", 1.0 * u238_wt)
        # Compose UO2: U + O2
        uo2.add_nuclide("U235", 1.0 * u235_wt)
        uo2.add_nuclide("U238", 1.0 * u238_wt)
        # Use natural oxygen isotopic composition (2 oxygen atoms)
        uo2.add_nuclide("O16", 2.0 * 0.9976)
        uo2.add_nuclide("O17", 2.0 * 0.0004)
        uo2.add_nuclide("O18", 2.0 * 0.0020)
        uo2.set_density("g/cm3", spec.fuel_density_gcc)
        mats["fuel"] = uo2
    else:
        u_metal = openmc.Material(name="U metal")
        u235_wt = spec.fuel_enrichment_wt_pct / 100.0
        u238_wt = 1.0 - u235_wt
        u_metal.add_nuclide("U235", 1.0 * u235_wt)
        u_metal.add_nuclide("U238", 1.0 * u238_wt)
        u_metal.set_density("g/cm3", spec.fuel_density_gcc)
        mats["fuel"] = u_metal

    if spec.moderator == "water":
        h2o = openmc.Material(name="Water")
        h2o.add_nuclide("H1", 2.0)  # Use H-1 (protium) explicitly
        # Use natural oxygen isotopic composition
        h2o.add_nuclide("O16", 0.9976)
        h2o.add_nuclide("O17", 0.0004)
        h2o.add_nuclide("O18", 0.0020)
        h2o.set_density("g/cm3", 1.0)
        # Optional S(a,b): requires matching library; user can add as needed
        mats["moderator"] = h2o
    elif spec.moderator == "graphite":
        c = openmc.Material(name="Graphite")
        c.add_element("C", 1)
        c.set_density("g/cm3", 1.7)
        mats["moderator"] = c
    elif spec.moderator == "heavy_water":
        d2o = openmc.Material(name="Heavy Water")
        d2o.add_nuclide("H2", 2.0)  # Deuterium
        # Use natural oxygen isotopic composition
        d2o.add_nuclide("O16", 0.9976)
        d2o.add_nuclide("O17", 0.0004)
        d2o.add_nuclide("O18", 0.0020)
        d2o.set_density("g/cm3", 1.105)
        mats["moderator"] = d2o

    # Reflector material (if requested) — reuse moderator or graphite
    if spec.moderator is None:
        # Default reflector is graphite if none selected
        c = openmc.Material(name="Graphite")
        c.add_element("C", 1)
        c.set_density("g/cm3", 1.7)
        mats["reflector"] = c
    elif spec.moderator == "water":
        c = openmc.Material(name="Graphite (reflector)")
        c.add_element("C", 1)
        c.set_density("g/cm3", 1.7)
        mats["reflector"] = c
    else:
        # graphite moderator → use same for reflector
        mats["reflector"] = mats.get("moderator")

    return mats


def _build_geometry(geom: GeometrySpec, mats: Dict[str, "openmc.Material"]) -> "openmc.Geometry":
    if openmc is None:
        raise RuntimeError("OpenMC is not installed in this environment.")

    fuel_mat = mats["fuel"]
    moderator_mat = mats.get("moderator")
    reflector_mat = mats.get("reflector") if geom.add_reflector else None

    if geom.geometry_kind == "sphere":
        r_fuel = openmc.Sphere(r=geom.fuel_radius_cm)
        fuel_region = -r_fuel
        fuel_cell = openmc.Cell(name="fuel", fill=fuel_mat, region=fuel_region)

        current_surface = r_fuel
        cells = [fuel_cell]

        if moderator_mat is not None and geom.moderator_thickness_cm > 0.0:
            r_mod = openmc.Sphere(r=geom.fuel_radius_cm + geom.moderator_thickness_cm)
            mod_region = -r_mod & +r_fuel
            mod_cell = openmc.Cell(name="moderator", fill=moderator_mat, region=mod_region)
            cells.append(mod_cell)
            current_surface = r_mod

        if reflector_mat is not None and geom.reflector_thickness_cm > 0.0:
            r_ref = openmc.Sphere(r=current_surface.r + geom.reflector_thickness_cm)
            ref_region = -r_ref & +current_surface
            ref_cell = openmc.Cell(name="reflector", fill=reflector_mat, region=ref_region)
            cells.append(ref_cell)
            current_surface = r_ref

        outer_boundary = openmc.Sphere(r=current_surface.r * 3.0, boundary_type="vacuum")
        void_region = +current_surface & -outer_boundary
        void_cell = openmc.Cell(name="void", region=void_region)
        cells.append(void_cell)

        root = openmc.Universe(cells=cells)
        geometry = openmc.Geometry(root)
        return geometry

    elif geom.geometry_kind == "cylinder":
        r_fuel = openmc.ZCylinder(r=geom.fuel_radius_cm)
        zmin = openmc.ZPlane(z0=-geom.height_cm / 2)
        zmax = openmc.ZPlane(z0=+geom.height_cm / 2)

        fuel_region = -r_fuel & +zmin & -zmax
        fuel_cell = openmc.Cell(name="fuel", fill=fuel_mat, region=fuel_region)
        cells = [fuel_cell]

        current_surface = r_fuel
        if moderator_mat is not None and geom.moderator_thickness_cm > 0.0:
            r_mod = openmc.ZCylinder(r=geom.fuel_radius_cm + geom.moderator_thickness_cm)
            mod_region = -r_mod & +r_fuel & +zmin & -zmax
            mod_cell = openmc.Cell(name="moderator", fill=moderator_mat, region=mod_region)
            cells.append(mod_cell)
            current_surface = r_mod

        if reflector_mat is not None and geom.reflector_thickness_cm > 0.0:
            r_ref = openmc.ZCylinder(r=current_surface.r + geom.reflector_thickness_cm)
            ref_region = -r_ref & +current_surface & +zmin & -zmax
            ref_cell = openmc.Cell(name="reflector", fill=reflector_mat, region=ref_region)
            cells.append(ref_cell)
            current_surface = r_ref

        r_outer = openmc.ZCylinder(r=current_surface.r * 3.0)
        zmin_v = openmc.ZPlane(z0=zmin.z0 * 2, boundary_type="vacuum")
        zmax_v = openmc.ZPlane(z0=zmax.z0 * 2, boundary_type="vacuum")
        r_outer_v = openmc.ZCylinder(r=r_outer.r * 1.2)
        void_region = +current_surface & -r_outer & +zmin & -zmax | -r_outer_v | +zmin_v | -zmax_v
        void_cell = openmc.Cell(name="void", region=void_region)
        cells.append(void_cell)

        root = openmc.Universe(cells=cells)
        geometry = openmc.Geometry(root)
        return geometry

    else:
        raise ValueError("geometry_kind must be 'sphere' or 'cylinder'")


def _build_source(src: SourceSpec) -> "openmc.Source":
    if openmc is None:
        raise RuntimeError("OpenMC is not installed in this environment.")

    position = Point(src.position_cm)
    angle = Isotropic() if src.direction_isotropic else Isotropic()

    if src.source_type == "pp":
        # p-p fusion: very rare in fusors, produces 0.42 MeV average from beta+ decay
        # Neutrons only from secondary reactions, model as very low energy
        e = np.array([0.1e6, 0.42e6, 1.0e6])
        p = np.array([0.2, 0.6, 0.2])
        energy = Tabular(e, p, interpolation="linear-linear")
    elif src.source_type == "dd":
        # D-D fusion has two branches:
        # 50%: D + D → ³He + n (2.45 MeV neutron)
        # 50%: D + D → T + p (no direct neutron)
        # Model the 2.45 MeV neutron with realistic width
        e = np.array([2.2e6, 2.35e6, 2.45e6, 2.55e6, 2.7e6])
        p = np.array([0.05, 0.20, 0.50, 0.20, 0.05])
        energy = Tabular(e, p, interpolation="linear-linear")
    elif src.source_type == "dt":
        # D-T fusion: D + T → ⁴He + n (14.1 MeV neutron)
        # Highest cross section, most neutrons
        # Broader spectrum due to ion temperature in plasma
        e = np.array([13.5e6, 13.8e6, 14.0e6, 14.1e6, 14.2e6, 14.4e6, 14.7e6])
        p = np.array([0.02, 0.10, 0.25, 0.30, 0.20, 0.10, 0.03])
        energy = Tabular(e, p, interpolation="linear-linear")
    elif src.source_type == "dt_generator":
        # Sealed-tube D-T generator: forward-peaked angular distribution and
        # slightly broadened 14 MeV spectrum. We approximate E-θ correlation by
        # using an anisotropic angle distribution and an independent energy PDF.
        e = np.array([13.5e6, 13.8e6, 14.0e6, 14.1e6, 14.2e6, 14.4e6, 14.7e6])
        p = np.array([0.02, 0.10, 0.25, 0.30, 0.20, 0.10, 0.03])
        energy = Tabular(e, p, interpolation="linear-linear")
        # Forward-peaked mu=cos(theta) distribution
        mu = np.array([-1.0, -0.5, 0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 1.0])
        w =  np.array([ 0.02, 0.03, 0.05, 0.10, 0.15, 0.18,  0.20,  0.17, 0.10])
        angle = PolarAzimuthal(mu=Tabular(mu, w, interpolation="linear-linear"))
        return openmc.Source(space=position, angle=angle, energy=energy)
    elif src.source_type == "watt_u235":
        # Representative Watt parameters for U-235 fission spectrum
        a_eV = 0.988e6
        b_inv_eV = 2.249e-6
        energy = Watt(a=a_eV, b=b_inv_eV)
    elif src.source_type == "watt_pu239":
        # Representative Watt parameters for Pu-239 fission spectrum
        a_eV = 0.799e6
        b_inv_eV = 2.68e-6
        energy = Watt(a=a_eV, b=b_inv_eV)
    elif src.source_type == "watt_custom":
        if src.watt_a_eV is None or src.watt_b_inv_eV is None:
            raise ValueError("watt_a_eV and watt_b_inv_eV required for watt_custom source")
        energy = Watt(a=src.watt_a_eV, b=src.watt_b_inv_eV)
    elif src.source_type == "maxwell":
        kT = src.maxwell_kT_eV if src.maxwell_kT_eV is not None else 2.0e3
        energy = Maxwell(theta=kT)
    elif src.source_type == "custom":
        if src.custom_energy_eV is None or src.custom_pdf is None:
            raise ValueError("custom_energy_eV and custom_pdf required for custom source")
        energy = Tabular(src.custom_energy_eV, src.custom_pdf, interpolation="linear-linear")
    else:
        raise ValueError(
            "source_type must be one of: 'pp', 'dd', 'dt', 'watt_u235', 'watt_pu239', 'watt_custom', 'maxwell', 'custom'"
        )

    return openmc.Source(space=position, angle=angle, energy=energy)


def build_fixed_source_model(
    geometry_kind: str = "sphere",
    fuel_enrichment_wt_pct: float = 5.0,
    fuel_density_gcc: float = 10.5,
    moderator: Optional[str] = "water",
    fuel_radius_cm: float = 5.0,
    moderator_thickness_cm: float = 5.0,
    add_reflector: bool = False,
    reflector_thickness_cm: float = 5.0,
    height_cm: float = 10.0,
    source_type: str = "dd",
    maxwell_kT_eV: Optional[float] = None,
    watt_a_eV: Optional[float] = None,
    watt_b_inv_eV: Optional[float] = None,
) -> "openmc.Model":
    if openmc is None:
        raise RuntimeError("OpenMC is not installed in this environment.")

    mats = _build_materials(
        MaterialSpec(
            fuel_form="uo2",
            fuel_enrichment_wt_pct=fuel_enrichment_wt_pct,
            fuel_density_gcc=fuel_density_gcc,
            moderator=moderator,
        )
    )
    geom = _build_geometry(
        GeometrySpec(
            geometry_kind=geometry_kind,
            fuel_radius_cm=fuel_radius_cm,
            moderator_thickness_cm=moderator_thickness_cm,
            height_cm=height_cm,
            add_reflector=add_reflector,
            reflector_thickness_cm=reflector_thickness_cm,
        ),
        mats,
    )
    src = _build_source(
        SourceSpec(
            source_type=source_type,
            maxwell_kT_eV=maxwell_kT_eV,
            watt_a_eV=watt_a_eV,
            watt_b_inv_eV=watt_b_inv_eV,
        )
    )

    model = openmc.Model()
    model.materials = openmc.Materials(list(set(mats.values())))
    model.geometry = geom
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.source = src
    settings.particles = 10000
    settings.batches = 50
    model.settings = settings

    # Basic tallies: total flux and energy-binned flux
    tallies = openmc.Tallies()
    tally_flux = openmc.Tally(name="flux")
    tally_flux.filters = [openmc.CellFilter([c for c in model.geometry.get_all_cells().values()])]
    tally_flux.scores = ["flux"]
    tallies.append(tally_flux)

    e_edges = np.logspace(0, 8, 101)  # 1 eV to 100 MeV
    tally_ebin = openmc.Tally(name="flux_ebin")
    tally_ebin.filters = [openmc.EnergyFilter(e_edges)]
    tally_ebin.scores = ["flux"]
    tallies.append(tally_ebin)

    model.tallies = tallies
    return model


def run_model(model: "openmc.Model", particles: int = 10000, batches: int = 50,
              progress_callback=None) -> Optional["openmc.StatePoint"]:
    """Run the OpenMC model and return the statepoint.
    
    Args:
        model: OpenMC model to run
        particles: Number of particles per batch
        batches: Number of batches
        progress_callback: Optional callback function(current_batch, total_batches, message)
    """
    if openmc is None:
        raise RuntimeError("OpenMC is not installed in this environment.")

    model.settings.particles = particles
    model.settings.batches = batches
    
    # Adjust for run_mode
    if model.settings.run_mode == "eigenvalue":
        # Set eigenvalue-specific settings if needed
        model.settings.generations_per_batch = 100
        model.settings.inactive = 10

    # Run with progress tracking if callback provided
    if progress_callback and model.settings.run_mode == "fixed source":
        import subprocess
        import re
        import glob
        from collections import deque
        
        # Export model first
        model.export_to_model_xml()
        
        # Run OpenMC with output capture
        process = subprocess.Popen(
            ['openmc'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        batch_pattern = re.compile(r'Simulating batch\s+(\d+)')
        tail = deque(maxlen=200)
        
        for line in process.stdout:
            tail.append(line.rstrip())
            match = batch_pattern.search(line)
            if match:
                current_batch = int(match.group(1))
                progress_callback(current_batch, batches, f"Simulating batch {current_batch}/{batches}")
        
        process.wait()
        if process.returncode != 0:
            err = "\n".join(tail)
            raise RuntimeError(f"OpenMC simulation failed. Last output:\n{err}")
        
        # Find the statepoint file
        sp_files = glob.glob("statepoint.*.h5")
        if not sp_files:
            raise RuntimeError("No statepoint file generated")
        sp_path = sorted(sp_files)[-1]  # Get the latest
    else:
        # Run normally without progress tracking
        sp_path = model.run()

    # Try to open statepoint; if HDF5 attribute issues arise, fall back to a clean run
    try:
        return openmc.StatePoint(sp_path, autolink=False)
    except Exception:
        # Fallback: run via Python API (no progress) and reopen
        sp_path2 = model.run()
        return openmc.StatePoint(sp_path2, autolink=False)


# --- Fusor-cylinder specialized builder ---

def _material_by_name(name: Optional[str], enrichment_wt_pct: float, fuel_density_gcc: float) -> Optional["openmc.Material"]:
    if name is None or name == "none":
        return None
    name = name.lower()
    if name == "water":
        m = openmc.Material(name="Water")
        m.add_nuclide("H1", 2.0)  # Use H-1 (protium) explicitly
        # Use natural oxygen isotopic composition
        m.add_nuclide("O16", 0.9976)
        m.add_nuclide("O17", 0.0004) 
        m.add_nuclide("O18", 0.0020)
        m.set_density("g/cm3", 1.0)
        return m
    if name == "heavy_water":
        m = openmc.Material(name="Heavy Water")
        m.add_nuclide("H2", 2.0)  # Deuterium
        # Use natural oxygen isotopic composition
        m.add_nuclide("O16", 0.9976)
        m.add_nuclide("O17", 0.0004)
        m.add_nuclide("O18", 0.0020)
        m.set_density("g/cm3", 1.105)
        return m
    if name == "graphite":
        m = openmc.Material(name="Graphite")
        m.add_element("C", 1)
        m.set_density("g/cm3", 1.7)
        return m
    if name == "beryllium":
        m = openmc.Material(name="Beryllium")
        m.add_element("Be", 1)
        m.set_density("g/cm3", 1.85)
        return m
    if name == "lead":
        m = openmc.Material(name="Lead")
        m.add_element("Pb", 1)
        m.set_density("g/cm3", 11.34)
        return m
    if name == "steel":
        m = openmc.Material(name="Steel")
        m.add_element("Fe", 0.98)
        m.add_element("C", 0.02)
        m.set_density("g/cm3", 7.85)
        return m
    if name == "tungsten_carbide":
        m = openmc.Material(name="Tungsten Carbide")
        m.add_element("W", 1)
        m.add_element("C", 1)
        m.set_density("g/cm3", 15.6)
        return m
    if name == "gold":
        m = openmc.Material(name="Gold")
        m.add_element("Au", 1)
        m.set_density("g/cm3", 19.3)
        return m
    if name == "uo2":
        m = openmc.Material(name="UO2")
        u235_wt = enrichment_wt_pct / 100.0
        u238_wt = 1.0 - u235_wt
        m.add_nuclide("U235", 1.0 * u235_wt)
        m.add_nuclide("U238", 1.0 * u238_wt)
        # Use natural oxygen isotopic composition (2 oxygen atoms)
        m.add_nuclide("O16", 2.0 * 0.9976)
        m.add_nuclide("O17", 2.0 * 0.0004)
        m.add_nuclide("O18", 2.0 * 0.0020)
        m.set_density("g/cm3", fuel_density_gcc)
        return m
    if name == "th_o2":
        m = openmc.Material(name="ThO2")
        # Thorium dioxide (thoria), natural thorium is essentially Th-232
        m.add_nuclide("Th232", 1.0)
        # Two oxygen atoms with natural isotopic composition
        m.add_nuclide("O16", 2.0 * 0.9976)
        m.add_nuclide("O17", 2.0 * 0.0004)
        m.add_nuclide("O18", 2.0 * 0.0020)
        m.set_density("g/cm3", fuel_density_gcc)
        return m
    if name == "u_metal":
        m = openmc.Material(name="U metal")
        u235_wt = enrichment_wt_pct / 100.0
        u238_wt = 1.0 - u235_wt
        m.add_nuclide("U235", 1.0 * u235_wt)
        m.add_nuclide("U238", 1.0 * u238_wt)
        m.set_density("g/cm3", fuel_density_gcc)
        return m
    if name == "th_metal":
        m = openmc.Material(name="Th metal")
        m.add_nuclide("Th232", 1.0)
        m.set_density("g/cm3", fuel_density_gcc)
        return m
    # Fallback: void
    return None


def build_fusor_cylindrical_model(
    height_cm: float,
    vacuum_radius_cm: float,
    moderator_thickness_cm: float,
    fuel_thickness_cm: float,
    reflector_thickness_cm: float,
    shielding_thickness_cm: float,
    moderator_material: Optional[str],
    fuel_material: str,
    fuel_enrichment_wt_pct: float,
    fuel_density_gcc: float,
    reflector_material: Optional[str],
    shielding_material: Optional[str],
    source_type: str = "dd",
    maxwell_kT_eV: Optional[float] = None,
    watt_a_eV: Optional[float] = None,
    watt_b_inv_eV: Optional[float] = None,
    run_mode: str = "fixed source",
    # New options for DU optimization
    inner_reflector_material: Optional[str] = None,
    inner_reflector_thickness_cm: float = 0.0,
    structural_material: Optional[str] = None,
    structural_thickness_cm: float = 0.0,
    gamma_shield_material: Optional[str] = None,
    gamma_shield_thickness_cm: float = 0.0,
    end_reflector_thickness_cm: float = 0.0,
    # Multi-source arrangement
    n_sources_azimuthal: int = 1,
    n_sources_axial: int = 1,
    source_ring_radius_fraction: float = 0.9,
) -> "openmc.Model":
    if openmc is None:
        raise RuntimeError("OpenMC is not installed in this environment.")

    # Materials
    moderator_mat = _material_by_name(moderator_material, fuel_enrichment_wt_pct, fuel_density_gcc)
    fuel_mat = _material_by_name(fuel_material, fuel_enrichment_wt_pct, fuel_density_gcc)
    reflector_mat = _material_by_name(reflector_material, fuel_enrichment_wt_pct, fuel_density_gcc)
    shielding_mat = _material_by_name(shielding_material, fuel_enrichment_wt_pct, fuel_density_gcc)
    inner_reflector_mat = _material_by_name(inner_reflector_material, fuel_enrichment_wt_pct, fuel_density_gcc)
    structural_mat = _material_by_name(structural_material, fuel_enrichment_wt_pct, fuel_density_gcc)
    gamma_shield_mat = _material_by_name(gamma_shield_material, fuel_enrichment_wt_pct, fuel_density_gcc)

    materials = [m for m in [moderator_mat, fuel_mat, reflector_mat, shielding_mat, 
                             inner_reflector_mat, structural_mat, gamma_shield_mat] if m is not None]

    # Geometry: concentric Z-cylinders
    # Radial surfaces (inner to outer)
    r_vac = openmc.ZCylinder(r=vacuum_radius_cm)
    r_inner_ref = openmc.ZCylinder(r=r_vac.r + max(0.0, inner_reflector_thickness_cm))
    r_mod = openmc.ZCylinder(r=r_inner_ref.r + max(0.0, moderator_thickness_cm))
    r_fuel = openmc.ZCylinder(r=r_mod.r + max(0.0, fuel_thickness_cm))
    r_refl = openmc.ZCylinder(r=r_fuel.r + max(0.0, reflector_thickness_cm))
    r_struct = openmc.ZCylinder(r=r_refl.r + max(0.0, structural_thickness_cm))
    r_gamma = openmc.ZCylinder(r=r_struct.r + max(0.0, gamma_shield_thickness_cm))

    # Axial surfaces: add end reflector slabs inside vacuum ends
    zmin_v = openmc.ZPlane(z0=-height_cm / 2, boundary_type="vacuum")
    zmax_v = openmc.ZPlane(z0=+height_cm / 2, boundary_type="vacuum")
    zmin_i = openmc.ZPlane(z0=(-height_cm / 2) + max(0.0, end_reflector_thickness_cm))
    zmax_i = openmc.ZPlane(z0=(+height_cm / 2) - max(0.0, end_reflector_thickness_cm))

    cells = []
    # Central vacuum column
    vac_region = -r_vac & +zmin_i & -zmax_i
    vac_cell = openmc.Cell(name="vacuum", region=vac_region)
    cells.append(vac_cell)

    # Inner reflector (booster liner)
    if inner_reflector_thickness_cm > 0 and inner_reflector_mat is not None:
        ir_region = -r_inner_ref & +r_vac & +zmin_i & -zmax_i
        ir_cell = openmc.Cell(name="inner_reflector", fill=inner_reflector_mat, region=ir_region)
        cells.append(ir_cell)

    # Moderator shell
    if moderator_thickness_cm > 0 and moderator_mat is not None:
        mod_region = -r_mod & +r_inner_ref & +zmin_i & -zmax_i
        mod_cell = openmc.Cell(name="moderator", fill=moderator_mat, region=mod_region)
        cells.append(mod_cell)

    # Fuel shell
    if fuel_thickness_cm > 0 and fuel_mat is not None:
        fuel_region = -r_fuel & +r_mod & +zmin_i & -zmax_i
        fuel_cell = openmc.Cell(name="fuel", fill=fuel_mat, region=fuel_region)
        cells.append(fuel_cell)

    # Outer fast reflector shell
    if reflector_thickness_cm > 0 and reflector_mat is not None:
        refl_region = -r_refl & +r_fuel & +zmin_i & -zmax_i
        refl_cell = openmc.Cell(name="reflector", fill=reflector_mat, region=refl_region)
        cells.append(refl_cell)

    # Structural shell (e.g., WC/steel)
    if structural_thickness_cm > 0 and structural_mat is not None:
        struct_region = -r_struct & +r_refl & +zmin_v & -zmax_v
        struct_cell = openmc.Cell(name="structural", fill=structural_mat, region=struct_region)
        cells.append(struct_cell)

    # Gamma shield shell (e.g., lead)
    if gamma_shield_thickness_cm > 0 and gamma_shield_mat is not None:
        gamma_region = -r_gamma & +r_struct & +zmin_v & -zmax_v
        gamma_cell = openmc.Cell(name="gamma_shield", fill=gamma_shield_mat, region=gamma_region)
        cells.append(gamma_cell)

    # End reflector slabs inside r_refl to reduce axial leakage
    if end_reflector_thickness_cm > 0 and reflector_mat is not None:
        end_bot = openmc.Cell(name="end_reflector_bottom", region=(-r_refl & +zmin_v & -zmin_i), fill=reflector_mat)
        end_top = openmc.Cell(name="end_reflector_top", region=(-r_refl & +zmax_i & -zmax_v), fill=reflector_mat)
        cells.extend([end_bot, end_top])

    # Outer vacuum boundary
    r_outer = openmc.ZCylinder(r=max(r_gamma.r if gamma_shield_thickness_cm > 0 else (r_struct.r if structural_thickness_cm > 0 else r_refl.r), 1e-3) * 1.5, boundary_type="vacuum")
    outer_region = + (r_gamma if gamma_shield_thickness_cm > 0 else (r_struct if structural_thickness_cm > 0 else r_refl)) & -r_outer & +zmin_v & -zmax_v
    outer_cell = openmc.Cell(name="outer_void", region=outer_region)
    cells.append(outer_cell)

    universe = openmc.Universe(cells=cells)
    geometry = openmc.Geometry(universe)

    # Source(s): point/disk sources placed near inner radius, optionally multiple
    sources = []
    if source_type in {"pp", "dd", "dt", "dt_generator", "watt_u235", "watt_pu239", "watt_custom", "maxwell", "custom"}:
        # Ring radius for sources (inside the vacuum gap)
        ring_r = max(1e-3, r_vac.r * max(0.1, min(0.99, source_ring_radius_fraction)))
        # Axial range for sources (within inner region)
        z_low = zmin_i.z0
        z_high = zmax_i.z0
        if n_sources_axial <= 1:
            z_positions = [0.5 * (z_low + z_high)]
        else:
            z_positions = list(np.linspace(z_low, z_high, n_sources_axial))
        # Azimuthal positions
        if n_sources_azimuthal <= 1:
            phis = [0.0]
        else:
            phis = [2*np.pi * k / n_sources_azimuthal for k in range(n_sources_azimuthal)]

        # Small finite source size
        eps_r = min(0.1, 0.05 * ring_r)
        eps_z = min(0.5, 0.02 * height_cm)

        for zc in z_positions:
            for phi in phis:
                x = ring_r * np.cos(phi)
                y = ring_r * np.sin(phi)
                space = Box(lower_left=(x-eps_r, y-eps_r, zc-eps_z), upper_right=(x+eps_r, y+eps_r, zc+eps_z), only_fissionable=False)
                s = _build_source(SourceSpec(
                    source_type=source_type,
                    maxwell_kT_eV=maxwell_kT_eV,
                    watt_a_eV=watt_a_eV,
                    watt_b_inv_eV=watt_b_inv_eV,
                ))
                s.space = space
                sources.append(s)
    else:
        raise ValueError("Unsupported source_type for fusor model")

    # Settings
    settings = openmc.Settings()
    settings.run_mode = run_mode

    if run_mode == "fixed source":
        settings.source = sources if len(sources) > 1 else sources[0]
        settings.particles = 20000
        settings.batches = 80
    elif run_mode == "eigenvalue":
        settings.generations_per_batch = 100
        settings.inactive = 10
        settings.batches = 100  # Adjust as needed

    # Tallies
    tallies = openmc.Tallies()

    # Energy-binned flux (global)
    e_edges = np.logspace(0, 8, 201)
    t_flux_ebin = openmc.Tally(name="flux_ebin")
    t_flux_ebin.filters = [openmc.EnergyFilter(e_edges)]
    t_flux_ebin.scores = ["flux"]
    tallies.append(t_flux_ebin)

    # Region-wise heating and fuel kappa-fission / nu-fission
    all_cells = geometry.get_all_cells()
    region_names = ["vacuum", "moderator", "fuel", "reflector", "shield"]
    for rn in region_names:
        cells_named = [c for c in all_cells.values() if c.name == rn]
        if not cells_named:
            continue
        tf = openmc.Tally(name=f"heating_{rn}")
        tf.filters = [openmc.CellFilter(cells_named)]
        tf.scores = ["heating"]
        tallies.append(tf)
        if rn == "fuel":
            tk = openmc.Tally(name="kappa_fission_fuel")
            tk.filters = [openmc.CellFilter(cells_named)]
            tk.scores = ["kappa-fission"]
            tallies.append(tk)

            # Add nu-fission to estimate fission neutrons per source neutron
            tnu = openmc.Tally(name="nu_fission_fuel")
            tnu.filters = [openmc.CellFilter(cells_named)]
            tnu.scores = ["nu-fission"]
            tallies.append(tnu)

    model = openmc.Model(geometry=geometry, materials=openmc.Materials(materials), settings=settings, tallies=tallies)
    return model


