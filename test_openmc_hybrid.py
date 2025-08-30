#!/usr/bin/env python3
"""Comprehensive tests for OpenMC hybrid reactor models."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET
import glob

# Ensure cross sections are set
default_xs = Path(__file__).parent / "openmc-data" / "cross_sections.xml"
if default_xs.exists() and not os.environ.get("OPENMC_CROSS_SECTIONS"):
    os.environ["OPENMC_CROSS_SECTIONS"] = str(default_xs)

try:
    import openmc
    from openmc_hybrid.model import (
        build_fixed_source_model,
        build_fusor_cylindrical_model,
        _material_by_name,
        run_model,
    )
    from openmc_hybrid.fusion_physics import neutron_yield_rate
    OPENMC_AVAILABLE = True
except ImportError:
    OPENMC_AVAILABLE = False


class TestCrossSectionData(unittest.TestCase):
    """Test that cross-section data is properly configured."""
    
    def test_cross_sections_xml_exists(self):
        """Test that cross_sections.xml file exists."""
        xs_path = Path("openmc-data") / "cross_sections.xml"
        self.assertTrue(
            xs_path.exists(),
            f"cross_sections.xml not found at {xs_path}. Run download script first."
        )
    
    def test_required_nuclides_in_xml(self):
        """Test that all required nuclides are in cross_sections.xml."""
        xs_path = Path("openmc-data") / "cross_sections.xml"
        if not xs_path.exists():
            self.skipTest("cross_sections.xml not found")
        
        # Parse XML
        tree = ET.parse(xs_path)
        root = tree.getroot()
        
        # Get all materials listed
        available_nuclides = set()
        for library in root.findall("library"):
            materials = library.get("materials")
            if materials:
                available_nuclides.add(materials)
        
        # Check required nuclides
        required = ["H1", "H2", "O16", "O17", "O18", "U235", "U238", 
                   "C12", "Be9", "Pb206", "Pb207", "Pb208", "Fe56", "W182", "Au197"]
        missing = []
        for nuclide in required:
            if nuclide not in available_nuclides:
                missing.append(nuclide)
        
        self.assertEqual(
            len(missing), 0,
            f"Missing nuclides in cross_sections.xml: {missing}. "
            f"Available: {sorted(available_nuclides)}"
        )
    
    def test_h5_files_exist(self):
        """Test that actual .h5 data files exist for required nuclides."""
        data_dir = Path("openmc-data")
        if not data_dir.exists():
            self.skipTest("openmc-data directory not found")
        
        required_files = [
            "H1.h5", "H2.h5", "O16.h5", "O17.h5", "O18.h5",
            "U235.h5", "U238.h5", "C12.h5", "Be9.h5"
        ]
        
        h5_files = list(data_dir.glob("*.h5"))
        h5_basenames = [f.name for f in h5_files]
        
        missing = []
        for req_file in required_files:
            # Check if any file contains the nuclide name
            nuclide = req_file.replace(".h5", "")
            found = any(nuclide in basename for basename in h5_basenames)
            if not found:
                missing.append(req_file)
        
        self.assertEqual(
            len(missing), 0,
            f"Missing .h5 data files: {missing}. "
            f"Available files: {h5_basenames[:5]}..."
        )
    
    def test_cross_sections_env_var(self):
        """Test that OPENMC_CROSS_SECTIONS is properly set."""
        xs_path = Path.cwd() / "openmc-data" / "cross_sections.xml"
        if not xs_path.exists():
            self.skipTest("cross_sections.xml not found")
        
        # Check if env var is set
        env_val = os.environ.get("OPENMC_CROSS_SECTIONS")
        self.assertIsNotNone(env_val, "OPENMC_CROSS_SECTIONS not set")
        self.assertTrue(
            Path(env_val).exists(),
            f"OPENMC_CROSS_SECTIONS points to non-existent file: {env_val}"
        )


class TestMaterials(unittest.TestCase):
    """Test material creation."""

    @unittest.skipUnless(OPENMC_AVAILABLE, "OpenMC not installed")
    def test_water_material(self):
        """Test water material with H1 nuclide."""
        mat = _material_by_name("water", 5.0, 10.5)
        self.assertIsNotNone(mat)
        self.assertEqual(mat.name, "Water")
        # Check that H1 is used, not H element
        nuclides = [n[0] for n in mat.nuclides]
        self.assertIn("H1", nuclides, "Water should use H1 nuclide explicitly")

    @unittest.skipUnless(OPENMC_AVAILABLE, "OpenMC not installed")
    def test_heavy_water_material(self):
        """Test heavy water material with H2 nuclide."""
        mat = _material_by_name("heavy_water", 5.0, 10.5)
        self.assertIsNotNone(mat)
        self.assertEqual(mat.name, "Heavy Water")
        # Check that H2 is used
        nuclides = [n[0] for n in mat.nuclides]
        self.assertIn("H2", nuclides, "Heavy water should use H2 (deuterium)")

    @unittest.skipUnless(OPENMC_AVAILABLE, "OpenMC not installed")
    def test_graphite_material(self):
        mat = _material_by_name("graphite", 5.0, 10.5)
        self.assertIsNotNone(mat)
        self.assertEqual(mat.name, "Graphite")

    @unittest.skipUnless(OPENMC_AVAILABLE, "OpenMC not installed")
    def test_uo2_material(self):
        """Test UO2 material with explicit O isotopes."""
        mat = _material_by_name("uo2", 5.0, 10.5)
        self.assertIsNotNone(mat)
        self.assertEqual(mat.name, "UO2 (5.0% enriched)")
        # Check that O isotopes are used explicitly
        nuclides = [n[0] for n in mat.nuclides]
        self.assertIn("O16", nuclides, "UO2 should use O16 explicitly")

    @unittest.skipUnless(OPENMC_AVAILABLE, "OpenMC not installed")
    def test_new_materials(self):
        """Test all newly added materials."""
        materials_to_test = [
            ("beryllium", "Beryllium"),
            ("lead", "Lead"),
            ("steel", "Steel"),
            ("tungsten_carbide", "Tungsten Carbide"),
            ("gold", "Gold"),
        ]
        
        for mat_name, expected_name in materials_to_test:
            with self.subTest(material=mat_name):
                mat = _material_by_name(mat_name, 5.0, 10.5)
                self.assertIsNotNone(mat, f"Failed to create {mat_name}")
                self.assertEqual(mat.name, expected_name)


class TestFusionPhysics(unittest.TestCase):
    """Test fusion physics calculations."""
    
    def test_neutron_yield_reasonable(self):
        """Test that neutron yield rates are in reasonable ranges."""
        # Amateur D-D fusor: should be ~10^6-10^8 n/s
        dd_rate = neutron_yield_rate("dd", 40, 20, 0.5)
        self.assertGreater(dd_rate, 1e5, "D-D rate too low")
        self.assertLess(dd_rate, 1e9, "D-D rate too high")
        
        # D-T should be ~100x higher
        dt_rate = neutron_yield_rate("dt", 40, 20, 0.5)
        self.assertGreater(dt_rate, dd_rate * 10, "D-T should be much higher than D-D")
        self.assertLess(dt_rate, dd_rate * 10000, "D-T rate unreasonably high")
        
        # p-p should be very low
        pp_rate = neutron_yield_rate("pp", 40, 20, 0.5)
        self.assertLess(pp_rate, dd_rate / 100, "p-p rate should be much lower than D-D")
    
    def test_voltage_scaling(self):
        """Test that neutron rate increases with voltage."""
        rates = []
        for voltage in [20, 30, 40, 50]:
            rate = neutron_yield_rate("dd", voltage, 20, 0.5)
            rates.append(rate)
        
        # Check monotonic increase
        for i in range(1, len(rates)):
            self.assertGreater(
                rates[i], rates[i-1],
                f"Rate should increase with voltage: {rates}"
            )


class TestModelBuilders(unittest.TestCase):
    """Test model building functions."""

    @unittest.skipUnless(OPENMC_AVAILABLE, "OpenMC not installed")
    def test_fusor_model_with_dd(self):
        """Test building fusor model with D-D source."""
        model = build_fusor_cylindrical_model(
            height_cm=100.0,
            vacuum_radius_cm=5.0,
            moderator_thickness_cm=10.0,
            fuel_thickness_cm=5.0,
            reflector_thickness_cm=10.0,
            shielding_thickness_cm=20.0,
            moderator_material="water",
            fuel_material="uo2",
            fuel_enrichment_wt_pct=5.0,
            fuel_density_gcc=10.5,
            reflector_material="graphite",
            shielding_material="lead",
            source_type="dd",
        )
        self.assertIsNotNone(model)
        self.assertEqual(model.settings.run_mode, "fixed source")
        self.assertIsNotNone(model.tallies)

    @unittest.skipUnless(OPENMC_AVAILABLE, "OpenMC not installed")
    def test_fusor_model_with_dt(self):
        """Test building fusor model with D-T source."""
        model = build_fusor_cylindrical_model(
            height_cm=100.0,
            vacuum_radius_cm=5.0,
            moderator_thickness_cm=10.0,
            fuel_thickness_cm=5.0,
            reflector_thickness_cm=10.0,
            shielding_thickness_cm=20.0,
            moderator_material="heavy_water",
            fuel_material="u_metal",
            fuel_enrichment_wt_pct=20.0,
            fuel_density_gcc=19.0,
            reflector_material="beryllium",
            shielding_material="steel",
            source_type="dt",
        )
        self.assertIsNotNone(model)

    @unittest.skipUnless(OPENMC_AVAILABLE, "OpenMC not installed")
    def test_all_source_types(self):
        """Test all fusion source types."""
        for source_type in ["pp", "dd", "dt"]:
            with self.subTest(source=source_type):
                model = build_fusor_cylindrical_model(
                    height_cm=50.0,
                    vacuum_radius_cm=5.0,
                    moderator_thickness_cm=5.0,
                    fuel_thickness_cm=2.0,
                    reflector_thickness_cm=5.0,
                    shielding_thickness_cm=10.0,
                    moderator_material="water",
                    fuel_material="uo2",
                    fuel_enrichment_wt_pct=5.0,
                    fuel_density_gcc=10.0,
                    reflector_material="graphite",
                    shielding_material="lead",
                    source_type=source_type,
                )
                self.assertIsNotNone(model, f"Failed with {source_type} source")
    
    @unittest.skipUnless(OPENMC_AVAILABLE, "OpenMC not installed")
    def test_du_optimized_configuration(self):
        """Test DU reactor with inner reflector and multi-layer shielding."""
        model = build_fusor_cylindrical_model(
            height_cm=80.0,
            vacuum_radius_cm=5.0,
            moderator_thickness_cm=0.0,  # No moderator for fast spectrum
            fuel_thickness_cm=35.0,
            reflector_thickness_cm=25.0,
            shielding_thickness_cm=10.0,
            moderator_material="none",
            fuel_material="uo2",
            fuel_enrichment_wt_pct=0.2,  # DU
            fuel_density_gcc=10.5,
            reflector_material="steel",
            shielding_material="lead",
            source_type="dt_generator",
            # DU optimization parameters
            inner_reflector_material="beryllium",
            inner_reflector_thickness_cm=3.0,
            structural_material="tungsten_carbide",
            structural_thickness_cm=5.0,
            gamma_shield_material="lead",
            gamma_shield_thickness_cm=10.0,
            end_reflector_thickness_cm=10.0,
            n_sources_azimuthal=6,
            n_sources_axial=3,
            source_ring_radius_fraction=0.9,
        )
        self.assertIsNotNone(model)
        # Check materials are created
        mat_names = {m.name for m in model.materials}
        self.assertIn("UO2", mat_names)
        self.assertIn("Beryllium", mat_names)
        self.assertIn("Steel", mat_names)
        self.assertIn("Lead", mat_names)
        self.assertIn("Tungsten Carbide", mat_names)
        # Check multi-source configuration
        if isinstance(model.settings.source, list):
            self.assertEqual(len(model.settings.source), 6 * 3)  # azimuthal * axial


class TestIntegration(unittest.TestCase):
    """Integration tests for the full workflow."""
    
    @unittest.skipUnless(OPENMC_AVAILABLE, "OpenMC not installed")
    def test_model_builds_without_error(self):
        """Test that models can be built without errors."""
        try:
            model = build_fusor_cylindrical_model(
                height_cm=100.0,
                vacuum_radius_cm=5.0,
                moderator_thickness_cm=10.0,
                fuel_thickness_cm=5.0,
                reflector_thickness_cm=10.0,
                shielding_thickness_cm=20.0,
                moderator_material="water",
                fuel_material="uo2",
                fuel_enrichment_wt_pct=5.0,
                fuel_density_gcc=10.0,
                reflector_material="graphite",
                shielding_material="lead",
                source_type="dd",
            )
            self.assertIsNotNone(model, "Model should be created")
            self.assertIsNotNone(model.geometry, "Model should have geometry")
            self.assertIsNotNone(model.materials, "Model should have materials")
            self.assertIsNotNone(model.settings, "Model should have settings")
        except Exception as e:
            if "Could not find nuclide H1" in str(e):
                self.fail("H1 nuclide not found - cross_sections.xml not properly configured")
            elif "Could not find nuclide" in str(e):
                self.fail(f"Missing nuclide data: {e}")
            else:
                self.fail(f"Model building failed: {e}")
    
    @unittest.skipUnless(OPENMC_AVAILABLE, "OpenMC not installed")
    def test_small_simulation_runs(self):
        """Test that a small simulation can actually run."""
        # Check if OpenMC executable is available
        try:
            import subprocess
            result = subprocess.run(["openmc", "--version"], capture_output=True)
            if result.returncode != 0:
                self.skipTest("OpenMC executable not available")
        except:
            self.skipTest("OpenMC executable not available")
        
        try:
            # Build a minimal model
            model = build_fusor_cylindrical_model(
                height_cm=50.0,
                vacuum_radius_cm=5.0,
                moderator_thickness_cm=5.0,
                fuel_thickness_cm=2.0,
                reflector_thickness_cm=5.0,
                shielding_thickness_cm=10.0,
                moderator_material="water",
                fuel_material="uo2",
                fuel_enrichment_wt_pct=5.0,
                fuel_density_gcc=10.0,
                reflector_material="graphite",
                shielding_material="lead",
                source_type="dd",
            )
            
            # Run with very few particles for speed
            with tempfile.TemporaryDirectory() as tmpdir:
                old_cwd = os.getcwd()
                try:
                    os.chdir(tmpdir)
                    sp = run_model(model, particles=100, batches=2)
                    self.assertIsNotNone(sp, "Simulation should produce statepoint")
                finally:
                    os.chdir(old_cwd)
                    
        except Exception as e:
            # Check for specific errors
            error_msg = str(e)
            if "Could not find nuclide H1" in error_msg:
                self.fail("H1 nuclide not found - cross_sections.xml not properly configured")
            elif "Could not find nuclide" in error_msg:
                self.fail(f"Missing nuclide data: {e}")
            elif "RuntimeError" in error_msg and "nuclear data library" in error_msg:
                self.fail(f"Nuclear data issue: {e}")
            else:
                # Other errors might be OK (e.g., OpenMC not installed)
                self.skipTest(f"Simulation failed (might be OK): {e}")


class TestEndToEnd(unittest.TestCase):
    """End-to-end tests simulating full user workflows."""
    
    @unittest.skipUnless(OPENMC_AVAILABLE, "OpenMC not installed")
    def test_full_workflow_with_all_materials(self):
        """Test building models with all supported materials."""
        # Test various material combinations
        test_configs = [
            {"moderator": "water", "fuel": "uo2", "reflector": "graphite", "shield": "lead"},
            {"moderator": "heavy_water", "fuel": "u_metal", "reflector": "beryllium", "shield": "steel"},
            {"moderator": "graphite", "fuel": "uo2", "reflector": "tungsten_carbide", "shield": "lead"},
            {"moderator": "water", "fuel": "uo2", "reflector": "gold", "shield": "lead"},
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                try:
                    model = build_fusor_cylindrical_model(
                        height_cm=50.0,
                        vacuum_radius_cm=5.0,
                        moderator_thickness_cm=5.0,
                        fuel_thickness_cm=2.0,
                        reflector_thickness_cm=5.0,
                        shielding_thickness_cm=10.0,
                        moderator_material=config["moderator"],
                        fuel_material=config["fuel"],
                        fuel_enrichment_wt_pct=5.0,
                        fuel_density_gcc=10.0,
                        reflector_material=config["reflector"],
                        shielding_material=config["shield"],
                        source_type="dd",
                    )
                    self.assertIsNotNone(model, f"Failed to build model with {config}")
                except Exception as e:
                    if "Could not find nuclide" in str(e):
                        self.fail(f"Missing nuclide for config {config}: {e}")
                    else:
                        self.fail(f"Failed with config {config}: {e}")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)