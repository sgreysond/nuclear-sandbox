# Nuclear Sandbox - Hybrid Reactor Neutronics Simulator

A Python-based neutron flux simulator for hybrid nuclear reactor designs, focusing on subcritical assemblies driven by external neutron sources (fusors, sealed-tube generators). Built on OpenMC for accurate Monte Carlo neutron transport.

## Features

- **Multiple neutron sources**: IEC fusors (p-p, D-D, D-T), sealed-tube D-T generators
- **Flexible fuel options**: Enriched/depleted uranium (UO₂, U metal), thorium (ThO₂, Th metal)
- **Optimization for different goals**:
  - Thermal spectrum (LEU, compact) - high multiplication with modest enrichment
  - Fast spectrum (DU, low-cost) - leverage U-238 fast fission with cheap fuel
- **Advanced geometry features**:
  - Inner/outer reflectors for neutron economy
  - Multi-layer shielding (structural, gamma)
  - End reflectors to reduce axial leakage
  - Multi-source arrays for improved coupling
- **Interactive Streamlit GUI** with real-time visualization and progress tracking
- **Net power metric** estimating thermal + fusion output minus electrical input
- **Comprehensive testing** suite with unit, integration, and end-to-end tests

## Quick Start

### Prerequisites

- macOS or Linux (Windows via WSL)
- ~2GB disk space for cross-section data
- 8GB+ RAM recommended for simulations

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nuclear-sandbox.git
cd nuclear-sandbox
```

2. Run the setup script:
```bash
./setup.sh
```

This will:
- Install micromamba (conda package manager)
- Create the conda environment with OpenMC and dependencies
- Download ENDF/B-VIII.0 nuclear cross-section data
- Configure environment variables
- Run verification tests

3. Activate the environment:
```bash
source activate.sh
```

### Running the Application

Start the Streamlit GUI:
```bash
bin/micromamba run -n nuclear-sandbox streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### Quick Examples

#### Thermal LEU Configuration (high multiplication)
- Click "Apply preset: Thermal (LEU, compact)"
- Uses heavy water moderator, 10% enriched UO₂, beryllium reflector
- Achieves multiplication M ≈ 7-10 with good neutron economy

#### Fast DU Configuration (low cost)
- Click "Apply preset: Fast (DU, low-cost)"
- Uses depleted uranium (0.2% U-235), no moderator, thick reflectors
- Leverages U-238 fast fission with 14 MeV neutrons from D-T source

### Parameter Optimization

A helper script performs a coarse grid search to find the smallest
cylindrical design that yields net-positive thermal power after accounting for
the neutron source electrical input:

```bash
python scripts/optimize_reactor.py
```

This requires OpenMC nuclear data and can take several minutes to run.

## Project Structure

```
nuclear-sandbox/
├── openmc_hybrid/          # Core OpenMC model building
│   ├── model.py           # Geometry, materials, sources
│   └── fusion_physics.py  # Fusor neutron yield calculations
├── scripts/               # Utility scripts
│   ├── download_xs.py     # Download cross-section data
│   └── regenerate_xs_xml.py # Rebuild cross-section index
├── app.py                 # Streamlit GUI application
├── test_openmc_hybrid.py  # Test suite
├── environment.yml        # Conda environment specification
├── setup.sh              # One-click setup script
└── README.md             # This file
```

## Key Concepts

### Subcritical Multiplication
- k_eff < 1.0 ensures safety (cannot go critical)
- Multiplication M ≈ 1/(1 - k_eff) amplifies source neutrons
- External source controls power level

### Neutron Sources
- **IEC Fusors**: Electrostatic confinement, ~10⁶-10⁸ n/s typical
- **Sealed-tube generators**: Beam-target D-T, ~10⁸-10¹⁰ n/s, better n/W efficiency
- **Source arrays**: Multiple sources improve spatial coupling

### Optimization Strategies
- **Thermal spectrum**: Maximize thermal fission in U-235, requires moderation
- **Fast spectrum**: Exploit U-238 fast fission and Be(n,2n) multiplication
- **Reflectors**: Return escaping neutrons, beryllium best for fast systems
- **Geometry**: Minimize vacuum gaps, optimize thickness ratios

## Testing

Run the test suite:
```bash
bin/micromamba run -n nuclear-sandbox python -m unittest discover
```

Run specific tests:
```bash
# Test DU configuration
bin/micromamba run -n nuclear-sandbox python -m unittest test_openmc_hybrid.TestModelBuilders.test_du_optimized_configuration

# Test cross-section data
bin/micromamba run -n nuclear-sandbox python -m unittest test_openmc_hybrid.TestCrossSectionData
```

## Development

### Adding New Materials

Edit `openmc_hybrid/model.py` and add to `_material_by_name()`:
```python
if name == "your_material":
    m = openmc.Material(name="Your Material")
    m.add_element("X", 1.0)
    m.set_density("g/cm3", density)
    return m
```

### Adding New Source Types

Edit `openmc_hybrid/model.py` and add to `_build_source()`:
```python
elif src.source_type == "your_source":
    # Define energy spectrum
    energy = Tabular(energies_eV, probabilities, interpolation="linear-linear")
    # Define angular distribution if needed
    angle = PolarAzimuthal(...)
```

## Safety Notice

This is a simulation tool for educational and research purposes. Any physical implementation of nuclear systems requires appropriate licensing, safety analysis, and regulatory approval.

## License

MIT License - See LICENSE file for details

## Acknowledgments

- OpenMC development team for the Monte Carlo engine
- NNDC for ENDF/B-VIII.0 cross-section data
- Streamlit for the interactive GUI framework

## Contributing

Contributions welcome! Please ensure:
- All tests pass
- New features include tests
- Code follows existing style
- Documentation is updated

## Contact

For questions or collaboration: [your email]