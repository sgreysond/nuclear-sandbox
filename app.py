import os
import traceback
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Hybrid Neutronics Sandbox", layout="wide")

st.title("Hybrid Neutronics Sandbox (OpenMC)")
st.caption("Configure a simple fixed-source model and run OpenMC")

# Auto-detect and set cross sections if not already set
if not os.environ.get("OPENMC_CROSS_SECTIONS"):
	default_xs = Path(__file__).parent / "openmc-data" / "cross_sections.xml"
	if default_xs.exists():
		os.environ["OPENMC_CROSS_SECTIONS"] = str(default_xs)

try:
	from openmc_hybrid.model import (
		build_fixed_source_model,
		build_fusor_cylindrical_model,
		run_model,
	)
	from openmc_hybrid.fusion_physics import (
		neutron_yield_rate,
		power_balance,
	)
	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	import openmc
except Exception:
	st.error("Python dependencies not fully installed. Ensure the conda env from environment.yml is active.")
	st.stop()


with st.sidebar:
	st.header("Configuration Mode")
	mode = st.selectbox("Mode", ["Simple (toy)", "Fusor cylinder"], index=1)

	if mode == "Simple (toy)":
		st.subheader("Geometry")
		geometry_kind = st.selectbox("Kind", ["sphere", "cylinder"], index=0)
		fuel_radius_cm = st.number_input("Fuel radius [cm]", 1.0, 100.0, 5.0, 0.5)
		moderator_thickness_cm = st.number_input("Moderator thickness [cm]", 0.0, 100.0, 5.0, 0.5)
		add_reflector = st.checkbox("Add reflector", value=False)
		reflector_thickness_cm = st.number_input("Reflector thickness [cm]", 0.0, 100.0, 5.0, 0.5)
		height_cm = st.number_input("Height (cylinder) [cm]", 1.0, 200.0, 10.0, 0.5)

		st.subheader("Materials")
		fuel_enrichment_wt_pct = st.slider("U-235 enrichment [wt%]", 0.2, 20.0, 5.0, 0.1)
		fuel_density_gcc = st.number_input("Fuel density [g/cc]", 1.0, 20.0, 10.5, 0.1)
		moderator = st.selectbox("Moderator", ["water", "graphite", "heavy_water", "none"], index=0)
		moderator_opt = None if moderator == "none" else moderator

		st.subheader("Source")
		source_type = st.selectbox(
			"Source type",
			["dd", "dt", "watt_u235", "watt_pu239", "watt_custom", "maxwell"],
			index=0,
		)
		watt_a_eV = watt_b_inv_eV = maxwell_kT_eV = None
		if source_type == "watt_custom":
			watt_a_eV = st.number_input("Watt a [eV]", 1e5, 5e6, 9.88e5, 1e4, format="%e")
			watt_b_inv_eV = st.number_input("Watt b [1/eV]", 1e-7, 1e-5, 2.249e-6, 1e-7, format="%e")
		elif source_type == "maxwell":
			maxwell_kT_eV = st.number_input("Maxwell kT [eV]", 1.0, 1e6, 2.0e3, 1.0, format="%e")

		st.subheader("Run")
		particles = int(st.number_input("Particles", 1000, 5_000_000, 50_000, 1000))
		batches = int(st.number_input("Batches", 1, 10_000, 50, 1))
		do_run = st.button("Run OpenMC")

	else:
		st.subheader("Fusor geometry (cylindrical)")
		# Preset for thermal LEU (compact)
		if st.button("Apply preset: Thermal (LEU, compact)", key="apply_preset_thermal_leu"):
			st.session_state["height_cm"] = 50.0
			st.session_state["vacuum_radius_cm"] = 5.0
			st.session_state["moderator_thickness_cm"] = 10.0
			st.session_state["fuel_thickness_cm"] = 20.0
			st.session_state["reflector_thickness_cm"] = 15.0
			st.session_state["shielding_thickness_cm"] = 10.0

			st.session_state["moderator_material"] = "heavy_water"
			st.session_state["fuel_material"] = "uo2"
			st.session_state["fuel_enrichment_wt_pct"] = 10.0
			st.session_state["fuel_density_gcc"] = 10.5
			st.session_state["reflector_material"] = "beryllium"
			st.session_state["shielding_material"] = "lead"

			st.session_state["source_type"] = "dt"
			st.session_state["fusor_voltage_kV"] = 50.0
			st.session_state["fusor_current_mA"] = 15.0
			st.session_state["fusor_pressure_mTorr"] = 1.0

			st.session_state["particles"] = 100000
			st.session_state["batches"] = 80

			try:
				st.rerun()
			except Exception:
				st.experimental_rerun()
		# Preset for fast DU (low-cost)
		if st.button("Apply preset: Fast (DU, low-cost)", key="apply_preset_fast_du"):
			st.session_state["height_cm"] = 80.0
			st.session_state["vacuum_radius_cm"] = 5.0
			st.session_state["moderator_thickness_cm"] = 0.0
			st.session_state["fuel_thickness_cm"] = 35.0
			st.session_state["reflector_thickness_cm"] = 25.0
			st.session_state["shielding_thickness_cm"] = 10.0

			st.session_state["moderator_material"] = "none"
			st.session_state["fuel_material"] = "uo2"
			st.session_state["fuel_enrichment_wt_pct"] = 0.2
			st.session_state["fuel_density_gcc"] = 10.5
			st.session_state["reflector_material"] = "steel"
			st.session_state["shielding_material"] = "lead"

			st.session_state["source_type"] = "dt"
			st.session_state["fusor_voltage_kV"] = 40.0
			st.session_state["fusor_current_mA"] = 10.0
			st.session_state["fusor_pressure_mTorr"] = 2.0

			st.session_state["particles"] = 100000
			st.session_state["batches"] = 80

			try:
				st.rerun()
			except Exception:
				st.experimental_rerun()

		height_cm = st.number_input("Active height [cm]", 1.0, 1000.0, 50.0, 1.0, key="height_cm")
		vacuum_radius_cm = st.number_input("Vacuum radius [cm]", 0.5, 200.0, 5.0, 0.5, key="vacuum_radius_cm")
		moderator_thickness_cm = st.number_input("Moderator thickness [cm]", 0.0, 200.0, 10.0, 0.5, key="moderator_thickness_cm")
		fuel_thickness_cm = st.number_input("Fuel thickness [cm]", 0.0, 200.0, 20.0, 0.5, key="fuel_thickness_cm")
		reflector_thickness_cm = st.number_input("Reflector thickness [cm]", 0.0, 200.0, 15.0, 0.5, key="reflector_thickness_cm")
		shielding_thickness_cm = st.number_input("Shielding thickness [cm]", 0.0, 200.0, 10.0, 0.5, key="shielding_thickness_cm")

		st.subheader("Materials")
		moderator_material = st.selectbox("Moderator", ["water", "graphite", "heavy_water", "none"], index=2, key="moderator_material")  # Heavy water default
		fuel_material = st.selectbox("Fuel", ["uo2", "u_metal", "th_o2", "th_metal"], index=0, key="fuel_material")
		fuel_enrichment_wt_pct = st.slider("Fuel U-235 enrichment [wt%]", 0.2, 20.0, 10.0, 0.1, key="fuel_enrichment_wt_pct")  # 10% default
		fuel_density_gcc = st.number_input("Fuel density [g/cc]", 1.0, 20.0, 10.5, 0.1, key="fuel_density_gcc")
		reflector_material = st.selectbox("Reflector", ["graphite", "beryllium", "steel", "tungsten_carbide", "gold", "water", "heavy_water", "none"], index=1, key="reflector_material")  # Beryllium default
		shielding_material = st.selectbox("Shielding", ["lead", "graphite", "steel", "water", "heavy_water", "none"], index=0, key="shielding_material")

		st.subheader("Source (fusion or generator)")
		source_type = st.selectbox(
			"Source type",
			["pp", "dd", "dt", "dt_generator"],
			index=2,
			key="source_type",
			help=(
				"p-p (H-H): Proton-proton fusion, very low neutron yield\n"
				"D-D: Deuterium fusion, ~2.45 MeV neutrons, moderate cross section\n"
				"D-T: Deuterium-tritium fusion, ~14.1 MeV neutrons, highest cross section\n"
				"D-T sealed-tube generator: forward-peaked 14 MeV spectrum"
			)
		)
		watt_a_eV = watt_b_inv_eV = maxwell_kT_eV = None
		if source_type == "watt_custom":
			watt_a_eV = st.number_input("Watt a [eV]", 1e5, 5e6, 9.88e5, 1e4, format="%e")
			watt_b_inv_eV = st.number_input("Watt b [1/eV]", 1e-7, 1e-5, 2.249e-6, 1e-7, format="%e")
		elif source_type == "maxwell":
			maxwell_kT_eV = st.number_input("Maxwell kT [eV]", 1.0, 1e6, 2.0e3, 1.0, format="%e")

		st.caption("Electrical/source assumptions drive power estimates")
		if source_type == "dt_generator":
			st.info("Sealed-tube D-T generator: enter nameplate neutron rate below.")
			calc_neutron_rate = 1e8
		else:
			fusor_voltage_kV = st.number_input("Fusor voltage [kV]", 1.0, 200.0, 50.0, 1.0, key="fusor_voltage_kV")
			fusor_current_mA = st.number_input("Fusor current [mA]", 0.1, 500.0, 20.0, 0.1, key="fusor_current_mA")
			fusor_pressure_mTorr = st.number_input("Operating pressure [mTorr]", 0.001, 10.0, 1.0, 0.1, key="fusor_pressure_mTorr")
			# Calculate realistic neutron rate based on fusion physics
			calc_neutron_rate = neutron_yield_rate(source_type, fusor_voltage_kV, fusor_current_mA, fusor_pressure_mTorr)
		
		# Show realistic estimates and typical values
		st.caption("Neutron production estimates:")
		col1, col2 = st.columns(2)
		with col1:
			st.metric("Calculated rate", f"{calc_neutron_rate:.2e} n/s")
		with col2:
			typical_rates = {"pp": "~10²-10⁴", "dd": "~10⁶-10⁸", "dt": "~10⁸-10¹⁰"}
			st.metric("Typical amateur fusor", f"{typical_rates.get(source_type, '?')} n/s")
		
		# Allow override with reasonable default
		default_rate = max(calc_neutron_rate, 1e6 if source_type == "dd" else 1e8 if source_type == "dt" else 1e3)
		source_rate_n_per_s = st.number_input(
			"Neutron source rate [n/s] (override if needed)", 
			min_value=1.0,  # Allow any positive value
			max_value=1e15, 
			value=default_rate, 
			step=1e3,
			format="%e",
			help="Use calculated value or override with experimental/measured rate",
			key="source_rate_n_per_s",
		)

		st.subheader("Run & Source Layout")
		col_run1, col_run2 = st.columns(2)
		with col_run1:
			particles = int(st.number_input("Particles", 1000, 5_000_000, 100_000, 1000, key="particles"))
			batches = int(st.number_input("Batches", 1, 10_000, 80, 1, key="batches"))
			compute_keff = st.checkbox("Also compute k_eff (diagnostic eigenvalue run)", value=False, key="compute_keff",
				help="Runs the same geometry in eigenvalue mode to report k_eff and multiplication M≈1/(1−k_eff).")
		with col_run2:
			n_sources_azimuthal = int(st.number_input("Azimuthal sources (ring)", 1, 32, 6, 1, key="n_sources_azimuthal"))
			n_sources_axial = int(st.number_input("Axial sources", 1, 32, 3, 1, key="n_sources_axial"))
			source_ring_radius_fraction = float(st.slider("Source ring radius (fraction of inner radius)", 0.1, 0.99, 0.9, 0.01, key="source_ring_radius_fraction"))
			do_run = st.button("Run OpenMC")


st.write("Environment")
xs = os.environ.get("OPENMC_CROSS_SECTIONS")
if not xs:
	default_xs = os.path.join(os.getcwd(), "openmc-data", "cross_sections.xml")
	if os.path.exists(default_xs):
		os.environ["OPENMC_CROSS_SECTIONS"] = default_xs
		xs = default_xs
st.code(f"OPENMC_CROSS_SECTIONS={xs}")
if not xs:
	st.warning("OPENMC_CROSS_SECTIONS is not set. Place data under openmc-data/ or set the path above.")


if do_run:
	try:
		if mode == "Simple (toy)":
			model = build_fixed_source_model(
				geometry_kind=geometry_kind,
				fuel_enrichment_wt_pct=fuel_enrichment_wt_pct,
				fuel_density_gcc=fuel_density_gcc,
				moderator=moderator_opt,
				fuel_radius_cm=fuel_radius_cm,
				moderator_thickness_cm=moderator_thickness_cm,
				add_reflector=add_reflector,
				reflector_thickness_cm=reflector_thickness_cm,
				height_cm=height_cm,
				source_type=source_type,
				maxwell_kT_eV=maxwell_kT_eV,
				watt_a_eV=watt_a_eV,
				watt_b_inv_eV=watt_b_inv_eV,
			)
		else:
			model = build_fusor_cylindrical_model(
				height_cm=height_cm,
				vacuum_radius_cm=vacuum_radius_cm,
				moderator_thickness_cm=moderator_thickness_cm,
				fuel_thickness_cm=fuel_thickness_cm,
				reflector_thickness_cm=reflector_thickness_cm,
				shielding_thickness_cm=shielding_thickness_cm,
				moderator_material=moderator_material,
				fuel_material=fuel_material,
				fuel_enrichment_wt_pct=fuel_enrichment_wt_pct,
				fuel_density_gcc=fuel_density_gcc,
				reflector_material=reflector_material,
				shielding_material=shielding_material,
				source_type=source_type,
				maxwell_kT_eV=maxwell_kT_eV,
				watt_a_eV=watt_a_eV,
				watt_b_inv_eV=watt_b_inv_eV,
				# DU optimization knobs
				inner_reflector_material="beryllium" if moderator_material in ("none", None) else None,
				inner_reflector_thickness_cm=3.0 if moderator_material in ("none", None) else 0.0,
				structural_material="tungsten_carbide",
				structural_thickness_cm=5.0,
				gamma_shield_material="lead",
				gamma_shield_thickness_cm=10.0,
				end_reflector_thickness_cm=10.0,
				n_sources_azimuthal=n_sources_azimuthal,
				n_sources_axial=n_sources_axial,
				source_ring_radius_fraction=source_ring_radius_fraction,
			)

		# Run with progress bar
		progress_bar = st.progress(0)
		status_text = st.empty()
		
		def update_progress(current, total, message):
			progress = current / total
			progress_bar.progress(progress)
			status_text.text(message)
		
		sp = run_model(model, particles=particles, batches=batches, progress_callback=update_progress)
		progress_bar.empty()
		status_text.empty()

		st.success("Run complete")
		st.subheader("Tallies")
		try:
			t_flux = sp.get_tally(name="flux")
			st.write(t_flux)
		except Exception:
			st.info("No cell-aggregated flux tally available")

		try:
			t_ebin = sp.get_tally(name="flux_ebin")
			df = t_ebin.get_pandas_dataframe()
			if not df.empty:
				x = df["energy low [eV]"].to_numpy()
				y = df["mean"].to_numpy()
				fig, ax = plt.subplots()
				ax.step(x, y, where="post")
				ax.set_xscale("log")
				ax.set_yscale("log")
				ax.set_xlabel("Energy [eV]")
				ax.set_ylabel("Flux")
				ax.grid(True, which="both", ls=":", alpha=0.5)
				st.pyplot(fig)
		except Exception:
			st.info("No energy-binned flux tally available")

		# Multiplication diagnostics
		if mode == "Fusor cylinder":
			st.subheader("Multiplication & Power Analysis")
			col1, col2 = st.columns(2)
			
			with col1:
				try:
					tnu = sp.get_tally(name="nu_fission_fuel")
					nu_df = tnu.get_pandas_dataframe()
					if not nu_df.empty:
						fission_neutrons_per_source = float(nu_df['mean'].sum())
						# Estimate multiplication M = 1 + fission_neutrons_per_source
						M_estimate = 1 + fission_neutrons_per_source
						st.metric("Multiplication (M)", f"{M_estimate:.2f}")
						st.metric("Fission neutrons/source", f"{fission_neutrons_per_source:.3f}")
				except Exception:
					pass

		# Power estimates (fusor mode)
		if mode == "Fusor cylinder":
			height_m = float(height_cm) / 100.0
			P_elec_W = float(fusor_voltage_kV * 1e3 * fusor_current_mA * 1e-3)
			st.markdown(f"**Electrical input:** {P_elec_W:,.1f} W  |  {P_elec_W/height_m:,.1f} W/m")
			
			# Region-wise heating (normalized by source strength)
			# OpenMC heating score is in eV per source particle
			# Convert to watts: eV/particle * particles/sec * 1.602e-19 J/eV = W
			regions = ["vacuum", "moderator", "fuel", "reflector", "shield"]
			rows = []
			total_W = 0.0
			eV_to_J = 1.602176634e-19
			
			for rn in regions:
				try:
					t = sp.get_tally(name=f"heating_{rn}")
					d = t.get_pandas_dataframe()
					if not d.empty:
						# Get heating in eV per source particle
						heating_eV_per_particle = float(d['mean'].sum())
						std_eV_per_particle = float(d['std. dev.'].sum())
						
						# Convert to watts using actual source rate
						heating_W = heating_eV_per_particle * source_rate_n_per_s * eV_to_J
						std_W = std_eV_per_particle * source_rate_n_per_s * eV_to_J
						
						rows.append((rn, heating_W, std_W, heating_W/height_m))
						total_W += heating_W
				except Exception as e:
					pass
			
			if rows:
				import pandas as _pd
				df_heat = _pd.DataFrame(rows, columns=["Region", "Power [W]", "Std [W]", "W/m"])
				st.markdown("**Region heating** (neutron & gamma energy deposition)")
				st.dataframe(df_heat.style.format({
					"Power [W]": "{:.2e}",
					"Std [W]": "{:.2e}", 
					"W/m": "{:.2e}"
				}))
				st.markdown(f"**Total thermal power:** {total_W:.2e} W  ({total_W/height_m:.2e} W/m)")
				# Compare to fusion neutron power lower-bound
				try:
					E_n_eV = 2.45e6 if source_type == "dd" else 14.1e6 if source_type == "dt" else 1e5
					P_neutron_W = source_rate_n_per_s * E_n_eV * eV_to_J
					st.caption(f"Neutron power lower-bound (ignoring non-neutron channels): {P_neutron_W:.2e} W")
				except Exception:
					pass
			# Fuel fission power (kappa-fission)
			with col2:
				try:
					tk = sp.get_tally(name="kappa_fission_fuel")
					kappa_df = tk.get_pandas_dataframe()
					if not kappa_df.empty:
						# Convert from eV/particle to watts
						kappa_eV = float(kappa_df['mean'].sum())
						P_fiss_W = kappa_eV * source_rate_n_per_s * eV_to_J
						
						# Display in appropriate units
						if P_fiss_W < 1e-6:
							st.metric("Fuel fission power", f"{P_fiss_W*1e9:.1f} nW")
						elif P_fiss_W < 1e-3:
							st.metric("Fuel fission power", f"{P_fiss_W*1e6:.1f} μW")
						elif P_fiss_W < 1:
							st.metric("Fuel fission power", f"{P_fiss_W*1e3:.1f} mW")
						else:
							st.metric("Fuel fission power", f"{P_fiss_W:.2f} W")
						
						# Power gain
						E_n_eV = 14.1e6 if source_type == "dt" else 2.45e6 if source_type == "dd" else 1e5
						P_source = source_rate_n_per_s * E_n_eV * eV_to_J
						gain = P_fiss_W / P_source if P_source > 0 else 0
						st.metric("Power gain (fission/source)", f"{gain:.1f}×")
				except Exception:
					pass

		# Cross-section schematic (radial)
		if mode == "Fusor cylinder":
			st.markdown("### Reactor Cross-Section Visualization")
			
			fig2, (ax_side, ax_top) = plt.subplots(1, 2, figsize=(14, 6))
			
			# Calculate radii and properties for each region
			r0 = 0.0
			r1 = vacuum_radius_cm
			r2 = r1 + moderator_thickness_cm
			r3 = r2 + fuel_thickness_cm
			r4 = r3 + reflector_thickness_cm
			r5 = r4 + shielding_thickness_cm
			
			radii = [r1, r2, r3, r4, r5]
			labels = ["Vacuum\n(Fusor)", f"Moderator\n({moderator_material})", 
					 f"Fuel\n({fuel_material})", f"Reflector\n({reflector_material})", 
					 f"Shield\n({shielding_material})"]
			colors = ["lightgray", "skyblue", "orange", "lightgreen", "gray"]
			alphas = [0.3, 0.5, 0.7, 0.5, 0.6]
			
			# Side view (axial cross-section)
			ax_side.set_title("Side View - Axial Cross-Section", fontsize=12, fontweight='bold')
			ax_side.set_xlabel("Radius [cm]")
			ax_side.set_ylabel("Height [cm]")
			
			# Draw each cylindrical layer
			x_prev = 0.0
			for i, (r, lab, col, alpha) in enumerate(zip(radii, labels, colors, alphas)):
				if r > x_prev:  # Only draw if thickness > 0
					# Draw both sides of the cylinder
					rect_left = plt.Rectangle((-r, 0), r - x_prev, height_cm, 
											  facecolor=col, alpha=alpha, edgecolor='black', linewidth=0.5)
					rect_right = plt.Rectangle((x_prev, 0), r - x_prev, height_cm,
											   facecolor=col, alpha=alpha, edgecolor='black', linewidth=0.5)
					ax_side.add_patch(rect_left)
					ax_side.add_patch(rect_right)
					
					# Add labels
					if i == 0:  # Vacuum
						ax_side.text(0, height_cm/2, lab, ha='center', va='center', 
								   fontsize=9, fontweight='bold')
					else:
						ax_side.text((x_prev + r)/2, height_cm*0.9, lab.split('\n')[0], 
								   ha='center', va='center', fontsize=8)
				x_prev = r
			
			# Add centerline and dimensions
			ax_side.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
			ax_side.plot([0, r5], [height_cm*1.05, height_cm*1.05], 'k-', linewidth=0.5)
			ax_side.plot([0, 0], [height_cm*1.03, height_cm*1.07], 'k-', linewidth=0.5)
			ax_side.plot([r5, r5], [height_cm*1.03, height_cm*1.07], 'k-', linewidth=0.5)
			ax_side.text(r5/2, height_cm*1.08, f'{r5:.1f} cm', ha='center', fontsize=8)
			
			ax_side.set_xlim(-r5*1.1, r5*1.1)
			ax_side.set_ylim(-height_cm*0.05, height_cm*1.15)
			ax_side.grid(True, alpha=0.3)
			
			# Top view (radial cross-section)
			ax_top.set_title("Top View - Radial Cross-Section", fontsize=12, fontweight='bold')
			ax_top.set_xlabel("X [cm]")
			ax_top.set_ylabel("Y [cm]")
			
			# Draw concentric circles
			x_prev = 0.0
			for i, (r, lab, col, alpha) in enumerate(zip(radii, labels, colors, alphas)):
				if r > x_prev:
					circle = plt.Circle((0, 0), r, facecolor=col, alpha=alpha, 
									   edgecolor='black', linewidth=0.5)
					ax_top.add_patch(circle)
					
					# Add radial dimension lines
					angle = 45 * np.pi / 180
					ax_top.plot([x_prev*np.cos(angle), r*np.cos(angle)], 
							   [x_prev*np.sin(angle), r*np.sin(angle)], 
							   'k-', linewidth=0.3, alpha=0.5)
					
					# Add labels along diagonal
					mid_r = (x_prev + r) / 2
					ax_top.text(mid_r*np.cos(angle)*1.1, mid_r*np.sin(angle)*1.1, 
							   lab.split('\n')[0], fontsize=7, rotation=45)
				x_prev = r
			
			# Add neutron source indicator at center
			ax_top.plot(0, 0, 'r*', markersize=15, label=f'{source_type.upper()} fusion source')
			
			# Add dimension annotations
			ax_top.plot([0, r5], [0, 0], 'k-', linewidth=0.3, alpha=0.5)
			for r in radii:
				if r > 0:
					ax_top.plot([r, r], [-r5*0.02, r5*0.02], 'k-', linewidth=0.5)
					ax_top.text(r, -r5*0.08, f'{r:.1f}', ha='center', fontsize=7)
			
			ax_top.legend(loc='upper right', fontsize=9)
			ax_top.set_xlim(-r5*1.2, r5*1.2)
			ax_top.set_ylim(-r5*1.2, r5*1.2)
			ax_top.set_aspect('equal')
			ax_top.grid(True, alpha=0.3)
			
			plt.tight_layout()
			st.pyplot(fig2)

			# Optional k_eff diagnostic
			if compute_keff:
				st.subheader("k_eff Diagnostic")
				try:
					model_k = build_fusor_cylindrical_model(
						height_cm=height_cm,
						vacuum_radius_cm=vacuum_radius_cm,
						moderator_thickness_cm=moderator_thickness_cm,
						fuel_thickness_cm=fuel_thickness_cm,
						reflector_thickness_cm=reflector_thickness_cm,
						shielding_thickness_cm=shielding_thickness_cm,
						moderator_material=moderator_material,
						fuel_material=fuel_material,
						fuel_enrichment_wt_pct=fuel_enrichment_wt_pct,
						fuel_density_gcc=fuel_density_gcc,
						reflector_material=reflector_material,
						shielding_material=shielding_material,
						source_type=source_type,
						run_mode="eigenvalue",
					)
					# Short eigenvalue run (no progress meter for now)
					sp_k = run_model(model_k, particles=max(1000, particles//10), batches=max(20, batches//4))
					st.info("Eigenvalue run complete. Inspect OpenMC output for reported k_eff. Multiplication M ≈ 1/(1−k_eff).")
				except Exception as e:
					st.warning(f"k_eff diagnostic failed: {e}")

	except Exception as e:
		st.error(str(e))
		st.code(traceback.format_exc())


