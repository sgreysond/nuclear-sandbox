import os
import sys


def main() -> int:
    try:
        import openmc_data_downloader as odd
    except Exception as exc:
        sys.stderr.write("openmc_data_downloader is not installed in the environment.\n")
        return 1

    dest_root = os.path.join(os.getcwd(), "openmc-data")
    os.makedirs(dest_root, exist_ok=True)

    # Try ENDF/B-VIII.0 first, fall back to VII.1
    lib_candidates = [
        os.environ.get("XS_LIB", "ENDFB-8.0-NNDC"),
        "ENDFB-7.1-NNDC",
    ]

    xs_path = None
    last_err = None
    for lib in lib_candidates:
        try:
            odd.just_in_time_library_generator(
                libraries=[lib],
                particles=["neutron"],
                destination=dest_root,
                set_OPENMC_CROSS_SECTIONS=False,
                # Minimal useful set for our starter models (add Be, Pb, Fe, W, Au, Th)
                elements=["U", "Th", "O", "H", "C", "Be", "Pb", "Fe", "W", "Au"],
            )
            # Find cross_sections.xml under dest_root
            for dirpath, dirnames, filenames in os.walk(dest_root):
                if "cross_sections.xml" in filenames:
                    xs_path = os.path.join(dirpath, "cross_sections.xml")
                    break
            if xs_path:
                break
        except Exception as e:  # try next library
            last_err = e
            continue

    if not xs_path:
        if last_err:
            sys.stderr.write(f"Failed to obtain cross sections: {last_err}\n")
        else:
            sys.stderr.write("Failed to locate cross_sections.xml after download.\n")
        return 2

    # Persist the discovered path for the calling shell
    with open(os.path.join(os.getcwd(), ".openmc_xs_path"), "w") as f:
        f.write(xs_path)

    # Also print for convenience
    print(xs_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())


