from collections import namedtuple

Cosmology = namedtuple(
    "Cosmology",
    [
        "H0",  # Hubble constant at z=0 in km/s/Mpc
        "ombh2",  # Baryon density parameter at z=0
        "omch2",  # Cold dark matter density parameter at z=0
        "omlh2",  # Dark energy density parameter at z=0
        "omk",  # Curvature density parameter at z=0
        "tau",  # Optical depth to reionization
    ],
)


def cl(cosmology, ell):
    """
    Calculate the angular power spectrum C_l for a given cosmology
    and multipole moment l.

    Parameters:
    cosmology (Cosmology): Cosmological parameters.
    ell (int): Multipole moment.

    Returns:
    float: The angular power spectrum C_l.
    """
    # Placeholder for actual computation
    # This should involve using the cosmological parameters to compute C_l
    # For now, we return a dummy value
    return cosmology.As * (ell / 1000.0) ** (-cosmology.ns)
