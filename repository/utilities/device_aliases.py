"""Device alias management for SNAQ-Node-1 Cs experiment.

Reads config/device_aliases.json and creates human-readable attribute
aliases on the experiment object, so experiments can use e.g.
``self.dds_FORT`` instead of ``self.urukul1_ch1``.
"""

import json
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "config"


def load_device_aliases():
    """Load the device aliases JSON config."""
    with open(CONFIG_DIR / "device_aliases.json", "r") as f:
        return json.load(f)


def register_devices(experiment):
    """Register all hardware devices and create alias attributes.

    Call this from BaseExperiment.build(). It:
      1. Calls setattr_device() for every device referenced in the config.
      2. Creates alias attributes (e.g. experiment.dds_FORT -> experiment.urukul1_ch1).
      3. Registers CPLDs, samplers, zotino, and core devices.

    Parameters
    ----------
    experiment : EnvExperiment
        The ARTIQ experiment instance.
    """
    config = load_device_aliases()

    # --- Core devices ---
    experiment.setattr_device("core")
    experiment.setattr_device("core_dma")
    experiment.setattr_device("core_cache")

    # --- DAC ---
    experiment.setattr_device("zotino0")

    # --- ADC ---
    experiment.setattr_device("sampler0")

    # --- DDS channels ---
    registered_cplds = set()
    for alias, info in config["dds_aliases"].items():
        device_name = info["device"]
        cpld_name = info["cpld"]

        experiment.setattr_device(device_name)

        # Create alias: experiment.dds_FORT = experiment.urukul1_ch1
        setattr(experiment, alias, getattr(experiment, device_name))

        # Register CPLD once per board
        if cpld_name not in registered_cplds:
            experiment.setattr_device(cpld_name)
            registered_cplds.add(cpld_name)

    # --- TTL channels ---
    for alias, info in config["ttl_aliases"].items():
        device_name = info["device"]
        experiment.setattr_device(device_name)
        setattr(experiment, alias, getattr(experiment, device_name))

    # --- Edge counters for SPCM TTLs ---
    experiment.setattr_device("ttl0_counter")
    experiment.setattr_device("ttl1_counter")

    # --- K10CR1 waveplate rotators (NDSP) ---
    try:
        experiment.setattr_device("k10cr1_ndsp")
    except Exception:
        pass  # NDSP may not be running

    # Store config on experiment for runtime access
    experiment._device_config = config
