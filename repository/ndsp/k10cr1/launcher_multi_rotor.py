#!/usr/bin/env python3
"""Launch the K10CR1 NDSP server for SNAQ-Node-3 waveplates.

Usage:
  1. Activate ARTIQ environment (conda activate artiq7)
  2. cd useful_codes/ndsp/k10cr1/
  3. python launcher_multi_rotor.py

Prerequisites:
  pip install sipyco pylablib
"""

# you can install sipyco via nix, or from the github using pip
from sipyco.pc_rpc import simple_server_loop
from pylablib.devices import Thorlabs  # for Kinesis instrument control

import k10cr1_driver as k10cr1_driver

import sys
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Import device_db from project root (useful_codes/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from device_db import device_db


def main():
    logger.info("Starting K10CR1 NDSP server")
    # network settings
    device_name = "k10cr1_ndsp"
    host = device_db[device_name]["host"]
    port = device_db[device_name]["port"]

    sn_list = device_db[device_name]["sn_list"]
    nicknames = device_db[device_name]["nickname_list"]
    devices = [(name, {'conn':sn}) for name,sn in zip(nicknames,sn_list)]

    # instantiate driver object
    driver = k10cr1_driver.K10CR1_NDSP_Driver(devices)

    try:
        simple_server_loop(
            {device_name: driver},
            host,
            port
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
     main()
