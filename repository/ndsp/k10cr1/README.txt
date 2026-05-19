K10CR1 Waveplate Rotator NDSP Server
=====================================

Prerequisites:
  pip install sipyco pylablib

Setup:
  1. Connect K10CR1 rotators via USB
  2. Verify serial numbers match device_db.py:
     - 852_QWP: SN 55000617
     - 852_HWP: SN 55422734
  3. If scale units show "step", use APT software to configure
     (deg_to_pos = 136533)

Launch:
  1. Activate ARTIQ environment (conda activate artiq7)
  2. cd useful_codes/ndsp/k10cr1/
  3. python launcher_multi_rotor.py
  4. Server runs on localhost:8083

In experiment code:
  from subroutines.k10cr1_functions import move_to_target_deg
  # In build():
  self.setattr_device("k10cr1_ndsp")
  # In kernel:
  move_to_target_deg(self, name="852_QWP", target_deg=45.0)
