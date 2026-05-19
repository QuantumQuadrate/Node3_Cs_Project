"""K10CR1 Thorlabs waveplate rotator control functions.

Wrapper functions to call NDSP methods from ARTIQ kernels via RPC.
Relocated from archive/Devices_and_config/External Devices/Waveplates/Node3_K10CR1/.

Usage in experiments::

    from subroutines.k10cr1_functions import move_to_target_deg
    # ...
    move_to_target_deg(self, name="852_QWP", target_deg=45.0)
"""

from artiq.experiment import *


_NDSP_ERROR = (
    "K10CR1 NDSP not reachable. Start the launcher:\n"
    "    python repository/ndsp/k10cr1/launcher_multi_rotor.py"
)

_NDSP_EXC = (ConnectionRefusedError, ConnectionResetError, OSError)


def _ndsp(self):
    """Return the NDSP handle or raise with a helpful message.

    `k10cr1_ndsp` is registered by device_aliases.register_devices() and
    silently skipped if the device isn't in device_db.py — so a missing
    attribute means the NDSP wasn't reachable at build time.
    """
    ndsp = getattr(self, "k10cr1_ndsp", None)
    if ndsp is None:
        raise RuntimeError(_NDSP_ERROR)
    return ndsp


@rpc
def get_rotator_position(self, name: TStr) -> TFloat:
    """Return rotator position in device units."""
    try:
        return _ndsp(self).get_position(name)
    except _NDSP_EXC as e:
        raise RuntimeError(_NDSP_ERROR) from e


@rpc
def get_rotator_deg(self, name: TStr) -> TFloat:
    """Return rotator position in degrees."""
    try:
        pos = _ndsp(self).get_position(name)
    except _NDSP_EXC as e:
        raise RuntimeError(_NDSP_ERROR) from e
    return pos / self.deg_to_pos


@rpc
def move_to_target_deg(self, name: TStr, target_deg):
    """Move rotator to absolute position in degrees.

    Blocks until move is complete (wait_move is built into move_to).
    """
    target_pos = int(target_deg * self.deg_to_pos)
    try:
        _ndsp(self).move_to(target_pos, name)
    except _NDSP_EXC as e:
        raise RuntimeError(_NDSP_ERROR) from e


@rpc
def move_by_deg(self, name: TStr, target_deg):
    """Move rotator by relative degrees from current position."""
    target_pos = int(target_deg * self.deg_to_pos)
    try:
        _ndsp(self).move_by(target_pos, name)
    except _NDSP_EXC as e:
        raise RuntimeError(_NDSP_ERROR) from e


@rpc
def is_rotator_moving(self, name: TStr) -> TBool:
    """Check if rotator is currently moving."""
    try:
        return _ndsp(self).is_moving(name)
    except _NDSP_EXC as e:
        raise RuntimeError(_NDSP_ERROR) from e


@rpc
def go_to_home(self, name: TStr):
    """Home the rotator (blocks until complete)."""
    try:
        _ndsp(self).home(name)
    except _NDSP_EXC as e:
        raise RuntimeError(_NDSP_ERROR) from e


@rpc
def wait_move(self, name: TStr):
    """Wait for rotator move to complete."""
    try:
        _ndsp(self).wait_move(name)
    except _NDSP_EXC as e:
        raise RuntimeError(_NDSP_ERROR) from e
