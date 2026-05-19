"""Structured calibration reference management for AOM feedback.

Replaces the flat ref_voltages_5.txt with a JSON-based store that supports:
  - Named profiles (standard, FORT_low, FORT_high, blowaway, etc.)
  - Partial profiles that override only specific channels
  - Timestamps for staleness tracking
  - Safe save with overwrite protection

Usage::

    store = CalibrationStore()
    refs = store.get_refs("standard")          # -> [v0, v1, v2, v3, v4]
    store.save_refs("standard", {"FORT": 0.093, "MOT_X": 0.035})
    age = store.get_ref_age("standard")        # -> timedelta or None
"""

import json
import datetime
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "config"
CAL_PATH = CONFIG_DIR / "calibration_references.json"

# Canonical channel order (matches sampler channels 0-4)
CHANNEL_ORDER = ["MOT_X", "MOT_Y", "MOT_Z", "FORT", "excitation"]


class CalibrationStore:
    """Manage calibration reference profiles for AOM feedback."""

    def __init__(self, path=None):
        self.path = Path(path) if path else CAL_PATH
        self._data = self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path, "r") as f:
                return json.load(f)
        return {"active_profile": "standard", "profiles": {}}

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=4)

    @property
    def active_profile(self):
        return self._data.get("active_profile", "standard")

    @active_profile.setter
    def active_profile(self, name):
        if name not in self._data["profiles"]:
            raise KeyError(f"Profile '{name}' does not exist. "
                           f"Available: {self.list_profiles()}")
        self._data["active_profile"] = name
        self._save()

    def list_profiles(self):
        """Return list of all profile names."""
        return list(self._data["profiles"].keys())

    def get_profile_info(self, profile=None):
        """Return profile metadata (description, calibrated_at)."""
        profile = profile or self.active_profile
        p = self._data["profiles"].get(profile, {})
        return {
            "description": p.get("description", ""),
            "calibrated_at": p.get("calibrated_at"),
        }

    def get_refs(self, profile=None):
        """Get reference voltages as a list of 5 floats [MOT_X, MOT_Y, MOT_Z, FORT, exc].

        For partial profiles, missing channels fall back to the active profile.
        Returns -1.0 for uncalibrated channels.
        """
        profile = profile or self.active_profile
        base_profile = self.active_profile

        # Start with base profile values
        base = self._data["profiles"].get(base_profile, {})
        base_channels = base.get("channels", {})

        # Override with requested profile if different
        target = self._data["profiles"].get(profile, {})
        target_channels = target.get("channels", {})

        refs = []
        for ch_name in CHANNEL_ORDER:
            if ch_name in target_channels and "ref_voltage" in target_channels[ch_name]:
                refs.append(target_channels[ch_name]["ref_voltage"])
            elif ch_name in base_channels and "ref_voltage" in base_channels[ch_name]:
                refs.append(base_channels[ch_name]["ref_voltage"])
            else:
                refs.append(-1.0)

        return refs

    def get_ref(self, channel, profile=None):
        """Get a single channel's reference voltage."""
        profile = profile or self.active_profile
        p = self._data["profiles"].get(profile, {})
        channels = p.get("channels", {})
        if channel in channels and "ref_voltage" in channels[channel]:
            return channels[channel]["ref_voltage"]
        # Fallback to active profile
        if profile != self.active_profile:
            return self.get_ref(channel, self.active_profile)
        return -1.0

    def save_refs(self, voltages_dict, profile=None):
        """Save reference voltages to a profile.

        Parameters
        ----------
        voltages_dict : dict
            Maps channel name -> voltage. e.g. {"FORT": 0.093, "MOT_X": 0.035}
        profile : str or None
            Target profile. Defaults to active profile.
        """
        profile = profile or self.active_profile

        if profile not in self._data["profiles"]:
            self._data["profiles"][profile] = {
                "description": "",
                "calibrated_at": None,
                "channels": {},
            }

        p = self._data["profiles"][profile]

        for ch_name, voltage in voltages_dict.items():
            if ch_name not in p["channels"]:
                p["channels"][ch_name] = {}
            p["channels"][ch_name]["ref_voltage"] = voltage

        p["calibrated_at"] = datetime.datetime.now().isoformat()
        self._save()

    def save_refs_list(self, voltages_list, profile=None):
        """Save reference voltages from a flat list [MOT_X, MOT_Y, MOT_Z, FORT, exc].

        Parameters
        ----------
        voltages_list : list of float
            5 reference voltages in canonical channel order.
        profile : str or None
            Target profile. Defaults to active profile.
        """
        d = {}
        for i, ch_name in enumerate(CHANNEL_ORDER):
            if i < len(voltages_list):
                d[ch_name] = voltages_list[i]
        self.save_refs(d, profile=profile)

    def update_ref(self, channel, voltage, profile=None):
        """Update a single channel's reference voltage."""
        self.save_refs({channel: voltage}, profile=profile)

    def get_ref_age(self, profile=None):
        """Return age of calibration as timedelta, or None if never calibrated."""
        profile = profile or self.active_profile
        p = self._data["profiles"].get(profile, {})
        cal_time = p.get("calibrated_at")
        if cal_time is None:
            return None
        try:
            cal_dt = datetime.datetime.fromisoformat(cal_time)
            return datetime.datetime.now() - cal_dt
        except Exception:
            return None

    def is_stale(self, profile=None, max_hours=24):
        """Check if calibration is older than max_hours."""
        age = self.get_ref_age(profile)
        if age is None:
            return True  # Never calibrated = stale
        return age.total_seconds() > max_hours * 3600

    def create_profile(self, name, description="", channels=None):
        """Create a new calibration profile."""
        if name in self._data["profiles"]:
            raise KeyError(f"Profile '{name}' already exists.")
        self._data["profiles"][name] = {
            "description": description,
            "calibrated_at": None,
            "channels": channels or {},
        }
        self._save()

    def print_summary(self):
        """Print a formatted summary of all profiles."""
        print(f"Active profile: {self.active_profile}")
        print(f"{'Profile':<20} {'Calibrated':<25} {'Description'}")
        print("-" * 75)
        for name in self.list_profiles():
            info = self.get_profile_info(name)
            cal = info["calibrated_at"] or "NEVER"
            marker = " *" if name == self.active_profile else ""
            print(f"{name + marker:<20} {cal:<25} {info['description']}")
        print()
        print(f"{'Channel':<15} " + " ".join(f"{p:<12}" for p in self.list_profiles()))
        print("-" * (15 + 13 * len(self.list_profiles())))
        for ch in CHANNEL_ORDER:
            vals = []
            for p in self.list_profiles():
                v = self.get_ref(ch, p)
                vals.append(f"{v:>10.4f}  " if v >= 0 else "  uncal.    ")
            print(f"{ch:<15} " + " ".join(vals))
