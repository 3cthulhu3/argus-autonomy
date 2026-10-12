"""argus-autonomy: active wildfire-front estimation for a single UAV, in simulation."""

from .config import NOMINAL, WIND_MISMATCH, MissionConfig
from .mission import PLANNERS, MissionResult, make_scenario, run_mission

__all__ = ["MissionConfig", "NOMINAL", "WIND_MISMATCH", "PLANNERS",
           "run_mission", "make_scenario", "MissionResult"]
__version__ = "1.0.0"
