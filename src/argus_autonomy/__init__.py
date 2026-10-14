"""argus-autonomy: active wildfire-front estimation for a single UAV, in simulation."""

from .config import NOMINAL, WIND_MISMATCH, MissionConfig
from .mission import PLANNERS, MissionResult, make_scenario, run_mission

__all__ = [
           "NOMINAL",
           "PLANNERS",
           "WIND_MISMATCH",
           "MissionConfig",
           "MissionResult",
           "make_scenario",
           "run_mission",
]
__version__ = "1.1.0"
