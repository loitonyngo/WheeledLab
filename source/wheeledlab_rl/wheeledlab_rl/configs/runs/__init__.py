from .rss_cfgs import *

from wheeledlab_rl.utils.hydra import register_run_to_hydra

register_run_to_hydra("RSS_TIMETRIAL", RSS_TIMETRIAL)
register_run_to_hydra("RSS_OVERTAKE", RSS_OVERTAKE)

# register_run_to_hydra("RSS_DRIFT_CONFIG", RSS_DRIFT_CONFIG)
# register_run_to_hydra("RSS_ELEV_CONFIG", RSS_ELEV_CONFIG)
# register_run_to_hydra("RSS_VISUAL_CONFIG", RSS_VISUAL_CONFIG)