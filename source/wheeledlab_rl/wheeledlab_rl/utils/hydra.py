import functools
from typing import Dict
from collections.abc import Callable
import time

from typing import Any

try:
    import hydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import DictConfig, OmegaConf, MISSING
except ImportError:
    raise ImportError("Hydra is not installed. Please install it by running 'pip install hydra-core'.")

from isaaclab.utils.dict import update_class_from_dict
from isaaclab.envs.utils.spaces import replace_env_cfg_spaces_with_strings, replace_strings_with_env_cfg_spaces
from isaaclab.utils import replace_slices_with_strings, replace_strings_with_slices
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from wheeledlab_rl.configs import *
import wheeledlab_rl.configs as configs

cs = ConfigStore.instance()

def _consolidate_resolved_cfgs(run_cfg: RunConfig):
    assert run_cfg.env is not MISSING, "Environment configuration is missing"
    assert run_cfg.agent is not MISSING, "Agent configuration is missing"

    ####### MODIFY CONFIGS USING EXPOSED OVERRIDES ####### TODO: anyway to resolve these better?
    run_cfg.env.scene.num_envs = run_cfg.env_setup.num_envs
    run_cfg.env.scene.env_spacing = run_cfg.env_setup.env_spacing

    run_cfg.env.seed = run_cfg.agent.seed
    run_cfg.env.sim.device = run_cfg.train.device

    log = run_cfg.train.log
    if log.test_mode:
        log.no_log = True
        log.no_wandb = True
        log.video = False
        log.no_checkpoints = True


def rl_run_cfg_from_dict(run_cfg:DictConfig, run_config_name: str, cfg: Dict[str, Any], env_cfg_class=None, agent_cfg_class=None) -> RunConfig:
    '''Returns the RunConfig object from the dictionary representation of the configclass object. Used
    to recover @property values from composed configs
    TODO: implement for arbitrary configclasses
    '''
    tic = time.time()

    # Fill default run config with train, env, and agent of loaded config
    update_run_cfg: RunConfig = getattr(configs, run_config_name)() # default run config from module
    update_class_from_dict(update_run_cfg.train, run_cfg.train)
    update_run_cfg.env_setup.from_dict(cfg['env_setup'])
    update_run_cfg.agent_setup.from_dict(cfg['agent_setup'])
    update_run_cfg.train.from_dict(cfg['train'])

    # Construct configclasses for missing types
    if env_cfg_class:
        update_run_cfg.env = env_cfg_class()
        update_run_cfg.env.from_dict(cfg['env'])
    else:
        update_run_cfg.env = cfg['env']

    if agent_cfg_class:
        update_run_cfg.agent = agent_cfg_class()
        update_run_cfg.agent.from_dict(cfg['agent'])
    else:
        update_run_cfg.agent = cfg['agent']
    rl_run_cfg_from_dict_time = time.time() - tic

    print(f" rl_run_cfg_from_dict: {rl_run_cfg_from_dict_time:.4f} ")

    return update_run_cfg



def register_run_to_hydra(run_config_name: str, node: Any):
    """Load the configurations from the registry and update the Hydra configuration store."""
    # Initialize timers
    timers = {
        'total': time.time(),
        'store_initial_config': 0,
        'load_run_config': 0,
        'load_env_config': 0,
        'load_agent_config': 0,
        'replace_spaces': 0,
        'convert_to_dict': 0,
        'replace_slices_env': 0,
        'replace_slices_agent': 0,
        'store_final_config': 0
    }
    
    # register the task to Hydra
    tic = time.time()
    cs.store(name=run_config_name, node=node)
    timers['store_initial_config'] = time.time() - tic

    # load run configuration
    tic = time.time()
    run_cfg = cs.repo.get(run_config_name + ".yaml").node
    task_name = run_cfg.env_setup.task_name
    agent_cfg_entry_point = run_cfg.agent_setup.entry_point
    timers['load_run_config'] = time.time() - tic

    # load environment and agent configs
    tic = time.time()
    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    timers['load_env_config'] = time.time() - tic
    
    tic = time.time()
    agent_cfg = load_cfg_from_registry(task_name, agent_cfg_entry_point)
    timers['load_agent_config'] = time.time() - tic

    # replace gymnasium spaces with strings
    tic = time.time()
    replace_env_cfg_spaces_with_strings(env_cfg)
    timers['replace_spaces'] = time.time() - tic

    # convert configs to dictionary
    tic = time.time()
    env_cfg_dict = env_cfg.to_dict()
    if isinstance(agent_cfg, dict):
        agent_cfg_dict = agent_cfg
    else:
        agent_cfg_dict = agent_cfg.to_dict()
    timers['convert_to_dict'] = time.time() - tic

    # replace slices with strings
    tic = time.time()
    env_cfg_dict = replace_slices_with_strings(env_cfg_dict)
    timers['replace_slices_env'] = time.time() - tic
    
    tic = time.time()
    agent_cfg_dict = replace_slices_with_strings(agent_cfg_dict)
    timers['replace_slices_agent'] = time.time() - tic

    # store final configuration
    tic = time.time()
    run_cfg.env = env_cfg_dict
    run_cfg.agent = agent_cfg_dict
    timers['copy_cfg_dict'] = time.time() - tic

    tic = time.time()
    cs.store(name=run_config_name, node=run_cfg)
    timers['store_final_config'] = time.time() - tic

    # Calculate total time
    timers['total'] = time.time() - timers['total']

    # Print timing results
    print("\n Register Run to Hydra Configuration Registration Time Breakdown:")
    print(f"Total time: {timers['total']:.4f} seconds")
    print(f"  cs.store(name=run_config_name, node=node) : {timers['store_initial_config']:.4f} ")
    print(f"  Load run config: {timers['load_run_config']:.4f}")
    print(f"  env_cfg = load_cfg_from_registry: {timers['load_env_config']:.4f} ")
    print(f"  agent_cfg = load_cfg_from_registry: {timers['load_agent_config']:.4f}")
    print(f"  replace_env_cfg_spaces_with_strings(env_cfg): {timers['replace_spaces']:.4f} ")
    print(f"  env_cfg_dict = env_cfg.to_dict(): {timers['convert_to_dict']:.4f} ")
    print(f"  env_cfg_dict = replace_slices_with_strings: {timers['replace_slices_env']:.4f}")
    print(f"  agent_cfg_dict = replace_slices_with_strings: {timers['replace_slices_agent']:.4f}")
    print(f"  run_cfg.env = env_cfg_dict/agent_cfg_dict: {timers['copy_cfg_dict']:.4f} ")
    print(f"  cs.store(name=run_config_name, node=run_cfg): {timers['store_final_config']:.4f} ")

    return env_cfg, agent_cfg


# def hydra_run_config(run_config_name:str, node:Any, auto_resolve_conflicts=True) -> Callable:
def hydra_run_config(run_config_name:str, auto_resolve_conflicts=True) -> Callable:
    """Decorator to handle the Hydra configuration for a task.

    This decorator registers the task to Hydra and updates the environment and agent configurations from Hydra parsed
    command line arguments.

    Args:
        task_name: The name of the task.
        agent_cfg_entry_point: The entry point key to resolve the agent's configuration file.

    Returns:
        The decorated function with the envrionment's and agent's configurations updated from command line arguments.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # env_cfg, agent_cfg = register_run_to_hydra(run_config_name, node)
                
            tic = time.time()
            # Load configs from registries using run config name
            run_cfg = cs.repo.get(run_config_name + ".yaml").node
            if run_cfg is None:
                raise ValueError(f"Run config {run_config_name} not found in the Hydra registry.")
            run_cfg_time = time.time() - tic
            
            tic = time.time()
            task_name = run_cfg.env_setup.task_name
            task_name_time = time.time() - tic

            tic = time.time()
            env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
            env_load_cfg_from_registry_time = time.time() - tic

            tic = time.time()
            agent_cfg = load_cfg_from_registry(task_name, run_cfg.agent_setup.entry_point)
            agent_load_cfg_from_registry_time = time.time() - tic

            print("\n[INFO]: Wrapper Configuration Registration Time Breakdown:")

            print(f" run_cfg = cs.repo.get().node : {run_cfg_time:.4f}")
            print(f" task_name = run_cfg.env_setup.task_name : {task_name_time:.4f}")
            print(f" env_cfg = load_cfg_from_registry : {env_load_cfg_from_registry_time:.4f}")
            print(f" agent_cfg = load_cfg_from_registry : {agent_load_cfg_from_registry_time:.4f}")

            # define the new Hydra main function
            @hydra.main(config_name=run_config_name, version_base="1.3")
            def hydra_main(hydra_env_cfg: DictConfig, env_cfg=env_cfg, agent_cfg=agent_cfg,
                           run_cfg=run_cfg, run_config_name: str=run_config_name):

                tic = time.time()
                # convert to a native dictionary
                hydra_env_cfg = OmegaConf.to_container(hydra_env_cfg, resolve=True)
                to_container_time = time.time() - tic

                tic = time.time()
                # replace string with slices because OmegaConf does not support slices
                hydra_env_cfg = replace_strings_with_slices(hydra_env_cfg)
                replace_strings_slices_time = time.time() - tic

                tic = time.time()
                # update the configs with the Hydra command line arguments
                env_cfg.from_dict(hydra_env_cfg["env"])
                from_dict_time = time.time() - tic

                tic = time.time()
                # replace strings that represent gymnasium spaces because OmegaConf does not support them.
                # this must be done after converting the env configs from dictionary to avoid internal reinterpretations
                replace_strings_with_env_cfg_spaces(env_cfg)
                replace_strings_spaces_time = time.time() - tic

                tic = time.time()   
                # call the original function
                # run_cfg = node()
                # run_cfg._from_dict(hydra_env_cfg, env_cfg_class=env_cfg.__class__,
                #                    agent_cfg_class=agent_cfg.__class__)
                run_cfg = rl_run_cfg_from_dict(run_cfg, run_config_name, hydra_env_cfg, env_cfg_class=env_cfg.__class__,
                                          agent_cfg_class=agent_cfg.__class__)
                rl_from_dict_time = time.time() - tic

                tic = time.time()   
                # Resolve interdependencies between various config params (e.g. env.num_envs = env_setup.num_envs)
                if auto_resolve_conflicts:
                    _consolidate_resolved_cfgs(run_cfg)

                func(run_cfg, *args, **kwargs)
                resolve_time = time.time() - tic

                print(f" hydra_env_cfg = OmegaConf.to_container : {to_container_time:.4f}")
                print(f" hydra_env_cfg = replace_strings_with_slices : {replace_strings_slices_time:.4f}")
                print(f" env_cfg.from_dict() : {from_dict_time:.4f}")
                print(f" replace_strings_with_env_cfg_spaces(env_cfg) : {replace_strings_spaces_time:.4f}")
                print(f" run_cfg = rl_run_cfg_from_dict : {rl_from_dict_time:.4f}")
                print(f" if auto_resolve_conflicts : {resolve_time:.4f}")

            # call the new Hydra main function
            tic = time.time()
            hydra_main()
            hydra_time = time.time()-tic
            print(f" hydra_main() : {hydra_time:.4f}")

        return wrapper

    return decorator
