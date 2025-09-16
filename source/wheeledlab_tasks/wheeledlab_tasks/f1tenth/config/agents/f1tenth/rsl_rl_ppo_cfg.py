from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class F1TenthTimeTrialPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Training hyperparameters (adjust as needed)
    num_steps_per_env = 32 # 16 for 4096, 32 for 2048, 64 for 1024
    max_iterations = 1500
    save_interval = 50
    experiment_name = "ppo_f1tenth"
    empirical_normalization = False  # Disable if using proprioceptive-only obs

    # Policy architecture
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[512, 512, 256],
        activation="elu",
    )

    # PPO algorithm parameters
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=10,
        num_mini_batches=64,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
 

@configclass
class F1TenthOvertakePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Training hyperparameters (adjust as needed)
    num_steps_per_env = 16
    max_iterations = 1500
    save_interval = 50
    experiment_name = "ppo_f1tenth"
    empirical_normalization = False  # Disable if using proprioceptive-only obs

    # Policy architecture
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[512, 512],
        activation="elu",
    )

    # PPO algorithm parameters
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.02,
        num_learning_epochs=10,
        num_mini_batches=64,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
       
################# rule of thumb => num_steps_per_env * num_envs == 65,536 batch size (e.g. 4096*16)
################# For 4096 env 

# class F1TenthPPORunnerCfg(RslRlOnPolicyRunnerCfg):
#     # Training hyperparameters (adjust as needed)
#     num_steps_per_env = 16
#     max_iterations = 1500
#     save_interval = 50
#     experiment_name = "ppo_f1tenth"
#     empirical_normalization = False  # Disable if using proprioceptive-only obs

#     # Policy architecture
#     policy = RslRlPpoActorCriticCfg(
#         init_noise_std=1.0,
#         actor_hidden_dims=[256, 256],
#         critic_hidden_dims=[256, 256],
#         activation="elu",
#     )

#     # PPO algorithm parameters
#     algorithm = RslRlPpoAlgorithmCfg(
#         value_loss_coef=1.0,
#         use_clipped_value_loss=True,
#         clip_param=0.2,
#         entropy_coef=0.01,
#         num_learning_epochs=10,
#         num_mini_batches=64,
#         learning_rate=3.0e-4,
#         schedule="adaptive",
#         gamma=0.99,
#         lam=0.95,
#         desired_kl=0.01,
#         max_grad_norm=1.0,
#     )
