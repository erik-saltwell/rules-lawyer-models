import os

import wandb
from rules_lawyer_models.training import TrainingOptions, TrainingRunConfiguration


def initialize_wandb(run_configuration: TrainingRunConfiguration, training_options: TrainingOptions) -> wandb.Run:
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    os.environ["WANDB_PROJECT"] = run_configuration.target_model_name
    wandb.login()
    wandb_config = {
        "training_options": training_options.to_dict(),
        "run_configuration": run_configuration.to_dict(),
    }
    return wandb.init(project=run_configuration.target_model_name, config=wandb_config)
