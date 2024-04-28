import wandb
import time
import wandb
import multiprocessing
import threading


def log_to_wandb(log_info_dict):

    wandb.init(
        project="illujaxv9",
        entity="frtim",
        config=dict(log_info_dict['config']),
        tags=[log_info_dict["wandb_tag"]],
    )

    for log_dict in log_info_dict['log_dicts']:
        wandb.log(log_dict)

    wandb.finish()

def log_dicts_to_wandb(log_info_dicts):

    for log_info_dict in log_info_dicts:
        log_to_wandb(log_info_dict)

    # multiprocessing.set_start_method('spawn', force=True)
    
    # args_list = list(log_info_dicts)
    
    # with multiprocessing.Pool(10) as pool:
    #     pool.map(log_to_wandb, args_list)

    # print("done with pool for logging to wandb")