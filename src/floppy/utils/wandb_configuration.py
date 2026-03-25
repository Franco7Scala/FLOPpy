from dataclasses import dataclass


@dataclass
class WandbConfiguration:
    """
        Configuration settings for Weights & Biases (WandB) integration.
        Used to synchronize real-time tracking of computational workloads.
    """

    #: The name of the WandB project where the runs will be logged.
    project_name: str
    #: The group name to organize multiple runs together in the W&B dashboard.
    group_name: str
    #: The API key used for authenticating with the WandB server.
    reporter_key: str
