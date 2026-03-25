class WandbConfiguration:
    """
        Configuration settings for Weights & Biases (WandB) integration.
        Used to synchronize real-time tracking of computational workloads.
    """

    def __init__(self, project_name: str, group_name: str, reporter_key: str):
        #: The name of the WandB project where the runs will be logged.
        self.project_name = project_name
        #: The group name to organize multiple runs together in the W&B dashboard.
        self.group_name = group_name
        #: The API key used for authenticating with the WandB server.
        self.reporter_key = reporter_key
