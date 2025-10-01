from transformers import TrainerCallback
from clearml import Task, Logger

from dotenv import load_dotenv
import os


class ClearMLCallback(TrainerCallback):
    def __init__(
        self,
        task_name="ViTMAE Training",
        project_name="MAE",
        env_file="PATH_TO_.ENV_FILE",
    ):
        super().__init__()
        load_dotenv(env_file)
        
        api_host = os.getenv("CLEARML_API_HOST")
        web_host = os.getenv("CLEARML_WEB_HOST")
        files_host = os.getenv("CLEARML_FILES_HOST")

        access_key = os.getenv("CLEARML_API_ACCESS_KEY")
        secret_key = os.getenv("CLEARML_API_SECRET_KEY")
        
        self.task = Task.init(project_name=project_name, task_name=task_name)
        if api_host and web_host and files_host:
            self.task.set_credentials(
                api_host=api_host,
                web_host=web_host,
                files_host=files_host,
                key=access_key,
                secret=secret_key,
            )

        self.logger = self.task.get_logger()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        step = state.global_step

        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.logger.report_scalar(
                    title=key, series="train/eval", value=value, iteration=step
                )

        if "mae_loss" in logs:
            self.logger.report_scalar(
                title="MAE Loss", series="validation", value=logs["mae_loss"], iteration=step
            )
        if "bt_loss" in logs:
            self.logger.report_scalar(
                title="BT Loss", series="validation", value=logs["bt_loss"], iteration=step
            )
        if "eval/linear_probe_accuracy" in logs:
            self.logger.report_scalar(
                title="Linear Probe Accuracy", series="validation", value=logs["eval/linear_probe_accuracy"], iteration=step
            )
