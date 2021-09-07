import logging

import t5
from t5.models import MtfModel


logger = logging.getLogger(__name__)


class T5ModelHelper:

    def __init__(self):

        # model_parallelism, train_batch_size, keep_checkpoint_max
        self.DEFAULT_PARAMS_DICT = {
            "small": (1, 256, 40),
            "base": (2, 128, 28),
            "large": (8, 64, 24),
            "3B": (8, 16, 30),
            "11B": (8, 8, 30)
        }

    def get_default_params(self, model_size):
        return self.DEFAULT_PARAMS_DICT[model_size]

    @staticmethod
    def gen_model_dir_from_mixture(mixture_name, model_size, model_upper_dir):
        return f"{model_upper_dir}/{mixture_name}/{model_size}"

    def instantiate_model(self,
                          model_dir:str,
                          sequence_length:int,
                          model_size:str,
                          tpu_name:str,
                          tpu_zone:str,
                          tpu_topology:str='2x2',
                          user_batch_size: int = None,
                          user_model_parallelism: int = None,
                          user_keep_checkpoint_max: int = None,
                          learning_rate_schedule=None,
                          save_checkpoints_steps=None,
                          pretrained_name=None,
                          iterations_per_loop=100) -> MtfModel:

        model_parallelism, train_batch_size, keep_checkpoint_max = self.get_default_params(model_size=model_size)
        if user_batch_size and user_batch_size >= 1:
            train_batch_size = user_batch_size
        if user_model_parallelism and user_model_parallelism >= 1:
            model_parallelism = user_model_parallelism
        if user_keep_checkpoint_max and user_keep_checkpoint_max >= 1:
            keep_checkpoint_max = user_keep_checkpoint_max

        print(f"Instantiating the model...")
        model = t5.models.MtfModel(
            model_dir=model_dir,
            tpu=tpu_name,
            tpu_zone=tpu_zone,
            tpu_topology=tpu_topology,
            model_parallelism=model_parallelism,
            batch_size=train_batch_size,
            sequence_length=sequence_length,
            learning_rate_schedule=learning_rate_schedule,
            save_checkpoints_steps=save_checkpoints_steps,
            keep_checkpoint_max=keep_checkpoint_max,
            iterations_per_loop=iterations_per_loop, # steps per train loop
        )
        print(f"[DONE] Instantiating the model...")
        return model
