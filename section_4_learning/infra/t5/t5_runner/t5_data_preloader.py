import logging
from typing import List, Any

import os
import functools
import t5
import tensorflow.compat.v1 as tf
import seqio

from t5_runner.t5_mixtures import t5_mixtures_map

logger = logging.getLogger(__name__)
DEFAULT_OUTPUT_FEATURES = {
    "inputs":
        seqio.Feature(
            vocabulary=t5.data.get_default_vocabulary(), add_eos=True),
    "targets":
        seqio.Feature(
            vocabulary=t5.data.get_default_vocabulary(), add_eos=True)
}


def dataset_preprocessor(ds):
    def normalize_text(text):
        """Lowercase and remove quotes from a TensorFlow string."""
        text = tf.strings.lower(text)
        text = tf.strings.regex_replace(text, "'(.*)'", r"\1")
        return text

    def to_inputs_and_targets(ex):
        return {
            "inputs": normalize_text(ex["inputs"]),
            "targets": normalize_text(ex["targets"])
        }

    return ds.map(to_inputs_and_targets,
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

def tsv_get_path(a_data_dir, split):
    tsv_path = {
        "train": os.path.join(a_data_dir, "train.tsv"),
        "dev": os.path.join(a_data_dir, "dev.tsv"),
        "test": os.path.join(a_data_dir, "test.tsv")
    }
    return tsv_path[split]

def tsv_dataset_fn(split,
                   shuffle_files=False,
                   dataset=""):                             
                   # this function helps loading dataset. triples in dataset are stored as "{h} {r}\t{t}"
                   # see example in ./atomic2020_splits/comet_heads_3_0
    ds = tf.data.TextLineDataset(tsv_get_path(dataset, split))
    ds = ds.map(
        functools.partial(tf.io.decode_csv, record_defaults=["", ""], field_delim="\t", use_quote_delim=False),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    print(ds.map(lambda *ex: dict(zip(["inputs", "targets"], ex))))
    return ds.map(lambda *ex: dict(zip(["inputs", "targets"], ex)))


class T5DataHelper:

    @staticmethod
    def get_temp_mixture_name() -> str:
        return "temp_mixture"

    @staticmethod
    def get_temp_dataset_name() -> str:
        return "temp_dataset"

    def __init__(self,
                 mixture_dir: str,  # dir to dataset, e.g ./atomic2020_splits/comet_heads_3_0
                 is_local: bool = False
                 ):

        if is_local:
            datasets = [T5DataHelper.get_temp_dataset_name()]
            for d in datasets:
                self.register_dataset(dataset_name=d, dataset_path=mixture_dir)
            self.register_mixture(mixture_name=T5DataHelper.get_temp_mixture_name(), datasets=datasets)
            assert T5DataHelper.get_temp_mixture_name() in seqio.MixtureRegistry.names(), f"Temp mixture name: {T5DataHelper.get_temp_mixture_name()} is not in {seqio.MixtureRegistry.names()}"
        else:
            print(f"Registering all previously existing datasets...")
            for dataset_name, datasetinfo in beaker_dataset_file_map.items():
                if datasetinfo.cloud_path.startswith("gs://"):
                    print(f"Registering dataset {dataset_name}...")
                    dataset_path = datasetinfo.cloud_path
                    t5.data.set_tfds_data_dir_override(dataset_path)
                    self.register_dataset(dataset_name=dataset_name,
                                          splits=datasetinfo.cloud_splits_names_arr,
                                          dataset_path=dataset_path
                                          )
            print(f"Done registering datasets, now registering mixture")
            for mixture_name, datasetinfo_list in t5_mixtures_map.items():
                names_of_ds = [x.name for x in datasetinfo_list]
                self.register_mixture(mixture_name=mixture_name, datasets=names_of_ds)

    def register_dataset(self,
                         dataset_name: str,
                         dataset_path: str,
                         metrics_fns: List[Any] = [t5.evaluation.metrics.accuracy],
                         splits:List[str] = ["train", "dev", "test"]
                         ):
        logger.info(f"Registering dataset: {dataset_name}")
    
        seqio.TaskRegistry.remove(f"{dataset_name}")
        seqio.TaskRegistry.add(
            f"{dataset_name}",
            # Specify the task source.
            source=seqio.FunctionDataSource(
                # Supply a function which returns a tf.data.Dataset.
                dataset_fn=functools.partial(tsv_dataset_fn, dataset=dataset_path),
                splits=["train", "dev"]
            ),
            # Supply a list of functions that preprocess the input tf.data.Dataset.
            preprocessors=[
                dataset_preprocessor,
                seqio.preprocessors.tokenize_and_append_eos,
            ],
            # Lowercase targets before computing metrics.
            postprocess_fn=t5.data.postprocessors.lower_text,
            # We'll use accuracy as our evaluation metric.
            metric_fns=metrics_fns,
            output_features=DEFAULT_OUTPUT_FEATURES,
        )

    # need to register mixture/dataset, followed API of T5
    # it will be recorded in a global variable, 
    # then be used to load data
    def register_mixture(self, mixture_name, datasets: List[Any]):
        seqio.MixtureRegistry.remove(mixture_name)
        seqio.MixtureRegistry.add(
            mixture_name,
            [d for d in datasets],
            default_rate=1.0
        )
