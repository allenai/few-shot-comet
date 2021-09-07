import argparse
import logging
import subprocess
import sys

sys.path.insert(0, 't5_runner/')

from t5_runner.t5_data_preloader import T5DataHelper
from t5_runner.t5_mixtures import t5_mixtures_map
from t5_runner.t5_model_wrapper import T5ModelHelper

logger = logging.getLogger(__name__)


def setup_mixtures(mixture_dir: str, is_local:bool):
    print(f"Registering data mixtures...")
    return T5DataHelper(mixture_dir=mixture_dir, is_local=is_local)


def setup_model_helper_for_training(args):
    print(f"Setting up model helper for training...")
    model_helper = T5ModelHelper()
    return model_helper.instantiate_model(model_dir=args.model_dir_to_save,
                                          sequence_length=args.max_len,
                                          model_size=args.model_size,
                                          tpu_name=args.tpu_name,
                                          tpu_zone=args.tpu_zone,
                                          user_batch_size=args.batch_size_train,
                                          user_model_parallelism=args.model_parallelism,
                                          learning_rate_schedule=args.learning_rate_schedule,
                                          save_checkpoints_steps=args.save_checkpoints_steps,
                                          iterations_per_loop=args.iterations_per_loop)

def setup_model_helper_for_prediction(args):
    print(f"Setting up model helper for prediction...")
    model_helper = T5ModelHelper()
    return model_helper.instantiate_model(model_dir=args.prev_trained_model,
                                          sequence_length=args.max_len,
                                          model_size=args.model_size,
                                          tpu_name=args.tpu_name,
                                          tpu_zone=args.tpu_zone,
                                          user_batch_size=args.batch_size_predict,
                                          user_model_parallelism=args.model_parallelism)

def tpu(args):
    print(f"TPU module...")
    if args.on_off == "on":
        print(f"\nTurning on the TPU...")
        subprocess.run(f"ctpu up --name={args.tpu_name} --project={args.tpu_project} --zone={args.tpu_zone} --tpu-size={args.tpu_size}  --tpu-only --tf-version={args.tf_version} --gcp-network={args.tpu_network} --noconf", shell=True)
        print(f"[done]\n")
    elif args.on_off == "off":
        print(f"\nTurning off the TPU...")
        subprocess.run(f"ctpu pause --name={args.tpu_name} --zone={args.tpu_zone} --noconf", shell=True)
        print(f"[done]\n")


def predict(args):
    print(f"Prediction module...")
    assert len(args.input_csv) > 1 and (len(args.input_csv.split(",")) == len(args.output_csv.split(",")))
    aggr_prediction_output_csv = ",".join([x.strip() + ".aggr.jsonl" for x in args.output_csv.split(",")])

    t5_model = setup_model_helper_for_prediction(args)

    for input_file, output_file in zip(args.input_csv.split(","), args.output_csv.split(",")):
        print(f"\nPerforming prediction on {input_file}")
        t5_model.predict(input_file=input_file,
                         output_file=output_file,
                         checkpoint_steps=args.checkpoint_steps,
                         temperature=args.temperature,
                         beam_size=args.beam_size
                         )
        print(f"[done]")

def train(args):
    print(f"Training module...")
    is_local = args.mixture_name == "new"
    t5_mixtures_helper = setup_mixtures(mixture_dir=args.mixture_dir, is_local=is_local)
    if is_local:
        args.mixture_name = T5DataHelper.get_temp_mixture_name()
    t5_model = setup_model_helper_for_training(args)
    if args.func == "t":
        print(f"Training of the model begun with initial checkpoint {args.prev_trained_model}")
        t5_model.train(mixture_or_task_name=args.mixture_name, steps=args.num_steps, init_checkpoint=args.prev_trained_model if args.prev_trained_model else None)
        print(f"Training of the model completed.")
    elif args.func == "f":
        print(f"Fine-tuning of the model begun with initial checkpoint {args.prev_trained_model}")
        t5_model.finetune(mixture_or_task_name=args.mixture_name, finetune_steps=args.num_steps, pretrained_model_dir=args.prev_trained_model, pretrained_checkpoint_step=args.pretrained_checkpoint_step)
        print(f"Fine-tuning of the model completed.")


def evaluate(args):
    print(f"Evaluation module...")
    raise NotImplementedError("Evaluation is not yet implemented. We are still puzzled with BLEU vs. BLEURT! ")


def main():
    cli = argparse.ArgumentParser(prog='t5_run.py')     # parser.add_argument('--foo', action='store_true')
    subparsers = cli.add_subparsers(title='subcommands', dest="subcommand", description='valid subcommands', help='additional help to setup TPU and VM for T5, train, tune, predict with T5.')

    train_parser = subparsers.add_parser('train', help='Finetune or train a T5 model.')
    train_parser.add_argument("--prev_trained_model", required=True, help="Path to previous model to continue training from.")
    train_parser.add_argument("--func", required=True, choices=["f", "t"], help="Fine tune (f), train (t)")
    train_parser.add_argument("--model_dir_to_save", required=True, help="Where should the new checkpoints be saved?")
    train_parser.add_argument("--num_steps", required=True,  type=int, default=100, help=f"Number of steps for training (default: %(default)s)")
    train_parser.add_argument("--model_size", required=True, default="11B", help=f"T5 model size e.g., (small, 11B). (default: %(default)s)", choices=["small", "base", "3B", "large","11B"])
    train_parser.add_argument("--mixture_name", required=True, help=f"Pick a mixture name from: {t5_mixtures_map.keys()}, otherwise type 'new' (in this case a temporary in-memory mixture will be registered.)", choices=list(t5_mixtures_map.keys())+list(["new"]))
    train_parser.add_argument("--save_checkpoints_steps",  type=int, default=1000, help="Number of steps before checkpoint saved (default: %(default)s)")
    train_parser.add_argument("--batch_size_train", type=int, default=4, help=f"Batch size for training (default: %(default)s)")
    train_parser.add_argument("--model_parallelism", type=int, default=8, help=f"Model parallelism for training (default: %(default)s)")
    train_parser.add_argument("--block_size", type=int, default=512, help=f"Block size for training (default: %(default)s)")
    train_parser.add_argument("--max_len",  default=200, type=int, help="Max output length (number of byte pair tokens) (default: %(default)s)")
    train_parser.add_argument("--learning_rate_schedule", type=float, default=0.001, help=f"learning rate (default: %(default)s)")
    train_parser.add_argument("--tpu_size", default="v3-8", help=f"Type of TPU to run this experiment on (default: %(default)s)")
    train_parser.add_argument("--tpu_name", default="tpu-name", help=f"Name of TPU to run this experiment on (default: %(default)s)")
    train_parser.add_argument("--tpu_zone", default="europe-west4-a", help=f"Zone of TPU to run this experiment on (default: %(default)s)")
    train_parser.add_argument("--tpu_project", default="tpu-project", help=f"Project to bill the TPU (default: %(default)s)")
    train_parser.add_argument("--iterations_per_loop", type=int, default=100, help=f"iterations per train loop")
    train_parser.add_argument("--mixture_dir", default="gs://project-dir/tpu-dir", help=f"Data directory contains files for mixtures -- can also be a local_dir if mixture name does not exist -- (default: %(default)s)")
    train_parser.add_argument("--pretrained_checkpoint_step", default=-1, type=int, help="pretrained checkpoint step to start from")
    train_parser.set_defaults(handle=train)

    # "predict" command
    predict_parser = subparsers.add_parser('predict', help='Predictions from a T5 model.')
    predict_parser.add_argument("--prev_trained_model", required=True, help="Path to previous model to continue training from.")
    predict_parser.add_argument("--input_csv", required=True, help="A csv of file paths to perform prediction on.")
    predict_parser.add_argument("--output_csv", required=True, help="A csv of file paths to write outputs, corr. to input_csv.")
    predict_parser.add_argument("--batch_size_predict", type=int, default=4, help=f"Batch size for prediction (default: %(default)s)")
    predict_parser.add_argument("--model_parallelism", type=int, default=8, help=f"Model parallelism for prediction (default: %(default)s)")
    predict_parser.add_argument("--model_size", default="small", help=f"T5 model size e.g., (small, 11B). (default: %(default)s)", choices=["small","large","11B"]) # TODO check
    predict_parser.add_argument("--max_len",  default=200, type=int, help="Max output length (number of byte pair tokens) (default: %(default)s)")
    predict_parser.add_argument("--beam_size", default=1, type=int, help="a number >= 1 specifying the number of beams to use for beam search (default: %(default)s)")
    predict_parser.add_argument("--temperature", default=1.0, type=float, help="a value between 0 and 1 (must be 0 if beam_size > 1) 0.0 means argmax, 1.0 means sample according to predicted distribution. (default: %(default)s)")
    predict_parser.add_argument("--checkpoint_steps", default=-1, type=int, help="int If an int or list of ints, inference will be run on the checkpoint files in `model_dir` whose global steps are closest to the global steps provided. If None, run inference continuously waiting for new checkpoints. If -1, get the  latest checkpoint from the model directory.")
    predict_parser.add_argument("--tpu_size", default="v3-8", help=f"Type of TPU to run this experiment on (default: %(default)s)")
    predict_parser.add_argument("--tpu_name", default="tpu-name", help=f"Name of TPU to run this experiment on (default: %(default)s)")
    predict_parser.add_argument("--tpu_zone", default="europe-west4-a", help=f"Zone of TPU to run this experiment on (default: %(default)s)")
    predict_parser.add_argument("--tpu_project", default="tpu-project", help=f"Project to bill the TPU (default: %(default)s)")
    predict_parser.set_defaults(handle=predict)

    # "tpu" command
    tpu_parser = subparsers.add_parser('tpu', help='Start or stop the TPU.')
    tpu_parser.add_argument("--on_off", required=True, help=f"Should the TPU be turned on or off?", choices=["on", "off"])
    tpu_parser.add_argument("--tpu_size", default="v3-8", help=f"Type of TPU to run this experiment on (default: %(default)s)")
    tpu_parser.add_argument("--tpu_name", default="tpu-name", help=f"Name of TPU to run this experiment on (default: %(default)s)")
    tpu_parser.add_argument("--tpu_zone", default="europe-west4-a", help=f"Zone of TPU to run this experiment on (default: %(default)s)")
    tpu_parser.add_argument("--tpu_project", default="tpu-project", help=f"Project to bill the TPU (default: %(default)s)")
    tpu_parser.add_argument("--tpu_network", default="main", help=f"Network for TPU (default: %(default)s)")
    tpu_parser.add_argument("--tf_version", default="2.1", help=f"Version of Tensoflow to work with the TPU (default: %(default)s)")
    tpu_parser.set_defaults(handle=tpu)

    args = cli.parse_args()
    if hasattr(args, 'handle'):
        args.handle(args)
    else:
        cli.print_help()
    try:
        args.save_checkpoints_steps = int(args.num_steps / 10)
    except:
        pass


if __name__ == "__main__":
    main()
