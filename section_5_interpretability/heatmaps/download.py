import subprocess
import tempfile

from transformers import T5Config, T5ForConditionalGeneration, load_tf_weights_in_t5


def convert_model(base_model, path, new_path):
    model = T5ForConditionalGeneration(T5Config.from_pretrained(base_model))
    print("loading weights...")
    load_tf_weights_in_t5(model, None, path)
    model.eval()
    print("saving HF weights...")
    model.save_pretrained(new_path)

def run_system_process(cmd):
    print(f"Running command: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True)
    print(f"Command result: {res.stdout}")
    return res

def convert_tpu_model_gcs(gcs_dir_in="",
                          checkpoint="1007000",
                          base_model="t5-11b",
                          gcs_dir_out=""):

    model_dir = "./gcloud_temp"
    print(f"Converting checkpoint {checkpoint} from {gcs_dir_in} to {model_dir}...")
    print("Downloading model using gsutil...")
    run_system_process(f"gsutil cp  {gcs_dir_in}*{checkpoint}* {model_dir}")
    with open(f"{model_dir}/checkpoint", "w") as file:
        file.write(f'model_checkpoint_path: "model.ckpt-{checkpoint}"\n')
    out_dir = gcs_dir_out
    print(f"Converting model, saving output in {gcs_dir_out}")
    convert_model(base_model, model_dir, out_dir)


def main():
    print("Starting main...")
    gclod_path = "gs://gcloud-path"
    step_count = "1000170"
    size = "t5-11b"
    outdir = "./models/11b/heads"
    for gpath, ckpt, mode, outdir in [(gcloud_path, step_count, size, outdir)]:
        if gpath[-1] != '/':
            gpath += '/'
        convert_tpu_model_gcs(gcs_dir_in=gpath,checkpoint=ckpt,base_model=mode, gcs_dir_out=outdir)


if __name__ == "__main__":
    main()