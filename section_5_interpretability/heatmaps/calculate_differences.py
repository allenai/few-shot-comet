from transformers import T5ForConditionalGeneration
from transformers import T5Config
from torch import linalg as LA
import torch
from collections import defaultdict
import math
import statistics
import gc

def sim_matrix(a, b, eps=1e-10):
    total = 0
    for dim in range(a.shape[0]):
        cos = torch.nn.CosineSimilarity(dim=0, eps=eps)
        total += math.acos(min(float(cos(a[dim], b[dim]).item()), 1)) / 3.141
    return total / a.shape[0]

def process_results(store):
    keys = ['q', 'k', 'v', 'o', 'wi', 'wo', 'xq', 'xk', 'xv', 'xo']
    for key in keys:
        if key in store:
            for idx in range(24):
                store[key][idx] = store[key][idx]
            store[key] = store[key][:24]
    
def get_norm(mat, n=None):
    if n == None:
        return torch.tensor(LA.norm(mat).item())
    return torch.tensor(LA.norm(mat, n).item())

for dirname in ['./models/11b/heads']:  # for your case, replace dirname with the path to your model file
    seeds = [0, 1, 2, 3, 4]
    for seed in seeds:
        gc.collect()
        results_encoder = defaultdict(list)
        results_decoder = defaultdict(list)
        table_file_decoder = open(f'l1_decoder_{dirname.split("/")[-1]}-{seed}.tsv', 'w')
        table_file_encoder = open(f'l1_encoder_{dirname.split("/")[-1]}-{seed}.tsv', 'w')
        config = T5Config.from_pretrained(f'{dirname}-{seed}')
        model = T5ForConditionalGeneration.from_pretrained(f'{dirname}-{seed}', config=config)

        org_config = T5Config.from_pretrained(f'./models/11b')
        org_model = T5ForConditionalGeneration.from_pretrained(f'./models/11b', config=org_config)

        org_dict = org_model.state_dict()
        trained_dict = model.state_dict()

        for encoder_n in range(24):
            print("seed", seed, encoder_n)
            q_org = org_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.q.weight']
            q_new = trained_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.q.weight']
            results_encoder['q'].append(get_norm(q_org - q_new, 1))

            k_org = org_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.k.weight']
            k_new = trained_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.k.weight']
            results_encoder['k'].append(get_norm(k_org - k_new, 1))

            v_org = org_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.v.weight']
            v_new = trained_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.v.weight']
            results_encoder['v'].append(get_norm(v_org - v_new, 1))

            o_org = org_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.o.weight']
            o_new = trained_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.o.weight']
            results_encoder['o'].append(get_norm(o_org - o_new, 1))

            wo_org = org_dict[f'encoder.block.{encoder_n}.layer.1.DenseReluDense.wi.weight']
            wo_new = trained_dict[f'encoder.block.{encoder_n}.layer.1.DenseReluDense.wi.weight']
            results_encoder['wi'].append(get_norm(wo_org - wo_new, 1)) # sic

            wo_org = org_dict[f'encoder.block.{encoder_n}.layer.1.DenseReluDense.wo.weight']
            wo_new = trained_dict[f'encoder.block.{encoder_n}.layer.1.DenseReluDense.wo.weight']
            results_encoder['wo'].append(get_norm(wo_org - wo_new, 1))

        for decoder_n in range(24):
            print("seed", seed, decoder_n)
            q_org = org_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.q.weight']
            q_new = trained_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.q.weight']
            results_decoder['q'].append(LA.norm(q_org - q_new, 1))

            k_org = org_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.k.weight']
            k_new = trained_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.k.weight']
            results_decoder['k'].append(LA.norm(k_org - k_new, 1))

            v_org = org_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.v.weight']
            v_new = trained_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.v.weight']
            results_decoder['v'].append(LA.norm(v_org - v_new, 1))

            o_org = org_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.o.weight']
            o_new = trained_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.o.weight']
            results_decoder['o'].append(LA.norm(o_org - o_new, 1))

            q_org = org_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.q.weight']
            q_new = trained_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.q.weight']
            results_decoder['xq'].append(LA.norm(q_org - q_new, 1))

            k_org = org_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.k.weight']
            k_new = trained_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.k.weight']
            results_decoder['xk'].append(LA.norm(k_org - k_new, 1))

            v_org = org_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.v.weight']
            v_new = trained_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.v.weight']
            results_decoder['xv'].append(LA.norm(v_org - v_new, 1))

            o_org = org_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.o.weight']
            o_new = trained_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.o.weight']
            results_decoder['xo'].append(LA.norm(o_org - o_new, 1))

            wo_org = org_dict[f'decoder.block.{decoder_n}.layer.2.DenseReluDense.wi.weight']
            wo_new = trained_dict[f'decoder.block.{decoder_n}.layer.2.DenseReluDense.wi.weight']
            results_decoder['wi'].append(LA.norm(wo_org - wo_new, 1)) # sic

            wo_org = org_dict[f'decoder.block.{decoder_n}.layer.2.DenseReluDense.wo.weight']
            wo_new = trained_dict[f'decoder.block.{decoder_n}.layer.2.DenseReluDense.wo.weight']
            results_decoder['wo'].append(LA.norm(wo_org - wo_new, 1))

        process_results(results_decoder)
        process_results(results_encoder)
        for item in results_decoder:
            st = "\t".join([str(float(f.item())) for f in results_decoder[item]])
            table_file_decoder.write(f'{item}\t{st}\n')

        for item in results_encoder:
            st = "\t".join([str(float(f.item())) for f in results_encoder[item]])
            table_file_encoder.write(f'{item}\t{st}\n')

for dirname in ['./models/11b/heads']: # for your case, replace dirname with the path to your model file
    results_encoder = defaultdict(list)
    results_decoder = defaultdict(list)
    table_file_decoder = open(f'cossim_decoder_{dirname.split("/")[-1]}-moresigfigs.tsv', 'w')
    table_file_encoder = open(f'cossim_encoder_{dirname.split("/")[-1]}-moresigfigs.tsv', 'w')

    seeds = [0, 1, 2, 3, 4]
    for seed in seeds:
        config = T5Config.from_pretrained(f'{dirname}-{seed}')
        model = T5ForConditionalGeneration.from_pretrained(f'{dirname}-{seed}', config=config)

        org_config = T5Config.from_pretrained(f'./models/11b')
        org_model = T5ForConditionalGeneration.from_pretrained(f'./models/11b', config=org_config)

        org_dict = org_model.state_dict()
        trained_dict = model.state_dict()

        for encoder_n in range(24):
            q_org = org_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.q.weight']
            q_new = trained_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.q.weight']
            results_encoder['q'].append(sim_matrix(q_org, q_new))

            k_org = org_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.k.weight']
            k_new = trained_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.k.weight']
            results_encoder['k'].append(sim_matrix(k_org, k_new))

            v_org = org_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.v.weight']
            v_new = trained_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.v.weight']
            results_encoder['v'].append(sim_matrix(v_org, v_new))

            o_org = org_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.o.weight']
            o_new = trained_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.o.weight']
            results_encoder['o'].append(sim_matrix(o_org, o_new))

            wo_org = org_dict[f'encoder.block.{encoder_n}.layer.1.DenseReluDense.wi.weight']
            wo_new = trained_dict[f'encoder.block.{encoder_n}.layer.1.DenseReluDense.wi.weight']
            results_encoder['wi'].append(sim_matrix(wo_org, wo_new))

            wo_org = org_dict[f'encoder.block.{encoder_n}.layer.1.DenseReluDense.wo.weight']
            wo_new = trained_dict[f'encoder.block.{encoder_n}.layer.1.DenseReluDense.wo.weight']
            results_encoder['wo'].append(sim_matrix(wo_org, wo_new))

        for decoder_n in range(24):
            q_org = org_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.q.weight']
            q_new = trained_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.q.weight']
            results_decoder['q'].append(sim_matrix(q_org, q_new))

            k_org = org_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.k.weight']
            k_new = trained_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.k.weight']
            results_decoder['k'].append(sim_matrix(k_org, k_new))

            v_org = org_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.v.weight']
            v_new = trained_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.v.weight']
            results_decoder['v'].append(sim_matrix(v_org, v_new))

            o_org = org_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.o.weight']
            o_new = trained_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.o.weight']
            results_decoder['o'].append(sim_matrix(o_org, o_new))

            q_org = org_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.q.weight']
            q_new = trained_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.q.weight']
            results_decoder['xq'].append(sim_matrix(q_org, q_new))

            k_org = org_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.k.weight']
            k_new = trained_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.k.weight']
            results_decoder['xk'].append(sim_matrix(k_org, k_new))

            v_org = org_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.v.weight']
            v_new = trained_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.v.weight']
            results_decoder['xv'].append(sim_matrix(v_org, v_new))

            o_org = org_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.o.weight']
            o_new = trained_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.o.weight']
            results_decoder['xo'].append(sim_matrix(o_org, o_new))

            wo_org = org_dict[f'decoder.block.{decoder_n}.layer.2.DenseReluDense.wi.weight']
            wo_new = trained_dict[f'decoder.block.{decoder_n}.layer.2.DenseReluDense.wi.weight']
            results_decoder['wi'].append(sim_matrix(wo_org, wo_new))

            wo_org = org_dict[f'decoder.block.{decoder_n}.layer.2.DenseReluDense.wo.weight']
            wo_new = trained_dict[f'decoder.block.{decoder_n}.layer.2.DenseReluDense.wo.weight']
            results_decoder['wo'].append(sim_matrix(wo_org, wo_new))

    process_results(results_decoder)
    process_results(results_encoder)
    for item in results_decoder:
        st = "\t".join([str(float(f)) for f in results_decoder[item]])
        table_file_decoder.write(f'{item}\t{st}\n')

    for item in results_encoder:
        st = "\t".join([str(float(f)) for f in results_encoder[item]])
        table_file_encoder.write(f'{item}\t{st}\n')