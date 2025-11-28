import esm
import torch
import string
import os
import itertools
import numpy as np
import argparse
from Bio import SeqIO
from typing import List, Tuple
import pickle as pkl

# read the Multiple Sequence Alignment (MSA)
def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq)))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--esm_file', type=str, default=None, help='esm file')
    parser.add_argument('--msa_dir', type=str, default=None, help='msa dir')
    parser.add_argument('--pdb_id', type=str, default=None, help='pdb id')
    parser.add_argument('--save_dir', type=str, default=None, help='save dir')
    args = parser.parse_args()

    return args


# load model
def load_esm(esm_file):

    model, alphabet = esm.pretrained.load_model_and_alphabet(esm_file)
    model = model.eval()
    batch_converter = alphabet.get_batch_converter()

    return model, batch_converter

# inference of ESM-MSA-1b
def esm_inference(esm1b, esm1b_batch_converter, seq_list):
    # convert the sequence to tokens
    _, _, esm1b_batch_tokens = esm1b_batch_converter(seq_list)

    with torch.no_grad():
        results = esm1b(esm1b_batch_tokens, repr_layers=[12], return_contacts=True)
   
    # esm-msa-1b sequence representation
    token_representations = results["representations"][12].mean(1)
    
    sequence_representations = []
    for i, seq in enumerate(seq_list):
        sequence_representations.append(np.array(token_representations[i, 1 : len(seq[0][1]) + 1].cpu()))

    # return the esm-msa-1d and row-attentions
    return sequence_representations[0], np.squeeze(np.array(results['row_attentions'].cpu()))[:,:,1:,1:]

def extract_features(esm_file, msa_dir, pdb_id, save_dir):
    

    # load model and read msa
    esm1b, esm1b_batch_converter = load_esm(esm_file)
    msa_data = [read_msa(os.path.join(msa_dir, pdb_id+"_filter.a3m"), 512)]

    # inference
    esm_msa_1d, row_attentions = esm_inference(esm1b, esm1b_batch_converter, msa_data)
    features = { 'esm_msa_1d':esm_msa_1d, 'row_attentions':row_attentions}

    with open(os.path.join(save_dir, pdb_id+"_esm_msa.pkl"), 'wb') as f:
        pkl.dump(features, f, protocol = 3)

if __name__ == "__main__":

    # translation for read sequence
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    translation = str.maketrans(deletekeys)


    args = parse_args()
    
    esm_file = args.esm_file
    msa_dir = args.msa_dir
    pdb_id = args.pdb_id
    save_dir = args.save_dir

    extract_features(esm_file, msa_dir, pdb_id, save_dir)