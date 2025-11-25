import os
# import argparse
import numpy as np
import pickle as pkl
from LoadHHM import load_hmm
from tqdm import tqdm
from Bio.PDB.PDBParser import PDBParser
from joblib import Parallel, delayed, cpu_count


aa3to1 = {
   'ALA':'A', 'VAL':'V', 'PHE':'F', 'PRO':'P', 'MET':'M',
   'ILE':'I', 'LEU':'L', 'ASP':'D', 'GLU':'E', 'LYS':'K',
   'ARG':'R', 'SER':'S', 'THR':'T', 'TYR':'Y', 'HIS':'H',
   'CYS':'C', 'ASN':'N', 'GLN':'Q', 'TRP':'W', 'GLY':'G',
}

aa1to3 = {
    'A': 'ALA', 'V': 'VAL', 'F': 'PHE', 'P': 'PRO', 'M': 'MET',
    'I': 'ILE', 'L': 'LEU', 'D': 'ASP', 'E': 'GLU', 'K': 'LYS',
    'R': 'ARG', 'S': 'SER', 'T': 'THR', 'Y': 'TYR', 'H': 'HIS',
    'C': 'CYS', 'N': 'ASN', 'Q': 'GLN', 'W': 'TRP', 'G': 'GLY',
}


def pmap_multi(pickleable_fn, data, n_jobs=None, verbose=1, **kwargs):
  """
  Extends dgllife pmap function.

  Parallel map using joblib.

  Parameters
  ----------
  pickleable_fn : callable
      Function to map over data.
  data : iterable
      Data over which we want to parallelize the function call.
  n_jobs : int, optional
      The maximum number of concurrently running jobs. By default, it is one less than
      the number of CPUs.
  verbose: int, optional
      The verbosity level. If nonzero, the function prints the progress messages.
      The frequency of the messages increases with the verbosity level. If above 10,
      it reports all iterations. If above 50, it sends the output to stdout.
  kwargs
      Additional arguments for :attr:`pickleable_fn`.

  Returns
  -------
  list
      The i-th element of the list corresponds to the output of applying
      :attr:`pickleable_fn` to :attr:`data[i]`.
  """
  if n_jobs is None:
    n_jobs = cpu_count() - 1

  results = Parallel(n_jobs=n_jobs, verbose=verbose, timeout=None)(
    delayed(pickleable_fn)(*d, **kwargs) for i, d in enumerate(data)
  )

  return results


def rbf(D):
    # Distance radial basis function
    D_min, D_max, D_count = 2., 22., 64
    D_mu = np.linspace(D_min, D_max, D_count)
    D_mu = D_mu[None,:]
    D_sigma = (D_max - D_min) / D_count

    D = D.transpose(1, 2, 0)
    RBF = np.exp(-((D - D_mu) / D_sigma)**2)
    return RBF.transpose(2, 0, 1)


def get_seq_len(fa_file):
    
    f = open(fa_file, 'r')
    all_ = []
    for line in f:
        if line.startswith('>'):
            continue
        else:
            str_ = line.split('\n')[0]
            all_.append(str_)
    
    return ''.join(all_)

def cal_dist(coord1, coord2):
    diff_vector  = coord1 - coord2
    return np.sqrt(np.sum(diff_vector * diff_vector))


def gen_contact(l_pdb_id, r_pdb_id, pdb_folder, fa_folder, cutoff=8.0):

    l_pdb_file = os.path.join(pdb_folder, l_pdb_id+'_modified.pdb')
    r_pdb_file = os.path.join(pdb_folder, r_pdb_id+'_modified.pdb')

    l_fa_file = os.path.join(fa_folder, '{}.fasta'.format(l_pdb_id))
    r_fa_file = os.path.join(fa_folder, '{}.fasta'.format(r_pdb_id))

    l_seq, r_seq = get_seq_len(l_fa_file), get_seq_len(r_fa_file)
    N1, N2 = len(l_seq), len(r_seq)

    l_residue_list = []
    # index = 0
    for item in l_seq:
        residue = aa1to3[item]
        l_residue_list.append(residue)
        # index += 1

    r_residue_list = []
    # index = 0
    for item in r_seq:
        residue = aa1to3[item]
        r_residue_list.append(residue)
        # index += 1


    parser = PDBParser(PERMISSIVE=1)
    l_structure = parser.get_structure(l_pdb_id, l_pdb_file)
    l_model = l_structure[0]
    l_chain_ids = list(l_model.child_dict.keys())
    l_chain = l_model[l_chain_ids[0]]
    # l_chain_dict = list(l_chain.child_dict.keys())
    
    r_structure = parser.get_structure(r_pdb_id, r_pdb_file)
    r_model = r_structure[0]
    r_chain_ids = list(r_model.child_dict.keys())
    r_chain = r_model[r_chain_ids[0]]

    # l_seq_len, r_seq_len = len(l_chain), len(r_chain)

    dist_map = np.zeros([N1, N2])

    # dist_map = np.zeros([l_seq_len, r_seq_len])
    ii = 0
    for _, l_residue in enumerate(l_chain):
        jj = 0
        if l_residue.resname != l_residue_list[ii]:
            continue
        for _, r_residue in enumerate(r_chain):
            
            if r_residue.resname != r_residue_list[jj]:
                continue


            min_dist = float("inf")

            for l_atom in l_residue:
                for r_atom in r_residue:
                    if l_atom.element == 'H' or r_atom.element == 'H':
                        continue
                    else:
                        dist = cal_dist(l_atom.get_coord(), r_atom.get_coord())
                        if dist < min_dist:
                            min_dist = dist
            
            dist_map[ii, jj] = min_dist
            jj += 1

        ii += 1

    # import pdb; pdb.set_trace()
    contact_map = (dist_map < 8.0).astype(int)
    # import pdb; pdb.set_trace()     
    flatten_contact_map = contact_map.flatten()
    # contact_indices = np.where(contact_map.flatten() == 1)[0]

    return contact_map, flatten_contact_map

def gen_dict(pdb_id, seq, file_folder):
    seq_len = len(seq)
    # generate DCA_APC, DAC_DI, PSSM features
    if os.stat(os.path.join(file_folder, pdb_id+'_apc.mat')).st_size == 0 or os.stat(os.path.join(file_folder, pdb_id+'_di.mat')).st_size == 0 or os.stat(os.path.join(file_folder, pdb_id+'.hhm')).st_size == 0:
        print('cannot parse {} DCA_APC, DAC_DI, PSSM files'.format(pdb_id))
        return None
    else:
        DCA_APC = np.loadtxt(os.path.join(file_folder, pdb_id+'_apc.mat'))
        DCA_DI = np.loadtxt(os.path.join(file_folder, pdb_id+'_di.mat'))
        PSSM = load_hmm(os.path.join(file_folder, pdb_id+'.hhm'))['PSSM']
    # Load mon_distance
    if os.path.exists(os.path.join(file_folder, pdb_id+'_mon_distance.out')):
        try:
            # gen pre-dict
            residue_list = []
            for item in seq:
                residue = aa1to3[item]
                residue_list.append(residue)
            N = len(residue_list)
            all_pairs = [[((residue_list[i], residue_list[j]), (i, j)) for j in range(N)] for i in range(N)]
            flat_all_pairs = [x for row in all_pairs for x in row]

            mon_dist = np.zeros((seq_len, seq_len))
            count = 0
            # import pdb; pdb.set_trace()
            for line in open(os.path.join(file_folder, pdb_id+'_mon_distance.out'), 'rb').readlines():
                idx1, res1, idx2, res2, dist = line.split()[:5]

                if (res1.decode(), res2.decode()) in flat_all_pairs[count]:
                    # mon_dist[int(idx1)-1, int(idx2)-1] = float(dist)
                    ii, jj = flat_all_pairs[count][-1]
                    mon_dist[ii, jj] = float(dist)
                    count += 1
            # import pdb; pdb.set_trace()
        except:
            print('cannot parse {} mon_distance file'.format(pdb_id))
            return None
    else:
        print('Missing the {} mon_distance file'.format(pdb_id))
        return None
        # exit()

    # Load the SA
    if os.path.exists(os.path.join(file_folder, pdb_id+'_renum.rsa')):
        try:
            sa = np.ones((seq_len, 1)) * -1.0
            for line in open(os.path.join(file_folder, pdb_id+'_renum.rsa')).readlines():
                if line[:3] == "RES":
                    _, _, _, res_idx, AA, AAR, SS, SSR, BB, BBR = line.split()[:10]
                    sa[int(res_idx)-1, 0] = float(AA)
        except:
            print('cannot parse {} SA file'.format(pdb_id))
            return None

    elif os.path.exists(os.path.join(file_folder, pdb_id+'_renum.freesasa')):
        # import pdb; pdb.set_trace()
        try:

            residue_list = []
            index = 0
            for item in seq:
                residue = aa1to3[item]
                residue_list.append((residue, index))
                index += 1
            N = len(residue_list)
            sa = np.ones((seq_len, 1)) * -1.0
            count = 0
            for line in open(os.path.join(file_folder, pdb_id+'_renum.freesasa')).readlines():
                if line[:3] == "SEQ":
                    _, _, res_idx, res, _, AA = line.split()[:6]
                    # import pdb; pdb.set_trace()
                    if res == residue_list[count][0]:
                        ii = residue_list[count][-1]
                        sa[ii, 0] = float(AA)
                        count += 1
                        # sa[int(res_idx)-1, 0] = float(AA)
            # import pdb; pdb.set_trace()
        except:
            print('cannot parse {} SA file'.format(pdb_id))
            return None
    else:
        print('Missing the {} SA files generated by naccess/freesasa'.format(pdb_id))
        return None
        # exit()
    
    data = dict()
    data["PSSM"] = PSSM
    data['DCA_DI'] = np.expand_dims(DCA_DI, 0)
    data["DCA_APC"] = np.expand_dims(DCA_APC, 0)
    data["Mon_distance"] = np.expand_dims(mon_dist, 0)
    data['SA'] = sa

    # esm_msa features
    try:
        data_esm = pkl.load(open(os.path.join(file_folder, pdb_id+'_esm_msa.pkl'), 'rb'))
    except:
        print('cannot load {} esm file'.format(pdb_id))
        return None
    data['esm_msa_1d'] = data_esm['esm_msa_1d']
    data['row_attentions'] = data_esm['row_attentions']

    return data

def gen_homo_pkl(l_pdb_id, r_pdb_id, l_seq, file_folder, pdb_folder, fa_folder, save_dir):

    pdb_name = l_pdb_id.split('_')[0]
    l_data = gen_dict(l_pdb_id, l_seq, file_folder)
    l_seq_len = len(l_seq)
    if l_data is None:
        return 
    else:
        # generate receptor features
        rec1d = np.concatenate([l_data['PSSM'], l_data['esm_msa_1d']], axis=-1).transpose(1,0)
        rec2d = np.concatenate([l_data['DCA_DI'], l_data['DCA_APC'], l_data['row_attentions'].reshape(144, l_seq_len, l_seq_len)], axis=0)
        recsa = l_data['SA']
        intra_distA = l_data['Mon_distance']
        com2d = rec2d  # generate com features
        rec2d = np.concatenate([rec2d, rbf(intra_distA)], axis=0)

        # generate ligand features
        lig1d = rec1d
        lig2d = rec2d
        ligsa = recsa
        intra_distB = intra_distA

        contact_map, flatten_contact_map = gen_contact(l_pdb_id, r_pdb_id, pdb_folder, fa_folder)
        np.savez(os.path.join(save_dir, pdb_name+'.npz'),
                seqA = str(l_seq),
                seqB = str(l_seq),
                intra_distA = intra_distA.astype(np.float32),
                intra_distB = intra_distB.astype(np.float32),
                rec1d = rec1d.astype(np.float32),
                rec2d = rec2d.astype(np.float32),
                recsa = recsa.astype(np.float32),
                lig1d = lig1d.astype(np.float32),
                lig2d = lig2d.astype(np.float32),
                ligsa = ligsa.astype(np.float32),
                com2d = com2d.astype(np.float32),
                contact_map = contact_map.astype(np.float32),
                flatten_contact_map = flatten_contact_map.astype(np.float32),
            )

def gen_hetero_pkl(l_pdb_id, r_pdb_id, l_seq, r_seq, file_folder, pdb_folder, fa_folder, save_dir):

    pdb_name = l_pdb_id.split('_')[0]

    l_data, r_data = gen_dict(l_pdb_id, l_seq, file_folder), gen_dict(r_pdb_id, r_seq, file_folder)
    l_seq_len, r_seq_len = len(l_seq), len(r_seq)

    if l_data is None or r_data is None:
        return 
    else:
        # generate receptor features
        rec1d = np.concatenate([l_data['PSSM'], l_data['esm_msa_1d']], axis=-1).transpose(1,0)
        rec2d = np.concatenate([l_data['DCA_DI'], l_data['DCA_APC'], l_data['row_attentions'].reshape(144, l_seq_len, l_seq_len)], axis=0)
        recsa = l_data['SA']
        intra_distA = l_data['Mon_distance']
        rec2d = np.concatenate([rec2d, rbf(intra_distA)], axis=0)

        # generate ligand features
        lig1d = np.concatenate([r_data['PSSM'], r_data['esm_msa_1d']], axis=-1).transpose(1,0)
        lig2d = np.concatenate([r_data['DCA_DI'], r_data['DCA_APC'], r_data['row_attentions'].reshape(144, r_seq_len, r_seq_len)], axis=0)
        ligsa = r_data['SA']
        intra_distB = r_data['Mon_distance']
        lig2d = np.concatenate([lig2d, rbf(intra_distB)], axis=0)

        # generate paired features
        paired_pdb_id = l_pdb_id.split('_')[0] + "_l_r_u"
        com_dca_di = np.expand_dims(np.loadtxt(os.path.join(file_folder, paired_pdb_id+'_paired_di.mat')) , 0)
        com_dca_apc = np.expand_dims(np.loadtxt(os.path.join(file_folder, paired_pdb_id+'_paired_apc.mat')) , 0)

        try:
            com_esm_msa = pkl.load(open( os.path.join(file_folder, paired_pdb_id+'_esm_msa.pkl'), 'rb') )
        except:
            print('cannot load {} esm file'.format(paired_pdb_id))
            return None
        
        com_row_attn = com_esm_msa['row_attentions']
        com2d = np.concatenate([com_dca_di[:, :l_seq_len, l_seq_len:l_seq_len+r_seq_len], com_dca_apc[:, :l_seq_len, l_seq_len:l_seq_len+r_seq_len], com_row_attn.reshape(144, l_seq_len+r_seq_len, l_seq_len+r_seq_len)[:, :l_seq_len, l_seq_len:l_seq_len+r_seq_len]], axis=0)

        contact_map, flatten_contact_map = gen_contact(l_pdb_id, r_pdb_id, pdb_folder, fa_folder)

        np.savez(os.path.join(save_dir, pdb_name+'.npz'),
                seqA = str(l_seq),
                seqB = str(r_seq),
                intra_distA = intra_distA.astype(np.float32),
                intra_distB = intra_distB.astype(np.float32),
                rec1d = rec1d.astype(np.float32),
                rec2d = rec2d.astype(np.float32),
                recsa = recsa.astype(np.float32),
                lig1d = lig1d.astype(np.float32),
                lig2d = lig2d.astype(np.float32),
                ligsa = ligsa.astype(np.float32),
                com2d = com2d.astype(np.float32),
                contact_map = contact_map.astype(np.float32),
                flatten_contact_map = flatten_contact_map.astype(np.float32),
            )


def main():
    # fa_dir = '/home/duhao.d/duhao-data/tmp/tmp_fa/real_world_pdb'
    # file_dir = '/home/duhao.d/duhao-data/tmp/tmp_file/real_world_pdb'
    # pdb_dir = '/home/duhao.d/duhao-data/tmp/tmp_pdb/real_world_pdb'

    fa_dir = './tmp/fa'
    file_dir = './tmp/file'
    pdb_dir = './tmp/pdb'

    save_dir = './save'
    folders = os.listdir(fa_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    homo_list = []
    hetero_list = []

    for folder in tqdm(folders):
        fa_folder = os.path.join(fa_dir, folder)
        file_folder = os.path.join(file_dir, folder)
        pdb_folder = os.path.join(pdb_dir, folder)

        file_names = os.listdir(fa_folder)
        fa_file_names = []

        for file_name in file_names:
            if file_name.endswith('.fasta'):
                fa_file_names.append(file_name)
        
        l_fa_file = os.path.join(fa_folder, fa_file_names[0])
        r_fa_file = os.path.join(fa_folder, fa_file_names[1])
        
        assert os.path.exists(l_fa_file)
        assert os.path.exists(r_fa_file)
        
        l_seq, r_seq = get_seq_len(l_fa_file), get_seq_len(r_fa_file)
        l_pdb_id = l_fa_file.split('/')[-1].split('.')[0]
        r_pdb_id = r_fa_file.split('/')[-1].split('.')[0]

        pdb_name = l_pdb_id.split('_')[0]

        if os.path.exists(os.path.join(save_dir, '{}.npz'.format(pdb_name))):
            continue
        else:
            if l_seq != r_seq:
                hetero_list.append((l_pdb_id, r_pdb_id, l_seq, r_seq, file_folder, pdb_folder, fa_folder, save_dir))
            else:
                print(l_pdb_id, r_pdb_id)
                homo_list.append((l_pdb_id, r_pdb_id, l_seq, file_folder, pdb_folder, fa_folder, save_dir))
    
    # import pdb; pdb.set_trace()
    print('----generate hetero features----')
    pmap_multi(
        gen_hetero_pkl,
        hetero_list,
        n_jobs=1
    )
    print('----finish----')

    print('----generate homo features----')
    pmap_multi(
        gen_homo_pkl,
        homo_list,
        n_jobs=1
    )
    print('----finish----')
    

if __name__ == "__main__":

    main()