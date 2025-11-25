import os


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

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

def main():
    raw_pdb_dir = './example'
    pdb_id = '1m10'
    l_pdb_file_path, r_pdb_file_path = os.path.join(raw_pdb_dir, '{}_l_u.pdb'.format(pdb_id)), os.path.join(raw_pdb_dir, '{}_r_u.pdb'.format(pdb_id))
    l_pdb_id = l_pdb_file.split('/')[-1].split('.')[0]
    r_pdb_id = r_pdb_file.split('/')[-1].split('.')[0]

    # tmp dir
    tmp_pdb_dir = './tmp/pdb'
    tmp_fa_dir = './tmp/fa'
    tmp_file_dir = './tmp/file'
    mkdir(tmp_pdb_dir)
    mkdir(tmp_fa_dir)
    mkdir(tmp_file_dir)

    # generate fa file
    os.system('./gen_fa.sh {} {} {} {}'.format(l_pdb_id, raw_pdb_dir, tmp_pdb_dir, tmp_fa_dir))   # generate left fasta sequence
    os.system('./gen_fa.sh {} {} {} {}'.format(r_pdb_id, raw_pdb_dir, tmp_pdb_dir, tmp_fa_dir))   # generate right fasta sequence

    # compare the sequence
    l_seq = get_seq_len('{}/{}.fasta'.format(tmp_fa_dir, l_pdb_id))
    r_seq = get_seq_len('{}/{}.fasta'.format(tmp_fa_dir, r_pdb_id))

    if l_seq != r_seq:
        pdb_id_paired = l_pdb_id.split('_')[0] + "_l_r_u"
        os.system('./gen_homo.sh {} {} {} {}'.format(l_pdb_id, tmp_pdb_dir, tmp_fa_dir, tmp_file_dir))
        os.system('./gen_homo.sh {} {} {} {}'.format(r_pdb_id, tmp_pdb_dir, tmp_fa_dir, tmp_file_dir))
        os.system('./gen_hetero.sh {} {} {} {}'.format(tmp_file_dir, l_pdb_id, r_pdb_id, pdb_id_paired))
    else:
        os.system('./gen_homo.sh {} {} {} {}'.format(l_pdb_id, tmp_pdb_dir, tmp_fa_dir, tmp_file_dir))



if __name__ == "__main__":
    main()