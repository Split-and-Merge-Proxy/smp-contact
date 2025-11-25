# -----------------------------------------------------------------------------------------------------------------------------
# the input variables
pdb_id=$1
raw_pdb_dir=$2
tmp_pdb_dir=$3
tmp_fa_dir=$4

packages_dir="./packages"

if [ ! -f ${tmp_fa_dir}/${pdb_id}.fasta ];then
    printf "*%.0s" {1..128}; printf '%s\n'
    echo "Generating fasta of ${pdb_id}"
    # -----------------------------------------------------------------------------------------------------------------------------
    # clean the unstandard residues of target structure --> {pdb_id}_clean.pdb
    egrep -v "HOH|WAT" ${raw_pdb_dir}/${pdb_id}.pdb > ${tmp_pdb_dir}/${pdb_id}_clean.pdb

    # modify the unstandard residues to standard reisdues --> {pdb_id}_modified.pdb
    sed "s/MEX/CYS/g; s/HID/HIS/g; s/HIE/HIS/g; s/HIP/HIS/g; s/MSE/MET/g; s/ASX/ASN/g; s/GLX/GLN/g; s/TYS/TRP/g" ${tmp_pdb_dir}/${pdb_id}_clean.pdb > ${tmp_pdb_dir}/${pdb_id}_modified.pdb

    # generate the sequence --> {pdb_id}.fasta
    $packages_dir/pdb2fasta ${tmp_pdb_dir}/${pdb_id}_modified.pdb > ${tmp_fa_dir}/${pdb_id}.fasta

    # check the length of the sequence
    length=`awk '{if(NR>1){len=len+length($0)}}END{print len}' ${tmp_fa_dir}/${pdb_id}.fasta`
else
    echo "the fasta file of ${pdb_id} already exists"
fi

if [ "$length" -eq 0 ]; then
    echo "The length of the ${pdb_id} is empty"
    exit 2
fi