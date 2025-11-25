# -----------------------------------------------------------------------------------------------------------------------------
# input variables
tmp_file_dir=$1
pdb_id_1=$2
pdb_id_2=$3
pdb_id_paired=$4


# -----------------------------------------------------------------------------------------------------------------------------
# some path
packages_dir="./packages"
hhblits_bin=$(which hhblits)
hhmake_bin=$(which hhmake)
hhfilter_bin=$(which hhfilter)
UniRef_database="/your/path/to/UniRef30_2020_03/UniRef30_2020_03"
esm_msa_model="/your/path/to/esm_msa1_t12_100M_UR50S.pt"


# -----------------------------------------------------------------------------------------------------------------------------
# generate the paired MSA
python ./gen_paired_msa.py --msa_dir $tmp_file_dir --pdb_id_1 $pdb_id_1 --pdb_id_2 $pdb_id_2 --save_dir $tmp_file_dir --paired_pdb_id $pdb_id_paired


# generate the PSSM, DCA feats with the paired MSA
if [ -f ${tmp_file_dir}/${pdb_id_paired}_paired.a3m ];then

    # generate HHM file for PSSM
    if [ ! -f ${tmp_file_dir}/${pdb_id_paired}_paired.hhm ];then
        printf "*%.0s" {1..128}; printf '%s\n'
        echo "Generating the HHM file with ${pdb_id_paired}_paired.a3m"
        $hhmake_bin -i ${tmp_file_dir}/${pdb_id_paired}_paired.a3m -o ${tmp_file_dir}/${pdb_id_paired}_paired.hhm
    fi

    # generate target_paired.aln for DCA feats
    if [ ! -f ${tmp_file_dir}/${pdb_id_paired}_paired.aln ];then
        printf "*%.0s" {1..128}; printf '%s\n'
        echo "Generateing the ${pdb_id_paired}_paired.aln file"
        $packages_dir/reformat.pl $tmp_file_dir/${pdb_id_paired}_paired.a3m $tmp_file_dir/${pdb_id_paired}_paired.fas -r -l 2000 >/dev/null
        awk '{if(!($0~/^>/)){print}}' ${tmp_file_dir}/${pdb_id_paired}_paired.fas > ${tmp_file_dir}/${pdb_id_paired}_paired.aln
    fi


    if [ -f $tmp_file_dir/${pdb_id_paired}_paired.aln ];then

        # generate DCA-DI
        if [ ! -f $tmp_file_dir/${pdb_id_paired}_paired_di.mat ];then
            printf "*%.0s" {1..128}; printf '%s\n'
            $packages_dir/ccmpred $tmp_file_dir/${pdb_id_paired}_paired.aln $tmp_file_dir/${pdb_id_paired}_paired_di.mat -R -A
        fi

        # generate DCA-APC
        if [ ! -f $tmp_file_dir/${pdb_id_paired}_paired_apc.mat ];then
            printf "*%.0s" {1..128}; printf '%s\n'
            $packages_dir/ccmpred $tmp_file_dir/${pdb_id_paired}_paired.aln $tmp_file_dir/${pdb_id_paired}_paired_apc.mat -R
        fi

    else
        echo "Missing the ${pdb_id_paired}_paired.aln file for generating the DCA feats"
    fi

else
    echo "Processing DCA feats : Missing the input MSA (${pdb_id_paired}_paired.a3m) file"
    exit 2
fi

# -----------------------------------------------------------------------------------------------------------------------------
# generate the ESM-MSA feats
if [ -f $tmp_file_dir/${pdb_id_paired}_paired.a3m ];then

    if [ ! -f $tmp_file_dir/${pdb_id_paired}_esm_msa.pkl ];then
        printf "*%.0s" {1..128}; printf '%s\n'
        echo "Generating ESM-MSA feats with ${pdb_id_paired}_paired.a3m"
        $hhfilter_bin -i $tmp_file_dir/${pdb_id_paired}_paired.a3m -o $tmp_file_dir/${pdb_id_paired}_filter.a3m -diff 512
        python ./extract_esm_features.py --esm_file $esm_msa_model --msa_dir $tmp_file_dir --pdb_id $pdb_id_paired --save_dir $tmp_file_dir
    fi
else
    echo "Processing ESM-MSA feautres : Missing the input MSA (${pdb_id_paired}_paired.a3m) file"
    exit 2
fi