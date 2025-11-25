# -----------------------------------------------------------------------------------------------------------------------------
# the input variables
pdb_id=$1
tmp_pdb_dir=$2
tmp_fa_dir=$3
tmp_file_dir=$4

# -----------------------------------------------------------------------------------------------------------------------------
# some path
packages_dir="./packages"
hhblits_bin=$(which hhblits)
hhmake_bin=$(which hhmake)
hhfilter_bin=$(which hhfilter)
UniRef_database="/your/path/to/UniRef30_2020_03/UniRef30_2020_03"
esm_msa_model="/your/path/to/esm_msa1_t12_100M_UR50S.pt"


# -----------------------------------------------------------------------------------------------------------------------------
# generate the MSA from hhblits
if [ -f ${tmp_fa_dir}/${pdb_id}.fasta ];then
	if [ ! -f ${tmp_file_dir}/${pdb_id}.a3m ];then
        printf "*%.0s" {1..128}; printf '%s\n'
		echo "Generating MSA of ${pdb_id}"
		$hhblits_bin -i ${tmp_fa_dir}/${pdb_id}.fasta -d $UniRef_database -cpu 16 -oa3m ${tmp_file_dir}/${pdb_id}.a3m -n 3 -e 0.001 -id 99 -cov 0.4
	fi
else
	echo "Missing the sequence file of ${pdb_id}"
    exit 2
fi

# generate MSA feats from pdb_id.a3m 
if [ -f ${tmp_file_dir}/${pdb_id}.a3m ];then

    # generate HHM file for PSSM
    if [ ! -f ${tmp_file_dir}/${pdb_id}.hhm ];then
        printf "*%.0s" {1..128}; printf '%s\n'
		echo "Generating the HHM file with ${pdb_id}.a3m"
	    $hhmake_bin -i ${tmp_file_dir}/${pdb_id}.a3m -o ${tmp_file_dir}/${pdb_id}.hhm
    fi

    # generate target.aln
    if [ ! -f ${tmp_file_dir}/${pdb_id}.aln ];then
		printf "*%.0s" {1..128}; printf '%s\n'
        echo "Generating the ${pdb_id}.aln file for DCA feats"
        $packages_dir/reformat.pl ${tmp_file_dir}/${pdb_id}.a3m ${tmp_file_dir}/${pdb_id}.fas -r -l 2000 >/dev/null
        awk '{if(!($0~/^>/)){print}}' ${tmp_file_dir}/${pdb_id}.fas > ${tmp_file_dir}/${pdb_id}.aln
    fi


    # generate the DCA feats with target.aln
    if [ -f ${tmp_file_dir}/${pdb_id}.aln ];then

        # generate DCA-DI
    	if [ ! -f ${tmp_file_dir}/${pdb_id}_di.mat ];then
            printf "*%.0s" {1..128}; printf '%s\n'
			echo "Generating the DCA-DI file with ${pdb_id}.aln"
	    	$packages_dir/ccmpred ${tmp_file_dir}/${pdb_id}.aln ${tmp_file_dir}/${pdb_id}_di.mat -R -A
    	fi

        # generate DCA-APC
        if [ ! -f ${tmp_file_dir}/${pdb_id}_apc.mat ];then
            printf "*%.0s" {1..128}; printf '%s\n'
			echo "Generating the DCA-APC file with ${pdb_id}.aln"
            $packages_dir/ccmpred ${tmp_file_dir}/${pdb_id}.aln ${tmp_file_dir}/${pdb_id}_apc.mat -R
        fi

    else
        echo "Missing the ${pdb_id}.aln for generate the DCA feats"
        exit 2
    fi

else
    echo "Processing DCA feats : Missing the input MSA (${pdb_id}.a3m) file"
    exit 2
fi

# -----------------------------------------------------------------------------------------------------------------------------
# generate the ESM-MSA feats
if [ -f ${tmp_file_dir}/${pdb_id}.a3m ];then

    if [ ! -f ${tmp_file_dir}/${pdb_id}_esm_msa.pkl ];then
        printf "*%.0s" {1..128}; printf '%s\n'
		echo "Generating ESM-MSA feats with ${pdb_id}.a3m"
     	$hhfilter_bin -i ${tmp_file_dir}/${pdb_id}.a3m -o ${tmp_file_dir}/${pdb_id}_filter.a3m -diff 512
	    python ./extract_esm_features.py --esm_file $esm_msa_model --msa_dir $tmp_file_dir --pdb_id $pdb_id --save_dir $tmp_file_dir
	fi
else
    echo "Processing ESM-MSA feautres : Missing the input MSA (${pdb_id}.a3m) file"
    exit 2
fi


# -----------------------------------------------------------------------------------------------------------------------------
# renumber the residue id from {pdb_id}_modified.pdb --> {pdb_id}_renum.pdb
awk '{s=substr($0,18,10);
    if(substr($1,1,4)=="ATOM"||substr($1,1,6)=="HETATM"){
        if(s!=s0)n++;
        printf"%s%4d %s\n",substr($0,1,22),n,substr($0,28);
        s0=s
    }
    else{
             print
    }
}'  ${tmp_pdb_dir}/${pdb_id}_modified.pdb > ${tmp_pdb_dir}/${pdb_id}_renum.pdb


# generate the structure feats including intra-dist, SA
if [ -f ${tmp_pdb_dir}/${pdb_id}_renum.pdb ];then
    printf "*%.0s" {1..128}; printf '%s\n'
    echo "Generating SA features"

	${packages_dir}/freesasa ${tmp_pdb_dir}/${pdb_id}_renum.pdb --format=seq -o ${tmp_file_dir}/${pdb_id}_renum.freesasa
	
    # name_software=$(basename $sa_software)
    # if [ $name_software = "naccess" ];then
    	# ${sa_software} ${pdb_id}_renum.pdb 
    # elif [ $name_software = "freesasa" ];then
        # ${sa_software} ${pdb_id}_renum.pdb --format=seq -o ${pdb_id}_renum.freesasa
    # else
        # echo "Please set the right software (naccess/freesasa) for SA"
        # exit 2
    # fi

	printf "*%.0s" {1..128}; printf '%s\n'
    echo "Generating MonDistance feature"
   	${packages_dir}/dis_rec_lig_all ${tmp_pdb_dir}/${pdb_id}_renum.pdb ${tmp_pdb_dir}/${pdb_id}_renum.pdb -o ${tmp_file_dir}/${pdb_id}_mon_distance.out
else
 	echo "Processing structure features: Missing the target pdb file"
	exit 2
fi