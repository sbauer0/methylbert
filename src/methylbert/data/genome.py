import argparse

# NCBI-style accession to chr name mapping for hg38
NCBI_TO_CHR = {
    "NC_000001.11": "chr1",  "NC_000002.12": "chr2",
    "NC_000003.12": "chr3",  "NC_000004.12": "chr4",
    "NC_000005.10": "chr5",  "NC_000006.12": "chr6",
    "NC_000007.14": "chr7",  "NC_000008.11": "chr8",
    "NC_000009.12": "chr9",  "NC_000010.11": "chr10",
    "NC_000011.10": "chr11", "NC_000012.12": "chr12",
    "NC_000013.11": "chr13", "NC_000014.9":  "chr14",
    "NC_000015.10": "chr15", "NC_000016.10": "chr16",
    "NC_000017.11": "chr17", "NC_000018.12": "chr18",
    "NC_000019.10": "chr19", "NC_000020.11": "chr20",
    "NC_000021.9":  "chr21", "NC_000022.11": "chr22",
    "NC_000023.11": "chrX",  "NC_000024.10": "chrY"
}



def _kmers(original_string, kmer=3):
    sentence = ""
    #original_string = original_string.replace("\n", "")
    i = 0
    while i < len(original_string)-kmer:
        sentence += original_string[i:i+kmer] + " "
        i += 1
    
    return sentence[:-1].strip("\"")


def pretrain_data_preprocess(f_ref, k=3, seq_len=510, f_output=None):
    '''
        Generate N bp length k-mers sequence from reference genome as pretrain data

        f_ref : str
            path to the reference fasta file
        k : int
            Number for k-mers (default=3)
        seq_len: int
            Base-pair length of generated sequences (default=510)
        f_output : str
            path to the output file, an appropriate name 
            will be automatically assigned if not given

    '''

    fp_ref = open(f_ref, "r")
    if f_output == None:
        f_output = f_ref + "_%dmers.txt"%(k)
    
    fp_out = open(f_output, "w")
    line = fp_ref.readline().strip().upper()
    cur_line="" # keeping an incomplete N bp line
    collect_data=False
    valid_chromosomes = ["CHR"+str(i) for i in range(22)]
    valid_chromosomes += ["CHRX", "CHRY"]
    
    while line:
        n_missing = line.count("N")

        if n_missing > 0:
            # Missing DNA base in the line -> reset the line 
            #line = fp_ref.readline().strip().upper()
            line=fp_ref.readline().strip().upper()
            cur_line=""
            continue
        elif ( line.count(">") > 0 ):
            # New chromosome or there are some missing bases at the middle
            # We restart a 510 bp piece
            chromosome = line.split(">")[1]
            collect_data = chromosome in valid_chromosomes
            if collect_data:
                print("Collect sequences in %s"%(chromosome))
            line = fp_ref.readline().strip().upper()
            cur_line = "" 
            continue

        if collect_data:
            cur_line += line
            line_length = len(cur_line)
            if line_length >= seq_len:
                new_line = cur_line[:seq_len]
                cur_line = cur_line[seq_len:]
                sentence = _kmers(new_line, kmer=k)
                fp_out.write(sentence + "\n")

        # get a new line 
        line = fp_ref.readline().strip().upper()
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kmer",
        default=1,
        type=int,
        help="K-mer",
    )
    parser.add_argument(
        "--length",
        default=10000,
        type=int,
        help="Length of the sampled sequence",
    )
    parser.add_argument(
        "--file_path",
        default=None,
        type=str,
        help="The path of the file to be processed",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        help="The path of the processed data",
    )
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = parse_args()
    pretrain_data_preprocess(args.file_path, k=args.kmer, 
                             seq_len=args.length, f_output=args.output_path)

