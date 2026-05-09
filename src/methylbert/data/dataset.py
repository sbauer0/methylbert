import gc
import multiprocessing as mp
import random
from copy import deepcopy
from functools import partial
import json
import glob
import os


import numpy as np
import torch
from torch.utils.data import Dataset

from methylbert.data.vocab import MethylVocab

import bisect
from methylbert.data.nanopore.featurize import (
         STATE_NON_CPG,
         STATE_UNKNOWN,
        )


def _line2tokens_pretrain(l, tokenizer, max_len=120):
    '''
        convert a text line into a list of tokens converted by tokenizer
    '''
    l = l.strip().split(" ")
    tokened = tokenizer.to_seq(l)
    if len(tokened) > max_len:
        return tokened[:max_len]
    else:
        return tokened + [tokenizer.pad_index for _ in range(max_len - len(tokened))]

def _parse_line(l, headers):
	# Check the header
	if not all([h in headers for h in ["dna_seq", "methyl_seq", "ctype", "dmr_ctype", "dmr_label"]]):
		raise ValueError("The header must contain dna_seq, methyl_seq, ctype, dmr_ctype, dmr_label")

	# Separate n-mers tokens and labels from each line
	l = l.split("\t")  # don't add strip; some columns may be None
	if len(headers) == len(l):
		l = {k: v for k, v in zip(headers, l)}
	else:
		raise ValueError(f"Only {len(headers)} elements are in the input file header, whereas the line has {len(l)} elements.")

	# Cell-type label is binary (whether the cell type corresponds to the DMR cell type)
	l["ctype_label"] = int(l["ctype"] == l["dmr_ctype"])
	l["dmr_label"] = int(l["dmr_label"])

	return l


def _line2tokens_finetune(l, tokenizer, max_len=150, headers=None):
	# parsed line!

	l["dna_seq"] = l["dna_seq"].split(" ")
	l["dna_seq"] = [[f] for f in tokenizer.to_seq(l["dna_seq"])]
	l["methyl_seq"] = [int(m) for m in l["methyl_seq"]]

	if len(l["dna_seq"]) > max_len:
		l["dna_seq"] = l["dna_seq"][:max_len]
		l["methyl_seq"] = l["methyl_seq"][:max_len]
	else:
		cur_seq_len=len(l["dna_seq"])
		l["dna_seq"] = l["dna_seq"]+[[tokenizer.pad_index] for k in range(max_len-cur_seq_len)]
		l["methyl_seq"] = l["methyl_seq"] + [2 for k in range(max_len-cur_seq_len)]

	return l

class MethylBertDataset(Dataset):
	def __init__(self):
		pass

	def __len__(self):
		return self.lines.shape[0] if type(self.lines) == np.array else len(self.lines)


class MethylBertPretrainDataset(MethylBertDataset):
	def __init__(self, f_path: str, vocab: MethylVocab, seq_len: int, random_len=False, n_cores=50):

		self.vocab = vocab
		self.seq_len = seq_len
		self.f_path = f_path
		self.random_len = random_len

		# Define a range of tokens to mask based on k-mers
		self.mask_list = self._get_mask()

		# Read all text files and convert the raw sequence into tokens
		with open(self.f_path, "r") as f_input:
			print("Open data : %s"%f_input)
			raw_seqs = f_input.read().splitlines()

		print("Total number of sequences : ", len(raw_seqs))

		# Multiprocessing for the sequence tokenisation
		with mp.Pool(n_cores) as pool:
			line_labels = pool.map(partial(_line2tokens_pretrain,
								           tokenizer=self.vocab,
								           max_len=self.seq_len), raw_seqs)
			del raw_seqs
			print("Lines are processed")
			self.lines = torch.squeeze(torch.tensor(np.array(line_labels, dtype=np.int16)))
		del line_labels
		gc.collect()

	def __getitem__(self, index):

		dna_seq = self.lines[index].clone()

		# Random len
		if self.random_len and np.random.random() < 0.5:
			dna_seq = dna_seq[:random.randint(5, self.seq_len)]

		# Padding
		if dna_seq.shape[0] < self.seq_len:
			pad_num = self.seq_len-dna_seq.shape[0]
			dna_seq = torch.cat((dna_seq,
								torch.tensor([self.vocab.pad_index for i in range(pad_num)], dtype=torch.int16)))

		# Mask
		masked_dna_seq, dna_seq, bert_mask = self._masking(dna_seq)
		#print(dna_seq, masked_dna_seq,"\n=============================================\n")
		return {"bert_input": masked_dna_seq,
				"bert_label": dna_seq,
				"bert_mask" : bert_mask}

	def subset_data(self, n_seq: int):
		self.lines = random.sample(self.lines, n_seq)

	def _get_mask(self):
		'''
			Relative positions from the centre of masked region
			e.g) [-1, 0, 1] for 3-mers
		'''
		half_length = int(self.vocab.kmers/2)
		mask_list = [-1*half_length + i for i in range(half_length)] + [i for i in range(1, half_length+1)]
		if self.vocab.kmers % 2 == 0:
			mask_list = mask_list[:-1]

		return mask_list

	"""def _masking(self, inputs: torch.Tensor, threshold=0.15):
		
			Modified version of masking token function
			Originally developed by Huggingface (datacollator) and DNABERT

			https://github.com/huggingface/transformers/blob/9a24b97b7f304fa1ceaaeba031241293921b69d3/src/transformers/data/data_collator.py#L747

			https://github.com/jerryji1993/DNABERT/blob/bed72fc0694a7b04f7e980dc9ce986e2bb785090/examples/run_pretrain.py#L251

			Added additional tasks to handle each sequence
			Lines using tokenizer were modified due to different tokenizer object structure

		

		labels = inputs.clone()

		# Sample tokens with given probability threshold
		probability_matrix = torch.full(labels.shape, threshold) # tensor filled with 0.15

		# Handle special tokens and padding
		special_tokens_mask = [
			val < 5 for val in labels.tolist()
		]
		probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
		#padding_mask = labels.eq(self.vocab.pad_index)
		#probability_matrix.masked_fill_(padding_mask, value=0.0)

		masked_indices = torch.bernoulli(probability_matrix).bool() # get masked tokens based on bernoulli only within non-special tokens

		# change masked indices
		masked_index = deepcopy(masked_indices)

		# This function handles each sequence
		end = torch.where(probability_matrix!=0)[0].tolist()[-1] # end of the sequence
		mask_centers = set(torch.where(masked_index==1)[0].tolist()) # mask locations

		new_centers = deepcopy(mask_centers)
		for center in mask_centers:
			for mask_number in self.mask_list:# add neighbour loci
				current_index = center + mask_number
				if current_index <= end and current_index >= 0:
					new_centers.add(current_index)

		new_centers = list(new_centers)

		masked_indices[new_centers] = True

		# Avoid loss calculation on unmasked tokens
		labels[~masked_indices] = -100

		# 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
		indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
		inputs[indices_replaced] = self.vocab.mask_index

		# 10% of the time, we replace masked input tokens with random word
		indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
		random_words = torch.randint(
            low=self.vocab.mask_index + 1,
            high=len(self.vocab),
            size=labels.shape,
            dtype=torch.int16,
        )
		inputs[indices_random] = random_words[indices_random]

		# The rest of the time (10% of the time) we keep the masked input tokens unchanged

		inputs         = torch.cat((inputs,         torch.tensor([self.vocab.pad_index])))
		labels         = torch.cat((labels,         torch.tensor([-100])))
		masked_indices = torch.cat((masked_indices, torch.tensor([False])))

		# Place EOS in the new slot at index end + 1.
		inputs[end + 1] = self.vocab.eos_index

		# Prepend SOS to all three tensors. Final length: seq_len + 2 = 512.
		labels         = torch.cat((torch.tensor([-100]),                  labels))
		inputs         = torch.cat((torch.tensor([self.vocab.sos_index]),  inputs))
		masked_indices = torch.cat((torch.tensor([False]),                 masked_indices))

		return inputs, labels, masked_indices"""
	
	def _masking(self, inputs: torch.Tensor, threshold=0.15):
		"""
		Spaced-center masking.

		- Centers selected sequentially with Bernoulli(threshold), enforcing
		a minimum spacing of len(self.mask_list) + 1 = k positions between
		centers so no two blocks overlap.
		- Each center's k-mer block (center + neighbors from self.mask_list)
		is corrupted: neighbors always become [MASK]; the center follows
		80/10/10 mask/random/keep.
		- Loss is computed only at centers (not at neighbors).
		- For random replacement, the center's flanking bases are preserved
		and only the middle base is changed (k=3 only — see note).
		"""
		labels = inputs.clone()

		# Sequential center selection with spacing.
		# Skip specials (token id < 5).
		seq_len = inputs.shape[0]
		spacing = len(self.mask_list) + 1   # for k=3 this is 3
		centers = []
		p = 0
		while p < seq_len:
			if inputs[p].item() < 5:
				p += 1
				continue
			if torch.rand(1).item() < threshold:
				centers.append(p)
				p += spacing
			else:
				p += 1

		end = torch.where(inputs >= 5)[0]
		end = end[-1].item() if len(end) > 0 else seq_len - 1

		# Build the expanded set (centers + neighbors), bounded to [0, end].
		# Neighbors are always present in the corrupted input but not in labels.
		block_offsets = [0] + self.mask_list
		expanded_positions = set()
		for c in centers:
			for d in block_offsets:
				q = c + d
				if 0 <= q <= end:
					expanded_positions.add(q)

		# Loss only at centers: zero out labels everywhere else.
		label_mask = torch.zeros(seq_len, dtype=torch.bool)
		label_mask[centers] = True
		labels[~label_mask] = -100

		# Apply corruption.
		# Neighbors: always [MASK].
		for q in expanded_positions:
			if q not in centers:   # neighbor
				inputs[q] = self.vocab.mask_index

		# Centers: 80/10/10.
		for c in centers:
			r = torch.rand(1).item()
			if r < 0.8:
				inputs[c] = self.vocab.mask_index
			elif r < 0.9:
				inputs[c] = self._constrained_random_replacement(inputs[c].item())
			# else: keep original

		# masked_indices for downstream methylation-hiding: union of centers
		# and neighbors (everything in the corrupted-input region).
		masked_indices = torch.zeros(seq_len, dtype=torch.bool)
		masked_indices[list(expanded_positions)] = True

		# --- length 512 layout, unchanged from previous fix ---
		inputs         = torch.cat((inputs,         torch.tensor([self.vocab.pad_index])))
		labels         = torch.cat((labels,         torch.tensor([-100])))
		masked_indices = torch.cat((masked_indices, torch.tensor([False])))

		inputs[end + 1] = self.vocab.eos_index

		labels         = torch.cat((torch.tensor([-100]),                 labels))
		inputs         = torch.cat((torch.tensor([self.vocab.sos_index]), inputs))
		masked_indices = torch.cat((torch.tensor([False]),                masked_indices))

		return inputs, labels, masked_indices
	
	

class MethylBertFinetuneDataset(MethylBertDataset):
	def __init__(self, f_path: str, vocab: MethylVocab, seq_len: int, n_cores: int=10, n_seqs = None):
		'''
		MethylBERT dataset

		f_path: str
			File path to the processed input file
		vocab: MethylVocab
			MethylVocab object to convert DNA and methylation pattern sequences
		seq_len: int
			Length for the processed sequences
		n_cores: int
			Number of cores for multiprocessing
		n_seqs: int
			Number of sequences to subset the input (default: None, do not make a subset)

		'''
		self.vocab = vocab
		self.seq_len = seq_len
		self.f_path = f_path

		# Read all text files and convert the raw sequence into tokens
		with open(self.f_path, "r") as f_input:
			raw_seqs = f_input.read().splitlines()

		# Check if there's a header
		self.headers = raw_seqs[0].split("\t")
		raw_seqs = raw_seqs[1:]

		if n_seqs is not None:
			raw_seqs = raw_seqs[:n_seqs]
		print("Total number of sequences : ", len(raw_seqs))

		# Multiprocessing for the sequence tokenisation
		with mp.Pool(n_cores) as pool:
			self.lines = pool.map(partial(_parse_line,
								   headers=self.headers), raw_seqs)
			del raw_seqs
		gc.collect()
		self.set_dmr_labels = set([l["dmr_label"] for l in self.lines])

		self.ctype_label_count = self._get_cls_num()
		print("# of reads in each label: ", self.ctype_label_count)

	def _get_cls_num(self):
		# unique labels
		ctype_labels=[l["ctype_label"] for l in self.lines]
		labels = list(set(ctype_labels))
		label_count = np.zeros(len(labels))
		for l in labels:
			label_count[l] = sum(np.array(ctype_labels) == l)
		return label_count

	def num_dmrs(self):
		return max(len(self.set_dmr_labels), max(self.set_dmr_labels)+1) # +1 is for the label 0

	def subset_data(self, n_seq):
		self.lines = self.lines[:n_seq]

	def __getitem__(self, index):
		line = deepcopy(self.lines[index])

		item = _line2tokens_finetune(
			l=line,
			tokenizer=self.vocab, max_len=self.seq_len, headers=self.headers)

		item["dna_seq"] = torch.squeeze(torch.tensor(np.array(item["dna_seq"], dtype=np.int32)))
		item["methyl_seq"] = torch.squeeze(torch.tensor(np.array(item["methyl_seq"], dtype=np.int8)))

		# Special tokens (SOS, EOS)
		end = torch.where(item["dna_seq"]!=self.vocab.pad_index)[0].tolist()[-1] + 1 # end of the read
		if end < item["dna_seq"].shape[0]:
			item["dna_seq"][end] = self.vocab.eos_index
			item["methyl_seq"][end] = 2
		else:
			item["dna_seq"][-1] = self.vocab.eos_index
			item["methyl_seq"][-1] = 2
		item["dna_seq"] = torch.cat((torch.tensor([self.vocab.sos_index]), item["dna_seq"]))
		item["methyl_seq"] = torch.cat((torch.tensor([2]), item["methyl_seq"]))

		return item


class MethylBertPretrainDatasetBinary(MethylBertDataset):
    """
    Memory-mapped pretraining dataset that reads two-int binary shards
    produced by methylbert.data.nanopore.preprocess.

    Each shard in ``data_dir`` is three files sharing a basename:
        <name>.tokens.bin  flat int16 array of shape (n_rows, window_len)
        <name>.states.bin  flat int8  array of shape (n_rows, window_len)
        <name>.json        sidecar with {n_rows, window_len, vocab_size, ...}

    For each item, returns a dict:
        bert_input  : (window_len + 1,) int  — DNA tokens after MLM corruption,
                                                with SOS prepended and EOS at
                                                the last non-pad position.
        bert_label  : (window_len + 1,) int  — original DNA tokens at MLM-selected
                                                positions, -100 elsewhere.
        bert_mask   : (window_len + 1,) bool — True at MLM-selected positions
                                                (used by the original MethylBERT
                                                code; preserved for compatibility).
        methyl_seq  : (window_len + 1,) int  — methylation states in {0, 1, 2, 3}.
                                                Mirrors bert_input; per Option B,
                                                MLM-selected positions are reset
                                                to STATE_UNKNOWN to prevent the
                                                CpG-context from leaking the
                                                masked DNA token's identity.
                                                SOS and EOS positions are
                                                STATE_NON_CPG.
    """

    def __init__(self, data_dir: str, vocab: MethylVocab, seq_len: int,
                 random_len: bool = False):
        '''
            data_dir : str
                Directory containing matching .tokens.bin / .states.bin / .json
                triples (as produced by the preprocessor driver). Other files
                in the directory (e.g., manifest.tsv) are ignored.
            vocab : MethylVocab
                Vocabulary. Must match the vocab used during preprocessing.
            seq_len : int
                Window length (must match the value recorded in each shard's
                JSON sidecar). Items returned by __getitem__ are seq_len + 1
                long after SOS prepend.
            random_len : bool
                If True, randomly truncate sequences (and pad back) to simulate
                variable read lengths during training. Default: False.
        '''
        self.vocab = vocab
        self.seq_len = seq_len
        self.random_len = random_len
        self.data_dir = data_dir

        # Same k-mer-aware mask expansion as the text-based dataset.
        self.mask_list = self._get_mask()

        # Discover shard JSON sidecars.
        json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
        if len(json_files) == 0:
            raise FileNotFoundError(
                f"No shard JSON files found in {data_dir}"
            )

        self.tokens_memmaps = []
        self.states_memmaps = []
        self.offsets = [0]   # cumulative row counts across shards
        total_rows = 0

        for f_json in json_files:
            with open(f_json, "r") as f:
                meta = json.load(f)

            # Validate schema.
            if "n_rows" not in meta or "window_len" not in meta:
                raise ValueError(
                    f"{f_json} does not look like a nanopore shard sidecar "
                    f"(missing n_rows/window_len). Did you point at an old-format dir?"
                )
            if meta["window_len"] != seq_len:
                raise ValueError(
                    f"window_len mismatch in {f_json}: "
                    f"metadata says {meta['window_len']}, dataset expects {seq_len}"
                )
            if meta["vocab_size"] != len(vocab):
                raise ValueError(
                    f"vocab_size mismatch in {f_json}: "
                    f"metadata says {meta['vocab_size']}, vocab has {len(vocab)}"
                )

            tokens_file = meta.get("tokens_file") or (
                os.path.basename(f_json).replace(".json", ".tokens.bin")
            )
            states_file = meta.get("states_file") or (
                os.path.basename(f_json).replace(".json", ".states.bin")
            )
            f_tokens = os.path.join(os.path.dirname(f_json), tokens_file)
            f_states = os.path.join(os.path.dirname(f_json), states_file)
            if not os.path.exists(f_tokens):
                raise FileNotFoundError(
                    f"Tokens file missing for {f_json}: {f_tokens}"
                )
            if not os.path.exists(f_states):
                raise FileNotFoundError(
                    f"States file missing for {f_json}: {f_states}"
                )

            n_rows = int(meta["n_rows"])
            if n_rows == 0:
                # Empty shard — skip without trying to memmap a zero-length file.
                continue

            mm_tokens = np.memmap(f_tokens, dtype=np.int16, mode="r",
                                  shape=(n_rows, seq_len))
            mm_states = np.memmap(f_states, dtype=np.int8, mode="r",
                                  shape=(n_rows, seq_len))
            self.tokens_memmaps.append(mm_tokens)
            self.states_memmaps.append(mm_states)
            total_rows += n_rows
            self.offsets.append(total_rows)

        self.total_rows = total_rows
        print(f"Loaded {len(self.tokens_memmaps)} shards, "
              f"{self.total_rows:,} total rows.")

    def __len__(self):
        return self.total_rows

    def _locate(self, index: int):
        '''Map a global index to (shard_idx, local_idx) within that shard.'''
        shard_idx = bisect.bisect_right(self.offsets, index) - 1
        local_idx = index - self.offsets[shard_idx]
        return shard_idx, local_idx

    def __getitem__(self, index):
        shard_idx, local_idx = self._locate(index)

        # Copy out of the memmap into independent tensors so that downstream
        # mutations (masking, EOS placement) don't touch the mmap.
        dna_seq = torch.tensor(
            np.array(self.tokens_memmaps[shard_idx][local_idx], dtype=np.int16),
            dtype=torch.int16,
        )
        methyl_seq = torch.tensor(
            np.array(self.states_memmaps[shard_idx][local_idx], dtype=np.int8),
            dtype=torch.int8,
        )

        # Optional random-length truncation (applied in lockstep to both arrays).
        if self.random_len and np.random.random() < 0.5:
            new_len = random.randint(5, self.seq_len)
            dna_seq = dna_seq[:new_len]
            methyl_seq = methyl_seq[:new_len]

        # Pad back to seq_len if truncation shortened the sequence.
        if dna_seq.shape[0] < self.seq_len:
            pad_num = self.seq_len - dna_seq.shape[0]
            dna_seq = torch.cat((
                dna_seq,
                torch.tensor([self.vocab.pad_index] * pad_num, dtype=torch.int16),
            ))
            methyl_seq = torch.cat((
                methyl_seq,
                torch.tensor([STATE_NON_CPG] * pad_num, dtype=torch.int8),
            ))

        # Compute the EOS position BEFORE calling _masking, so we can mark the
        # methylation track at that position as STATE_NON_CPG. _masking computes
        # the same value internally as `end` and overwrites dna_seq[end] with
        # the EOS token; we just need the index.
        non_pad_positions = (dna_seq != self.vocab.pad_index).nonzero(as_tuple=True)[0]
        if len(non_pad_positions) > 0:
            end_pos = non_pad_positions[-1].item()
        else:
            end_pos = dna_seq.shape[0] - 1   # all-pad sequence: shouldn't happen

        # Apply MLM masking on DNA. Returns arrays of length seq_len + 1
        # (one extra for the prepended SOS).
        masked_dna_seq, dna_label, masked_index = self._masking(dna_seq)

        # Build the methylation track parallel to masked_dna_seq (length seq_len + 2).
        # Prepend STATE_NON_CPG for the SOS slot AND append one for the EOS slot.
        methyl_seq = torch.cat((
            torch.tensor([STATE_NON_CPG], dtype=torch.int8),
            methyl_seq,
            torch.tensor([STATE_NON_CPG], dtype=torch.int8),
        ))

        # Hide methylation at every MLM-selected position.
        methyl_seq[masked_index] = STATE_UNKNOWN

        # Reaffirm SOS and EOS positions as STATE_NON_CPG (defensive — should already
        # hold from the cat, but masked_index could in principle touch them).
        methyl_seq[0] = STATE_NON_CPG
        eos_in_padded = end_pos + 2   # +1 for the EOS placement, +1 more for SOS prepend
        if 0 <= eos_in_padded < methyl_seq.shape[0]:
            methyl_seq[eos_in_padded] = STATE_NON_CPG

        return {
            "bert_input": masked_dna_seq,
            "bert_label": dna_label,
            "bert_mask": masked_index,
            "methyl_seq": methyl_seq,
        }

    # Reuse the masking logic unchanged from MethylBertPretrainDataset.
    _get_mask = MethylBertPretrainDataset._get_mask
    _masking = MethylBertPretrainDataset._masking

    def _constrained_random_replacement(self, original_token_id: int) -> int:
        """
        Given a 3-mer token id, return a token id whose 3-mer string differs
        from the original only at the middle base. Samples uniformly from the
        3 possible substitutions (excluding the original).

        Only valid for k=3. For other k, fall back to arbitrary replacement
        or generalize this function.
        """
        if self.vocab.kmers != 3:
            # Fallback: arbitrary non-special replacement.
            return torch.randint(
                low=self.vocab.mask_index + 1,
                high=len(self.vocab),
                size=(1,),
            ).item()

        original_kmer = self.vocab.itos[original_token_id]
        bases = ["A", "C", "G", "T"]
        candidates = [b for b in bases if b != original_kmer[1]]
        new_middle = candidates[torch.randint(0, 3, (1,)).item()]
        new_kmer = original_kmer[0] + new_middle + original_kmer[2]
        return self.vocab.stoi[new_kmer]