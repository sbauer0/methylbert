import gc
import multiprocessing as mp
import random
from copy import deepcopy
from functools import partial

import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset

from methylbert.data.vocab import MethylVocab


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

	def _masking(self, inputs: torch.Tensor, threshold=0.15):
		"""
			Moidfied version of masking token function
			Originally developed by Huggingface (datacollator) and DNABERT

			https://github.com/huggingface/transformers/blob/9a24b97b7f304fa1ceaaeba031241293921b69d3/src/transformers/data/data_collator.py#L747

			https://github.com/jerryji1993/DNABERT/blob/bed72fc0694a7b04f7e980dc9ce986e2bb785090/examples/run_pretrain.py#L251

			Added additional tasks to handle each sequence
			Lines using tokenizer were modified due to different tokenizer object structure

		"""

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
		random_words = torch.randint(len(self.vocab), labels.shape, dtype=torch.int16)
		inputs[indices_random] = random_words[indices_random]

		# The rest of the time (10% of the time) we keep the masked input tokens unchanged

		# Special tokens (SOS, EOS)
		if end < inputs.shape[0]:
			inputs[end] = self.vocab.eos_index
		else:
			inputs[-1] = self.vocab.eos_index

		labels = torch.cat((torch.tensor([-100]), labels))
		inputs = torch.cat((torch.tensor([self.vocab.sos_index]), inputs))
		masked_index = torch.cat((torch.tensor([False]), masked_index))


		return inputs, labels, masked_index

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

import json
import glob
from pathlib import Path


class MethylBertPretrainDatasetBinary(MethylBertDataset):
    def __init__(self, data_dir: str, vocab: MethylVocab, seq_len: int,
                 random_len: bool = False):
        '''
            Memory-mapped pretraining dataset that reads tokenized int16 binary
            files produced by pretrain_data_preprocess_5base_binary.

            Expects data_dir to contain matching .bin / .json file pairs.
            Each .bin file is a flat int16 array of shape (n_reads, seq_len).
            The .json sidecar holds the n_reads count and seq_len.

            data_dir : str
                Directory containing .bin and .json file pairs.
            vocab : MethylVocab
                Vocabulary. Must match the vocab used during preprocessing.
            seq_len : int
                Sequence length. Must match the seq_len used during
                preprocessing.
            random_len : bool
                If True, randomly truncate sequences to simulate variable
                read lengths during training. Default: False.
        '''
        self.vocab = vocab
        self.seq_len = seq_len
        self.random_len = random_len
        self.data_dir = data_dir

        # Define a range of tokens to mask based on k-mers (same as text version)
        self.mask_list = self._get_mask()

        # Discover .bin / .json pairs
        json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
        if len(json_files) == 0:
            raise FileNotFoundError(f"No .json metadata files found in {data_dir}")

        self.memmaps = []
        self.offsets = [0]   # cumulative read counts per shard
        total_reads = 0

        for f_json in json_files:
            with open(f_json, "r") as f:
                meta = json.load(f)
            f_bin = f_json.replace(".json", ".bin")
            if not os.path.exists(f_bin):
                raise FileNotFoundError(f"Binary file missing for {f_json}: {f_bin}")
            if meta["seq_len"] != seq_len:
                raise ValueError(
                    f"seq_len mismatch in {f_json}: "
                    f"metadata says {meta['seq_len']}, dataset expects {seq_len}"
                )
            if meta["vocab_size"] != len(vocab):
                raise ValueError(
                    f"vocab_size mismatch in {f_json}: "
                    f"metadata says {meta['vocab_size']}, vocab has {len(vocab)}"
                )

            mm = np.memmap(f_bin, dtype=np.int16, mode="r",
                           shape=(meta["n_reads"], seq_len))
            self.memmaps.append(mm)
            total_reads += meta["n_reads"]
            self.offsets.append(total_reads)

        self.total_reads = total_reads
        print(f"Loaded {len(self.memmaps)} shards, {self.total_reads:,} total reads.")

    def __len__(self):
        return self.total_reads

    def _locate(self, index: int):
        '''
            Map a global index to (shard_idx, local_idx) within that shard.
        '''
        # binary search over cumulative offsets
        import bisect
        shard_idx = bisect.bisect_right(self.offsets, index) - 1
        local_idx = index - self.offsets[shard_idx]
        return shard_idx, local_idx

    def __getitem__(self, index):
        shard_idx, local_idx = self._locate(index)

        # Copy the row out of the memmap into an independent tensor so
        # downstream mutations (masking) don't touch the mmap.
        dna_seq = torch.tensor(
            np.array(self.memmaps[shard_idx][local_idx], dtype=np.int16),
            dtype=torch.int16
        )

        # Random length truncation (optional)
        if self.random_len and np.random.random() < 0.5:
            dna_seq = dna_seq[:random.randint(5, self.seq_len)]

        # Padding if truncation shortened the sequence
        if dna_seq.shape[0] < self.seq_len:
            pad_num = self.seq_len - dna_seq.shape[0]
            dna_seq = torch.cat((
                dna_seq,
                torch.tensor([self.vocab.pad_index] * pad_num, dtype=torch.int16)
            ))

        # Apply MLM masking (reuses logic from the text-based class)
        masked_dna_seq, dna_seq, bert_mask = self._masking(dna_seq)

        return {
            "bert_input": masked_dna_seq,
            "bert_label": dna_seq,
            "bert_mask": bert_mask
        }

    # --- Reuse the masking logic unchanged from MethylBertPretrainDataset ---
    _get_mask = MethylBertPretrainDataset._get_mask
    _masking = MethylBertPretrainDataset._masking