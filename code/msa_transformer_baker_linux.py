#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
from sklearn import metrics
import xlwt
import numpy as np
import matplotlib.pyplot as plt
import sys
#sys.path.append("F:/Paper2Code/esm-MSA/esm-master/")#/home/mqz/esm-MSA/esm-master
sys.path.append("/home/mqz/esm-MSA/esm-master")
import esm
import torch
from Bio import SeqIO
import itertools
from typing import List, Tuple
import string
import os
print("start ...")

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
depth_threshold = 1024

test_data_path = "/home/mqz/esm-MSA/baker31_dataset_min10.cpkl"
saved_models = "/home/mqz/esm-MSA/baker_result/"
saved_result = "/home/mqz/esm-MSA/baker_result/"
MSA_add = "/home/mqz/esm-MSA/msas_baker/"
pt_add = "/home/mqz/pretrain_model/esm_msa1_t12_100M_UR50S.pt"
'''
test_data_path = "F:/ubuntu_file/data/Baker_dataset/baker31_dataset_min10.cpkl"
saved_models = "F:/ubuntu_file/result/test/"
saved_result = "F:/ubuntu_file/result/test/"
MSA_add = "F:/ubuntu_file/data/Baker_dataset/msas_baker_idcov/"
pt_add = "F:/Paper2Code/esm-MSA/esm_msa1_t12_100M_UR50S.pt"#"/home/mqz/pretrain_model/esm_msa1_t12_100M_UR50S.pt"
'''
#excel_name = "msa_transformer_original_model_HVIDB_result_1024_seed123456.xls"

seed = 123456
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed_all(seed) # 为所有GPU设置随机种子

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    translation = str.maketrans(deletekeys)
    return sequence.translate(translation)
def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq)))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]
def get_aupr(rec_ls, pre_ls):
    pr_value = 0.0
    for ix in range(len(rec_ls[:-1])):
        x_right, x_left = rec_ls[ix], rec_ls[ix + 1]
        y_top, y_bottom = pre_ls[ix], pre_ls[ix + 1]
        temp_area = abs(x_right - x_left) * (y_top + y_bottom) * 0.5
        pr_value += temp_area
    return pr_value
def testset_evaluate(preds, labels,L):
    # aupr auc top50 aupr top50auc
    labels = labels.reshape([-1])
    preds = preds.reshape([-1])
    metric = []

    # aupr 0
    precision_ls, recall_ls, thresholds = metrics.precision_recall_curve(labels, preds)
    metric.append(get_aupr(recall_ls, precision_ls))

    # auc 1
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
    metric.append(metrics.auc(fpr, tpr))

    # top
    sort_result = np.zeros((2, len(labels)))
    sort_result[0, :] = preds
    sort_result[1, :] = labels
    sort_results = np.transpose(sort_result).tolist()
    sort_results.sort(reverse=True)
    sort_results = np.array(sort_results)
    # top L/5 acc 2
    metric.append(sum(sort_results[:int(L / 5), 1] == 1) / int(L / 5))
    # top L/10 acc 3
    metric.append(sum(sort_results[:int(L / 10), 1] == 1) / int(L / 10))
    # top L/20 acc 4
    metric.append( sum(sort_results[:int( L/ 20), 1]==1)  /int( L/ 20))
    # top 50 acc 5
    metric.append(sum(sort_results[:50, 1]==1)/50)
    # top 20 acc 6
    metric.append(sum(sort_results[:20, 1] == 1) / 20)
    # top 10 acc 7
    metric.append(sum(sort_results[:10, 1] == 1) / 10)
    # top 5 acc 8
    metric.append(sum(sort_results[:5, 1] == 1) / 5)
    return metric

test_dataset = pickle.load(open(test_data_path, 'rb'))

msa_transformer, msa_alphabet = esm.pretrained.load_model_and_alphabet_local(pt_add)
msa_batch_converter = msa_alphabet.get_batch_converter()
for i,protein in enumerate(test_dataset):

    MSA_address = MSA_add + protein["Complex_code"] + "_id90cov75.fas"
    print(protein["Complex_code"]+" starting")
    max_depth = len(open(MSA_address).readlines())/2
    if max_depth<depth_threshold:
        msa_depth = int(max_depth)
    else:
        msa_depth = depth_threshold
    msa_data = [read_msa(MSA_address, msa_depth),]
    msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(msa_data)
    with torch.no_grad():
         results = msa_transformer.pretrain_attention(msa_batch_tokens)
    # pred
    contacts_result = results["contacts"].detach().cpu().numpy()[0]
    row_attentions = results["row_attentions"].detach().cpu().numpy()
    contacts_result_avg = results["attentions_avg"].detach().cpu().numpy()
    contacts_result_max = results["attentions_max"].detach().cpu().numpy()
    contacts_result_min = results["attentions_min"].detach().cpu().numpy()
    np.savetxt(os.path.join(saved_result,protein["Complex_code"]+".txt" ),contacts_result)
    f = open(os.path.join(saved_result, protein["Complex_code"] + "_row_attentions.pkl"), 'wb')
    pickle.dump(row_attentions, f)
    f.close()
    np.savetxt(os.path.join(saved_result, protein["Complex_code"] + "_avg.txt"), contacts_result_avg)
    np.savetxt(os.path.join(saved_result, protein["Complex_code"] + "_max.txt"), contacts_result_max)
    np.savetxt(os.path.join(saved_result, protein["Complex_code"] + "_min.txt"), contacts_result_min)

    print(protein["Complex_code"]+" finish")