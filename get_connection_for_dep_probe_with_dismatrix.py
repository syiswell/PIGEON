from transformers import BertModel, BertTokenizer
import torch
import torch.nn.functional as F
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import os
import unicodedata
import json
import math
from dataloaders.dataloader_sent_pt_copy import load_data_instances
from utils.utils import context_models
from configs.config import shared_configs
import gensim
from collections import defaultdict
import geoopt as gt
import time


def _run_strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


def match_tokenized_to_untokenized(subwords, sentence):
    token_subwords = np.zeros(len(sentence))
    sentence = [_run_strip_accents(x) for x in sentence]
    token_ids, subwords_str, current_token, current_token_normalized = [-1] * len(subwords), "", 0, None
    for i, subword in enumerate(subwords):
        if subword in ["[CLS]", "[SEP]"]: continue

        while current_token_normalized is None:
            current_token_normalized = sentence[current_token].lower()

        if subword.startswith("[UNK]"):
            unk_length = int(subword[6:])
            subwords[i] = subword[:5]
            subwords_str += current_token_normalized[len(subwords_str):len(subwords_str) + unk_length]
        else:
            subwords_str += subword[2:] if subword.startswith("##") else subword
        if not current_token_normalized.startswith(subwords_str):
            return False

        token_ids[i] = current_token
        token_subwords[current_token] += 1
        if current_token_normalized == subwords_str:
            subwords_str = ""
            current_token += 1
            current_token_normalized = None

    assert current_token_normalized is None
    while current_token < len(sentence):
        assert not sentence[current_token]
        current_token += 1
    assert current_token == len(sentence)

    return token_ids


def get_all_subword_id(mapping, idx):
    current_id = mapping[idx]
    id_for_all_subwords = [tmp_id for tmp_id, v in enumerate(mapping) if v == current_id]
    return id_for_all_subwords


def get_iw2sent_iw_dis(args, iw_reps):
    iw_num = iw_reps.size(0) - 1
    iw_dis = []
    for i in range(iw_num):
        iw_dis.append(tensor2list(get_dis(args, iw_reps[0], iw_reps[1+i])))
    
    return iw_dis


def get_re2re_hidden_states_onemask(args, model,
                                    sample_re1, sample_re1_id, re1_sent_import_word_span,
                                    sample_re2, sample_re2_id, re2_sent_import_word_span,
                                    mask_id, mode='same'):
    # 1. Generate mask indices
    # re_all_layers_matrix_as_list = [[] for _ in range(LAYER)]
    
    all_sent2sent_matrices = {}
    min_value1, max_value1 = 999, -999
    min_value2, max_value2 = 999, -999
    for sent_i, (sent1, sent1_tokens, sent1_import_word_spans) in enumerate(
            zip(sample_re1_id, sample_re1, re1_sent_import_word_span)):
        for sent_j, (sent2, sent2_tokens, sent2_import_word_spans) in enumerate(
                zip(sample_re2_id, sample_re2, re2_sent_import_word_span)):
            # print("sent ", sent)
            # print("sent_tokens ", sent_tokens)
            # print("sent_import_word_spans", sent_import_word_spans)
            # print()
            
            if mode == 'same' and sent_i > sent_j:
                continue
            
            if len(sent1_import_word_spans) == 0 or len(sent2_import_word_spans) == 0:
                all_sent2sent_matrices[(sent_i, sent_j)] = None
            else:
                tmp1_indexed_tokens = list(sent1)
                tmp_sent1_mask_iw_list = [list(tmp1_indexed_tokens) for _ in range(0, len(sent1_import_word_spans))]
                for i, (s_idx, e_idx) in enumerate(sent1_import_word_spans):
                    tmp_sent1_mask_iw_list[i][s_idx:e_idx + 1] = [mask_id] * (e_idx + 1 - s_idx)
                
                tmp2_indexed_tokens = list(sent2)
                tmp_sent2_mask_iw_list = [list(tmp2_indexed_tokens) for _ in range(0, len(sent2_import_word_spans))]
                for i, (s_idx, e_idx) in enumerate(sent2_import_word_spans):
                    tmp_sent2_mask_iw_list[i][s_idx:e_idx + 1] = [mask_id] * (e_idx + 1 - s_idx)
                
                sent2sent_matrices1 = []  # [len(sent1_import_word_spans), 1+len(sent2_import_word_spans), 768]
                
                for i in range(0, len(sent1_import_word_spans)):
                    one_batch = [tmp1_indexed_tokens + tmp2_indexed_tokens[1:]]  # 1: deleted the CLS in the sent2
                    for j in range(0, len(sent2_import_word_spans)):
                        one_batch.append(tmp1_indexed_tokens + tmp_sent2_mask_iw_list[j][1:])
                    
                    # print("one_batch ", one_batch)
                    
                    # 2. Convert one batch to PyTorch tensors
                    mask_tensor = torch.tensor([[1 for _ in one_sent] for one_sent in one_batch]).to(args.device)
                    segments_tensor = torch.tensor([[0] * len(tmp1_indexed_tokens) + [1] * (len(tmp2_indexed_tokens) - 1)
                                                    for _ in one_batch]).to(args.device)
                    tokens_tensor = torch.tensor(one_batch).to(args.device)
                    
                    # print("tokens_tensor", tokens_tensor.size())
                    # print("mask_tensor", mask_tensor.size())
                    
                    # 3. get all hidden states for one batch
                    with torch.no_grad():
                        model_outputs = model(tokens_tensor, attention_mask=mask_tensor, token_type_ids=segments_tensor)
                        last_layer = model_outputs[0]  # [1+len(sent2_import_word_spans), sent1_len+sent2_len, 768]
                        # all_layers = model_outputs[-1]  # 12 layers + embedding layer [13, batch_size, seq_len, hidden_size]
                        
                        iw2sent_dis_list = get_iw2sent_iw_dis(args,
                                                              last_layer[:,
                                                              sent1_import_word_spans[i][0]:sent1_import_word_spans[i][
                                                                                                1] + 1].mean(1))
                        if min_value1 > min(iw2sent_dis_list):
                            min_value1 = min(iw2sent_dis_list)
                        if max_value1 < max(iw2sent_dis_list):
                            max_value1 = max(iw2sent_dis_list)
                        sent2sent_matrices1.append(iw2sent_dis_list)
                
                sent2sent_matrices2 = []  # [len(sent2_import_word_spans), len(sent1_import_word_spans)]
                
                for i in range(0, len(sent2_import_word_spans)):
                    one_batch = [tmp1_indexed_tokens + tmp2_indexed_tokens[1:]]  # 1: deleted the CLS in the sent2
                    for j in range(0, len(sent1_import_word_spans)):
                        one_batch.append(tmp_sent1_mask_iw_list[j] + tmp2_indexed_tokens[1:])
                    
                    # print("one_batch ", one_batch)
                    
                    # 2. Convert one batch to PyTorch tensors
                    mask_tensor = torch.tensor([[1 for _ in one_sent] for one_sent in one_batch]).to(args.device)
                    segments_tensor = torch.tensor([[0] * len(tmp1_indexed_tokens) + [1] * (len(tmp2_indexed_tokens) - 1)
                                                    for _ in one_batch]).to(args.device)
                    tokens_tensor = torch.tensor(one_batch).to(args.device)
                    
                    
                    # print("tokens_tensor", tokens_tensor.size())
                    # print("mask_tensor", mask_tensor.size())
                    
                    # 3. get all hidden states for one batch
                    with torch.no_grad():
                        model_outputs = model(tokens_tensor, attention_mask=mask_tensor, token_type_ids=segments_tensor)
                        last_layer = model_outputs[0]  # [1+len(sent1_import_word_spans), sent1_len+sent2_len, 768]
                        # all_layers = model_outputs[-1]  # 12 layers + embedding layer [13, batch_size, seq_len, hidden_size]
                        
                        iw2sent_dis_list = get_iw2sent_iw_dis(args,
                                                              last_layer[:, sent2_import_word_spans[i][0] + len(
                                                                  tmp1_indexed_tokens) - 1:
                                                                            sent2_import_word_spans[i][1] + len(
                                                                                tmp1_indexed_tokens) - 1 + 1].mean(1)) # more bigger more important
                        if min_value2 > min(iw2sent_dis_list):
                            min_value2 = min(iw2sent_dis_list)
                        if max_value2 < max(iw2sent_dis_list):
                            max_value2 = max(iw2sent_dis_list)
                        sent2sent_matrices2.append(iw2sent_dis_list)

                all_sent2sent_matrices[(sent_i, sent_j)] = (sent2sent_matrices1, sent2sent_matrices2)
        
    return all_sent2sent_matrices, (min_value1, max_value1, min_value2, max_value2)  # [sent_pair_num, 2, iw_num1, iw_num2]


def get_re2re_hidden_states_twomask(args, model,
                                    sample_re1, sample_re1_id, re1_sent_import_word_span,
                                    sample_re2, sample_re2_id, re2_sent_import_word_span,
                                    mask_id, mode='same'):
    # 1. Generate mask indices
    # re_all_layers_matrix_as_list = [[] for _ in range(LAYER)]
    
    all_sent2sent_matrices = {}
    min_value1, max_value1 = 999, -999
    min_value2, max_value2 = 999, -999
    for sent_i, (sent1, sent1_tokens, sent1_import_word_spans) in enumerate(
            zip(sample_re1_id, sample_re1, re1_sent_import_word_span)):
        for sent_j, (sent2, sent2_tokens, sent2_import_word_spans) in enumerate(
                zip(sample_re2_id, sample_re2, re2_sent_import_word_span)):

            if mode == 'same' and sent_i > sent_j:
                continue

            # print("sent_i sent_j", sent_i, sent_j)
            # print("sent1_tokens sent2_tokens", sent1_tokens, sent2_tokens)
            # print("sent_tokens ", sent_tokens)
            # print("sent_import_word_spans", sent_import_word_spans)
            # print()
            
            if len(sent1_import_word_spans) == 0 or len(sent2_import_word_spans) == 0:
                all_sent2sent_matrices[(sent_i, sent_j)] = None
            else:
                tmp1_indexed_tokens = list(sent1)
                tmp_sent1_mask_iw_list = [list(tmp1_indexed_tokens) for _ in range(0, len(sent1_import_word_spans))]
                # for i in range(0, len(sent1_import_word_spans)):
                for i, (s_idx, e_idx) in enumerate(sent1_import_word_spans):
                    tmp_sent1_mask_iw_list[i][s_idx:e_idx + 1] = [mask_id] * (e_idx + 1 - s_idx)
                
                tmp2_indexed_tokens = list(sent2)
                tmp_sent2_mask_iw_list = [list(tmp2_indexed_tokens) for _ in range(0, len(sent2_import_word_spans))]
                # for j in range(0, len(sent2_import_word_spans)):
                for i, (s_idx, e_idx) in enumerate(sent2_import_word_spans):
                    tmp_sent2_mask_iw_list[i][s_idx:e_idx + 1] = [mask_id] * (e_idx + 1 - s_idx)
                
                sent2sent_matrices1 = []  # [len(sent1_import_word_spans), len(sent2_import_word_spans)]
                for i in range(0, len(sent1_import_word_spans)):
                    one_batch = [tmp_sent1_mask_iw_list[i] + tmp2_indexed_tokens[1:]]  # 1: deleted the CLS in the sent2
                    for j in range(0, len(sent2_import_word_spans)):
                        one_batch.append(tmp_sent1_mask_iw_list[i] + tmp_sent2_mask_iw_list[j][1:])
                    
                    # print("one_batch ", one_batch)
                    
                    # 2. Convert one batch to PyTorch tensors
                    mask_tensor = torch.tensor([[1 for _ in one_sent] for one_sent in one_batch]).to(args.device)
                    segments_tensor = torch.tensor([[0] * len(tmp1_indexed_tokens) + [1] * (len(tmp2_indexed_tokens)-1)
                                                    for _ in one_batch]).to(args.device)
                    tokens_tensor = torch.tensor(one_batch).to(args.device)
                    
                    # print("tokens_tensor", tokens_tensor.size())
                    # print("tokens_tensor", tokens_tensor.size())
                    # print("mask_tensor", mask_tensor.size())
                    
                    # 3. get all hidden states for one batch
                    with torch.no_grad():
                        model_outputs = model(tokens_tensor, attention_mask=mask_tensor, token_type_ids=segments_tensor)
                        last_layer = model_outputs[0]  # [1+len(sent2_import_word_spans), sent1_len+sent2_len, 768]
                        # all_layers = model_outputs[-1]  # 12 layers + embedding layer [13, batch_size, seq_len, hidden_size]
                        
                        iw2sent_dis_list = get_iw2sent_iw_dis(args,
                            last_layer[:, sent1_import_word_spans[i][0]:sent1_import_word_spans[i][1] + 1].mean(1)) # [len(sent2_import_word_spans)]
                        if min_value1 > min(iw2sent_dis_list):
                            min_value1 = min(iw2sent_dis_list)
                        if max_value1 < max(iw2sent_dis_list):
                            max_value1 = max(iw2sent_dis_list)
                        sent2sent_matrices1.append(iw2sent_dis_list)
                
                sent2sent_matrices2 = []  # [len(sent2_import_word_spans), len(sent1_import_word_spans)]
                for i in range(0, len(sent2_import_word_spans)):
                    one_batch = [tmp1_indexed_tokens + tmp_sent2_mask_iw_list[i][1:]]  # 1: deleted the CLS in the sent2
                    for j in range(0, len(sent1_import_word_spans)):
                        one_batch.append(tmp_sent1_mask_iw_list[j] + tmp_sent2_mask_iw_list[i][1:])
                    
                    # print("one_batch ", one_batch)
                    
                    # 2. Convert one batch to PyTorch tensors
                    mask_tensor = torch.tensor([[1 for _ in one_sent] for one_sent in one_batch]).to(args.device)
                    segments_tensor = torch.tensor([[0] * len(tmp1_indexed_tokens) + [1] * (len(tmp2_indexed_tokens) - 1)
                                                    for _ in one_batch]).to(args.device)
                    tokens_tensor = torch.tensor(one_batch).to(args.device)
                    
                    # print("tokens_tensor", tokens_tensor.size())
                    # print("mask_tensor", mask_tensor.size())
                    
                    # 3. get all hidden states for one batch
                    with torch.no_grad():
                        model_outputs = model(tokens_tensor, attention_mask=mask_tensor, token_type_ids=segments_tensor) # need ??????????????????????????????????????????????
                        last_layer = model_outputs[0]  # [1+len(sent1_import_word_spans), sent1_len+sent2_len, 768]
                        # all_layers = model_outputs[-1]  # 12 layers + embedding layer [13, batch_size, seq_len, hidden_size]

                        iw2sent_dis_list = get_iw2sent_iw_dis(args,
                                                              last_layer[:, sent2_import_word_spans[i][0] + len(
                                                                  tmp1_indexed_tokens) - 1:
                                                                            sent2_import_word_spans[i][1] + len(
                                                                                tmp1_indexed_tokens) - 1 + 1].mean(1)) # [len(sent1_import_word_spans)]
                        if min_value2 > min(iw2sent_dis_list):
                            min_value2 = min(iw2sent_dis_list)
                        if max_value2 < max(iw2sent_dis_list):
                            max_value2 = max(iw2sent_dis_list)
                        sent2sent_matrices2.append(iw2sent_dis_list)

                all_sent2sent_matrices[(sent_i, sent_j)] = (sent2sent_matrices1, sent2sent_matrices2)
        
    return all_sent2sent_matrices, (min_value1, max_value1, min_value2, max_value2) # [sent_pair_num, 2, iw_num1, iw_num2]


def get_dis(args, base_state, state):
    if isinstance(base_state, np.generic):
        if args.metric == 'dist':
            dis = np.linalg.norm(base_state - state)
        if args.metric == 'cos':
            dis = np.dot(base_state, state) / (np.linalg.norm(base_state) * np.linalg.norm(state))
        if args.metric == 'poincare':
            base_state = torch.Tensor(base_state).to(args.device)
            state = torch.Tensor(state).to(args.device)
            norm_b = ball.expmap0(base_state)
            norm_s = ball.expmap0(state)
            dis = ball.dist(norm_b, norm_s).sum(-1)
            
    elif torch.is_tensor(base_state):
        if args.metric == 'dist':
            dis = torch.linalg.norm(base_state - state)
        if args.metric == 'cos':
            dis = torch.dot(base_state, state) / (torch.linalg.norm(base_state) * torch.linalg.norm(state))
        if args.metric == 'poincare':
            norm_b = ball.expmap0(base_state)
            norm_s = ball.expmap0(state)
            dis = ball.dist(norm_b, norm_s).sum(-1)
    else:
        raise ValueError
        
    return dis


def tensor2numpy(data):
    if args.device == "cuda":
        data = data.cpu().numpy()
    else:
        data = data.numpy()

    return data


def tensor2list(data):
    if args.device == "cuda":
        data = data.cpu().tolist()
    else:
        data = data.tolist()
    
    return data


def min_max_normalization(matrix, min_value, max_value):
    return (matrix - min_value) / (max_value - min_value)


def get_weight_for_sent_pair_present_rev2rep(re2re_dis_matrix_as_dic, adj, min_max_values, re1_bert_tokens, re1_sent_import_word_spans,
                             re2_bert_tokens, re2_sent_import_word_spans):
    print("min_max_values", min_max_values)
    re2re_sent_pair_num = len(re2re_dis_matrix_as_dic)
    
    if args.norm_score == 'sample':
        min_value1, max_value1, min_value2, max_value2 = min_max_values
    
    sent2sent_weight_sum = {0: defaultdict(int), 1: defaultdict(int)}
    sent2sent_weight_avg = {0: defaultdict(int), 1: defaultdict(int)}
    sent2sent_weight_1 = {0: defaultdict(int), 1: defaultdict(int)}
    sent2sent_weight_2 = {0: defaultdict(int), 1: defaultdict(int)}
    
    sent2sent_weight_matrix_sum = np.zeros([len(re1_sent_import_word_spans), len(re2_sent_import_word_spans)])
    sent2sent_weight_matrix_avg = np.zeros([len(re1_sent_import_word_spans), len(re2_sent_import_word_spans)])
    sent2sent_weight_matrix_1 = np.zeros([len(re1_sent_import_word_spans), len(re2_sent_import_word_spans)])
    sent2sent_weight_matrix_2 = np.zeros([len(re1_sent_import_word_spans), len(re2_sent_import_word_spans)])
    
    for pair, sent2sent_matrices in re2re_dis_matrix_as_dic.items():
        sent_ind1, sent_ind2 = pair[0], pair[1]
        label = adj[sent_ind1, sent_ind2]
        
        print("sent_ind1, sent_ind2", sent_ind1, sent_ind2)
        if sent2sent_matrices != None:
            sent1_bert_tokens = re1_bert_tokens[sent_ind1]
            sent1_import_word_spans = re1_sent_import_word_spans[sent_ind1]
            
            sent2_bert_tokens = re2_bert_tokens[sent_ind2]
            sent2_import_word_spans = re2_sent_import_word_spans[sent_ind2]
            
            sent1_import_word = []
            for j, (s_idx, e_idx) in enumerate(sent1_import_word_spans):
                sent1_import_word.append("".join(sent1_bert_tokens[s_idx: e_idx + 1]))
            
            sent2_import_word = []
            for j, (s_idx, e_idx) in enumerate(sent2_import_word_spans):
                sent2_import_word.append("".join(sent2_bert_tokens[s_idx: e_idx + 1]))
            
            sent2sent_matrix1_as_list, sent2sent_matrix2_as_list = sent2sent_matrices
            # print("sent2sent_matrix1_as_list", sent2sent_matrix1_as_list)
            # print("sent2sent_matrix2_as_list", sent2sent_matrix2_as_list)
            
            # print("sent2sent_matrix2_as_list == sent2sent_matrix2_as_list",
            #       np.array(sent2sent_matrix1_as_list) == np.array(sent2sent_matrix2_as_list).T)
            
            for iw_idx, iw_2sent_dis in enumerate(sent2sent_matrix1_as_list):
                print("sent2sent_matrix1_as_list")
                print("iw_idx", iw_idx)
                print("sent1_iw",
                      sent1_bert_tokens[sent1_import_word_spans[iw_idx][0]:sent1_import_word_spans[iw_idx][1] + 1])
                print("sent2_iw", sent2_import_word)
                print("iw_2sent_dis", iw_2sent_dis)
            
            for iw_idx, iw_2sent_dis in enumerate(sent2sent_matrix2_as_list):
                print("sent2sent_matrix2_as_list")
                print("iw_idx", iw_idx)
                print("sent2_iw",
                      sent2_bert_tokens[sent2_import_word_spans[iw_idx][0]:sent2_import_word_spans[iw_idx][1] + 1])
                print("sent1_iw", sent1_import_word)
                print("iw_2sent_dis", iw_2sent_dis)
            
            print("label", label)
            # print("sent2sent_matrix1_sum", np.sum(sent2sent_matrix1_as_list) / (len(sent2sent_matrix1_as_list) * len(sent2sent_matrix1_as_list[0])))
            # print("sent2sent_matrix2_sum", np.sum(sent2sent_matrix2_as_list) / (len(sent2sent_matrix2_as_list) * len(sent2sent_matrix2_as_list[0])))
            #  there is not relation between the result and the true sentences in the AC pair
            
            sent2sent_matrix1 = np.array(sent2sent_matrix1_as_list)
            sent2sent_matrix2 = np.array(sent2sent_matrix2_as_list)
            if args.norm_score == 'sent':
                min_value1, max_value1 = np.min(sent2sent_matrix1), np.max(sent2sent_matrix1)
                min_value2, max_value2 = np.min(sent2sent_matrix2), np.max(sent2sent_matrix2)
            
            if args.norm_score == 'sent' or args.norm_score == 'sample':
                sent2sent_norm_matrix1 = min_max_normalization(sent2sent_matrix1, min_value1, max_value1)
                sent2sent_norm_matrix2 = min_max_normalization(sent2sent_matrix2, min_value2, max_value2)
            else:
                sent2sent_norm_matrix1 = sent2sent_matrix1
                sent2sent_norm_matrix2 = sent2sent_matrix2
            
            print("sent2sent_norm_matrix1 min, sent2sent_norm_matrix1 max", np.min(sent2sent_norm_matrix1),
                  np.max(sent2sent_norm_matrix1))
            print("sent2sent_norm_matrix2 min, sent2sent_norm_matrix2 max", np.min(sent2sent_norm_matrix2),
                  np.max(sent2sent_norm_matrix2))
            
            sent2sent_norm_avg = (sent2sent_norm_matrix1 + sent2sent_norm_matrix2.T) / 2
            print("sent2sent_norm_matrix_average min, sent2sent_norm_matrix_average max", np.min(sent2sent_norm_avg),
                  np.max(sent2sent_norm_avg))
            
            if args.metric == 'poincare':
                threshold = 0.7
            elif args.metric == 'dist':
                threshold = 0.2
            # print("sent2sent weight1", np.sum(sent2sent_norm_matrix1 > threshold) / sent2sent_norm_matrix1.size)
            # print("sent2sent weight2", np.sum(sent2sent_norm_matrix2 > threshold) / sent2sent_norm_matrix2.size)
            
            print("sent2sent weight1", np.sum(sent2sent_norm_matrix1 > threshold))
            print("sent2sent weight2", np.sum(sent2sent_norm_matrix2 > threshold))
            print("sent2sent weight avg", np.sum(sent2sent_norm_avg > threshold))
            print("sent2sent weight all", np.sum(sent2sent_norm_matrix1 > threshold) +
                  np.sum(sent2sent_norm_matrix2 > threshold))
            
            if label == 1:
                sent2sent_weight_1[1][np.sum(sent2sent_norm_matrix1 > threshold)] += 1
                sent2sent_weight_2[1][np.sum(sent2sent_norm_matrix2 > threshold)] += 1
                sent2sent_weight_avg[1][np.sum(sent2sent_norm_avg > threshold)] += 1
                
                sent2sent_weight_sum[1][np.sum(sent2sent_norm_matrix1 > threshold) +
                                        np.sum(sent2sent_norm_matrix2 > threshold)] += 1
            else:
                sent2sent_weight_1[0][np.sum(sent2sent_norm_matrix1 > threshold)] += 1
                sent2sent_weight_2[0][np.sum(sent2sent_norm_matrix2 > threshold)] += 1
                sent2sent_weight_avg[0][np.sum(sent2sent_norm_avg > threshold)] += 1
                
                sent2sent_weight_sum[0][np.sum(sent2sent_norm_matrix1 > threshold) +
                                        np.sum(sent2sent_norm_matrix2 > threshold)] += 1
            
            sent2sent_weight_matrix_sum[sent_ind1, sent_ind2] = np.sum(sent2sent_norm_matrix1 > threshold) + \
                                                                np.sum(sent2sent_norm_matrix2 > threshold)
            sent2sent_weight_matrix_avg[sent_ind1, sent_ind2] = np.sum(sent2sent_norm_avg > threshold)
            sent2sent_weight_matrix_1[sent_ind1, sent_ind2] = np.sum(sent2sent_norm_matrix1 > threshold)
            sent2sent_weight_matrix_2[sent_ind1, sent_ind2] = np.sum(sent2sent_norm_matrix2 > threshold)
            
            # print("sent2sent_norm_matrix1", sent2sent_norm_matrix1)
            # print("sent2sent_norm_matrix2", sent2sent_norm_matrix2)
    
    print("sent2sent_weight_matrix_sum", sent2sent_weight_matrix_sum)
    print("sent2sent_weight_matrix_avg", sent2sent_weight_matrix_avg)
    print("sent2sent_weight_matrix_1", sent2sent_weight_matrix_1)
    print("sent2sent_weight_matrix_2", sent2sent_weight_matrix_2)
    
    print("sent2sent_weight_matrix_sum * adj", sent2sent_weight_matrix_sum * adj)
    print("sent2sent_weight_matrix_avg * adj", sent2sent_weight_matrix_avg * adj)
    print("sent2sent_weight_matrix_1 * adj", sent2sent_weight_matrix_1 * adj)
    print("sent2sent_weight_matrix_2 * adj", sent2sent_weight_matrix_2 * adj)
    
    print("sent2sent_weight_matrix_sum * (1-adj)", sent2sent_weight_matrix_sum * (1 - adj))
    print("sent2sent_weight_matrix_avg * (1-adj)", sent2sent_weight_matrix_avg * (1 - adj))
    print("sent2sent_weight_matrix_1 * (1-adj)", sent2sent_weight_matrix_1 * (1 - adj))
    print("sent2sent_weight_matrix_2 * (1-adj)", sent2sent_weight_matrix_2 * (1 - adj))
    
    print("sent2sent_weight_1", sent2sent_weight_1)
    print("sent2sent_weight_2", sent2sent_weight_2)
    print("sent2sent_weight_avg", sent2sent_weight_avg)
    print("sent2sent_weight_sum", sent2sent_weight_sum)


def get_weight_for_sent_pair_rev2rep(re2re_dis_matrix_as_dic, adj, min_max_values,
                                     re1_sents_words, re1_import_word_indexes,
                                     re2_sents_words, re2_import_word_indexes):
    
    if args.norm_score == 'sample':
        min_value1, max_value1, min_value2, max_value2 = min_max_values
    
    sent2sent_weight_matrix_1 = np.zeros([len(re1_import_word_indexes), len(re2_import_word_indexes)])

    for pair, sent2sent_matrices in re2re_dis_matrix_as_dic.items():
        sent_ind1, sent_ind2 = pair[0], pair[1]
        label = adj[sent_ind1, sent_ind2]
        
        # if sent2sent_matrices == None:
        #     sent2sent_weight_matrix_1[sent_ind1, sent_ind2] = 0
        
        if sent2sent_matrices != None:
            sent2sent_matrix1_as_list, sent2sent_matrix2_as_list = sent2sent_matrices

            sent2sent_matrix1 = np.array(sent2sent_matrix1_as_list)
            sent2sent_matrix2 = np.array(sent2sent_matrix2_as_list)
            if args.norm_score == 'sent':
                min_value1, max_value1 = np.min(sent2sent_matrix1), np.max(sent2sent_matrix1)
                min_value2, max_value2 = np.min(sent2sent_matrix2), np.max(sent2sent_matrix2)
            
            if args.norm_score == 'sent' or args.norm_score == 'sample':
                sent2sent_norm_matrix1 = min_max_normalization(sent2sent_matrix1, min_value1, max_value1)
                sent2sent_norm_matrix2 = min_max_normalization(sent2sent_matrix2, min_value2, max_value2)
            else:
                sent2sent_norm_matrix1 = sent2sent_matrix1
                sent2sent_norm_matrix2 = sent2sent_matrix2
            
            # if args.metric == 'poincare':
            #     threshold = args.poincare_threshold # default: 0.7
            # elif args.metric == 'dist':
            #     threshold = args.dist_threshold # default: 0.2
            if np.sum(sent2sent_norm_matrix1 > threshold) > args.inter_cor_threshold: # default: 2
                sent2sent_weight_matrix_1[sent_ind1, sent_ind2] = np.sum(sent2sent_norm_matrix1 > threshold)
    
    # max_all_s2s_num = -1
    max_s2s_num = np.max(sent2sent_weight_matrix_1)
    sent2sent1_weight_matrix = np.zeros([len(re1_import_word_indexes), len(re2_import_word_indexes)])
    for sent_i in range(len(re1_import_word_indexes)):
        if args.max_cor_num_type == "all":
            max_sent_i_num = 209
        elif args.max_cor_num_type == "sample":
            max_sent_i_num = max_s2s_num
        elif args.max_cor_num_type == "sent":
            max_sent_i_num = np.max(sent2sent_weight_matrix_1[sent_i])
            # if max_all_s2s_num < max_sent_i_num:
                # max_all_s2s_num = max_sent_i_num
        else:
            raise ValueError
        if max_sent_i_num == 0:
            max_sent_i_num = 1
        sent2sent1_weight_matrix[sent_i] = sent2sent_weight_matrix_1[sent_i] / (max_sent_i_num) # + (sent2sent_weight_matrix_1[sent_i] != 0)
    
    sent2sent_weight_matrix_2 = sent2sent_weight_matrix_1.T
    sent2sent2_weight_matrix = np.zeros([len(re2_import_word_indexes), len(re1_import_word_indexes)])
    for sent_i in range(len(re2_import_word_indexes)):
        if args.max_cor_num_type == "all":
            max_sent_i_num = 209
        elif args.max_cor_num_type == "sample":
            max_sent_i_num = max_s2s_num
        elif args.max_cor_num_type == "sent":
            max_sent_i_num = np.max(sent2sent_weight_matrix_2[sent_i])
            # if max_all_s2s_num < max_sent_i_num:
            #     max_all_s2s_num = max_sent_i_num
        else:
            raise ValueError
        if max_sent_i_num == 0:
            max_sent_i_num = 1

        sent2sent2_weight_matrix[sent_i] = sent2sent_weight_matrix_2[sent_i] / (max_sent_i_num) #+ (sent2sent_weight_matrix_2[sent_i] != 0)
    
    return sent2sent1_weight_matrix, sent2sent2_weight_matrix #, max_all_s2s_num


def get_weight_for_sent_pair_present_re2re(re2re_dis_matrix_as_dic, adj, min_max_values, re1_bert_tokens, re1_sent_import_word_spans,
                              re2_bert_tokens, re2_sent_import_word_spans):
    print("min_max_values", min_max_values)
    re2re_sent_pair_num = len(re2re_dis_matrix_as_dic)
    
    if args.norm_score == 'sample':
        min_value1, max_value1, min_value2, max_value2 = min_max_values
    
    sent2sent_weight_sum = {0: defaultdict(int), 1: defaultdict(int)}
    sent2sent_weight_avg = {0: defaultdict(int), 1: defaultdict(int)}
    sent2sent_weight_1 = {0: defaultdict(int), 1: defaultdict(int)}
    sent2sent_weight_2 = {0: defaultdict(int), 1: defaultdict(int)}
    
    sent2sent_weight_matrix_sum = np.zeros([len(re1_sent_import_word_spans), len(re2_sent_import_word_spans)])
    sent2sent_weight_matrix_avg = np.zeros([len(re1_sent_import_word_spans), len(re2_sent_import_word_spans)])
    sent2sent_weight_matrix_1 = np.zeros([len(re1_sent_import_word_spans), len(re2_sent_import_word_spans)])
    sent2sent_weight_matrix_2 = np.zeros([len(re1_sent_import_word_spans), len(re2_sent_import_word_spans)])
    
    for pair, sent2sent_matrices in re2re_dis_matrix_as_dic.items():
        sent_ind1, sent_ind2 = pair[0], pair[1]
        label = adj[sent_ind1, sent_ind2]
        
        print("sent_ind1, sent_ind2", sent_ind1, sent_ind2)
        if sent2sent_matrices != None:
            sent1_bert_tokens = re1_bert_tokens[sent_ind1]
            sent1_import_word_spans = re1_sent_import_word_spans[sent_ind1]
            
            sent2_bert_tokens = re2_bert_tokens[sent_ind2]
            sent2_import_word_spans = re2_sent_import_word_spans[sent_ind2]
            
            sent1_import_word = []
            for j, (s_idx, e_idx) in enumerate(sent1_import_word_spans):
                sent1_import_word.append("".join(sent1_bert_tokens[s_idx: e_idx + 1]))
            
            sent2_import_word = []
            for j, (s_idx, e_idx) in enumerate(sent2_import_word_spans):
                sent2_import_word.append("".join(sent2_bert_tokens[s_idx: e_idx + 1]))
            
            sent2sent_matrix1_as_list, sent2sent_matrix2_as_list = sent2sent_matrices
            # print("sent2sent_matrix1_as_list", sent2sent_matrix1_as_list)
            # print("sent2sent_matrix2_as_list", sent2sent_matrix2_as_list)
            
            # print("sent2sent_matrix2_as_list == sent2sent_matrix2_as_list",
            #       np.array(sent2sent_matrix1_as_list) == np.array(sent2sent_matrix2_as_list).T)
            
            for iw_idx, iw_2sent_dis in enumerate(sent2sent_matrix1_as_list):
                print("sent2sent_matrix1_as_list")
                print("iw_idx", iw_idx)
                print("sent1_iw",
                      sent1_bert_tokens[sent1_import_word_spans[iw_idx][0]:sent1_import_word_spans[iw_idx][1] + 1])
                print("sent2_iw", sent2_import_word)
                print("iw_2sent_dis", iw_2sent_dis)
            
            for iw_idx, iw_2sent_dis in enumerate(sent2sent_matrix2_as_list):
                print("sent2sent_matrix2_as_list")
                print("iw_idx", iw_idx)
                print("sent2_iw",
                      sent2_bert_tokens[sent2_import_word_spans[iw_idx][0]:sent2_import_word_spans[iw_idx][1] + 1])
                print("sent1_iw", sent1_import_word)
                print("iw_2sent_dis", iw_2sent_dis)
            
            print("label", label)
            # print("sent2sent_matrix1_sum", np.sum(sent2sent_matrix1_as_list) / (len(sent2sent_matrix1_as_list) * len(sent2sent_matrix1_as_list[0])))
            # print("sent2sent_matrix2_sum", np.sum(sent2sent_matrix2_as_list) / (len(sent2sent_matrix2_as_list) * len(sent2sent_matrix2_as_list[0])))
            #  there is not relation between the result and the true sentences in the AC pair
            
            sent2sent_matrix1 = np.array(sent2sent_matrix1_as_list)
            sent2sent_matrix2 = np.array(sent2sent_matrix2_as_list)
            if args.norm_score == 'sent':
                min_value1, max_value1 = np.min(sent2sent_matrix1), np.max(sent2sent_matrix1)
                min_value2, max_value2 = np.min(sent2sent_matrix2), np.max(sent2sent_matrix2)
            
            if args.norm_score == 'sent' or args.norm_score == 'sample':
                sent2sent_norm_matrix1 = min_max_normalization(sent2sent_matrix1, min_value1, max_value1)
                sent2sent_norm_matrix2 = min_max_normalization(sent2sent_matrix2, min_value2, max_value2)
            else:
                sent2sent_norm_matrix1 = sent2sent_matrix1
                sent2sent_norm_matrix2 = sent2sent_matrix2
            
            print("sent2sent_norm_matrix1 min, sent2sent_norm_matrix1 max", np.min(sent2sent_norm_matrix1),
                  np.max(sent2sent_norm_matrix1))
            print("sent2sent_norm_matrix2 min, sent2sent_norm_matrix2 max", np.min(sent2sent_norm_matrix2),
                  np.max(sent2sent_norm_matrix2))
            
            sent2sent_norm_avg = (sent2sent_norm_matrix1 + sent2sent_norm_matrix2.T) / 2
            print("sent2sent_norm_matrix_average min, sent2sent_norm_matrix_average max", np.min(sent2sent_norm_avg),
                  np.max(sent2sent_norm_avg))
            
            if args.metric == 'poincare':
                threshold = 0.7
            elif args.metric == 'dist':
                threshold = 0.2
            # print("sent2sent weight1", np.sum(sent2sent_norm_matrix1 > threshold) / sent2sent_norm_matrix1.size)
            # print("sent2sent weight2", np.sum(sent2sent_norm_matrix2 > threshold) / sent2sent_norm_matrix2.size)
            
            print("sent2sent weight1", np.sum(sent2sent_norm_matrix1 > threshold))
            print("sent2sent weight2", np.sum(sent2sent_norm_matrix2 > threshold))
            print("sent2sent weight avg", np.sum(sent2sent_norm_avg > threshold))
            print("sent2sent weight all", np.sum(sent2sent_norm_matrix1 > threshold) +
                  np.sum(sent2sent_norm_matrix2 > threshold))
            
            if label == 1:
                sent2sent_weight_1[1][np.sum(sent2sent_norm_matrix1 > threshold)] += 1
                sent2sent_weight_2[1][np.sum(sent2sent_norm_matrix2 > threshold)] += 1
                sent2sent_weight_avg[1][np.sum(sent2sent_norm_avg > threshold)] += 1
                
                sent2sent_weight_sum[1][np.sum(sent2sent_norm_matrix1 > threshold) +
                                        np.sum(sent2sent_norm_matrix2 > threshold)] += 1
            else:
                sent2sent_weight_1[0][np.sum(sent2sent_norm_matrix1 > threshold)] += 1
                sent2sent_weight_2[0][np.sum(sent2sent_norm_matrix2 > threshold)] += 1
                sent2sent_weight_avg[0][np.sum(sent2sent_norm_avg > threshold)] += 1
                
                sent2sent_weight_sum[0][np.sum(sent2sent_norm_matrix1 > threshold) +
                                        np.sum(sent2sent_norm_matrix2 > threshold)] += 1
            
            sent2sent_weight_matrix_sum[sent_ind1, sent_ind2] = np.sum(sent2sent_norm_matrix1 > threshold) + \
                                                                np.sum(sent2sent_norm_matrix2 > threshold)
            sent2sent_weight_matrix_avg[sent_ind1, sent_ind2] = np.sum(sent2sent_norm_avg > threshold)
            sent2sent_weight_matrix_1[sent_ind1, sent_ind2] = np.sum(sent2sent_norm_matrix1 > threshold)
            sent2sent_weight_matrix_2[sent_ind1, sent_ind2] = np.sum(sent2sent_norm_matrix2 > threshold)
            
            # print("sent2sent_norm_matrix1", sent2sent_norm_matrix1)
            # print("sent2sent_norm_matrix2", sent2sent_norm_matrix2)
    
    print("sent2sent_weight_matrix_sum", sent2sent_weight_matrix_sum)
    print("sent2sent_weight_matrix_avg", sent2sent_weight_matrix_avg)
    print("sent2sent_weight_matrix_1", sent2sent_weight_matrix_1)
    print("sent2sent_weight_matrix_2", sent2sent_weight_matrix_2)
    
    print("sent2sent_weight_matrix_sum * adj", sent2sent_weight_matrix_sum * adj)
    print("sent2sent_weight_matrix_avg * adj", sent2sent_weight_matrix_avg * adj)
    print("sent2sent_weight_matrix_1 * adj", sent2sent_weight_matrix_1 * adj)
    print("sent2sent_weight_matrix_2 * adj", sent2sent_weight_matrix_2 * adj)
    
    print("sent2sent_weight_matrix_sum * (1-adj)", sent2sent_weight_matrix_sum * (1 - adj))
    print("sent2sent_weight_matrix_avg * (1-adj)", sent2sent_weight_matrix_avg * (1 - adj))
    print("sent2sent_weight_matrix_1 * (1-adj)", sent2sent_weight_matrix_1 * (1 - adj))
    print("sent2sent_weight_matrix_2 * (1-adj)", sent2sent_weight_matrix_2 * (1 - adj))
    
    print("sent2sent_weight_1", sent2sent_weight_1)
    print("sent2sent_weight_2", sent2sent_weight_2)
    print("sent2sent_weight_avg", sent2sent_weight_avg)
    print("sent2sent_weight_sum", sent2sent_weight_sum)


def get_weight_for_sent_pair_re2re(re2re_dis_matrix_as_dic, adj, min_max_values,
                                   re1_sents_words, re1_import_word_indexes,
                                   re2_sents_words, re2_import_word_indexes,
                                   max_cor_num):
    
    if args.norm_score == 'sample':
        min_value1, max_value1, min_value2, max_value2 = min_max_values
    
    sent2sent_weight_matrix_1 = np.zeros([len(re1_import_word_indexes), len(re2_import_word_indexes)])
    
    for pair, sent2sent_matrices in re2re_dis_matrix_as_dic.items():
        sent_ind1, sent_ind2 = pair[0], pair[1]
        label = adj[sent_ind1, sent_ind2]
        
        # if sent2sent_matrices == None:
        #     sent2sent_weight_matrix_1[sent_ind1, sent_ind2] = 0
        #     sent2sent_weight_matrix_1[sent_ind2, sent_ind1] = 0
        
        if sent2sent_matrices != None:
            sent2sent_matrix1_as_list, sent2sent_matrix2_as_list = sent2sent_matrices
            
            sent2sent_matrix1 = np.array(sent2sent_matrix1_as_list)
            sent2sent_matrix2 = np.array(sent2sent_matrix2_as_list)
            if args.norm_score == 'sent':
                min_value1, max_value1 = np.min(sent2sent_matrix1), np.max(sent2sent_matrix1)
                min_value2, max_value2 = np.min(sent2sent_matrix2), np.max(sent2sent_matrix2)
            
            if args.norm_score == 'sent' or args.norm_score == 'sample':
                sent2sent_norm_matrix1 = min_max_normalization(sent2sent_matrix1, min_value1, max_value1)
                sent2sent_norm_matrix2 = min_max_normalization(sent2sent_matrix2, min_value2, max_value2)
            else:
                sent2sent_norm_matrix1 = sent2sent_matrix1
                sent2sent_norm_matrix2 = sent2sent_matrix2

            cor_num = np.sum(sent2sent_norm_matrix1 > threshold)
            
            if abs(sent_ind1 - sent_ind2) == 0:
                sent2sent_weight_matrix_1[sent_ind1, sent_ind2] = len(re1_import_word_indexes[sent_ind1])
            
            elif cor_num > args.intra_cor_threshold and abs(sent_ind1 - sent_ind2) <= args.intra_neighbour: # intra_neighbour: 3(default)
                # if abs(sent_ind1 - sent_ind2) == 0:
                #     sent2sent_weight_matrix_1[sent_ind1, sent_ind2] = len(re1_sent_import_word_spans)
                # else:
                if cor_num > len(re1_import_word_indexes[sent_ind1]):
                    sent2sent_weight_matrix_1[sent_ind1, sent_ind2] = len(re1_import_word_indexes[sent_ind1])
                else:
                    sent2sent_weight_matrix_1[sent_ind1, sent_ind2] = cor_num
                
                if cor_num > len(re2_import_word_indexes[sent_ind2]):
                    sent2sent_weight_matrix_1[sent_ind2, sent_ind1] = len(re2_import_word_indexes[sent_ind2])
                else:
                    sent2sent_weight_matrix_1[sent_ind2, sent_ind1] = cor_num
    
    # max_all_s2s_num = -1
    max_s2s_num = np.max(sent2sent_weight_matrix_1)
    for sent_i in range(len(re1_import_word_indexes)):
        # if args.max_cor_num_type == "all":
        #     max_sent_i_num = max_cor_num
        # elif args.max_cor_num_type == "sample":
        #     max_sent_i_num = max_s2s_num
        # elif args.max_cor_num_type == "sent":
        #     max_sent_i_num = np.max(sent2sent_weight_matrix_1[sent_i])
        #     # if max_all_s2s_num < max_sent_i_num:
        #     #     max_all_s2s_num = max_sent_i_num
        # else:
        #     raise ValueError
        max_sent_i_num = np.max(sent2sent_weight_matrix_1[sent_i])

        if max_sent_i_num == 0:
            max_sent_i_num = 1
    
        sent2sent_weight_matrix_1[sent_i] = sent2sent_weight_matrix_1[sent_i] / (max_sent_i_num) #+ (sent2sent_weight_matrix_1[sent_i] != 0) # why add 1?

    return sent2sent_weight_matrix_1 #, max_all_s2s_num


def get_dep_matrix(args, dataset, dis_matrix_dics):
    
    out = []
    global max_all_rev2rep_num
    global max_all_rev2rev_num
    global max_all_rep2rep_num

    for sample, instance_weight_matrices in tqdm(zip(dataset, dis_matrix_dics)):
        id = sample.id
        review_length = sample.review_length
        reply_length = sample.reply_length
        
        review_import_word_indexes = sample.review_import_word_indexes
        reply_import_word_indexes = sample.reply_import_word_indexes
        review_sents_words = sample.review_sents_words
        reply_sents_words = sample.reply_sents_words
        
        tags = sample.tags
        review_adj = sample.review_adj
        reply_adj = sample.reply_adj

        dic_id = instance_weight_matrices["id"] = id
        assert id == dic_id, print(id, dic_id)
        
        rev2rev_dis_matrix_as_dic = instance_weight_matrices["rev2rev_dis_matrix_as_dic"]
        rev2rev_min_max_values = instance_weight_matrices["rev2rev_min_max_values"]
        rev2rep_dis_matrix_as_dic = instance_weight_matrices["rev2rep_dis_matrix_as_dic"]
        rev2rep_min_max_values = instance_weight_matrices["rev2rep_min_max_values"]
        rep2rep_dis_matrix_as_dic = instance_weight_matrices["rep2rep_dis_matrix_as_dic"]
        rep2rep_min_max_values = instance_weight_matrices["rep2rep_min_max_values"]
        
        
        # rev2rev_min_max_values， rev2rep_min_max_values是否式相互独立，即rev中相似句子与rev2rep中相似句子的构建是否是无关的
        # 如果有关，则融合rev2rev_min_max_values和rev2rep_min_max_values

        rev2rep_weight, rep2rev_weight = get_weight_for_sent_pair_rev2rep(rev2rep_dis_matrix_as_dic, tags, rev2rep_min_max_values,
                                                          review_sents_words, review_import_word_indexes,
                                                          reply_sents_words, reply_import_word_indexes)
        # if max_all_rev2rep_num < max_rev2rep_num:
        #     max_all_rev2rep_num = max_rev2rep_num
        
        rev2rev_weight = get_weight_for_sent_pair_re2re(rev2rev_dis_matrix_as_dic, review_adj, rev2rev_min_max_values,
                                                        review_sents_words, review_import_word_indexes,
                                                        review_sents_words, review_import_word_indexes, max_cor_num=84)
        # if max_all_rev2rev_num < max_rev2rev_num:
        #     max_all_rev2rev_num = max_rev2rev_num
            
        rep2rep_weight = get_weight_for_sent_pair_re2re(rep2rep_dis_matrix_as_dic, reply_adj, rep2rep_min_max_values,
                                                        reply_sents_words, reply_import_word_indexes,
                                                        reply_sents_words, reply_import_word_indexes, max_cor_num=91)
        # if max_all_rep2rep_num < max_rep2rep_num:
        #     max_all_rep2rep_num = max_rep2rep_num
            
        instance_weight_matrices = {}
        instance_weight_matrices["id"] = id
        instance_weight_matrices["review2review_weights"] = rev2rev_weight.tolist()
        instance_weight_matrices["review2reply_weights"] = rev2rep_weight.tolist()
        instance_weight_matrices["reply2review_weights"] = rep2rev_weight.tolist()
        instance_weight_matrices["reply2reply_weights"] = rep2rep_weight.tolist()
        
        out.append(instance_weight_matrices)
    
    # print("max_all_rev2rep_num", max_all_rev2rep_num)
    # print("max_all_rev2rev_num", max_all_rev2rev_num)
    # print("max_all_rep2rep_num", max_all_rep2rep_num)
    return out
        

if __name__ == '__main__':
    start_time = time.time()

    args = shared_configs.get_args()
    # args.device = "cuda:2"
    # model_class, tokenizer_class, pretrained_weights = MODEL_CLASSES[args.model_type]
    # model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
    # tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=True)
    
    # special_tokens = ['[ENDL]', '[TAB]', '[LINE]', '[EQU]', '[URL]', '[NUM]', '[SPE]']
    
    # with open(args.bert_weights_path + "vocab.txt", 'wr') as fin:
    #     for special_token in special_tokens:

    args.device = 'cuda:2'
    # args.probing_mask = "onemask"
    # args.metric = 'poincare'
    
    # model_class = args.bert_weights_path.split("/")[-1]
    # model = context_models[model_class]['model'].from_pretrained(args.bert_weights_path, output_hidden_states=True).to(args.device)
    # tokenizer = context_models[model_class]['tokenizer'].from_pretrained(args.bert_weights_path,
    #                                                                      additional_special_tokens=special_tokens)
    ball = gt.Stereographic(-1)

    if args.metric == 'poincare':
        threshold = args.poincare_threshold  # default: 0.7
    elif args.metric == 'dist':
        threshold = args.dist_threshold  # default: 0.2

    max_all_rev2rep_num = -1
    max_all_rev2rev_num = -1
    max_all_rep2rep_num = -1
    train_sentence_packs = eval(json.load(open(args.data_path + '/train3.json')))
    train_out = []
    # load_dir = "/home/sunyang/hlt/PTG4AM/data/RR-submission-v2/pkl/"
    # output_dir = "/home/sunyang/hlt/PTG4AM/data/RR-submission-v2/ablation/"

    load_dir = "/home/sunyang/hlt/PTG4AM/data/RR-submission-v2/pkl_onemask/"
    output_dir = "/home/sunyang/hlt/PTG4AM/data/RR-submission-v2/ablation_onemask/"

    for start_sample in range(0, 3614, 100):
        if len(train_sentence_packs[start_sample:]) > 100:
            temp_sentence_packs = train_sentence_packs[start_sample:start_sample+100]
        else:
            temp_sentence_packs = train_sentence_packs[start_sample:]
        number = len(temp_sentence_packs)
        print("train ", start_sample)
        instances_train = load_data_instances(temp_sentence_packs, args, is_train=False)

        load_file = load_dir + f"train/dis_matrix_{str(start_sample)}_{str(start_sample+number-1)}.pkl"
        train_dis_matrix_dic = pickle.load(open(load_file, 'rb'))

        train_out.extend(get_dep_matrix(args, instances_train, train_dis_matrix_dic))
    output_file = output_dir + f"train_graph_matrix_probing_mask_{args.probing_mask}_" \
                               f"metric_{str(args.metric)}_threshold_{threshold}_" \
                               f"max_cor_num_type_{str(args.max_cor_num_type)}_" \
                               f"inter_cor_threshold_{args.inter_cor_threshold}_" \
                               f"intra_neighbour_{args.intra_neighbour}_" \
                               f"intra_cor_threshold{args.intra_cor_threshold}no1.json"
    # json.dump(train_out, open(output_file, 'w'))

    dev_sentence_packs = eval(json.load(open(args.data_path + '/dev3.json')))
    dev_out = []
    for start_sample in range(0, 474, 100):
        if len(dev_sentence_packs[start_sample:]) > 100:
            temp_sentence_packs = dev_sentence_packs[start_sample:start_sample+100]
        else:
            temp_sentence_packs = dev_sentence_packs[start_sample:]
        number = len(temp_sentence_packs)
        print("dev ", start_sample)
        instances_dev = load_data_instances(temp_sentence_packs, args, is_train=False)

        load_file = load_dir + f"dev/dis_matrix_{str(start_sample)}_{str(start_sample+number-1)}.pkl"
        dev_dis_matrix_dic = pickle.load(open(load_file, 'rb'))

        dev_out.extend(get_dep_matrix(args, instances_dev, dev_dis_matrix_dic))
    output_file = output_dir + f"dev_graph_matrix_probing_mask_{args.probing_mask}_" \
                               f"metric_{str(args.metric)}_threshold_{threshold}_" \
                               f"max_cor_num_type_{str(args.max_cor_num_type)}_" \
                               f"inter_cor_threshold_{args.inter_cor_threshold}_" \
                               f"intra_neighbour_{args.intra_neighbour}_" \
                               f"intra_cor_threshold{args.intra_cor_threshold}no1.json"
    # json.dump(dev_out, open(output_file, 'w'))

    test_sentence_packs = eval(json.load(open(args.data_path + '/test3.json')))
    test_out = []
    for start_sample in range(0, 473, 100):
        if len(test_sentence_packs[start_sample:]) > 100:
            temp_sentence_packs = test_sentence_packs[start_sample:start_sample + 100]
        else:
            temp_sentence_packs = test_sentence_packs[start_sample:]
        number = len(temp_sentence_packs)
        print("test", start_sample)
        instances_test = load_data_instances(temp_sentence_packs, args, is_train=False)
    
        load_file = load_dir + f"test/dis_matrix_{str(start_sample)}_{str(start_sample+number-1)}.pkl"
        test_dis_matrix_dic = pickle.load(open(load_file, 'rb'))

        test_out.extend(get_dep_matrix(args, instances_test, test_dis_matrix_dic))
    output_file = output_dir + f"test_graph_matrix_probing_mask_{args.probing_mask}_" \
                               f"metric_{str(args.metric)}_threshold_{threshold}_" \
                               f"max_cor_num_type_{str(args.max_cor_num_type)}_" \
                               f"inter_cor_threshold_{args.inter_cor_threshold}_" \
                               f"intra_neighbour_{args.intra_neighbour}_" \
                               f"intra_cor_threshold{args.intra_cor_threshold}no1.json"
    # json.dump(test_out, open(output_file, 'w'))
    
    print(time.time()-start_time)
