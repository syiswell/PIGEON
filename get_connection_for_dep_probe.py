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
from dataloaders.dataloader_sent_pt import load_data_instances
from utils.utils import context_models
from configs.config import shared_configs
import gensim



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
                
                cur_len = len(tmp1_indexed_tokens)
                
                tmp2_indexed_tokens = list(sent2)
                
                if cur_len + len(tmp2_indexed_tokens) > 513:
                    tmp2_indexed_tokens = tmp2_indexed_tokens[:512-cur_len] + tmp2_indexed_tokens[-1:]
                tmp_sent2_mask_iw_list = [list(tmp2_indexed_tokens) for _ in range(0, len(sent2_import_word_spans))]
                # for j in range(0, len(sent2_import_word_spans)):
                for i, (s_idx, e_idx) in enumerate(sent2_import_word_spans):
                    if cur_len + e_idx >= 512:
                        break
                    tmp_sent2_mask_iw_list[i][s_idx:e_idx + 1] = [mask_id] * (e_idx + 1 - s_idx)
                
                sent2sent_matrices1 = []  # [len(sent1_import_word_spans), len(sent2_import_word_spans)]
                for i in range(0, len(sent1_import_word_spans)):
                    one_batch = [tmp_sent1_mask_iw_list[i] + tmp2_indexed_tokens[1:]]  # 1: deleted the CLS in the sent2
                    for j in range(0, len(sent2_import_word_spans)):
                        if cur_len + sent2_import_word_spans[j][1] >= 512:
                            break
                        one_batch.append(tmp_sent1_mask_iw_list[i] + tmp_sent2_mask_iw_list[j][1:])
                    
                    # print("one_batch ", one_batch)
                    
                    # 2. Convert one batch to PyTorch tensors
                    mask_tensor = torch.tensor([[1 for _ in one_sent] for one_sent in one_batch]).to(args.device)
                    segments_tensor = torch.tensor([[0] * len(tmp1_indexed_tokens) + [1] * (len(tmp2_indexed_tokens)-1)
                                                    for _ in one_batch]).to(args.device)
                    tokens_tensor = torch.tensor(one_batch).to(args.device)
                    
                    # print("tokens_tensor", tokens_tensor.size())
                    # print("mask_tensor", mask_tensor.size())
                    # print("segments_tensor", segments_tensor.size())

                    
                    # 3. get all hidden states for one batch
                    with torch.no_grad():
                        model_outputs = model(tokens_tensor, attention_mask=mask_tensor, token_type_ids=segments_tensor)
                        last_layer = model_outputs[0]  # [1+len(sent2_import_word_spans), sent1_len+sent2_len, 768]
                        # all_layers = model_outputs[-1]  # 12 layers + embedding layer [13, batch_size, seq_len, hidden_size]
                        
                        iw2sent_dis_list = get_iw2sent_iw_dis(args,
                            last_layer[:, sent1_import_word_spans[i][0]:sent1_import_word_spans[i][1] + 1].mean(1)) # [len(sent2_import_word_spans)]

                        # iw2sent_dis_list = last_layer[:,sent1_import_word_spans[i][0]:sent1_import_word_spans[i][1] + 1].mean(1)  # [len(sent2_import_word_spans)]

                        if min_value1 > min(iw2sent_dis_list):
                            min_value1 = min(iw2sent_dis_list)
                        if max_value1 < max(iw2sent_dis_list):
                            max_value1 = max(iw2sent_dis_list)
                        sent2sent_matrices1.append(iw2sent_dis_list)
                
                sent2sent_matrices2 = []  # [len(sent2_import_word_spans), len(sent1_import_word_spans)]
                for i in range(0, len(sent2_import_word_spans)):
                    if cur_len + sent2_import_word_spans[i][1] >= 512:
                        break
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
                        # iw2sent_dis_list = last_layer[:, sent2_import_word_spans[i][0] + len(
                        #                                           tmp1_indexed_tokens) - 1:
                        #                                                     sent2_import_word_spans[i][1] + len(
                        #                                                         tmp1_indexed_tokens) - 1 + 1].mean(
                        #                                           1)  # [len(sent1_import_word_spans)]

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
    elif torch.is_tensor(base_state):
        if args.metric == 'dist':
            dis = torch.linalg.norm(base_state - state)
        if args.metric == 'cos':
            dis = torch.dot(base_state, state) / (torch.linalg.norm(base_state) * torch.linalg.norm(state))
    else:
        raise ValueError
        
    return dis


def get_inter_sents_dis(args, one_layer_matrix, re_sent_import_word_spans, all_sent_base_reps, count_all_sent_base_reps):
    # delete the sentence with iw_num=0
    
    all_inter_base_masked_sents_d = []   # [sent_num, iw_num, sent_num, 1]
    count_all_inter_base_masked_sents_d = []   # [sent_num, iw_num, sent_num, 1]
    all_inter_base_sents_d = []  # [sent_num, sent_num, 1]
    count_all_inter_base_sents_d = []  # [sent_num, sent_num, 1]
    
    for sent_idx, sent_matrix in enumerate(one_layer_matrix):  # sent_matrix : [batch_size, seq_len, hidden_size]
        sent_import_word_spans = re_sent_import_word_spans[sent_idx]
        assert len(sent_import_word_spans) == sent_matrix.size(0) - 1
        
        inter_sents_base_dises = []  # [sent_num, 1]
        count_inter_sents_base_dises = []
        if args.inter_type == "CLS":
            for sent_base_reps in all_sent_base_reps:
                inter_sents_base_dis = get_dis(args, sent_matrix[0, 0], sent_base_reps)
                # print("inter_sents_base_dis ", inter_sents_base_dis)
                inter_sents_base_dises.append(inter_sents_base_dis)
            
            for count_sent_base_reps in count_all_sent_base_reps:
                count_inter_sents_base_dis = get_dis(args, sent_matrix[0, 0], count_sent_base_reps)
                count_inter_sents_base_dises.append(count_inter_sents_base_dis)
        
        elif args.inter_type == "ALL":
            for sent_base_reps in all_sent_base_reps:
                inter_sents_base_dis = get_dis(args, sent_matrix[0, 1:-1].mean(0), sent_base_reps)
                inter_sents_base_dises.append(inter_sents_base_dis)
            
            for count_sent_base_reps in count_all_sent_base_reps:
                count_inter_sents_base_dis = get_dis(args, sent_matrix[0, 1:-1].mean(0), count_sent_base_reps)
                count_inter_sents_base_dises.append(count_inter_sents_base_dis)
        else:
            raise ValueError
        
        # print("sent_idx all_inter_base_sents_d", inter_sents_base_dises)
        # print("sent_idx count_inter_sents_base_dises", count_inter_sents_base_dises)
        
        all_inter_base_sents_d.append(torch.tensor(inter_sents_base_dises).to(args.device)) # [sent_num]
        count_all_inter_base_sents_d.append(torch.tensor(count_inter_sents_base_dises).to(args.device))
        
        if len(sent_import_word_spans) == 0:
            all_inter_base_masked_sents_d.append(None)
            count_all_inter_base_masked_sents_d.append(None)
            continue
        
        inter_base_masked_sents_d = []  # [iw_num, sent_num, 1]
        count_inter_base_masked_sents_d = []  # [iw_num, sent_num, 1]
        for k, import_word_span in enumerate(sent_import_word_spans):
            if args.inter_type == "CLS":
                masked_sent_rep = sent_matrix[k + 1, 0]
            elif args.inter_type == "ALL":
                masked_sent_rep = sent_matrix[k + 1, 1:-1].mean(0)
            
            sent_base_masked_d = []  # [sent_num, 1]
            for s_i, sent_base_reps in enumerate(all_sent_base_reps):
                inter_masked_sent_dis = get_dis(args, sent_base_reps, masked_sent_rep)
                # 1.
                # inter_sents_d = inter_sents_base_dises[s_i] - inter_masked_sent_dis
                # 2.
                inter_sents_d = inter_masked_sent_dis - inter_sents_base_dises[s_i]
                # 3
                # inter_sents_d = inter_masked_sent_dis
                
                sent_base_masked_d.append(inter_sents_d)
            inter_base_masked_sents_d.append(sent_base_masked_d)
            
            count_sent_base_masked_d = []  # [sent_num, 1]
            for s_i, count_sent_base_reps in enumerate(count_all_sent_base_reps):
                count_inter_masked_sent_dis = get_dis(args, count_sent_base_reps, masked_sent_rep)
                # 1.
                # count_inter_sents_d = count_inter_sents_base_dises[s_i] - count_inter_masked_sent_dis
                # 2.
                count_inter_sents_d = count_inter_masked_sent_dis - count_inter_sents_base_dises[s_i]
                # 3.
                # count_inter_sents_d = count_inter_masked_sent_dis
                
                count_sent_base_masked_d.append(count_inter_sents_d)
            count_inter_base_masked_sents_d.append(count_sent_base_masked_d)
        
        all_inter_base_masked_sents_d.append(torch.tensor(inter_base_masked_sents_d).to(args.device))  # [sent_num, iw_num, sent_num, 1]
        count_all_inter_base_masked_sents_d.append(torch.tensor(count_inter_base_masked_sents_d).to(args.device))  # [sent_num, iw_num, sent_num, 1]
    
    return all_inter_base_masked_sents_d, all_inter_base_sents_d,\
           count_all_inter_base_masked_sents_d, count_all_inter_base_sents_d


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


def get_weight_for_sent_pair(re2re_dis_matrix_as_dic, adj, min_max_values, re1_bert_tokens, re1_sent_import_word_spans,
                             re2_bert_tokens, re2_sent_import_word_spans):
    
    re2re_sent_pair_num = len(re2re_dis_matrix_as_dic)

    if args.norm_score == 'sample':
        min_value1, max_value1, min_value2, max_value2 = min_max_values
    
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
            
            threshold = 0.15
            # print("sent2sent weight1", np.sum(sent2sent_norm_matrix1 > threshold) / sent2sent_norm_matrix1.size)
            # print("sent2sent weight2", np.sum(sent2sent_norm_matrix2 > threshold) / sent2sent_norm_matrix2.size)

            print("sent2sent weight1", np.sum(sent2sent_norm_matrix1 > threshold))
            print("sent2sent weight2", np.sum(sent2sent_norm_matrix2 > threshold))
            
            print("sent2sent_norm_matrix1", sent2sent_norm_matrix1)
            print("sent2sent_norm_matrix2", sent2sent_norm_matrix2)


def get_dep_matrix(args, model, tokenizer, dataset):

    mask_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
    model.eval()

    # LAYER = int(args.probing_layers)
    # LAYER += 1  # embedding layer
    
    out = []

    for sample in tqdm(dataset):
    
        id = sample.id
        review_bert_tokens = sample.review_bert_tokens # [sent_num, seq_len]
        review_length = sample.review_length
        review_bert_tokens_id = sample.review_bert_tokens_id
        review_num_tokens = sample.review_num_tokens
        review_sent_import_word_spans = sample.review_sent_import_word_span
        
        reply_bert_tokens = sample.reply_bert_tokens
        reply_length = sample.reply_length
        reply_bert_tokens_id = sample.reply_bert_tokens_id
        reply_num_tokens = sample.reply_num_tokens
        reply_sent_import_word_spans = sample.reply_sent_import_word_spans
        
        review_import_word_indexes = sample.review_import_word_indexes
        reply_import_word_indexes = sample.reply_import_word_indexes
        review_sents_words = sample.review_sents_words
        reply_sents_words = sample.reply_sents_words
        review_import_span2word_list = sample.review_import_span2word_list
        reply_import_span2word_list = sample.reply_import_span2word_list
        
        tags = sample.tags
        review_adj = sample.review_adj
        reply_adj = sample.reply_adj
        
        # there are two method to claculate the connection between words in sentence pair.
        if args.probing_mask == 'twomask':
            # # [sent_pair_num, 2, iw_num1, iw_num2]
            # first mask a word x in S construct the S/x and extract the represent of x in S/x, then mask another word y in the S/x to construct the S/{x,y} and extract the represent of x in S/{x,y}

            rev2rev_dis_matrix_as_dic, rev2rev_min_max_values = get_re2re_hidden_states_twomask(args, model,
                                                                    review_bert_tokens, review_bert_tokens_id,
                                                                    review_sent_import_word_spans,
                                                                    review_bert_tokens, review_bert_tokens_id,
                                                                    review_sent_import_word_spans,
                                                                    mask_id, mode='same')

            rev2rep_dis_matrix_as_dic, rev2rep_min_max_values = get_re2re_hidden_states_twomask(args, model,
                                                                        review_bert_tokens, review_bert_tokens_id,
                                                                        review_sent_import_word_spans,
                                                                        reply_bert_tokens, reply_bert_tokens_id,
                                                                        reply_sent_import_word_spans,
                                                                        mask_id, mode='diff')

            rep2rep_dis_matrix_as_dic, rep2rep_min_max_values = get_re2re_hidden_states_twomask(args, model,
                                                                      reply_bert_tokens, reply_bert_tokens_id,
                                                                      reply_sent_import_word_spans,
                                                                      reply_bert_tokens, reply_bert_tokens_id,
                                                                      reply_sent_import_word_spans,
                                                                      mask_id, mode='same')  # [layer, sent_num, batch_size, seq_len, hidden_size]
        elif args.probing_mask == 'onemask':
            # first extract the represent of x in S, then mask another word y in the S to construct the S/y and extract the represent of x in the S/y
    
            # # [sent_pair_num, 2, iw_num1, iw_num2]
            rev2rev_dis_matrix_as_dic, rev2rev_min_max_values = get_re2re_hidden_states_onemask(args, model,
                                                                        review_bert_tokens, review_bert_tokens_id,
                                                                        review_sent_import_word_spans,
                                                                        review_bert_tokens, review_bert_tokens_id,
                                                                        review_sent_import_word_spans,
                                                                        mask_id, mode='same')

            rev2rep_dis_matrix_as_dic, rev2rep_min_max_values = get_re2re_hidden_states_onemask(args, model,
                                                                        review_bert_tokens, review_bert_tokens_id,
                                                                        review_sent_import_word_spans,
                                                                        reply_bert_tokens, reply_bert_tokens_id,
                                                                        reply_sent_import_word_spans,
                                                                        mask_id, mode='diff')

            rep2rep_dis_matrix_as_dic, rep2rep_min_max_values = get_re2re_hidden_states_onemask(args, model,
                                                                        reply_bert_tokens, reply_bert_tokens_id,
                                                                        reply_sent_import_word_spans,
                                                                        reply_bert_tokens, reply_bert_tokens_id,
                                                                        reply_sent_import_word_spans,
                                                                        mask_id, mode='same')
        
        # rev2rev_min_max_values， rev2rep_min_max_values是否式相互独立，即rev中相似句子与rev2rep中相似句子的构建是否是无关的
        # 如果有关，则融合rev2rev_min_max_values和rev2rep_min_max_values
        
        # rev2rep_weight = get_weight_for_sent_pair(rev2rep_dis_matrix_as_dic, tags, rev2rep_min_max_values,
        #                                           review_bert_tokens, review_sent_import_word_spans,
        #                                           reply_bert_tokens, reply_sent_import_word_spans)
        
        # rev2rev_weight = get_weight_for_sent_pair(rev2rev_dis_matrix_as_dic, review_adj, rev2rev_min_max_values,
        #                                           review_bert_tokens, review_sent_import_word_spans,
        #                                           review_bert_tokens, review_sent_import_word_spans)
        
        # rep2rev_weight = get_weight_for_sent_pair(rep2rep_dis_matrix_as_dic, reply_adj,
        #                                           reply_bert_tokens, reply_sent_import_word_spans,
        #                                           reply_bert_tokens, reply_sent_import_word_spans)
            
        instance_weight_matrices = {}
        # instance_weight_matrices["review2review_weights"] = rev2rev_weight
        # instance_weight_matrices["review2reply_weights"] = rev2rep_weight
        # instance_weight_matrices["reply2review_weights"] = rep2rev_weight

        instance_weight_matrices["id"] = id
        instance_weight_matrices["rev2rev_dis_matrix_as_dic"] = rev2rev_dis_matrix_as_dic
        instance_weight_matrices["rev2rev_min_max_values"] = rev2rev_min_max_values
        instance_weight_matrices["rev2rep_dis_matrix_as_dic"] = rev2rep_dis_matrix_as_dic
        instance_weight_matrices["rev2rep_min_max_values"] = rev2rep_min_max_values
        instance_weight_matrices["rep2rep_dis_matrix_as_dic"] = rep2rep_dis_matrix_as_dic
        instance_weight_matrices["rep2rep_min_max_values"] = rep2rep_min_max_values
        
        out.append(instance_weight_matrices)
        
    return out
        

if __name__ == '__main__':
    
    args = shared_configs.get_args()
    # args.device = "cuda:2"
    # model_class, tokenizer_class, pretrained_weights = MODEL_CLASSES[args.model_type]
    # model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
    # tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=True)
    
    special_tokens = ['[ENDL]', '[TAB]', '[LINE]', '[EQU]', '[URL]', '[NUM]', '[SPE]']
    
    # with open(args.bert_weights_path + "vocab.txt", 'wr') as fin:
    #     for special_token in special_tokens:
    
    model_class = args.bert_weights_path.split("/")[-1]
    model = context_models[model_class]['model'].from_pretrained(args.bert_weights_path, output_hidden_states=True).to(args.device)
    tokenizer = context_models[model_class]['tokenizer'].from_pretrained(args.bert_weights_path,
                                                                         additional_special_tokens=special_tokens)
    dataset = args.probing_dataset
    
    train_sentence_packs = eval(json.load(open(args.data_path + '/train3.json'.format(dataset))))
    dev_sentence_packs = eval(json.load(open(args.data_path + '/dev3.json'.format(dataset))))
    test_sentence_packs = eval(json.load(open(args.data_path + '/test3.json'.format(dataset))))
    print("args.data_path", args.data_path, dataset)
    print(len(train_sentence_packs), len(dev_sentence_packs), len(test_sentence_packs))
    
    # all_instances = instances_train + instances_dev + instances_test
    # all_instances = instances_train[:1]
    if args.probing_dataset == 'train':
        if len(train_sentence_packs[args.start_sample:]) > 100:
            train_sentence_packs = train_sentence_packs[args.start_sample:args.start_sample+100]
        else:
            train_sentence_packs = train_sentence_packs[args.start_sample:]
        number = len(train_sentence_packs)
        print(len(train_sentence_packs))
        all_instances = load_data_instances(train_sentence_packs, tokenizer, args, is_train=False)
    elif args.probing_dataset == 'dev':
        if len(dev_sentence_packs[args.start_sample:]) > 100:
            dev_sentence_packs = dev_sentence_packs[args.start_sample:args.start_sample+100]
        else:
            dev_sentence_packs = dev_sentence_packs[args.start_sample:]
        number = len(dev_sentence_packs)
        print(len(dev_sentence_packs))
        all_instances = load_data_instances(dev_sentence_packs, tokenizer, args, is_train=False)
    else:
        if len(test_sentence_packs[args.start_sample:]) > 100:
            test_sentence_packs = test_sentence_packs[args.start_sample:args.start_sample+100]
        else:
            test_sentence_packs = test_sentence_packs[args.start_sample:]
        number = len(test_sentence_packs)
        print(len(test_sentence_packs))
        all_instances = load_data_instances(test_sentence_packs, tokenizer, args, is_train=False)

    out = get_dep_matrix(args, model, tokenizer, all_instances)
    
    print(args.start_sample, args.start_sample+number-1)
    output_dir = args.data_path + "/pkl/"
    output_file = output_dir + f"{args.probing_dataset}/dis_matrix_{str(args.start_sample)}_{str(args.start_sample+number-1)}.pkl"
    print("output_file", output_file)
    pickle.dump(out, open(output_file, 'wb'))
