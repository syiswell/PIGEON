import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from copy import deepcopy
import numpy as np
import math
from utils.basic_utils import get_index_positions, flat_list_of_lists
from modeling.embedder import Embedder, TokenEmbedder
from utils.utils import context_models, extract_arguments
from modeling.graph import *
from torchcrf import CRF
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pad_sequence, pack_padded_sequence


class BertProber(nn.Module):
    def __init__(self, cfg):
        super(BertProber, self).__init__()
        self.use_probing_topic = cfg.use_probing_topic

        self.bert = context_models['bert-base-uncased']['model'].from_pretrained(cfg.bert_weights_path)

        # if cfg.token_embedding:
        # 	self.embedder = TokenEmbedder(bertModel, cfg)
        # else:
        # 	self.embedder = Embedder(bertModel)
        if cfg.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.rep_type = cfg.rep_type

        self.top_iw = cfg.word_top

    def bert_emb(self, re_list):
        sent_tokens_list = [sent for sent_list in re_list for sent in sent_list]

        ids_padding_list, mask_list = self.padding_and_mask(sent_tokens_list)
        ids_padding_tensor = torch.tensor(ids_padding_list).cuda() # [batch_size*sent_num, sent_len]
        mask_tensor = torch.tensor(mask_list).cuda()
        bert_outputs = self.bert(ids_padding_tensor, attention_mask=mask_tensor)

        return bert_outputs.last_hidden_state


    def padding_and_mask(self, ids_list):
        max_len = max([len(x) for x in ids_list])
        mask_list = []
        ids_padding_list = []
        for ids in ids_list:
            mask = [1.] * len(ids) + [0.] * (max_len - len(ids))
            ids = ids + [0] * (max_len - len(ids))
            mask_list.append(mask)
            ids_padding_list.append(ids)
        return ids_padding_list, mask_list


    def forward(self, review_embedder_input, reply_embedder_input,
                review_num_tokens, reply_num_tokens,
                review_lengths, reply_lengths):
        '''
        :param review_embedder_input:
        :param reply_embedder_input:
        :param reply_sorted_iw_spans:
        :param review_lengths:
        :param reply_lengths:
        :return:
        '''

        review_token_feature = self.bert_emb(review_embedder_input)  # review_token_feature : [batch_size * sent_num, max_sent_len, dim]
        reply_token_feature = self.bert_emb(reply_embedder_input)  # reply_feature: [batch_size, num_sents, bert_feature_dim]

        assert review_token_feature.size(0) == sum(review_lengths)
        if self.rep_type == "mean":

            review_feature_batch = torch.split(review_token_feature, review_lengths, 0)
            review_feature = []
            for i in range(len(review_lengths)):
                review_sent_feature_list = []
                for j in range(len(review_num_tokens[i])):
                    review_sent_feature_list.append(review_feature_batch[i][j, 1:review_num_tokens[i][j]+1].mean(0))
                review_sent_feature = torch.stack(review_sent_feature_list) # [rev_sent_num, hidden_size]
                review_feature.append(review_sent_feature)

            reply_feature_batch = torch.split(reply_token_feature, reply_lengths, 0)
            reply_feature = []
            for i in range(len(reply_lengths)):
                reply_sent_feature_list = []
                for j in range(len(reply_num_tokens[i])):
                    reply_sent_feature_list.append(reply_feature_batch[i][j, 1:reply_num_tokens[i][j] + 1].mean(0))
                reply_sent_feature = torch.stack(reply_sent_feature_list)  # [rep_sent_num, hidden_size]
                reply_feature.append(reply_sent_feature)
        elif self.rep_type == "cls":
            review_feature_batch = torch.split(review_token_feature, review_lengths, 0)
            review_feature = []
            for i in range(len(review_lengths)):
                review_sent_feature_list = []
                for j in range(len(review_num_tokens[i])):
                    review_sent_feature_list.append(review_feature_batch[i][j, 0])
                review_sent_feature = torch.stack(review_sent_feature_list)  # [rev_sent_num, hidden_size]
                review_feature.append(review_sent_feature)

            reply_feature_batch = torch.split(reply_token_feature, reply_lengths, 0)
            reply_feature = []
            for i in range(len(reply_lengths)):
                reply_sent_feature_list = []
                for j in range(len(reply_num_tokens[i])):
                    reply_sent_feature_list.append(reply_feature_batch[i][j, 0])
                reply_sent_feature = torch.stack(reply_sent_feature_list)  # [rep_sent_num, hidden_size]
                reply_feature.append(reply_sent_feature)
        else:
            raise ValueError

        return review_feature, reply_feature


class TGModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.graph_type = config.graph_type
        self.hidden_size = config.hidden_size
        self.psg_layer_num = config.tg_layer_num

        self.dropout = config.dropout

        self.use_sent_rep = config.use_sent_rep

        self.psg_hidden_size = config.bert_output_size

        if self.hidden_size != self.psg_hidden_size:
            self.dec_bilstm = nn.LSTM(self.psg_hidden_size, self.hidden_size // 2,
                                       num_layers=1, bidirectional=True, batch_first=True)

        self.psg_layers = nn.ModuleList()
        for i in range(self.psg_layer_num):
            self.psg_layers.append(Intra_Inter_GCN4(self.hidden_size, residual=config.graph_residual))

        self.cl_proj_dim = config.cl_proj_dim
        if self.cl_proj_dim == 0:
            self.cl_proj_dim = self.hidden_size

        self.cl_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.cl_proj_dim, bias=False),
        )

        self.am_bilstm = nn.LSTM(self.hidden_size, self.hidden_size // 2,
                                 num_layers=1, bidirectional=True, batch_first=True)

        # self.align_linear = nn.Linear(4*self.hidden_size, self.hidden_size)

        # self.dual_graph = DualCrossAttention_concat(config, self.hidden_size, self.hidden_size)
        # self.dual_graph = re2graph(config, self.hidden_size, self.hidden_size, self.dropout)

        self.pair_bilstm = nn.LSTM(self.hidden_size * 2, self.hidden_size,
                                   num_layers=1, bidirectional=True, batch_first=True)
        self.pair_2_bilstm = nn.LSTM(self.hidden_size * 2, self.hidden_size,
                                     num_layers=1, bidirectional=True, batch_first=True)

        self.am_hidden2tag = nn.Linear(self.hidden_size, config.num_tags)
        self.pair_hidden2tag = nn.Linear(self.hidden_size * 2, config.num_tags)
        self.pair_2_hidden2tag = nn.Linear(self.hidden_size * 2, config.num_tags)

        self.am_crf = CRF(config.num_tags, batch_first=True)
        self.pair_crf = CRF(config.num_tags, batch_first=True)
        self.pair_2_crf = CRF(config.num_tags, batch_first=True)

        self.dropout = nn.Dropout(p=config.dropout)

    def padding_and_mask(self, ids_list):
        max_len = max([len(x) for x in ids_list])
        mask_list = []
        ids_padding_list = []
        for ids in ids_list:
            mask = [1.] * len(ids) + [0.] * (max_len - len(ids))
            ids = ids + [0] * (max_len - len(ids))
            mask_list.append(mask)
            ids_padding_list.append(ids)
        return ids_padding_list, mask_list

    def am_tagging(self, re_features, re_lengths, tags_list, mode='train'):
        tags_list, tags_mask = self.padding_and_mask(tags_list)
        tags_tensor = torch.tensor(tags_list).cuda() # [batch_size, max_seq_length]
        tags_mask_tensor = torch.tensor(tags_mask).cuda() # [batch_size, max_seq_length]

        # re_features : tensor [batch_size, max_sent_num, hidden_size)]
        re_features_packed = pack_padded_sequence(re_features, torch.tensor(re_lengths),
                                                  batch_first=True, enforce_sorted=False)
        re_lstm_out_packed, _ = self.am_bilstm(re_features_packed)

        # re_features : list [(sent_num, hidden_size)]
        # re_emb_packed = pack_sequence(re_features, enforce_sorted=False)
        # re_lstm_out_packed, _ = self.am_bilstm(re_emb_packed) #

        re_lstm_out_padded, _ = pad_packed_sequence(re_lstm_out_packed, batch_first=True) # [batch_size, max_seq_length, hidden_size]

        # re_lstm_out = re_lstm_out_padded[tags_mask_tensor.bool()]
        # tags_prob = self.am_hidden2tag(re_lstm_out)
        # tags_prob_list = torch.split(tags_prob, re_lengths, 0)
        # tags_prob_padded = pad_sequence(tags_prob_list, batch_first=True)

        # re_lstm_out_padded_2 = re_lstm_out_padded + re_features # residual connection

        tags_prob_padded = self.am_hidden2tag(re_lstm_out_padded)
        tags_prob_padded = tags_prob_padded * tags_mask_tensor.unsqueeze(-1)

        loss, pred_args = None, None
        if mode == 'train':
            loss = -1 * self.am_crf(tags_prob_padded, tags_tensor, mask=tags_mask_tensor.byte())
        else:
            pred_args = self.am_crf.decode(tags_prob_padded, tags_mask_tensor.byte())

        # re_lstm_out_list = []
        # for batch_i, sent_num in enumerate(re_lengths):
        # 	re_lstm_out_list.append(re_lstm_out_padded[batch_i][:sent_num])

        return loss, pred_args, re_lstm_out_padded


    def forward(self,
                review_feature_list,
                reply_feature_list,
                review_masks, reply_masks,
                review_lengths, reply_lengths,
                rev2rev_weights, rev2rep_weights,
                rep2rev_weights, rep2rep_weights,
                rev2rev_weights_gcl, rev2rep_weights_gcl,
                rep2rev_weights_gcl, rep2rep_weights_gcl,
                review_ibose_list, reply_ibose_list,
                rev_arg_2_rep_arg_tags_list,
                rep_arg_2_rev_arg_tags_list):

        batch_size = len(review_feature_list)

        # if self.graph_type == "single":
        review_emb_paded = pad_sequence(review_feature_list, batch_first=True)
        reply_emb_paded = pad_sequence(reply_feature_list, batch_first=True)
        assert review_masks.size() == review_emb_paded.size()[:2]

        review_reps_paded = self.dropout(review_emb_paded)
        reply_reps_paded = self.dropout(reply_emb_paded)

        batch_size, max_review_sents_num, _ = review_reps_paded.size()
        _, max_reply_sents_num, _ = reply_reps_paded.size()

        # use_pos_emb in here is bad

        if self.hidden_size != self.psg_hidden_size:
            # 	review_reps_paded = self.dec_layer(review_reps_paded)
            # 	reply_reps_paded = self.dec_layer(reply_reps_paded)
            review_dec_packed = pack_padded_sequence(review_reps_paded, torch.tensor(review_lengths),
                                                      batch_first=True, enforce_sorted=False)
            review_dec_packed, _ = self.dec_bilstm(review_dec_packed)
            review_reps_paded, _ = pad_packed_sequence(review_dec_packed,
                                                        batch_first=True)  # [batch_size, max_re_sents_num, hidden_size]
            reply_dec_packed = pack_padded_sequence(reply_reps_paded, torch.tensor(reply_lengths),
                                                     batch_first=True, enforce_sorted=False)
            reply_dec_packed, _ = self.dec_bilstm(reply_dec_packed)
            reply_reps_paded, _ = pad_packed_sequence(reply_dec_packed, batch_first=True)


        layer_reps = []

        review_gate_attn_sum = torch.zeros((batch_size, max_review_sents_num)).to(self.config.device)
        reply_gate_attn_sum = torch.zeros((batch_size, max_reply_sents_num)).to(self.config.device)

        review_reps_paded_gcl = review_reps_paded
        reply_reps_paded_gcl = reply_reps_paded
        for i in range(self.psg_layer_num):
            review_reps_paded_gcl, reply_reps_paded_gcl, \
            review_gcn_rep1_gcl, review_gcn_rep2_gcl, \
            reply_gcn_rep1_gcl, reply_gcn_rep2_gcl = self.psg_layers[i](review_reps_paded_gcl, reply_reps_paded_gcl,
                                                                rev2rev_weights_gcl, rep2rep_weights_gcl,
                                                                rev2rep_weights_gcl, rep2rev_weights_gcl)

            review_reps_paded, reply_reps_paded,  \
            review_gcn_rep1, review_gcn_rep2, \
            reply_gcn_rep1, reply_gcn_rep2 = self.psg_layers[i](review_reps_paded, reply_reps_paded,
                                                                     rev2rev_weights, rep2rep_weights,
                                                                     rev2rep_weights, rep2rev_weights)

            layer_reps.append([review_reps_paded, reply_reps_paded])


        review_reps_gcl1 = self.cl_proj(review_reps_paded)
        reply_reps_gcl1 = self.cl_proj(reply_reps_paded)
        review_reps_gcl2 = self.cl_proj(review_reps_paded_gcl)
        reply_reps_gcl2 = self.cl_proj(reply_reps_paded_gcl)

        review_gcn_rep1_gcl1 = self.cl_proj(review_gcn_rep1)
        review_gcn_rep2_gcl1 = self.cl_proj(review_gcn_rep2)
        reply_gcn_rep1_gcl1 = self.cl_proj(reply_gcn_rep1)
        reply_gcn_rep2_gcl1 = self.cl_proj(reply_gcn_rep2)
        review_gcn_rep1_gcl2 = self.cl_proj(review_gcn_rep1_gcl)
        review_gcn_rep2_gcl2 = self.cl_proj(review_gcn_rep2_gcl)
        reply_gcn_rep1_gcl2 = self.cl_proj(reply_gcn_rep1_gcl)
        reply_gcn_rep2_gcl2 = self.cl_proj(reply_gcn_rep2_gcl)


        # if self.cl_layer != "graph" and self.cl_layer != "bert" and self.cl_layer != "lstm":
        # 	raise ValueError


        # review
        rev_crf_loss, _, rev_lstm_out_padded = self.am_tagging(review_reps_paded, review_lengths, review_ibose_list,
                                                               mode='train')
        # reply
        rep_crf_loss, _, rep_lstm_out_padded = self.am_tagging(reply_reps_paded, reply_lengths, reply_ibose_list,
                                                               mode='train')

        rev_mix_reps_paded = review_reps_paded + rev_lstm_out_padded # hope to capture the graph and sequence information
        rep_mix_reps_paded = reply_reps_paded + rep_lstm_out_padded # need to improve

        rev_mix_reps_list = []
        for batch_i, sent_num in enumerate(review_lengths):
            rev_mix_reps_list.append(rev_mix_reps_paded[batch_i][:sent_num])
        rep_mix_reps_list = []
        for batch_i, sent_num in enumerate(reply_lengths):
            rep_mix_reps_list.append(rep_mix_reps_paded[batch_i][:sent_num])

        # pair
        review_args_rep = []
        pair_tags_list = [] # [all_arg_num, rep_sent_num]
        rev_args_span_list = []
        for batch_i, rev_arg_2_rep_arg_tags in enumerate(rev_arg_2_rep_arg_tags_list):
            rev_args_span = []
            for rev_arg_span, rep_arg_tags in rev_arg_2_rep_arg_tags.items():
                review_args_rep.append(
                    rev_mix_reps_list[batch_i][rev_arg_span[0]:rev_arg_span[1] + 1].mean(dim=-2))
                pair_tags_list.append(rep_arg_tags)
                rev_args_span.append(rev_arg_span)
            rev_args_span_list.append(rev_args_span)

        review_args_rep_tensor = torch.stack(review_args_rep) # [all_arg_num, hidden_size]
        num_args_list = [len(d) for d in rev_arg_2_rep_arg_tags_list]
        review_args_rep_list = torch.split(review_args_rep_tensor, num_args_list, 0)

        # num_args_list = []
        rep_with_rev_rep_list = [] # [rev_arg_num, rep_sent_num, dim]
        for batch_i, review_args_rep in enumerate(review_args_rep_list):
            # num_args_list.append(review_args_rep.shape[0])
            for arg_idx in range(review_args_rep.size(0)):
                args_rep = review_args_rep[arg_idx].unsqueeze(0).repeat(reply_lengths[batch_i], 1)
                rep_with_rev_rep = torch.cat([rep_mix_reps_list[batch_i], args_rep], dim=-1) # [rep_sent_num, dim]
                # rep_with_rev_rep = rep_mix_reps_list[batch_i] * args_rep
                # rep_with_rev_rep = rep_mix_reps_list[batch_i] + args_rep

                rep_with_rev_rep_list.append(rep_with_rev_rep)

        rep_with_rev_rep_packed = pack_sequence(rep_with_rev_rep_list, enforce_sorted=False)
        rep_with_rev_lstm_out_packed, _ = self.pair_bilstm(rep_with_rev_rep_packed)
        rep_with_rev_lstm_out_padded, _ = pad_packed_sequence(rep_with_rev_lstm_out_packed, batch_first=True) # [rev_arg_num, rep_sent_num, dim]

        # rep_tags_len_list = [len(tags) for tags in pair_tags_list]
        # rep_max_tags_len = max(rep_tags_len_list)
        pair_tags_list, tags_mask = self.padding_and_mask(pair_tags_list)
        pair_tags_tensor = torch.tensor(pair_tags_list).cuda()
        pair_tags_mask_tensor = torch.tensor(tags_mask).cuda()

        # rep_with_rev_lstm_out = rep_with_rev_lstm_out_padded[pair_tags_mask_tensor.bool()]
        # pair_tags_prob = self.pair_hidden2tag(rep_with_rev_lstm_out)
        # pair_tags_prob_list = torch.split(pair_tags_prob, num_sent_list, 0)
        # pair_tags_prob_padded = pad_sequence(pair_tags_prob_list, batch_first=True)
        pair_tags_prob_padded = self.pair_hidden2tag(rep_with_rev_lstm_out_padded)
        # print("pair_tags_prob_padded", pair_tags_prob_padded.size(), pair_tags_mask_tensor.size())
        pair_tags_prob_padded = pair_tags_prob_padded * pair_tags_mask_tensor.unsqueeze(-1)

        pair_loss = -1 * self.pair_crf(pair_tags_prob_padded, pair_tags_tensor,
                                       mask=pair_tags_mask_tensor.byte())

        reply_args_rep = []
        pair_tags_2_list = []
        rep_args_span_list = []
        for batch_i, rep_arg_2_rev_arg_tags in enumerate(rep_arg_2_rev_arg_tags_list):
            rep_args_span = []
            for rep_arg_span, rev_arg_tags in rep_arg_2_rev_arg_tags.items():
                reply_args_rep.append(
                    rep_mix_reps_list[batch_i][rep_arg_span[0]:rep_arg_span[1] + 1].mean(dim=-2))
                pair_tags_2_list.append(rev_arg_tags)
                rep_args_span.append(rep_arg_span)
            rep_args_span_list.append(rep_args_span)

        reply_args_rep_tensor = torch.stack(reply_args_rep)
        num_args_list = [len(d) for d in rep_arg_2_rev_arg_tags_list]
        reply_args_rep_list = torch.split(reply_args_rep_tensor, num_args_list, 0)

        # num_args_list = []
        rev_with_rep_rep_list = [] # [rep_arg_num, rev_sent_num, 2*dim]
        for batch_i, reply_args_rep in enumerate(reply_args_rep_list):
            # num_args_list.append(reply_args_rep.shape[0])
            for arg_idx in range(reply_args_rep.size(0)):
                args_rep = reply_args_rep[arg_idx].unsqueeze(0).repeat(review_lengths[batch_i], 1)
                rev_with_rep_rep = torch.cat([rev_mix_reps_list[batch_i], args_rep], dim=-1) # [rev_sent_num, 2*dim]
                # rev_with_rep_rep = rev_mix_reps_list[batch_i] * args_rep # [rev_sent_num, dim]
                # rev_with_rep_rep = rev_mix_reps_list[batch_i] + args_rep

                rev_with_rep_rep_list.append(rev_with_rep_rep)

        rev_with_rep_rep_packed = pack_sequence(rev_with_rep_rep_list, enforce_sorted=False)
        rev_with_rep_lstm_out_packed, _ = self.pair_2_bilstm(rev_with_rep_rep_packed)
        rev_with_rep_lstm_out_padded, _ = pad_packed_sequence(rev_with_rep_lstm_out_packed, batch_first=True)

        # rev_tags_len_list = [len(tags) for tags in pair_tags_2_list]
        # rev_max_tags_len = max(rev_tags_len_list)
        pair_tags_2_list, tags_mask = self.padding_and_mask(pair_tags_2_list)
        pair_tags_2_tensor = torch.tensor(pair_tags_2_list).cuda()
        pair_tags_mask_tensor = torch.tensor(tags_mask).cuda() # [arg_num, sent_num]

        # rev_with_rep_lstm_out = rev_with_rep_lstm_out_padded[pair_tags_mask_tensor.bool()]
        # pair_tags_2_prob = self.pair_2_hidden2tag(rev_with_rep_lstm_out)
        # pair_tags_2_prob_list = torch.split(pair_tags_2_prob, num_sent_list, 0)
        # pair_tags_2_prob_padded = pad_sequence(pair_tags_2_prob_list, batch_first=True)
        pair_tags_2_prob_padded = self.pair_2_hidden2tag(rev_with_rep_lstm_out_padded)
        pair_tags_2_prob_padded = pair_tags_2_prob_padded * pair_tags_mask_tensor.unsqueeze(-1)

        pair_2_loss = -1 * self.pair_2_crf(pair_tags_2_prob_padded, pair_tags_2_tensor,
                                           mask=pair_tags_mask_tensor.byte())


        # elif self.graph_type == "dual":
        #
        # 	pass

        return rev_crf_loss + rep_crf_loss, pair_loss + pair_2_loss, \
               review_reps_gcl1, reply_reps_gcl1, review_reps_gcl2, reply_reps_gcl2, \
               review_gcn_rep1_gcl1, review_gcn_rep2_gcl1, reply_gcn_rep1_gcl1, reply_gcn_rep2_gcl1, \
               review_gcn_rep1_gcl2, review_gcn_rep2_gcl2, reply_gcn_rep1_gcl2, reply_gcn_rep2_gcl2


    def predict(self, review_feature_list,
                reply_feature_list,
                review_masks, reply_masks,
                review_lengths, reply_lengths,
                rev2rev_weights, rev2rep_weights,
                rep2rev_weights, rep2rep_weights,
                review_ibose_list=None, reply_ibose_list=None,):

        batch_size = len(review_feature_list)


        # if self.graph_type == "single":

        review_emb_paded = pad_sequence(review_feature_list, batch_first=True)
        reply_emb_paded = pad_sequence(reply_feature_list, batch_first=True)
        assert review_masks.size() == review_emb_paded.size()[:2]

        review_reps_paded = self.dropout(review_emb_paded)
        reply_reps_paded = self.dropout(reply_emb_paded)

        batch_size, max_review_sents_num, _ = review_reps_paded.size()
        _, max_reply_sents_num, _ = reply_reps_paded.size()

        if self.hidden_size != self.psg_hidden_size:
            # 	review_reps_paded = self.dec_layer(review_reps_paded)
            # 	reply_reps_paded = self.dec_layer(reply_reps_paded)
            review_dec_packed = pack_padded_sequence(review_reps_paded, torch.tensor(review_lengths),
                                                     batch_first=True, enforce_sorted=False)
            review_dec_packed, _ = self.dec_bilstm(review_dec_packed)
            review_reps_paded, _ = pad_packed_sequence(review_dec_packed,
                                                       batch_first=True)  # [batch_size, max_seq_length, hidden_size]
            reply_dec_packed = pack_padded_sequence(reply_reps_paded, torch.tensor(reply_lengths),
                                                    batch_first=True, enforce_sorted=False)
            reply_dec_packed, _ = self.dec_bilstm(reply_dec_packed)
            reply_reps_paded, _ = pad_packed_sequence(reply_dec_packed, batch_first=True)

        layer_reps = []
        for i in range(self.psg_layer_num):
            review_reps_paded, reply_reps_paded, \
            review_gcn_rep1, review_gcn_rep2, \
            reply_gcn_rep1, reply_gcn_rep2 = self.psg_layers[i](review_reps_paded, reply_reps_paded,
                                                                     rev2rev_weights, rep2rep_weights,
                                                                     rev2rep_weights, rep2rev_weights)

            layer_reps.append([review_reps_paded, reply_reps_paded])


        # review
        _, pred_rev_args, rev_lstm_out_padded = self.am_tagging(review_reps_paded, review_lengths,
                                                                   review_ibose_list, mode='pred')
        pred_rev_args_list = extract_arguments(pred_rev_args)

        # reply
        _, pred_rep_args, rep_lstm_out_padded = self.am_tagging(reply_reps_paded, reply_lengths,
                                                                   reply_ibose_list, mode='pred')
        pred_rep_args_list = extract_arguments(pred_rep_args)


        rev_mix_reps_paded = review_reps_paded + rev_lstm_out_padded  # hope to capture the graph and sequence information
        rep_mix_reps_paded = reply_reps_paded + rep_lstm_out_padded  # need to improve


        rev_mix_reps_list = []
        for batch_i, sent_num in enumerate(review_lengths):
            rev_mix_reps_list.append(rev_mix_reps_paded[batch_i][:sent_num])
        rep_mix_reps_list = []
        for batch_i, sent_num in enumerate(reply_lengths):
            rep_mix_reps_list.append(rep_mix_reps_paded[batch_i][:sent_num])

        # pair
        pred_args_pair_dict_list = []
        review_args_rep = []
        for batch_i, pred_arguments in enumerate(pred_rev_args_list):
            for rev_arg_span in pred_arguments:
                review_args_rep.append(rev_mix_reps_list[batch_i][rev_arg_span[0]:rev_arg_span[1] + 1].mean(dim=-2))

        if review_args_rep == []:
            pred_args_pair_dict_list = [{} for t in pred_rev_args_list]
        else:
            review_args_rep_tensor = torch.stack(review_args_rep) # [rev_all_arg_num, dim]
            num_args_list = [len(d) for d in pred_rev_args_list]
            review_args_rep_list = torch.split(review_args_rep_tensor, num_args_list, 0) # [batch_size, rev_arg_num,dim]

            rev_with_rep_rep_list = [] # [rev_all_arg_num, reply_sent_num, 2*dim]
            for batch_i, review_args_rep in enumerate(review_args_rep_list):
                for arg_idx in range(review_args_rep.size(0)):
                    args_rep = review_args_rep[arg_idx].unsqueeze(0).repeat(reply_lengths[batch_i], 1)
                    rev_with_rep_rep = torch.cat([rep_mix_reps_list[batch_i], args_rep], dim=-1)
                    # rev_with_rep_rep = rep_mix_reps_list[batch_i] * args_rep
                    # rev_with_rep_rep = rep_mix_reps_list[batch_i] + args_rep

                    rev_with_rep_rep_list.append(rev_with_rep_rep)

            rev_with_rep_rep_packed = pack_sequence(rev_with_rep_rep_list, enforce_sorted=False)
            rev_with_rep_lstm_out_packed, _ = self.pair_bilstm(rev_with_rep_rep_packed)
            rev_with_rep_lstm_out_padded, rep_len_list = pad_packed_sequence(rev_with_rep_lstm_out_packed,
                                                                             batch_first=True)

            max_rep_len = max(rep_len_list)
            tags_mask = [[1] * rep_len + [0] * (max_rep_len - rep_len) for rep_len in rep_len_list]
            pair_tags_mask_tensor = torch.tensor(tags_mask).cuda()

            # rev_with_rep_lstm_out = rev_with_rep_lstm_out_padded[pair_tags_mask_tensor.bool()]
            # pair_tags_prob = self.pair_hidden2tag(rev_with_rep_lstm_out)
            # pair_tags_prob_list = torch.split(pair_tags_prob, num_sent_list, 0)
            # pair_tags_prob_padded = pad_sequence(pair_tags_prob_list, batch_first=True)
            pair_tags_prob_padded = self.pair_hidden2tag(rev_with_rep_lstm_out_padded) # [rev_all_arg_num, max_reply_sent_num, 6]
            pair_tags_prob_padded = pair_tags_prob_padded * pair_tags_mask_tensor.unsqueeze(-1)

            pred_pair_rep_args_tag = self.pair_crf.decode(pair_tags_prob_padded, pair_tags_mask_tensor.byte()) # [rev_all_arg_num, max_reply_sent_num]
            pred_pair_rep_args_list = extract_arguments(pred_pair_rep_args_tag) # [rev_arg_num, rep_arg_num]

            # prob
            pred_pair_rep_args_prob_list = [] # [rev_arg_num, rep_arg_num]
            for arg_idx in range(pair_tags_prob_padded.size(0)):
                full_emission = pair_tags_prob_padded[arg_idx]
                full_mask = pair_tags_mask_tensor[arg_idx].byte()
                full_tags = torch.tensor(pred_pair_rep_args_tag[arg_idx]).cuda() # [max_reply_sent_num]
                pred_pair_rep_args_probs = []
                for pred_span in pred_pair_rep_args_list[arg_idx]:
                    start, end = pred_span
                    # emissions: (seq_length, batch_size, num_tags)
                    # tags: (seq_length, batch_size)
                    # mask: (seq_length, batch_size)
                    emission = full_emission[start: end + 1].unsqueeze(1)  # [1, end-start+1]
                    mask = full_mask[start: end + 1].unsqueeze(1)  # [1, end-start+1]
                    tags = full_tags[start: end + 1].unsqueeze(1)  # [1, end-start+1]
                    numerator = self.pair_crf._compute_score(emission, tags, mask)
                    denominatr = self.pair_crf._compute_normalizer(emission, mask)
                    llh = numerator - denominatr
                    prob = llh.exp()
                    pred_pair_rep_args_probs.append(prob.tolist()[0])
                pred_pair_rep_args_prob_list.append(pred_pair_rep_args_probs)

            i = 0
            for pred_arguments in pred_rev_args_list:
                pred_args_pair_dict = {}
                for args in pred_arguments:
                    pred_args_pair_dict[args] = (pred_pair_rep_args_list[i],
                                                 pred_pair_rep_args_prob_list[i])
                    i += 1
                pred_args_pair_dict_list.append(pred_args_pair_dict)


        pred_args_pair_2_dict_list = []
        reply_args_rep = []
        for batch_i, pred_arguments in enumerate(pred_rep_args_list):
            for rep_arg_span in pred_arguments:
                reply_args_rep.append(rep_mix_reps_list[batch_i][rep_arg_span[0]:rep_arg_span[1] + 1].mean(dim=-2))

        if reply_args_rep == []:
            pred_args_pair_2_dict_list = [{} for t in pred_rep_args_list]
        else:
            reply_args_rep_tensor = torch.stack(reply_args_rep)
            num_args_list = [len(d) for d in pred_rep_args_list]
            reply_args_rep_list = torch.split(reply_args_rep_tensor, num_args_list, 0)

            rep_with_rev_rev_list = []
            for batch_i, reply_args_rep in enumerate(reply_args_rep_list):
                for arg_idx in range(reply_args_rep.shape[0]):
                    args_rep = reply_args_rep[arg_idx].unsqueeze(0).repeat(review_lengths[batch_i], 1)
                    rep_with_rev_rev = torch.cat([rev_mix_reps_list[batch_i], args_rep], dim=-1)
                    # rep_with_rev_rev = rev_mix_reps_list[batch_i] * args_rep
                    # rep_with_rev_rev = rev_mix_reps_list[batch_i] + args_rep

                    rep_with_rev_rev_list.append(rep_with_rev_rev)

            rep_with_rev_rev_packed = pack_sequence(rep_with_rev_rev_list, enforce_sorted=False)
            rep_with_rev_lstm_out_packed, _ = self.pair_2_bilstm(rep_with_rev_rev_packed)
            rep_with_rev_lstm_out_padded, rev_len_list = pad_packed_sequence(rep_with_rev_lstm_out_packed,
                                                                             batch_first=True)

            max_rev_len = max(rev_len_list)
            tags_mask_2 = [[1] * rev_len + [0] * (max_rev_len - rev_len) for rev_len in rev_len_list]
            pair_tags_2_mask_tensor = torch.tensor(tags_mask_2).cuda()

            # rev_with_rep_lstm_out = rev_with_rep_lstm_out_padded[pair_tags_mask_tensor.bool()]
            # pair_tags_2_prob = self.pair_2_hidden2tag(rev_with_rep_lstm_out)
            # pair_tags_2_prob_list = torch.split(pair_tags_2_prob, num_sent_list, 0)
            # pair_tags_2_prob_padded = pad_sequence(pair_tags_2_prob_list, batch_first=True)
            pair_tags_2_prob_padded = self.pair_2_hidden2tag(rep_with_rev_lstm_out_padded)
            pair_tags_2_prob_padded = pair_tags_2_prob_padded * pair_tags_2_mask_tensor.unsqueeze(-1)

            pred_pair_rep_args_tag_2 = self.pair_2_crf.decode(pair_tags_2_prob_padded, pair_tags_2_mask_tensor.byte())
            pred_pair_rep_args_2_list = extract_arguments(pred_pair_rep_args_tag_2)

            # prob
            pred_pair_rep_args_prob_2_list = []
            for idx in range(pair_tags_2_prob_padded.size(0)):
                full_emission = pair_tags_2_prob_padded[idx]
                full_mask = pair_tags_2_mask_tensor[idx].byte()
                full_tags = torch.tensor(pred_pair_rep_args_tag_2[idx]).cuda()
                pred_pair_rep_args_probs = []
                for pred_span in pred_pair_rep_args_2_list[idx]:
                    start, end = pred_span
                    emission = full_emission[start: end + 1].unsqueeze(1)  # [1, end-start+1]
                    mask = full_mask[start: end + 1].unsqueeze(1)  # [1, end-start+1]
                    tags = full_tags[start:end + 1].unsqueeze(1)  # [1, end-start+1]
                    numerator = self.pair_2_crf._compute_score(emission, tags, mask)
                    denominatr = self.pair_2_crf._compute_normalizer(emission, mask)
                    llh = numerator - denominatr
                    prob = llh.exp()
                    pred_pair_rep_args_probs.append(prob.tolist()[0])
                pred_pair_rep_args_prob_2_list.append(pred_pair_rep_args_probs)

            i = 0
            for pred_arguments in pred_rep_args_list:
                pred_args_pair_2_dict = {}
                for args in pred_arguments:
                    pred_args_pair_2_dict[args] = (pred_pair_rep_args_2_list[i],
                                                   pred_pair_rep_args_prob_2_list[i])
                    i += 1
                pred_args_pair_2_dict_list.append(pred_args_pair_2_dict)

        # elif self.graph_type == 'dual':
        # 	pass

        return pred_rev_args_list, pred_rep_args_list, pred_args_pair_dict_list, pred_args_pair_2_dict_list


class RobertaProber(nn.Module):
    def __init__(self, cfg):
        super(RobertaProber, self).__init__()

        robertaModel = context_models['bert-base-uncased']['model'].from_pretrained(cfg.bert_weights_path)
        if cfg.token_embedding:
            self.embedder = TokenEmbedder(robertaModel, cfg)
        else:
            self.embedder = Embedder(robertaModel)

        self.top_iw = cfg.word_top

    def forward(self, review_embedder_input, reply_embedder_input, review_sorted_iw_spans, reply_sorted_iw_spans):
        pass


class LongFormerProber(nn.Module):
    def __init__(self, cfg):
        super(LongFormerProber, self).__init__()

        robertaModel = context_models['bert-base-uncased']['model'].from_pretrained(cfg.bert_weights_path)
        if cfg.token_embedding:
            self.embedder = TokenEmbedder(robertaModel, cfg)
        else:
            self.embedder = Embedder(robertaModel)

        self.top_iw = cfg.word_top

    def forward(self, review_embedder_input, reply_embedder_input,
                review_num_tokens, reply_num_tokens,
                review_lengths, reply_lengths):



        pass