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


class Biaffine(nn.Module):
    def __init__(self, in1_features, in2_features, out_features, bias=(True, True)):
        super(Biaffine, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.linear = nn.Linear(in_features=self.linear_input_size,
                                out_features=self.linear_output_size,
                                bias=False)

        # self.reset_parameters()

    def reset_parameters(self):
        W = np.zeros((self.linear_output_size, self.linear_input_size), dtype=np.float32)
        self.linear.weight.data.copy_(torch.from_numpy(W))

    def forward(self, input1, input2):
        batch_size, dim1 = input1.size()
        batch_size, dim2 = input2.size()
        if self.bias[0]:
            ones = input1.data.new(batch_size, 1).zero_().fill_(1)
            input1 = torch.cat((input1, Variable(ones)), dim=1)
            dim1 += 1
        if self.bias[1]:
            ones = input2.data.new(batch_size, 1).zero_().fill_(1)
            input2 = torch.cat((input2, Variable(ones)), dim=1)
            dim2 += 1

        affine = self.linear(input1)

        affine = affine.view(batch_size, self.out_features, dim2)
        input2 = input2.unsqueeze(2) # [b, in2_feature, 1]

        biaffine = torch.bmm(affine, input2)

        biaffine = biaffine.contiguous().view(batch_size, self.out_features)

        return biaffine

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'in1_features=' + str(self.in1_features) \
            + ', in2_features=' + str(self.in2_features) \
            + ', out_features=' + str(self.out_features) + ')'


# module for neural topic model
class NTM(nn.Module):
    def __init__(self, opt, hidden_dim=500, l1_strength=0.001):
        super(NTM, self).__init__()
        self.input_dim = opt.bow_vocab_size
        self.topic_num = opt.topic_num
        topic_num = opt.topic_num
        self.fc11 = nn.Linear(self.input_dim, hidden_dim)
        self.fc12 = nn.Linear(hidden_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, topic_num)
        self.fc22 = nn.Linear(hidden_dim, topic_num)
        self.fcs = nn.Linear(self.input_dim, hidden_dim, bias=False)
        self.fcg1 = nn.Linear(topic_num, topic_num)
        self.fcg2 = nn.Linear(topic_num, topic_num)
        self.fcg3 = nn.Linear(topic_num, topic_num)
        self.fcg4 = nn.Linear(topic_num, topic_num)
        self.fcd1 = nn.Linear(topic_num, self.input_dim)
        self.l1_strength = torch.FloatTensor([l1_strength]).to(opt.device)

    def encode(self, x):
        e1 = F.relu(self.fc11(x))
        e1 = F.relu(self.fc12(e1))
        e1 = e1.add(self.fcs(x))
        return self.fc21(e1), self.fc22(e1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def generate(self, h):
        g1 = torch.tanh(self.fcg1(h))
        g1 = torch.tanh(self.fcg2(g1))
        g1 = torch.tanh(self.fcg3(g1))
        g1 = torch.tanh(self.fcg4(g1))
        g1 = g1.add(h)
        return g1

    def decode(self, z):
        d1 = F.softmax(self.fcd1(z), dim=1)
        return d1

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        g = self.generate(z)
        return z, g, self.decode(g), mu, logvar # _, _, recon_x, mu, logvar

    def print_topic_words(self, vocab_dic, fn, n_top_words=10):
        beta_exp = self.fcd1.weight.data.cpu().numpy().T
        print("Writing to %s" % fn)
        fw = open(fn, 'w')
        for k, beta_k in enumerate(beta_exp):
            topic_words = [vocab_dic[w_id] for w_id in np.argsort(beta_k)[:-n_top_words - 1:-1]]
            print('Topic {}: {}'.format(k, ' '.join(topic_words)))
            fw.write('{}\n'.format(' '.join(topic_words)))
        fw.close()

    # Reconstruction + KL divergence losses summed over all elements and batch
    # x: bow_vector
    def loss(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def l1_penalty(self, para):
        return nn.L1Loss()(para, torch.zeros_like(para))


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
	            review_sorted_iw_spans, reply_sorted_iw_spans,
	            review_lengths, reply_lengths):
		'''
		:param review_embedder_input:
		:param reply_embedder_input:
		:param review_sorted_iw_spans: [batch_size, sent_num, iw_num]
		:param reply_sorted_iw_spans:
		:param review_lengths:
		:param reply_lengths:
		:return:
		'''
		
		review_token_feature = self.bert_emb(review_embedder_input)  # review_token_feature : [batch_size * sent_num, max_sent_len, dim]
		reply_token_feature = self.bert_emb(reply_embedder_input)  # reply_feature: [batch_size, num_sents, bert_feature_dim]
		
		assert review_token_feature.size(0) == sum(review_lengths)
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
		
		if self.use_probing_topic:
			review_pt_feature = self.get_probing_topic_feature(review_feature_batch, review_sorted_iw_spans, review_num_tokens)
			reply_pt_feature = self.get_probing_topic_feature(reply_feature_batch, reply_sorted_iw_spans, reply_num_tokens)
		else:
			review_pt_feature = None
			reply_pt_feature = None
		
		return review_pt_feature, review_feature, reply_pt_feature, reply_feature
	
	def get_probing_topic_feature(self, re_token_feature, re_sorted_iw_spans, re_num_tokens):
		re_pt_features = []
		for batch_idx, re_sorted_iw_span in enumerate(re_sorted_iw_spans):
			sents_pt_features = []
			for sent_idx, sent_sorted_iw_span in enumerate(re_sorted_iw_span):
				if sent_sorted_iw_span == None or sent_sorted_iw_span == []:
					sents_pt_features.append(re_token_feature[batch_idx][sent_idx, 1:re_num_tokens[batch_idx][sent_idx]+1].mean(-2))
				else:
					spans_pt_feature = []  # [span_num, dim]
					if len(sent_sorted_iw_span) >= self.top_iw:
						for span in sent_sorted_iw_span[:self.top_iw]:
							# print("span ", span)
							# spans_pt_feature.append(
							# 	re_token_feature[batch_idx][sent_idx, span[0]:span[1] + 1].mean(-2))
							spans_pt_feature.append(
								re_token_feature[batch_idx][sent_idx, span[0]:span[1] + 1])
					else:
						for span in sent_sorted_iw_span:
							# spans_pt_feature.append(
							# 	re_token_feature[batch_idx][sent_idx, span[0]:span[1] + 1].mean(-2))
							spans_pt_feature.append(
								re_token_feature[batch_idx][sent_idx, span[0]:span[1] + 1])
					# sents_pt_features.append(torch.stack(spans_pt_feature).mean(0))
					sents_pt_features.append(torch.cat(spans_pt_feature, 0).mean(0))
					
			re_pt_features.append(torch.stack(sents_pt_features))
		
		return re_pt_features # list: [batch_size, sent_num, dim]


class TGModel(nn.Module):
	
	def __init__(self, config):
		super().__init__()
		
		self.config = config
		self.graph_type = config.graph_type
		
		self.hidden_size = config.hidden_size
		
		self.tg_layer_num = config.tg_layer_num
		
		self.dropout = config.dropout
		
		self.use_sent_rep = config.use_sent_rep
		self.use_probing_topic = config.use_probing_topic
		self.use_ntm = config.use_ntm
		self.use_pos_emb = config.use_pos_emb
		self.use_topic_project = config.use_topic_project
		
		if self.use_ntm:
			self.topic_num = config.topic_num
			self.ntm = NTM(config)
			if config.warm_up_ntm == -1:
				self.ntm.load_state_dict(torch.load(config.ntm_model))
			
			if self.use_topic_project:
				self.topic_project = nn.Linear(self.topic_num, self.hidden_size)
				self.topic_num = self.hidden_size

		
		if self.graph_type == "single": # concat
			if self.use_probing_topic and self.use_ntm:
				if self.use_sent_rep:
					self.tg_hidden_size = config.bert_output_size + config.bert_output_size + self.topic_num
				else:
					self.tg_hidden_size = config.bert_output_size + self.topic_num
			elif self.use_probing_topic and not self.use_ntm:
				if self.use_sent_rep:
					self.tg_hidden_size = config.bert_output_size + config.bert_output_size
				else:
					self.tg_hidden_size = config.bert_output_size
			elif not self.use_probing_topic and self.use_ntm:
				if self.use_sent_rep:
					self.tg_hidden_size = config.bert_output_size + self.topic_num
				else:
					self.tg_hidden_size = self.topic_num
			else:
				self.tg_hidden_size = config.bert_output_size
			
			# if self.hidden_size != self.tg_hidden_size:
			# 	self.dec_layer = nn.Linear(self.tg_hidden_size, self.hidden_size)
			
			if self.use_pos_emb:
				self.K = 192
				self.gamma = config.gamma
				self.rel_pos_emb = nn.Embedding(self.K*2+1, 50)
				# self.rep_pos_emb = nn.Embedding(self.K*2+1, 50)
			
			if self.hidden_size != self.tg_hidden_size:
				self.dec_bilstm = nn.LSTM(self.tg_hidden_size, self.hidden_size // 2,
				                           num_layers=1, bidirectional=True, batch_first=True)
			
			self.tg_layers = nn.ModuleList()
			for i in range(self.tg_layer_num):
				self.tg_layers.append(re2graph(self.config, self.hidden_size, self.hidden_size, self.dropout))
			
			# self.linear_align = nn.Linear(4 * self.hidden_size, self.hidden_size)
			# self.edge_transform_layer = nn.Sequential(
			# 	nn.Linear(self.hidden_size, self.hidden_size // 2),
			# 	nn.LeakyReLU(),  # better
			# 	nn.Dropout(self.dropout),
			# 	nn.Linear(self.hidden_size // 2, self.hidden_size // 2),
			# 	nn.LeakyReLU(),
			# 	nn.Dropout(self.dropout),
			# 	nn.Linear(self.hidden_size // 2, 2),
			# )
		
		# elif self.graph_type == "dual": # semnatic and topic
		#
		# 	self.sg_layers = nn.ModuleList()
		# 	self.tg_layers = nn.ModuleList()
		# 	for i in range(self.tg_layer_num):
		# 		self.sg_layers.append(re2graph(self.config, self.hidden_size, self.dropout))
		# 		self.tg_layers.append(re2graph(self.config, self.hidden_size, self.dropout))
		#
		# 	self.edge_link_input_size = self.hidden_size + self.hidden_size + self.bow_rep_size
		# 	self.edge_transform_layer = nn.Sequential(
		# 		nn.Linear(self.edge_link_input_size, self.hidden_size),
		# 		nn.LeakyReLU(),  # better
		# 		nn.Dropout(self.dropout),
		# 		nn.Linear(self.hidden_size, self.hidden_size),
		# 		nn.LeakyReLU(),
		# 		nn.Dropout(self.dropout),
		# 		nn.Linear(self.hidden_size, 2),
		# 	)
		# 	self.bia_edge_link_layer = Biaffine(self.hidden_size, self.hidden_size, 2,
		# 	                                    bias=(True, True))
		else:
			raise ValueError
		
		self.am_bilstm = nn.LSTM(self.hidden_size, self.hidden_size // 2,
		                         num_layers=1, bidirectional=True, batch_first=True)
		
		# self.align_linear = nn.Linear(4*self.hidden_size, self.hidden_size)
		
		# self.dual_graph = DualCrossAttention_concat(config, self.hidden_size, self.hidden_size)
		# self.dual_graph = re2graph(config, self.hidden_size, self.hidden_size, self.dropout)
		if self.use_pos_emb:
			self.pair_bilstm = nn.LSTM(self.hidden_size * 2 + 50, self.hidden_size,
									   num_layers=1, bidirectional=True, batch_first=True)
			self.pair_2_bilstm = nn.LSTM(self.hidden_size * 2 + 50, self.hidden_size,
										 num_layers=1, bidirectional=True, batch_first=True)
		else:
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
	            review_pt_feature_list, review_feature_list, # list batch_size  [(sent_num, hidden_size), ...]
	            reply_pt_feature_list, reply_feature_list,
	            review_bow_vectors, reply_bow_vectors, # [all_review_num_sents, bow_vocab_size]
	            review_masks, reply_masks,
	            review_lengths, reply_lengths,
	            review_ibose_list, reply_ibose_list,
	            rev_arg_2_rep_arg_tags_list,
	            rep_arg_2_rev_arg_tags_list):
		
		batch_size = len(review_feature_list)
		
		# NTM
		if self.use_ntm:
			review_bow_norm_vectors = F.normalize(review_bow_vectors)
			reply_bow_norm_vectors = F.normalize(reply_bow_vectors)
			_, review_bow_feature, review_recon, review_mu, review_logvar = self.ntm(review_bow_norm_vectors)
			_, reply_bow_feature, reply_recon, reply_mu, reply_logvar = self.ntm(reply_bow_norm_vectors)
			if self.use_topic_project: # useless
				review_bow_feature = self.topic_project(review_bow_feature)
				reply_bow_feature = self.topic_project(reply_bow_feature)
			review_bow_feature = torch.split(review_bow_feature, review_lengths, 0)
			reply_bow_feature = torch.split(reply_bow_feature, reply_lengths, 0)
		
		if self.graph_type == "single":
			if self.use_probing_topic and self.use_ntm:
				review_pt_feature_paded = pad_sequence(review_pt_feature_list, batch_first=True)
				reply_pt_feature_paded = pad_sequence(reply_pt_feature_list, batch_first=True)
				review_bow_feature_padded = pad_sequence(review_bow_feature, batch_first=True)
				reply_bow_feature_padded = pad_sequence(reply_bow_feature, batch_first=True)
				
				if self.use_sent_rep:
					review_emb_paded = pad_sequence(review_feature_list, batch_first=True)
					reply_emb_paded = pad_sequence(reply_feature_list, batch_first=True)
					
					review_reps_paded = torch.cat([review_pt_feature_paded, review_emb_paded, review_bow_feature_padded], -1)
					reply_reps_paded = torch.cat([reply_pt_feature_paded, reply_emb_paded, reply_bow_feature_padded], -1)
				else:
					review_reps_paded = torch.cat([review_pt_feature_paded, review_bow_feature_padded], -1)
					reply_reps_paded = torch.cat([reply_pt_feature_paded, reply_bow_feature_padded], -1)
			elif self.use_probing_topic and not self.use_ntm:
				review_pt_feature_paded = pad_sequence(review_pt_feature_list, batch_first=True)
				reply_pt_feature_paded = pad_sequence(reply_pt_feature_list, batch_first=True)
				if self.use_sent_rep:
					review_emb_paded = pad_sequence(review_feature_list, batch_first=True)
					reply_emb_paded = pad_sequence(reply_feature_list, batch_first=True)
					
					review_reps_paded = torch.cat([review_pt_feature_paded, review_emb_paded], -1)
					reply_reps_paded = torch.cat([reply_pt_feature_paded, reply_emb_paded], -1)
				else:
					review_reps_paded = review_pt_feature_paded
					reply_reps_paded = reply_pt_feature_paded
			elif not self.use_probing_topic and self.use_ntm:
				review_bow_feature_padded = pad_sequence(review_bow_feature, batch_first=True)
				reply_bow_feature_padded = pad_sequence(reply_bow_feature, batch_first=True)
				if self.use_sent_rep:
					review_emb_paded = pad_sequence(review_feature_list, batch_first=True)
					reply_emb_paded = pad_sequence(reply_feature_list, batch_first=True)
					
					review_reps_paded = torch.cat([review_emb_paded, review_bow_feature_padded], -1)
					reply_reps_paded = torch.cat([reply_emb_paded, reply_bow_feature_padded], -1)
				else:
					review_reps_paded = review_bow_feature_padded
					reply_reps_paded = reply_bow_feature_padded
			else:
				review_emb_paded = pad_sequence(review_feature_list, batch_first=True)
				reply_emb_paded = pad_sequence(reply_feature_list, batch_first=True)
				assert review_masks.size() == review_emb_paded.size()[:2]
				
				review_reps_paded = self.dropout(review_emb_paded)
				reply_reps_paded = self.dropout(reply_emb_paded)
			
			batch_size, max_review_sents_num, _ = review_reps_paded.size()
			_, max_reply_sents_num, _ = reply_reps_paded.size()
			
			# use_pos_emb in here is bad
			
			if self.hidden_size != self.tg_hidden_size:
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

			# if self.use_pos_emb:
			# 	review_pos_emb = self.rev_pos_emb(torch.arange(0, max_review_sents_num, dtype=torch.long).to(self.config.device))
			# 	review_reps_paded = review_reps_paded + review_pos_emb.view(1, max_review_sents_num, -1)
			# 	reply_pos_emb = self.rep_pos_emb(torch.arange(0, max_reply_sents_num, dtype=torch.long).to(self.config.device))
			# 	reply_reps_paded = reply_reps_paded + reply_pos_emb
			# 	review_reps_paded = review_reps_paded * review_masks.unsqueeze(-1)
			# 	reply_reps_paded = reply_reps_paded * reply_masks.unsqueeze(-1)
			
			layer_reps = []
			
			cross_attn_sum = torch.zeros((batch_size, max_review_sents_num, max_reply_sents_num)).to(self.config.device)
			review_attn_sum = torch.zeros((batch_size, max_review_sents_num, max_review_sents_num)).to(self.config.device)
			reply_attn_sum = torch.zeros((batch_size, max_reply_sents_num, max_reply_sents_num)).to(self.config.device)
			review_gate_attn_sum = torch.zeros((batch_size, max_review_sents_num)).to(self.config.device)
			reply_gate_attn_sum = torch.zeros((batch_size, max_reply_sents_num)).to(self.config.device)
			for i in range(self.tg_layer_num):
				review_reps_paded, reply_reps_paded, \
				review_att, reply_att, \
				review_cross_attention, reply_cross_attention, \
				review_gated_atts, reply_gated_atts = self.tg_layers[i](review_reps_paded, reply_reps_paded,
				                                                                  review_masks, reply_masks)
				review_attn_sum = self.config.ema * review_attn_sum + review_att
				reply_attn_sum = self.config.ema * reply_attn_sum + reply_att
				cross_attn_sum = self.config.ema * cross_attn_sum + \
				                 review_cross_attention + reply_cross_attention.permute(0, 2, 1)
				review_gate_attn_sum = self.config.ema * review_gate_attn_sum + review_gated_atts
				reply_gate_attn_sum = self.config.ema * reply_gate_attn_sum + reply_gated_atts
				
				# review_reps_paded, reply_reps_paded, \
				# review_att, reply_att = self.tg_layers[i](review_reps_paded, reply_reps_paded,
				#                                                         review_masks, reply_masks)
				
				
				
				layer_reps.append([review_reps_paded, reply_reps_paded])
			
			# review
			rev_crf_loss, _, rev_lstm_out_padded = self.am_tagging(review_reps_paded, review_lengths, review_ibose_list,
			                                                       mode='train')
			# reply
			rep_crf_loss, _, rep_lstm_out_padded = self.am_tagging(reply_reps_paded, reply_lengths, reply_ibose_list,
			                                                       mode='train')
			
			rev_mix_reps_paded = review_reps_paded + rev_lstm_out_padded # hope to capture the graph and sequence information
			rep_mix_reps_paded = reply_reps_paded + rep_lstm_out_padded # need to improve
			
			# rev_mix_reps_paded = rev_lstm_out_padded  # hope to capture the graph and sequence information
			# rep_mix_reps_paded = rep_lstm_out_padded  # need to improve
			
			# rev_mix_reps_paded = review_reps_paded  # hope to capture the graph and sequence information
			# rep_mix_reps_paded = reply_reps_paded  # need to improve
			
			# rev_mix_reps_paded = self.align_linear(torch.cat([review_reps_paded, rev_lstm_out_padded,
			#                                                   rev_lstm_out_padded - review_reps_paded,
			#                                                   review_reps_paded * rev_lstm_out_padded], -1)) # hope to capture the graph and sequence information
			# rep_mix_reps_paded = self.align_linear(torch.cat([reply_reps_paded, rep_lstm_out_padded,
			#                                                   rep_lstm_out_padded - reply_reps_paded,
			#                                                   rep_lstm_out_padded * reply_reps_paded], -1)) # need to improve
			
			# review_mask_inf = ((1.0 - review_masks) * -10000.0).view(-1, 1, review_masks.size(-1))
			# reply_mask_inf = ((1.0 - reply_masks) * -10000.0).view(-1, 1, reply_masks.size(-1))
			# rev_dual_reps_paded, rep_dual_reps_paded, \
			# (review_dual_attention, reply_dual_attention) = self.dual_graph(rev_mix_reps_paded,
			#                                                                 review_mask_inf,
			#                                                                 rep_mix_reps_paded,
			#                                                                 reply_mask_inf)
			
			# rev_dual_reps_paded, rep_dual_reps_paded, \
			# review_dual_att, reply_dual_att, \
			# review_dual_cross_attention, reply_dual_cross_attention, \
			# review_dual_gated_atts, reply_dual_gated_atts = self.dual_graph(rev_mix_reps_paded,
			#                                                                 rep_mix_reps_paded,
			#                                                                 review_masks,
			#                                                                 reply_masks)
			#
			# rev_mix_reps_paded = rev_dual_reps_paded + rev_mix_reps_paded
			# rep_mix_reps_paded = rep_dual_reps_paded + rep_mix_reps_paded
			
			
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
					# rep_with_rev_rep = torch.cat([rep_mix_reps_list[batch_i], args_rep], dim=-1) # [rep_sent_num, dim]
					# rep_with_rev_rep = rep_mix_reps_list[batch_i] * args_rep
					# rep_with_rev_rep = rep_mix_reps_list[batch_i] + args_rep
					if self.use_pos_emb:
						rev_arg_span = rev_args_span_list[batch_i][arg_idx]
						avg_pos = (rev_arg_span[0]+1 + rev_arg_span[1]+1) / 2 # or
						if review_lengths[batch_i] >= reply_lengths[batch_i]:
							proportion = review_lengths[batch_i] / reply_lengths[batch_i]
							temp_pos = np.arange(1, reply_lengths[batch_i]+1) * proportion
							rel_pos = (temp_pos - avg_pos) // proportion
						else:
							proportion = reply_lengths[batch_i] / review_lengths[batch_i]
							temp_pos = np.arange(1, reply_lengths[batch_i] + 1) // proportion
							rel_pos = (temp_pos - avg_pos) // 1

						rel_pos = torch.LongTensor(rel_pos).to(self.config.device)
						kernel_mutual = self.kernel_function(rel_pos)
						rel_pos = rel_pos + self.K # K is the max related position
						rel_pos_emb = self.rel_pos_emb(rel_pos)
						rel_pos_emb = torch.matmul(kernel_mutual, rel_pos_emb)
						rep_with_rev_rep = torch.cat([rep_mix_reps_list[batch_i], args_rep, rel_pos_emb], dim=-1) # [rep_sent_num, dim]
					else:
						rep_with_rev_rep = torch.cat([rep_mix_reps_list[batch_i], args_rep], dim=-1) # [rep_sent_num, dim]

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
					# rev_with_rep_rep = torch.cat([rev_mix_reps_list[batch_i], args_rep], dim=-1) # [rev_sent_num, 2*dim]
					# rev_with_rep_rep = rev_mix_reps_list[batch_i] * args_rep # [rev_sent_num, dim]
					# rev_with_rep_rep = rev_mix_reps_list[batch_i] + args_rep

					if self.use_pos_emb:
						rep_args_span = rep_args_span_list[batch_i][arg_idx]
						avg_pos = (rep_args_span[0] + 1 + rep_args_span[1] + 1) / 2
						if reply_lengths[batch_i] >= review_lengths[batch_i]:
							proportion = reply_lengths[batch_i] / review_lengths[batch_i]
							temp_pos = np.arange(1, review_lengths[batch_i] + 1) * proportion
							rel_pos = (temp_pos - avg_pos) // proportion
						else:
							proportion = review_lengths[batch_i] / reply_lengths[batch_i]
							temp_pos = np.arange(1, review_lengths[batch_i] + 1) // proportion
							rel_pos = (temp_pos - avg_pos) // 1

						rel_pos = torch.LongTensor(rel_pos).to(self.config.device)
						kernel_mutual = self.kernel_function(rel_pos)
						rel_pos = rel_pos + self.K  # K is the max related position
						rel_pos_emb = self.rel_pos_emb(rel_pos)
						rel_pos_emb = torch.matmul(kernel_mutual, rel_pos_emb)
						rev_with_rep_rep = torch.cat([rev_mix_reps_list[batch_i], args_rep, rel_pos_emb], dim=-1) # [rev_sent_num, 2*dim]
					else:
						rev_with_rep_rep = torch.cat([rev_mix_reps_list[batch_i], args_rep],
													 dim=-1)  # [rev_sent_num, 2*dim]

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
			
			
		elif self.graph_type == "dual":
		
			pass
		

		if self.use_ntm:
			return rev_crf_loss + rep_crf_loss, pair_loss + pair_2_loss, \
			       (review_recon, review_mu, review_logvar), (reply_recon, reply_mu, reply_logvar),\
			       review_attn_sum, reply_attn_sum, cross_attn_sum, review_gate_attn_sum, reply_gate_attn_sum
		else:
			return rev_crf_loss + rep_crf_loss, pair_loss + pair_2_loss, None, None, \
			       review_attn_sum, reply_attn_sum, cross_attn_sum, review_gate_attn_sum, reply_gate_attn_sum

	def kernel_function(self, rel_pos):
		n_couple = rel_pos.size(0)
		kernel_left = torch.cat([rel_pos.reshape(-1, 1)] * n_couple, dim=1)
		kernel = kernel_left - kernel_left.transpose(0, 1)
		return torch.exp(-(torch.pow(kernel, 2))/self.gamma) # the larger the more relevant

	def predict(self, review_pt_feature_list, review_feature_list, # list batch_size  [(sent_num, hidden_size), ...]
	            reply_pt_feature_list, reply_feature_list,
	            review_bow_vectors, reply_bow_vectors, # [all_reviews_num_sents, bow_vocab_size]
	            review_masks, reply_masks,
	            review_lengths, reply_lengths,
	            review_ibose_list=None, reply_ibose_list=None,):
		
		batch_size = len(review_feature_list)
		
		# NTM
		if self.use_ntm:
			review_bow_norm_vectors = F.normalize(review_bow_vectors)
			reply_bow_norm_vectors = F.normalize(reply_bow_vectors)
			_, review_bow_feature, review_recon, review_mu, review_logvar = self.ntm(review_bow_norm_vectors)
			_, reply_bow_feature, reply_recon, reply_mu, reply_logvar = self.ntm(reply_bow_norm_vectors)
			if self.use_topic_project:
				review_bow_feature = self.topic_project(review_bow_feature)
				reply_bow_feature = self.topic_project(reply_bow_feature)
			review_bow_feature = torch.split(review_bow_feature, review_lengths, 0)
			reply_bow_feature = torch.split(reply_bow_feature, reply_lengths, 0)

		if self.graph_type == "single":
			if self.use_probing_topic and self.use_ntm:
				review_pt_feature_paded = pad_sequence(review_pt_feature_list, batch_first=True)
				reply_pt_feature_paded = pad_sequence(reply_pt_feature_list, batch_first=True)
				review_bow_feature_padded = pad_sequence(review_bow_feature, batch_first=True)
				reply_bow_feature_padded = pad_sequence(reply_bow_feature, batch_first=True)
				
				if self.use_sent_rep:
					review_emb_paded = pad_sequence(review_feature_list, batch_first=True)
					reply_emb_paded = pad_sequence(reply_feature_list, batch_first=True)
					
					review_reps_paded = torch.cat(
						[review_pt_feature_paded, review_emb_paded, review_bow_feature_padded], -1)
					reply_reps_paded = torch.cat([reply_pt_feature_paded, reply_emb_paded, reply_bow_feature_padded],
					                             -1)
				else:
					review_reps_paded = torch.cat([review_pt_feature_paded, review_bow_feature_padded], -1)
					reply_reps_paded = torch.cat([reply_pt_feature_paded, reply_bow_feature_padded], -1)
			elif self.use_probing_topic and not self.use_ntm:
				review_pt_feature_paded = pad_sequence(review_pt_feature_list, batch_first=True)
				reply_pt_feature_paded = pad_sequence(reply_pt_feature_list, batch_first=True)
				if self.use_sent_rep:
					review_emb_paded = pad_sequence(review_feature_list, batch_first=True)
					reply_emb_paded = pad_sequence(reply_feature_list, batch_first=True)
					
					review_reps_paded = torch.cat([review_pt_feature_paded, review_emb_paded], -1)
					reply_reps_paded = torch.cat([reply_pt_feature_paded, reply_emb_paded], -1)
				else:
					review_reps_paded = review_pt_feature_paded
					reply_reps_paded = reply_pt_feature_paded
			elif not self.use_probing_topic and self.use_ntm:
				review_bow_feature_padded = pad_sequence(review_bow_feature, batch_first=True)
				reply_bow_feature_padded = pad_sequence(reply_bow_feature, batch_first=True)
				if self.use_sent_rep:
					review_emb_paded = pad_sequence(review_feature_list, batch_first=True)
					reply_emb_paded = pad_sequence(reply_feature_list, batch_first=True)
					
					review_reps_paded = torch.cat([review_emb_paded, review_bow_feature_padded], -1)
					reply_reps_paded = torch.cat([reply_emb_paded, reply_bow_feature_padded], -1)
				else:
					review_reps_paded = review_bow_feature_padded
					reply_reps_paded = reply_bow_feature_padded
			else:
				review_emb_paded = pad_sequence(review_feature_list, batch_first=True)
				reply_emb_paded = pad_sequence(reply_feature_list, batch_first=True)
				assert review_masks.size() == review_emb_paded.size()[:2]
				
				review_reps_paded = self.dropout(review_emb_paded)
				reply_reps_paded = self.dropout(reply_emb_paded)
			
			batch_size, max_review_sents_num, _ = review_reps_paded.size()
			_, max_reply_sents_num, _ = reply_reps_paded.size()
				
			if self.hidden_size != self.tg_hidden_size:
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
			
			# if self.use_pos_emb:
			# 	review_pos_emb = self.rev_pos_emb(
			# 		torch.arange(0, max_review_sents_num, dtype=torch.long).to(self.config.device))
			# 	review_reps_paded = review_reps_paded + review_pos_emb.view(1, max_review_sents_num, -1)
			# 	reply_pos_emb = self.rep_pos_emb(
			# 		torch.arange(0, max_reply_sents_num, dtype=torch.long).to(self.config.device))
			# 	reply_reps_paded = reply_reps_paded + reply_pos_emb
			# 	review_reps_paded = review_reps_paded * review_masks.unsqueeze(-1)
			# 	reply_reps_paded = reply_reps_paded * reply_masks.unsqueeze(-1)
			
			layer_reps = []
			
			cross_attn_sum = torch.zeros((batch_size, max_review_sents_num, max_reply_sents_num)).to(self.config.device)
			review_attn_sum = torch.zeros((batch_size, max_review_sents_num, max_review_sents_num)).to(
				self.config.device)
			reply_attn_sum = torch.zeros((batch_size, max_reply_sents_num, max_reply_sents_num)).to(self.config.device)
			review_gate_attn_sum = torch.zeros((batch_size, max_review_sents_num)).to(self.config.device)
			reply_gate_attn_sum = torch.zeros((batch_size, max_reply_sents_num)).to(self.config.device)
			for i in range(self.tg_layer_num):
				review_reps_paded, reply_reps_paded, \
				review_att, reply_att, \
				review_cross_attention, reply_cross_attention, \
				review_gated_atts, reply_gated_atts = self.tg_layers[i](review_reps_paded, reply_reps_paded,
				                                                        review_masks, reply_masks)
				review_attn_sum = self.config.ema * review_attn_sum + review_att
				reply_attn_sum = self.config.ema * reply_attn_sum + reply_att
				cross_attn_sum = self.config.ema * cross_attn_sum + \
				                 review_cross_attention + reply_cross_attention.permute(0, 2, 1)
				review_gate_attn_sum = self.config.ema * review_gate_attn_sum + review_gated_atts
				reply_gate_attn_sum = self.config.ema * reply_gate_attn_sum + reply_gated_atts
				
				# review_reps_paded, reply_reps_paded, \
				# review_att, reply_att = self.tg_layers[i](review_reps_paded, reply_reps_paded,
				#                                           review_masks, reply_masks)
				
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
			
			# rev_mix_reps_paded = rev_lstm_out_padded  # hope to capture the graph and sequence information
			# rep_mix_reps_paded = rep_lstm_out_padded  # need to improve
			
			# rev_mix_reps_paded = review_reps_paded  # hope to capture the graph and sequence information
			# rep_mix_reps_paded = reply_reps_paded  # need to improve
			
			# rev_mix_reps_paded = self.align_linear(torch.cat([review_reps_paded, rev_lstm_out_padded,
			#                                                   rev_lstm_out_padded - review_reps_paded,
			#                                                   review_reps_paded * rev_lstm_out_padded],
			#                                                  -1))  # hope to capture the graph and sequence information
			# rep_mix_reps_paded = self.align_linear(torch.cat([reply_reps_paded, rep_lstm_out_padded,
			#                                                   rep_lstm_out_padded - reply_reps_paded,
			#                                                   rep_lstm_out_padded * reply_reps_paded],
			#                                                  -1))  # need to improve
			
			# review_mask_inf = ((1.0 - review_masks) * -10000.0).view(-1, 1, review_masks.size(-1))
			# reply_mask_inf = ((1.0 - reply_masks) * -10000.0).view(-1, 1, reply_masks.size(-1))
			# rev_dual_reps_paded, rep_dual_reps_paded, \
			# (review_dual_attention, reply_dual_attention) = self.dual_graph(rev_mix_reps_paded,
			#                                                                 review_mask_inf,
			#                                                                 rep_mix_reps_paded,
			#                                                                 reply_mask_inf)
			# rev_dual_reps_paded, rep_dual_reps_paded, \
			# review_dual_att, reply_dual_att, \
			# review_dual_cross_attention, reply_dual_cross_attention, \
			# review_dual_gated_atts, reply_dual_gated_atts = self.dual_graph(rev_mix_reps_paded,
			#                                                                 rep_mix_reps_paded,
			#                                                                 review_masks,
			#                                                                 reply_masks)
			#
			# rev_mix_reps_paded = rev_dual_reps_paded + rev_mix_reps_paded
			# rep_mix_reps_paded = rep_dual_reps_paded + rep_mix_reps_paded
			
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
						# rev_with_rep_rep = torch.cat([rep_mix_reps_list[batch_i], args_rep], dim=-1)
						# rev_with_rep_rep = rep_mix_reps_list[batch_i] * args_rep
						# rev_with_rep_rep = rep_mix_reps_list[batch_i] + args_rep
						if self.use_pos_emb:
							rev_arg_span = pred_rev_args_list[batch_i][arg_idx]
							avg_pos = (rev_arg_span[0] + 1 + rev_arg_span[1] + 1) / 2
							if review_lengths[batch_i] >= reply_lengths[batch_i]:
								proportion = review_lengths[batch_i] / reply_lengths[batch_i]
								temp_pos = np.arange(1, reply_lengths[batch_i] + 1) * proportion
								rel_pos = (temp_pos - avg_pos) // proportion
							else:
								proportion = reply_lengths[batch_i] / review_lengths[batch_i]
								temp_pos = np.arange(1, reply_lengths[batch_i] + 1) // proportion
								rel_pos = (temp_pos - avg_pos) // 1

							rel_pos = torch.LongTensor(rel_pos).to(self.config.device)
							kernel_mutual = self.kernel_function(rel_pos)
							rel_pos = rel_pos + self.K  # K is the max related position
							rel_pos_emb = self.rel_pos_emb(rel_pos)
							rel_pos_emb = torch.matmul(kernel_mutual, rel_pos_emb)
							rev_with_rep_rep = torch.cat([rep_mix_reps_list[batch_i], args_rep, rel_pos_emb],
														 dim=-1)  # [rep_sent_num, dim]
						else:
							rev_with_rep_rep = torch.cat([rep_mix_reps_list[batch_i], args_rep], dim=-1)
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
						# rep_with_rev_rev = torch.cat([rev_mix_reps_list[batch_i], args_rep], dim=-1)
						# rep_with_rev_rev = rev_mix_reps_list[batch_i] * args_rep
						# rep_with_rev_rev = rev_mix_reps_list[batch_i] + args_rep
						if self.use_pos_emb:
							rep_args_span = pred_rep_args_list[batch_i][arg_idx]
							avg_pos = (rep_args_span[0] + 1 + rep_args_span[1] + 1) / 2 # or interate and avg
							if reply_lengths[batch_i] >= review_lengths[batch_i]:
								proportion = reply_lengths[batch_i] / review_lengths[batch_i]
								temp_pos = np.arange(1, review_lengths[batch_i] + 1) * proportion
								rel_pos = (temp_pos - avg_pos) // proportion
							else:
								proportion = review_lengths[batch_i] / reply_lengths[batch_i]
								temp_pos = np.arange(1, review_lengths[batch_i] + 1) // proportion
								rel_pos = (temp_pos - avg_pos) // 1

							rel_pos = torch.LongTensor(rel_pos).to(self.config.device)
							kernel_mutual = self.kernel_function(rel_pos)
							rel_pos = rel_pos + self.K  # K is the max related position
							rel_pos_emb = self.rel_pos_emb(rel_pos)
							rel_pos_emb = torch.matmul(kernel_mutual, rel_pos_emb)
							rep_with_rev_rev = torch.cat([rev_mix_reps_list[batch_i], args_rep, rel_pos_emb],
														 dim=-1)  # [rev_sent_num, 2*dim]
						else:
							rep_with_rev_rev = torch.cat([rev_mix_reps_list[batch_i], args_rep], dim=-1)

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
					
		elif self.graph_type == 'dual':
			pass
		
		
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
	
	def forward(self, review_embedder_input, reply_embedder_input, review_sorted_iw_spans, reply_sorted_iw_spans):
		pass