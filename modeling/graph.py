import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from transformers import BertModel, BertTokenizer
from copy import deepcopy
import numpy as np
import math
from utils.basic_utils import get_index_positions, flat_list_of_lists
import scipy.sparse as sp
from scipy.sparse import coo_matrix, csr_matrix
import torch.distributions as dist


class GraphConvolution(nn.Module):
	def __init__(self, in_features_dim, out_features_dim, activation=None, bias=True):
		super(GraphConvolution, self).__init__()
		self.in_features = in_features_dim
		self.out_features = out_features_dim
		self.activation = activation
		self.weight = nn.Parameter(torch.FloatTensor(in_features_dim, out_features_dim))
		if bias:
			self.bias = nn.Parameter(torch.FloatTensor(out_features_dim))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()
	
	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		# nn.init.xavier_uniform_(self.weight)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)
		# nn.init.zeros_(self.bias)
	
	def forward(self, infeatn, adj):
		'''
		infeatn: init feature(H，上一层的feature)
		adj: A
		'''
		support = torch.matmul(infeatn, self.weight)  # H*W  # (in_feat_dim, in_feat_dim) * (in_feat_dim, out_dim)
		output = torch.matmul(adj, support)  # A*H*W  # (in_feat_dim, in_feat_dim) * (in_feat_dim, out_dim)
		if self.bias is not None:
			output = output + self.bias
		
		if self.activation is not None:
			output = self.activation(output)
		
		return output
	

class Intra_Inter_GCN(nn.Module):
	def __init__(self, hidden_size, activation=None, residual=False):
		super(Intra_Inter_GCN, self).__init__()
		self.residual = residual
		self.rev_intra_gcn = GraphConvolution(hidden_size, hidden_size)
		self.rev2rep_inter_gcn = GraphConvolution(hidden_size, hidden_size)
		self.rep_intra_gcn = GraphConvolution(hidden_size, hidden_size)
		self.rep2rev_inter_gcn = GraphConvolution(hidden_size, hidden_size)
	
	def forward(self, review_input, reply_input, review_adj, reply_adj, rev2rep_adj, rep2rev_adj):
		review_gcn_rep = F.relu(self.rev_intra_gcn(review_input, review_adj)) + F.relu(self.rev2rep_inter_gcn(reply_input, rev2rep_adj))
		reply_gcn_rep = F.relu(self.rep_intra_gcn(reply_input, reply_adj)) + F.relu(self.rep2rev_inter_gcn(review_input, rep2rev_adj))

		if self.residual:
			review_gcn_rep = review_input + review_gcn_rep
			reply_gcn_rep = reply_input + reply_gcn_rep

		return review_gcn_rep, reply_gcn_rep


class Intra_Inter_GCN4(nn.Module):
	def __init__(self, hidden_size, activation=None, residual=False):
		super(Intra_Inter_GCN4, self).__init__()
		self.residual = residual
		self.rev_intra_gcn = GraphConvolution(hidden_size, hidden_size)
		self.rev2rep_inter_gcn = GraphConvolution(hidden_size, hidden_size)
		self.rep_intra_gcn = GraphConvolution(hidden_size, hidden_size)
		self.rep2rev_inter_gcn = GraphConvolution(hidden_size, hidden_size)

	def forward(self, review_input, reply_input, review_adj, reply_adj, rev2rep_adj, rep2rev_adj):
		review_gcn_rep1 = F.relu(self.rev_intra_gcn(review_input, review_adj))
		review_gcn_rep2 = F.relu(self.rev2rep_inter_gcn(reply_input, rev2rep_adj))
		review_gcn_rep = review_gcn_rep1 + review_gcn_rep2

		reply_gcn_rep1 = F.relu(self.rep_intra_gcn(reply_input, reply_adj))
		reply_gcn_rep2 = F.relu(self.rep2rev_inter_gcn(review_input, rep2rev_adj))
		reply_gcn_rep = reply_gcn_rep1 + reply_gcn_rep2

		if self.residual:
			review_gcn_rep = review_input + review_gcn_rep
			reply_gcn_rep = reply_input + reply_gcn_rep

		return review_gcn_rep, reply_gcn_rep, review_gcn_rep1, review_gcn_rep2, reply_gcn_rep1, reply_gcn_rep2


class Intra_Inter_GCN2(nn.Module):
	def __init__(self, config, hidden_size, residual=True):
		super(Intra_Inter_GCN2, self).__init__()
		self.residual = residual
		self.rev_intra_gat = GAT(config, hidden_size, hidden_size, residual=False)
		self.rep_intra_gat = GAT(config, hidden_size, hidden_size, residual=False)

		self.rev2rep_inter_gcn = GraphConvolution(hidden_size, hidden_size)
		self.rep2rev_inter_gcn = GraphConvolution(hidden_size, hidden_size)

	def forward(self, review_input, reply_input, review_mask, reply_mask, rev2rep_adj, rep2rev_adj):
		'''
		:param review_input: [batch_size, sent_num, hidden_size]
		:param reply_input:
		:param review_mask: [batch_size, sent_num]
		:param reply_mask:
		:return:
		'''

		batch_size, max_review_sent_num, _ = review_input.size()
		_, max_reply_sent_num, _ = reply_input.size()

		review_adj = review_mask.view(batch_size, 1, max_review_sent_num).repeat(1, max_review_sent_num, 1)
		review_reps, review_att = self.rev_intra_gat(review_input, review_adj)
		reply_adj = reply_mask.view(batch_size, 1, max_reply_sent_num).repeat(1, max_reply_sent_num, 1)
		reply_reps, reply_att = self.rep_intra_gat(reply_input, reply_adj)

		review_reps = F.relu(review_reps) + F.relu(self.rev2rep_inter_gcn(reply_input, rev2rep_adj))
		reply_reps = F.relu(reply_reps) + F.relu(self.rep2rev_inter_gcn(review_input, rep2rev_adj))

		if self.residual:
			review_reps = review_input + review_reps
			reply_reps = reply_input + reply_reps

		return review_reps, reply_reps


class Intra_Inter_GCN3(nn.Module):
	def __init__(self, config, hidden_size, residual=True):
		super(Intra_Inter_GCN3, self).__init__()
		self.residual = residual
		self.rev_intra_gat = GAT_pos(config, hidden_size, hidden_size, residual=False)
		self.rep_intra_gat = GAT_pos(config, hidden_size, hidden_size, residual=False)

		self.rev2rep_inter_gcn = GraphConvolution(hidden_size, hidden_size)
		self.rep2rev_inter_gcn = GraphConvolution(hidden_size, hidden_size)

	def forward(self, review_input, reply_input, review_mask, reply_mask, rev2rep_adj, rep2rev_adj):
		'''
		:param review_input: [batch_size, sent_num, hidden_size]
		:param reply_input:
		:param review_mask: [batch_size, sent_num]
		:param reply_mask:
		:return:
		'''

		batch_size, max_review_sent_num, _ = review_input.size()
		_, max_reply_sent_num, _ = reply_input.size()

		review_adj = review_mask.view(batch_size, 1, max_review_sent_num).repeat(1, max_review_sent_num, 1)
		review_reps, review_att = self.rev_intra_gat(review_input, review_adj)
		reply_adj = reply_mask.view(batch_size, 1, max_reply_sent_num).repeat(1, max_reply_sent_num, 1)
		reply_reps, reply_att = self.rep_intra_gat(reply_input, reply_adj)

		review_reps = F.relu(review_reps) + F.relu(self.rev2rep_inter_gcn(reply_input, rev2rep_adj))
		reply_reps = F.relu(reply_reps) + F.relu(self.rep2rev_inter_gcn(review_input, rep2rev_adj))

		if self.residual:
			review_reps = review_input + review_reps
			reply_reps = reply_input + reply_reps

		return review_reps, reply_reps


class GCN(nn.Module):
	def __init__(self, nfeat, nhid, n_layers, activation="", dropout=0.1, nclass=0):
		
		super(GCN, self).__init__()
		
		if activation == 'relu':
			self.activation = nn.ReLU()
		elif activation == 'leaky_relu':
			self.activation = nn.LeakyReLU()
		
		self.layers = nn.ModuleList()
		# input layer
		self.layers.append(GraphConvolution(nfeat, nhid, activation=self.activation))
		# hidden layers
		for i in range(n_layers - 1):
			self.layers.append(GraphConvolution(nhid, nhid, activation=self.activation))
		# output layer
		# self.layers.append(GraphConvolution(nhid, nclass))
		
		self.dropout = nn.Dropout(p=dropout)
	
	def forward(self, x, adj):
		h = x
		for i, layer in enumerate(self.layers):
			# if i != 0:
			#     h = self.dropout(h)
			h = layer(h, adj)
		return h


class GraphAttentionHead(nn.Module):
	"""
	Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
	"""
	def __init__(self, in_features, out_features, dropout, alpha, concat=True):
		super(GraphAttentionHead, self).__init__()
		self.dropout = dropout
		self.in_features = in_features
		self.out_features = out_features
		self.alpha = alpha
		self.concat = concat
		
		self.W = nn.Parameter(torch.empty(size=(in_features, out_features)).cuda())
		nn.init.xavier_uniform_(self.W.data, gain=1.414)
		self.a = nn.Parameter(torch.empty(size=( 2 *out_features, 1)).cuda())
		nn.init.xavier_uniform_(self.a.data, gain=1.414)
		
		self.leakyrelu = nn.LeakyReLU(self.alpha)
	
	def forward(self, h, adj):
		Wh = torch.matmul(h, self.W) # h.shape: (b, N, in_features), Wh.shape: (b, N, out_features)
		Wh = F.dropout(Wh, self.dropout, training=self.training)
		e = self._prepare_attentional_mechanism_input(Wh)
		
		zero_vec = -9e15 * torch.ones_like(e).cuda()
		attention = torch.where(adj > 0, e, zero_vec)
		attention = F.softmax(attention, dim=-1)
		attention = F.dropout(attention, self.dropout, training=self.training)
		h_prime = torch.matmul(attention, Wh)
		# print(self.a.data)
		if self.concat:
			return F.elu(h_prime)
		else:
			return h_prime
	
	def _prepare_attentional_mechanism_input(self, Wh):
		# Wh.shape (b, N, out_feature)
		# self.a.shape (2 * out_feature, 1)
		# Wh1&2.shape (b, N, 1)
		# e.shape (b, N, N)
		Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
		Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
		# broadcast add
		e = Wh1 + Wh2.transpose(2, 1)
		return self.leakyrelu(e)
	
	def __repr__(self):
		return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GATLayer(nn.Module):
	def __init__(self, nfeat, nhid, dropout, nheads, alpha=0.2, concat=True):
		"""Dense version of GAT."""
		super(GATLayer, self).__init__()
		
		self.concat = concat
		self.attentions = [GraphAttentionHead(nfeat, nhid, dropout=dropout, alpha=alpha, concat=concat) for _ in
		                   range(nheads)]
		for i, attention in enumerate(self.attentions):
			self.add_module('attention_{}'.format(i), attention)
	
	# self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
	
	def forward(self, x, adj):
		out = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
		return out


class MultiLayerGAT(nn.Module):
	def __init__(self, nfeat, nhid, nlayers, nheads, dropout=0.6, residual=True):
		"""Dense version of GAT."""
		super(MultiLayerGAT, self).__init__()
		
		assert nhid % nheads == 0
		self.layers = nn.ModuleList()
		# input layer
		self.layers.append(GATLayer(nfeat, nhid // nheads, dropout, nheads))
		# hidden layers
		for i in range(nlayers - 1):
			self.layers.append(GATLayer(nhid, nhid // nheads, dropout, nheads))
		
		self.dropout = nn.Dropout(p=dropout)
		self.residual = residual
	# self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
	
	def forward(self, h, adj):
		# h = self.dropout(x)
		for i, layer in enumerate(self.layers):
			hi = layer(h, adj)
			if self.residual:
				hi = h + hi
			h = self.dropout(hi)
		return h


class GAT(nn.Module):
	def __init__(self, config, in_features, out_features, dropout=0, alpha=0.2, concat=False, residual=True):
		super(GAT, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.alpha = alpha
		self.concat = concat
		self.residual = residual
		
		self.dropout = nn.Dropout(p=dropout)
		
		self.W = nn.Parameter(torch.empty(size=(self.in_features, self.out_features)).to(config.device))
		nn.init.xavier_uniform_(self.W.data, gain=1.414)
		self.a = nn.Parameter(torch.empty(size=(2 * self.out_features, 1)).to(config.device))
		nn.init.xavier_uniform_(self.a.data, gain=1.414)
		
		self.leakyrelu = nn.LeakyReLU(self.alpha)
	
	def forward(self, h, adj):
		# h = self.dropout(x)
		# print(h.size(), self.W.data.size())
		Wh = torch.matmul(h, self.W)  # h.shape: (b, N, in_features), Wh.shape: (b, N, out_features)
		Wh = self.dropout(Wh)
		
		e = self._prepare_attentional_mechanism_input(Wh)
		
		zero_vec = -9e15 * torch.ones_like(e).cuda()
		attention = torch.where(adj > 0, e, zero_vec)
		attention = F.softmax(attention, dim=-1)
		h_prime = torch.matmul(self.dropout(attention), Wh)
		
		if self.concat:
			out = F.elu(h_prime)
		else:
			out = h_prime
		
		if self.residual == True:
			out = h + out # remove the residual connection

		out = self.dropout(out)
		
		return out, attention
	
	def _prepare_attentional_mechanism_input(self, Wh):
		# Wh.shape (b, N, out_feature)
		# self.a.shape (2 * out_feature, 1)
		# Wh1&2.shape (b, N, 1)
		# e.shape (b, N, N)
		Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
		Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
		# broadcast add
		e = Wh1 + Wh2.transpose(2, 1)
		return self.leakyrelu(e)


class GAT_pos(nn.Module):
	def __init__(self, config, in_features, out_features, dropout=0, alpha=0.2, concat=False, residual=True):
		super(GAT_pos, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.alpha = alpha
		self.concat = concat
		self.residual = residual
		self.config = config

		self.dropout = nn.Dropout(p=dropout)

		self.W = nn.Parameter(torch.empty(size=(self.in_features, self.out_features)).to(config.device))
		nn.init.xavier_uniform_(self.W.data, gain=1.414)
		self.a = nn.Parameter(torch.empty(size=(2 * self.out_features, 1)).to(config.device))
		nn.init.xavier_uniform_(self.a.data, gain=1.414)

		self.leakyrelu = nn.LeakyReLU(self.alpha)

	def forward(self, h, adj):
		# h = self.dropout(x)
		# print(h.size(), self.W.data.size())
		Wh = torch.matmul(h, self.W)  # h.shape: (b, N, in_features), Wh.shape: (b, N, out_features)
		Wh = self.dropout(Wh)

		e = self._prepare_attentional_mechanism_input(Wh)
		pos_weights = self.relate_position_weights(adj.size(-1))
		e = e * pos_weights.unsqueeze(0).repeat(e.size(0), 1, 1)
		zero_vec = -9e15 * torch.ones_like(e).to(self.config.device)
		attention = torch.where(adj > 0, e, zero_vec)
		attention = F.softmax(attention, dim=-1)
		h_prime = torch.matmul(self.dropout(attention), Wh)

		if self.concat:
			out = F.elu(h_prime)
		else:
			out = h_prime

		if self.residual == True:
			out = h + out  # remove the residual connection

		out = self.dropout(out)

		return out, attention

	def _prepare_attentional_mechanism_input(self, Wh):
		# Wh.shape (b, N, out_feature)
		# self.a.shape (2 * out_feature, 1)
		# Wh1&2.shape (b, N, 1)
		# e.shape (b, N, N)
		Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
		Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
		# broadcast add
		e = Wh1 + Wh2.transpose(2, 1)
		return self.leakyrelu(e)

	def relate_position_weights(self, dimension):
		pw_matrix = []
		for i in range(1, dimension + 1):
			pw_matrix.append(list(range(i, 1, -1)) + list(range(1, dimension + 2 - i)))
		return 1 / torch.tensor(pw_matrix).to(self.config.device)

class DualCrossAttention_MH(nn.Module):
	def __init__(self, config, hidden_size):
		super(DualCrossAttention_MH, self).__init__()
		if hidden_size % config.bi_num_attention_heads != 0:
			raise ValueError(
				"The hidden size (%d) is not a multiple of the number of attention "
				"heads (%d)" % (hidden_size, config.bi_num_attention_heads)
			)
		
		self.hidden_size = hidden_size
		
		self.num_attention_heads = config.bi_num_attention_heads
		self.attention_head_size = int(
			self.hidden_size / config.bi_num_attention_heads
		)
		self.all_head_size = self.num_attention_heads * self.attention_head_size
		
		# self.scale = nn.Linear(1, self.num_attention_heads, bias=False)
		# self.scale_act_fn = ACT2FN['relu']
		
		self.query1 = nn.Linear(self.hidden_size, self.all_head_size)
		self.key1 = nn.Linear(self.hidden_size, self.all_head_size)
		self.value1 = nn.Linear(self.hidden_size, self.all_head_size)
		# self.logit1 = nn.Linear(config.hidden_size, self.num_attention_heads)
		
		self.dropout1 = nn.Dropout(config.dropout)
		
		self.query2 = nn.Linear(self.hidden_size, self.all_head_size)
		self.key2 = nn.Linear(self.hidden_size, self.all_head_size)
		self.value2 = nn.Linear(self.hidden_size, self.all_head_size)
		# self.logit2 = nn.Linear(config.hidden_size, self.num_attention_heads)
		self.dropout2 = nn.Dropout(config.dropout)
	
	# self.proj_1 = nn.Linear(self.hidden_size, self.hidden_size)
	# self.proj_2 = nn.Linear(self.hidden_size, self.hidden_size)
	
	def transpose_for_scores(self, x):
		new_x_shape = x.size()[:-1] + (
			self.num_attention_heads,
			self.attention_head_size,
		)
		x = x.view(*new_x_shape)
		return x.permute(0, 2, 1, 3)
	
	def forward(self, input_tensor1, attention_mask1, input_tensor2, attention_mask2, co_attention_mask=None,
	            use_co_attention_mask=False):
		# for input1.
		mixed_query_layer1 = self.query1(input_tensor1)
		mixed_key_layer1 = self.key1(input_tensor1)
		mixed_value_layer1 = self.value1(input_tensor1)
		# mixed_value_layer1 = input_tensor1
		
		query_layer1 = self.transpose_for_scores(mixed_query_layer1)
		key_layer1 = self.transpose_for_scores(mixed_key_layer1)
		value_layer1 = self.transpose_for_scores(mixed_value_layer1)
		# logit_layer1 = self.transpose_for_logits(mixed_logit_layer1)
		
		# for input:2
		mixed_query_layer2 = self.query2(input_tensor2)
		mixed_key_layer2 = self.key2(input_tensor2)
		mixed_value_layer2 = self.value2(input_tensor2)
		# mixed_value_layer2 = input_tensor2
		
		query_layer2 = self.transpose_for_scores(mixed_query_layer2)
		key_layer2 = self.transpose_for_scores(mixed_key_layer2)
		value_layer2 = self.transpose_for_scores(mixed_value_layer2)
		
		# Take the dot product between "query1" and "key2" to get the raw attention scores for value 2.
		attention_scores1 = torch.matmul(query_layer1, key_layer2.transpose(-1, -2))
		attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
		# Apply the attention mask is (precomputed for all layers in BertModel forward() function)
		
		# we can comment this line for single flow.
		attention_scores1 = attention_scores1 + attention_mask2
		# if use_co_attention_mask:
		#    attention_scores1 = attention_scores1 + co_attention_mask
		
		# Normalize the attention scores to probabilities.
		attention_probs1 = nn.Softmax(dim=-1)(attention_scores1)
		
		# This is actually dropping out entire tokens to attend to, which might
		# seem a bit unusual, but is taken from the original Transformer paper.
		
		context_layer1 = torch.matmul(self.dropout2(attention_probs1), value_layer2)
		context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
		new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size,)
		context_layer1 = context_layer1.view(*new_context_layer_shape1)
		
		# Take the dot product between "query2" and "key1" to get the raw attention scores for value 1.
		attention_scores2 = torch.matmul(query_layer2, key_layer1.transpose(-1, -2))
		attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)
		attention_scores2 = attention_scores2 + attention_mask1
		
		# Normalize the attention scores to probabilities.
		attention_probs2 = nn.Softmax(dim=-1)(attention_scores2)
		
		# This is actually dropping out entire tokens to attend to, which might
		# seem a bit unusual, but is taken from the original Transformer paper.
		
		context_layer2 = torch.matmul(self.dropout1(attention_probs2), value_layer1)
		context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()
		new_context_layer_shape2 = context_layer2.size()[:-2] + (self.all_head_size,)
		context_layer2 = context_layer2.view(*new_context_layer_shape2)
		
		# context_layer1 = self.proj_1(context_layer1)
		# context_layer2 = self.proj_2(context_layer2)
		
		return context_layer1, context_layer2, (attention_probs1, attention_probs2)


class DualCrossAttention_dot(nn.Module):
	def __init__(self, config, input_size, output_size):
		super(DualCrossAttention_dot, self).__init__()
		
		self.input_size = input_size
		self.output_size = output_size
		
		self.query1 = nn.Linear(self.input_size, self.output_size)
		self.key1 = nn.Linear(self.input_size, self.output_size)
		self.value1 = nn.Linear(self.input_size, self.output_size)
		# self.logit1 = nn.Linear(config.hidden_size, self.num_attention_heads)
		
		self.dropout = nn.Dropout(config.dropout)
		
		self.query2 = nn.Linear(self.input_size, self.output_size)
		self.key2 = nn.Linear(self.input_size, self.output_size)
		self.value2 = nn.Linear(self.input_size, self.output_size)
		# self.logit2 = nn.Linear(config.hidden_size, self.num_attention_heads)
	
	def forward(self, input_tensor1, attention_mask1, input_tensor2, attention_mask2):
		# for input1.
		query_layer1 = self.query1(input_tensor1)
		key_layer1 = self.key1(input_tensor1)
		value_layer1 = self.value1(input_tensor1)
		
		# for input2.
		query_layer2 = self.query2(input_tensor2)
		key_layer2 = self.key2(input_tensor2)
		value_layer2 = self.value2(input_tensor2)
		
		# Take the dot product between "query1" and "key2" to get the raw attention scores for value 2.
		attention_scores1 = torch.matmul(query_layer1, key_layer2.transpose(-1, -2))
		# attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
		# Apply the attention mask is (precomputed for all layers in BertModel forward() function)
		
		# we can comment this line for single flow.
		attention_scores1 = attention_scores1 + attention_mask2
		# if use_co_attention_mask:
		#    attention_scores1 = attention_scores1 + co_attention_mask
		
		# Normalize the attention scores to probabilities.
		attention_probs1 = F.softmax(attention_scores1, dim=-1)
		
		# This is actually dropping out entire tokens to attend to, which might
		# seem a bit unusual, but is taken from the original Transformer paper.
		
		context_layer1 = torch.matmul(self.dropout(attention_probs1), value_layer2)
		
		# Take the dot product between "query2" and "key1" to get the raw attention scores for value 1.
		attention_scores2 = torch.matmul(query_layer2, key_layer1.transpose(-1, -2))
		# attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)
		attention_scores2 = attention_scores2 + attention_mask1
		
		# Normalize the attention scores to probabilities.
		attention_probs2 = F.softmax(attention_scores2, dim=-1)
		
		# This is actually dropping out entire tokens to attend to, which might
		# seem a bit unusual, but is taken from the original Transformer paper.
		
		context_layer2 = torch.matmul(self.dropout(attention_probs2), value_layer1)
		
		return context_layer1, context_layer2, (attention_probs1, attention_probs2)


class CrossAttention_concat(nn.Module):
	def __init__(self, config, input_size, output_size, alpha=1e-2):
		super(CrossAttention_concat, self).__init__()
		
		self.input_size = input_size
		self.output_size = output_size
		
		self.W1 = nn.Parameter(torch.empty(size=(input_size, output_size)).to(config.device))
		nn.init.xavier_uniform_(self.W1.data, gain=1.414)
		self.a1 = nn.Parameter(torch.empty(size=(2 * output_size, 1)).to(config.device))
		nn.init.xavier_uniform_(self.a1.data, gain=1.414)
		
		self.dropout = nn.Dropout(p=config.dropout)
		
		self.alpha = alpha
		self.leakyrelu = nn.LeakyReLU(self.alpha)
	
	def _prepare_attentional_mechanism_input(self, Wh1, Wh2, a):
		# Wh.shape (b, N, out_feature)
		# self.a.shape (2 * out_feature, 1)
		# attention1&2.shape (b, N, 1)
		# e.shape (b, N, N)
		attention1 = torch.matmul(Wh1, a[:self.output_size, :])
		attention2 = torch.matmul(Wh2, a[self.output_size:, :])
		# broadcast add
		e = attention1 + attention2.transpose(2, 1)
		return self.leakyrelu(e)
	
	def forward(self, input_tensor1, input_tensor2, attention_mask2):
		# for input1.
		Wh11 = torch.matmul(input_tensor1, self.W1)  # h.shape: (b, N, in_features), Wh.shape: (b, N, out_features)
		Wh11 = self.dropout(Wh11)
		
		Wh12 = torch.matmul(input_tensor2, self.W1)  # h.shape: (b, N, in_features), Wh.shape: (b, N, out_features)
		Wh12 = self.dropout(Wh12)
		
		attention_scores1 = self._prepare_attentional_mechanism_input(Wh11, Wh12, self.a1)
		attention_scores1 = attention_scores1 + attention_mask2
		attention_probs1 = F.softmax(attention_scores1, dim=-1)
		context_layer1 = torch.matmul(self.dropout(attention_probs1), Wh12)
		
		return context_layer1, attention_probs1


class DualCrossAttention_concat(nn.Module):
	def __init__(self, config, input_size, output_size, alpha=1e-2):
		super(DualCrossAttention_concat, self).__init__()
		
		self.input_size = input_size
		self.output_size = output_size

		self.W1 = nn.Parameter(torch.empty(size=(input_size, output_size)).to(config.device))
		nn.init.xavier_uniform_(self.W1.data, gain=1.414)
		self.a1 = nn.Parameter(torch.empty(size=(2 * output_size, 1)).to(config.device))
		nn.init.xavier_uniform_(self.a1.data, gain=1.414)
		
		self.W2 = nn.Parameter(torch.empty(size=(input_size, output_size)).to(config.device))
		nn.init.xavier_uniform_(self.W2.data, gain=1.414)
		self.a2 = nn.Parameter(torch.empty(size=(2 * output_size, 1)).to(config.device))
		nn.init.xavier_uniform_(self.a2.data, gain=1.414)
		
		self.dropout = nn.Dropout(p=config.dropout)
		
		self.alpha = alpha
		self.leakyrelu = nn.LeakyReLU(self.alpha)
	
	def _prepare_attentional_mechanism_input(self, Wh1, Wh2, a):
		# Wh.shape (b, N, out_feature)
		# self.a.shape (2 * out_feature, 1)
		# attention1&2.shape (b, N, 1)
		# e.shape (b, N, N)
		attention1 = torch.matmul(Wh1, a[:self.output_size, :])
		attention2 = torch.matmul(Wh2, a[self.output_size:, :])
		# broadcast add
		e = attention1 + attention2.transpose(2, 1)
		return self.leakyrelu(e)
	
	def forward(self, input_tensor1, attention_mask1, input_tensor2, attention_mask2):
		# for input1.
		Wh11 = torch.matmul(input_tensor1, self.W1)  # h.shape: (b, N, in_features), Wh.shape: (b, N, out_features)
		Wh11 = self.dropout(Wh11)
		
		Wh12 = torch.matmul(input_tensor2, self.W1)  # h.shape: (b, N, in_features), Wh.shape: (b, N, out_features)
		Wh12 = self.dropout(Wh12)
		
		attention_scores1 = self._prepare_attentional_mechanism_input(Wh11, Wh12, self.a1)
		attention_scores1 = attention_scores1 + attention_mask2
		attention_probs1 = F.softmax(attention_scores1, dim=-1)
		context_layer1 = torch.matmul(self.dropout(attention_probs1), Wh12)
		
		# for input2.
		Wh22 = torch.matmul(input_tensor2, self.W2)  # h.shape: (b, N, in_features), Wh.shape: (b, N, out_features)
		Wh22 = self.dropout(Wh22)
		
		Wh21 = torch.matmul(input_tensor1, self.W2)  # h.shape: (b, N, in_features), Wh.shape: (b, N, out_features)
		Wh21 = self.dropout(Wh21)
		
		attention_scores2 = self._prepare_attentional_mechanism_input(Wh22, Wh21, self.a2)
		attention_scores2 = attention_scores2 + attention_mask1
		attention_probs2 = F.softmax(attention_scores2, dim=-1)
		context_layer2 = torch.matmul(self.dropout(attention_probs2), Wh21)
		
		return context_layer1, context_layer2, (attention_probs1, attention_probs2)


class re2graph(nn.Module):
	def __init__(self, config, input_size, output_size, dropout):
		super(re2graph, self).__init__()
		
		self.input_size = input_size
		self.output_size = output_size
		self.dropout = dropout
		self.config = config
		
		self.aggregation_type = config.aggregation_type
		self.cross_type = config.cross_type
		
		if self.aggregation_type == "one_stage":
			self.review_gat = GAT(config, self.input_size, self.output_size, self.dropout, residual=False)
			self.reply_gat = GAT(config, self.input_size, self.output_size, self.dropout, residual=False)
		elif self.aggregation_type == "two_stage":
			self.review_gat = GAT(config, self.input_size, self.output_size, self.dropout, residual=True)
			self.reply_gat = GAT(config, self.input_size, self.output_size, self.dropout, residual=True)
		
		# self.residual_layer = nn.Linear(self.hidden_size, self.hidden_size)
		# self.residual_layer = nn.Parameter(torch.ones([1]).to(config.device))
		
		if self.cross_type == "dot":
			self.cross_graph = DualCrossAttention_dot(config, self.input_size, self.output_size)
		elif self.cross_type == "concat":
			self.cross_graph = DualCrossAttention_concat(config, self.input_size, self.output_size)
		
		self.review_gate_unit = nn.Linear(2 * self.output_size, 1, bias=False)
		self.reply_gate_unit = nn.Linear(2 * self.output_size, 1, bias=False)
		
		# self.linear_align_review2reply = nn.Linear(4 * self.output_size, self.output_size)
		# self.linear_align_reply2review = nn.Linear(4 * self.output_size, self.output_size)
		
	def forward(self, review_input, reply_input, review_mask, reply_mask):
		'''
		:param review_input: [batch_size, sent_num, hidden_size]
		:param reply_input:
		:param review_mask: [batch_size, sent_num]
		:param reply_mask:
		:return:
		'''
		
		batch_size, max_review_sent_num, _ = review_input.size()
		_, max_reply_sent_num, _ = reply_input.size()
		
		review_adj = review_mask.view(batch_size, 1, max_review_sent_num).repeat(1, max_review_sent_num, 1)
		review_reps, review_att = self.review_gat(review_input, review_adj)
		reply_adj = reply_mask.view(batch_size, 1, max_reply_sent_num).repeat(1, max_reply_sent_num, 1)
		reply_reps, reply_att = self.reply_gat(reply_input, reply_adj)
		
		# print("review_reps ", review_reps.size())
		# print("reply_reps ", reply_reps.size())
		
		review_mask_inf = (1.0 - review_mask) * -10000.0
		reply_mask_inf = (1.0 - reply_mask) * -10000.0
		if self.cross_graph == "dot":
			review_mask_inf = review_mask_inf.view(-1, 1, 1, review_mask_inf.size(-1))
			reply_mask_inf = reply_mask_inf.view(-1, 1, 1, reply_mask_inf.size(-1))
		elif self.cross_graph == "concat":
			review_mask_inf = review_mask_inf.view(-1, 1, review_mask_inf.size(-1))
			reply_mask_inf = reply_mask_inf.view(-1, 1, reply_mask_inf.size(-1))
			
		if self.aggregation_type == "one_stage": # also consider the residual then cross
			
			review_node_from_graph, reply_node_from_graph, \
			(review_cross_attention, reply_cross_attention) = self.cross_graph(review_input, # replace it with review_reps (for strctural information)
			                                                                   review_mask_inf,
			                                                                   reply_input,
			                                                                   reply_mask_inf)
			# print("review_node_from_graph", review_node_from_graph.size())
			# print("reply_node_from_graph", reply_node_from_graph.size())
			review_gated_atts = torch.sigmoid(self.review_gate_unit(torch.cat([review_input, review_node_from_graph], -1))) # filter noise
			reply_gated_atts = torch.sigmoid(self.reply_gate_unit(torch.cat([reply_input, reply_node_from_graph], -1)))
			
			review_out = review_reps + review_input + review_gated_atts * review_node_from_graph
			reply_out = reply_reps + reply_input + reply_gated_atts * reply_node_from_graph
			
			# review_node_from_graph = self.linear_align_review2reply(torch.cat([review_input, review_node_from_graph,
			#                                                          review_input - review_node_from_graph,
			#                                                          review_input * review_node_from_graph], -1))
			# reply_node_from_graph = self.linear_align_review2reply(torch.cat([reply_input, reply_node_from_graph,
			#                                                         reply_input - reply_node_from_graph,
			#                                                         reply_input * reply_node_from_graph], -1))
			#
			# review_out = review_reps + review_input + review_node_from_graph
			# reply_out = reply_reps + reply_input + reply_node_from_graph
			
		elif self.aggregation_type == "two_stage":
			review_node_from_graph, reply_node_from_graph, \
			(review_cross_attention, reply_cross_attention) = self.cross_graph(review_reps,
			                                                                   # replace it with review_reps (for strctural information)
			                                                                   review_mask_inf,
			                                                                   reply_reps,
			                                                                   reply_mask_inf)
			
			review_gated_atts = torch.sigmoid(self.review_gate_unit(torch.cat([review_reps, review_node_from_graph], -1)))  # filter noise
			reply_gated_atts = torch.sigmoid(self.reply_gate_unit(torch.cat([reply_reps, reply_node_from_graph], -1)))
			
			review_out = review_reps + review_gated_atts * review_node_from_graph
			reply_out = reply_reps + reply_gated_atts * reply_node_from_graph
			
		return review_out, reply_out, review_att, reply_att, \
		       review_cross_attention, reply_cross_attention, \
		       review_gated_atts, reply_gated_atts


class re2graph_onlygat(nn.Module):
	def __init__(self, config, input_size, output_size, dropout):
		super(re2graph_onlygat, self).__init__()
		
		self.input_size = input_size
		self.output_size = output_size
		self.dropout = dropout
		self.config = config
		
		self.aggregation_type = config.aggregation_type
		self.cross_type = config.cross_type
		
		self.review_gat = GAT(config, self.input_size, self.output_size, self.dropout, residual=True)
		self.reply_gat = GAT(config, self.input_size, self.output_size, self.dropout, residual=True)
		
	def forward(self, review_input, reply_input, review_mask, reply_mask):
		'''
		:param review_input: [batch_size, sent_num, hidden_size]
		:param reply_input:
		:param review_mask: [batch_size, sent_num]
		:param reply_mask:
		:return:
		'''
		
		batch_size, max_review_sent_num, _ = review_input.size()
		_, max_reply_sent_num, _ = reply_input.size()
		
		review_adj = review_mask.view(batch_size, 1, max_review_sent_num).repeat(1, max_review_sent_num, 1)
		review_out, review_att = self.review_gat(review_input, review_adj)
		reply_adj = reply_mask.view(batch_size, 1, max_reply_sent_num).repeat(1, max_reply_sent_num, 1)
		reply_out, reply_att = self.reply_gat(reply_input, reply_adj)
		

		
		return review_out, reply_out, review_att, reply_att


class re2graph_onlycross(nn.Module):
	def __init__(self, config, input_size, output_size, dropout):
		super(re2graph_onlycross, self).__init__()
		
		self.input_size = input_size
		self.output_size = output_size
		self.dropout = dropout
		self.config = config
		
		self.aggregation_type = config.aggregation_type
		self.cross_type = config.cross_type
		
		self.cross_graph = DualCrossAttention_concat(config, self.input_size, self.output_size)
	
	def forward(self, review_input, reply_input, review_mask, reply_mask):
		'''
		:param review_input: [batch_size, sent_num, hidden_size]
		:param reply_input:
		:param review_mask: [batch_size, sent_num]
		:param reply_mask:
		:return:
		'''
		
		batch_size, max_review_sent_num, _ = review_input.size()
		_, max_reply_sent_num, _ = reply_input.size()
		
		# print("review_reps ", review_reps.size())
		# print("reply_reps ", reply_reps.size())
		
		review_mask_inf = (1.0 - review_mask) * -10000.0
		reply_mask_inf = (1.0 - reply_mask) * -10000.0
		review_mask_inf = review_mask_inf.view(-1, 1, review_mask_inf.size(-1))
		reply_mask_inf = reply_mask_inf.view(-1, 1, reply_mask_inf.size(-1))
		
		review_node_from_graph, reply_node_from_graph, \
		(review_cross_attention, reply_cross_attention) = self.cross_graph(review_input,
		                                                                   # replace it with review_reps (for strctural information)
		                                                                   review_mask_inf,
		                                                                   reply_input,
		                                                                   reply_mask_inf)
		review_out = review_node_from_graph
		reply_out = reply_node_from_graph
		
		
		return review_out, reply_out, review_cross_attention, reply_cross_attention,


class GCNConv(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, activation=None, bias=True):
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias

        if self.activation is not None:
            output = self.activation(output)

        return output