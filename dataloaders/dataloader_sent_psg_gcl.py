import math
import torch
import numpy as np
from transformers import BertTokenizer
import itertools
import scipy.sparse as sp


def get_spans(tags):
    """
    for spans
    """
    tags = tags.strip().split('<tag>')
    length = len(tags)
    spans = []
    start = -1
    for i in range(length):
        if tags[i].endswith('B'):
            if start != -1:
                spans.append([start, i - 1])
            start = i
        elif tags[i].endswith('O'):
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans


global zero_sent_bow

label2id = {'O': 0, 'B-Review': 1, 'I-Review': 2, 'E-Review': 3, 'S-Review': 4,
         'B-Reply': 1, 'I-Reply': 2, 'E-Reply': 3, 'S-Reply': 4,
         'B': 1, 'I': 2, 'E': 3, 'S': 4}

special_tokens = ['[ENDL]', '[TAB]', '[LINE]',
                      '[EQU]', '[URL]', '[NUM]',
                      '[SPE]']

class Instance(object):

    def __init__(self, tokenizer, graph_matrix, sentence_pack, args):
        self.id = sentence_pack['id']
        self.sentence = sentence_pack['sentence']
        self.last_review = sentence_pack['split_idx']
        self.sents = self.sentence.strip().split(' <sentsep> ')
        self.review = self.sents[:self.last_review + 1]
        self.reply = self.sents[self.last_review + 1:]



        tags = sentence_pack['tags']
        review_adj = sentence_pack['review_adj']
        reply_adj = sentence_pack['reply_adj']

        review_bio_list = sentence_pack['review_bio_list']
        reply_bio_list = sentence_pack['reply_bio_list']

        review_iobes_list = sentence_pack['review_iobes_list']
        reply_iobes_list = sentence_pack['reply_iobes_list']

        self.sen_length = len(self.sents)
        self.review_length = self.last_review + 1
        self.reply_length = self.sen_length - self.last_review - 1
        self.review_bert_tokens = []
        self.review_bert_tokens_id = []
        self.reply_bert_tokens = []
        self.reply_bert_tokens_id = []
        self.review_num_tokens = []
        self.reply_num_tokens = []


        self.rev2rev_weight = graph_matrix["review2review_weights"]
        assert len(self.rev2rev_weight) == self.review_length and len(self.rev2rev_weight[0]) == self.review_length
        self.rev2rep_weight = graph_matrix["review2reply_weights"]
        assert len(self.rev2rep_weight) == self.review_length and len(self.rev2rep_weight[0]) == self.reply_length
        self.rep2rev_weight = graph_matrix["reply2review_weights"]
        assert len(self.rep2rev_weight) == self.reply_length and len(self.rep2rev_weight[0]) == self.review_length
        self.rep2rep_weight = graph_matrix["reply2reply_weights"]
        assert len(self.rep2rep_weight) == self.reply_length and len(self.rep2rep_weight[0]) == self.reply_length


        for i, sent in enumerate(self.review):
            sent_tokens = sent.strip().split(" ")

            sent_tokens_for_bert = ['[CLS]']
            sen_tokens_len = 0
            for orig_pos, token in enumerate(sent_tokens):
                bert_tokens = tokenizer.tokenize(token)
                sen_tokens_len += len(bert_tokens)
                sent_tokens_for_bert += bert_tokens
            sent_tokens_for_bert = sent_tokens_for_bert[:args.max_bert_token-1]
            sent_tokens_for_bert.append('[SEP]')

            assert sen_tokens_len > 0, print("self.review", self.review, '\n', sent_tokens)

            self.review_bert_tokens.append(sent_tokens_for_bert)
            self.review_bert_tokens_id.append(tokenizer.convert_tokens_to_ids(sent_tokens_for_bert))
            self.review_num_tokens.append(min(sen_tokens_len, args.max_bert_token-2))

        for i, sent in enumerate(self.reply):
            sent_tokens = sent.strip().split(" ")

            sent_tokens_for_bert = ['[CLS]']
            sen_tokens_len = 0
            for orig_pos, token in enumerate(sent_tokens):
                bert_tokens = tokenizer.tokenize(token)
                sen_tokens_len += len(bert_tokens)
                sent_tokens_for_bert += bert_tokens
            sent_tokens_for_bert = sent_tokens_for_bert[:args.max_bert_token - 1]
            sent_tokens_for_bert.append('[SEP]')

            assert sen_tokens_len > 0, print("self.reply", self.reply, '\n', sent_tokens)

            self.reply_bert_tokens.append(sent_tokens_for_bert)
            self.reply_bert_tokens_id.append(tokenizer.convert_tokens_to_ids(sent_tokens_for_bert))
            self.reply_num_tokens.append(min(sen_tokens_len, args.max_bert_token-2))

        self.length = len(self.sents)

        self.tags = torch.tensor(tags)

        self.review_adj = torch.tensor(review_adj)
        self.reply_adj = torch.tensor(reply_adj)

        if args.encoding_scheme == 'BIO':
            review_bio_list = [label2id[label] for label in review_bio_list]
            reply_bio_list = [label2id[label] for label in reply_bio_list]
            self.review_iboes = review_bio_list
            self.reply_ibose = reply_bio_list
        elif args.encoding_scheme == 'IOBES':
            review_iobes_list = [label2id[label] for label in review_iobes_list]
            reply_iobes_list = [label2id[label] for label in reply_iobes_list]
            self.review_ibose = review_iobes_list
            self.reply_ibose = reply_iobes_list

        self.rev_arg_2_rep_arg_dict = sentence_pack['rev_arg_2_rep_arg_dict'][0]
        assert len(sentence_pack['rev_arg_2_rep_arg_dict']) == 1
        self.rep_arg_2_rev_arg_dict = sentence_pack['rep_arg_2_rev_arg_dict'][0]
        assert len(sentence_pack['rep_arg_2_rev_arg_dict']) == 1
        self.rev_arg_2_rep_arg_tags_dict = sentence_pack['rev_arg_2_rep_arg_tags_dict'][0]
        assert len(sentence_pack['rev_arg_2_rep_arg_tags_dict']) == 1
        self.rep_arg_2_rev_arg_tags_dict = sentence_pack['rep_arg_2_rev_arg_tags_dict'][0]
        assert len(sentence_pack['rep_arg_2_rev_arg_tags_dict']) == 1

        review_mask_sents = np.zeros(self.review_length)
        reply_mask_sents = np.zeros(self.reply_length)

        for review_arg, _ in self.rev_arg_2_rep_arg_dict.items():
            review_mask_sents[review_arg[0]:review_arg[1]+1] = 1

        for reply_arg, _ in self.rep_arg_2_rev_arg_dict.items():
            reply_mask_sents[reply_arg[0]:reply_arg[1]+1] = 1
        self.mask_sents = np.concatenate([review_mask_sents, reply_mask_sents], axis=0)



def symmetric_normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    # print("normalize: ", adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    #  D^(-1/2)AD^(-1/2)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo().A


def non_symmetric_normalize_adj(adj):
    """Non Symmetrically normalize adjacency matrix."""
    denom = np.sum(adj, axis=-1, keepdims=True) + 1
    return adj / denom

def non_symmetric_normalize_adj2(adj):
    """Non Symmetrically normalize adjacency matrix."""
    denom = np.sum(adj, axis=-1, keepdims=True)
    denom = (denom == 0) + denom
    return adj / denom


def load_data_instances(sentence_packs, graph_matrices, tokenizer, args):
    global zero_sent_bow
    zero_sent_bow = 0
    instances = list()
    # special_tokens = ['[ENDL]', '[TAB]', '[LINE]', '[EQU]', '[URL]', '[NUM]', '[SPE]']
    # tokenizer = context_models[args.bert_tokenizer_path]['tokenizer'].from_pretrained(args.bert_tokenizer_path, additional_special_tokens=special_tokens)
    if args.num_instances != -1:
        for sentence_pack, graph_matrix in zip(sentence_packs[:args.num_instances], graph_matrices[:args.num_instances]):
            instances.append(Instance(tokenizer, graph_matrix, sentence_pack, args))
    else:
        for sentence_pack, graph_matrix in zip(sentence_packs, graph_matrices):
            instances.append(Instance(tokenizer, graph_matrix, sentence_pack, args))
    print("Find %d zero bow" % zero_sent_bow)
    return instances


class DataIterator(object):
    def __init__(self, instances, args, batch_size):
        self.instances = instances
        self.args = args
        self.batch_size = batch_size
        self.batch_count = math.ceil(len(instances) / self.batch_size)
        self.max_bert_token = args.max_bert_token

    def __len__(self):
        return len(self.instances)

    def get_batch(self, batch_i):
        batch_size = min((batch_i + 1) * self.batch_size, len(self.instances)) - batch_i * self.batch_size

        max_review_num_sents = max([self.instances[i].review_length for i in range(batch_i * self.batch_size,
                                                                                   min((batch_i + 1) * self.batch_size,
                                                                                       len(self.instances)))])
        max_reply_num_sents = max([self.instances[i].reply_length for i in range(batch_i * self.batch_size,
                                                                                 min((batch_i + 1) * self.batch_size,
                                                                                     len(self.instances)))])
        review_bert_tokens = []
        reply_bert_tokens = []
        review_lengths = []
        reply_lengths = []

        review_ibose_list = []
        reply_ibose_list = []
        rev_arg_2_rep_arg_tags_list = []
        rep_arg_2_rev_arg_tags_list = []
        rev_arg_2_rep_arg_list = []
        rep_arg_2_rev_arg_list = []

        review_adjs = []
        reply_adjs = []
        pair_matrix = []

        review_masks = torch.zeros(batch_size, max_review_num_sents, dtype=torch.long).to(self.args.device)
        reply_masks = torch.zeros(batch_size, max_reply_num_sents, dtype=torch.long).to(self.args.device)

        rev2rev_weights = []
        rev2rep_weights = []
        rep2rev_weights = []
        rep2rep_weights = []

        review_num_tokens = []
        reply_num_tokens = []

        mask_sent_list = []

        for i in range(batch_i * self.batch_size, min((batch_i + 1) * self.batch_size, len(self.instances))):
            review_bert_tokens.append(self.instances[i].review_bert_tokens_id)
            reply_bert_tokens.append(self.instances[i].reply_bert_tokens_id)

            review_lengths.append(self.instances[i].review_length)
            reply_lengths.append(self.instances[i].reply_length)

            review_masks[i - batch_i * self.batch_size, :self.instances[i].review_length] = 1
            reply_masks[i - batch_i * self.batch_size, :self.instances[i].reply_length] = 1

            review_num_tokens.append(self.instances[i].review_num_tokens)  # from index=1
            reply_num_tokens.append(self.instances[i].reply_num_tokens)

            rev2rev_weights = self.instances[i].rev2rev_weight
            rev2rep_weights = self.instances[i].rev2rep_weight
            rep2rev_weights = self.instances[i].rep2rev_weight
            rep2rep_weights = self.instances[i].rep2rep_weight

            review_adjs.append(self.instances[i].review_adj)
            reply_adjs.append(self.instances[i].reply_adj)

            pair_matrix.append(self.instances[i].tags)

            mask_sent_list.append(self.instances[i].mask_sents)

            review_ibose_list.append(self.instances[i].review_ibose)
            reply_ibose_list.append(self.instances[i].reply_ibose)
            rev_arg_2_rep_arg_tags_list.append(self.instances[i].rev_arg_2_rep_arg_tags_dict)
            rep_arg_2_rev_arg_tags_list.append(self.instances[i].rep_arg_2_rev_arg_tags_dict)
            rev_arg_2_rep_arg_list.append(self.instances[i].rev_arg_2_rep_arg_dict)
            rep_arg_2_rev_arg_list.append(self.instances[i].rep_arg_2_rev_arg_dict)

        return review_bert_tokens, reply_bert_tokens, \
               review_num_tokens, reply_num_tokens, \
               review_lengths, reply_lengths, \
               review_masks, reply_masks, \
               rev2rev_weights, rev2rep_weights, \
               rep2rev_weights, rep2rep_weights,\
               review_adjs, reply_adjs, \
               pair_matrix, \
               mask_sent_list,\
               review_ibose_list, reply_ibose_list, \
               rev_arg_2_rep_arg_tags_list, rep_arg_2_rev_arg_tags_list, \
               rev_arg_2_rep_arg_list, rep_arg_2_rev_arg_list

# if __name__ == '__main__':
#
# 	special_tokens = ['[ENDL]', '[TAB]', '[LINE]', '[EQU]', '[URL]', '[NUM]', '[SPE]']
# 	tokenizer = context_models[cfg.model_class]['tokenizer'].from_pretrained(cfg.bert_weights_path,
# 	                                                                         additional_special_tokens=special_tokens)
#
# 	train_sentence_packs = eval(json.load(open(config.data_path + '/train3.json')))
# 	dev_sentence_packs = eval(json.load(open(config.data_path + '/dev3.json')))
# 	test_sentence_packs = eval(json.load(open(config.data_path + '/test3.json')))
#
# 	# if cfg.use_probing_topic:
# 	postfix = f"_iw_intra_{cfg.intra_type}_inter_{cfg.inter_type}_sent_top_{cfg.sent_top}iw_weight_norm{str(cfg.iw_weight_norm)}.json"
# 	train_sorted_iw_packs = json.load(open(config.data_path + '/train' + postfix))
# 	dev_sorted_iw_packs = json.load(open(config.data_path + '/dev' + postfix))
# 	test_sorted_iw_packs = json.load(open(config.data_path + '/test' + postfix))
# 	# else:
# 	# 	train_sorted_iw_packs = None
# 	# 	dev_sorted_iw_packs = None
# 	# 	test_sorted_iw_packs = None
#
# 	bow_dictionary = pickle.load(open(config.data_path + '/vocab.pkl', "rb"))
# 	config.bow_vocab_size = len(bow_dictionary)
#
# 	if cfg.encoding_scheme == 'IOBES':
# 		cfg.num_tags = 5
# 	else:
# 		cfg.num_tags = 3
#
# 	instances_train = load_data_instances(train_sentence_packs, train_sorted_iw_packs, bow_dictionary, tokenizer,
# 	                                      config)
# 	instances_dev = load_data_instances(dev_sentence_packs, dev_sorted_iw_packs, bow_dictionary, tokenizer, config)
# 	instances_test = load_data_instances(test_sentence_packs, test_sorted_iw_packs, bow_dictionary, tokenizer, config)