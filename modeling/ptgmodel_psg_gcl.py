from modeling.modules_psg_gcl import (BertProber, RobertaProber, LongFormerProber, TGModel)
from torch import nn
from utils.load_save import load_state_dict_with_mismatch
from utils.basic_utils import get_index_positions
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from utils.basic_utils import flat_list_of_lists
from utils.loss import GCL_Loss, GCL2_Loss
import math
# np.set_printoptions(threshold=np.inf)

class PTGModel(nn.Module):
    def __init__(self, config, encoder=None):
        super(PTGModel, self).__init__()
        self.config = config
        if config.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.encoder = encoder
        if self.encoder == "bert":
            self.prober = BertProber(config)
        elif self.encoder == "roberta":
            self.prober = RobertaProber(config)
        elif self.encoder == "longformer":
            self.prober = LongFormerProber(config)
        else:
            raise ValueError
        
        self.tgmodel = TGModel(config)

        if self.config.use_cl_loss:
            if self.config.cl_type == "gcl":
                self.cl_criterion = GCL_Loss(self.config.temperature)
            elif self.config.cl_type == "gcl2":
                self.cl_criterion = GCL2_Loss(self.config.temperature)
            else:
                raise ValueError
        self.drop_edge_type = self.config.drop_edge_type
        self.drop_edge_rate = self.config.drop_edge_rate

    def forward(self,
                review_bert_tokens,
                reply_bert_tokens,
                review_num_tokens, reply_num_tokens,
                review_lengths, reply_lengths,
                review_masks, reply_masks,
                rev2rev_weights, rev2rep_weights,
                rep2rev_weights, rep2rep_weights,
                review_adjs, reply_adjs,
                pair_matrix,
                mask_sent_list,
                review_ibose_list=None,
                reply_ibose_list=None,
                rev_arg_2_rep_arg_tags_list=None,
                rep_arg_2_rev_arg_tags_list=None,
                mode='train'):
    
        review_feature, reply_feature = self.prober(
            review_bert_tokens,
            reply_bert_tokens,
            review_num_tokens, reply_num_tokens,
            review_lengths,
            reply_lengths
        )

        if mode == 'train':
            crf_loss, pair_loss, cl_loss = self.train_mode(review_feature,  # [batch_size, review_sent_num, hidden_size]
                                                           reply_feature,  # [batch_size, review_sent_num, hidden_size]
                                                           review_masks,
                                                           reply_masks,
                                                           review_lengths,
                                                           reply_lengths,
                                                           rev2rev_weights, rev2rep_weights,
                                                           rep2rev_weights, rep2rep_weights,
                                                           review_adjs,
                                                           reply_adjs,
                                                           pair_matrix,
                                                           mask_sent_list,
                                                           review_ibose_list,
                                                           reply_ibose_list,
                                                           rev_arg_2_rep_arg_tags_list,
                                                           rep_arg_2_rev_arg_tags_list,
                                                           )
            result = (crf_loss, pair_loss, cl_loss)
            
        elif mode == 'val':
            pred_rev_args_list, pred_rep_args_list, \
            pred_args_pair_dict_list, pred_args_pair_2_dict_list = self.val_mode(
                review_feature,
                reply_feature,
                review_masks,
                reply_masks,
                review_lengths,
                reply_lengths,
                rev2rev_weights, rev2rep_weights,
                rep2rev_weights, rep2rep_weights,
                review_ibose_list,
                reply_ibose_list
            )
            result = (pred_rev_args_list, pred_rep_args_list, pred_args_pair_dict_list, pred_args_pair_2_dict_list)
        else:
            raise ValueError("Error model mode , please choice [train] or [val]")

        return result

    def train_mode(self,
                   review_feature,
                   reply_feature,
                   review_masks,
                   reply_masks,
                   review_lengths,
                   reply_lengths,
                   rev2rev_weights, rev2rep_weights,
                   rep2rev_weights, rep2rep_weights,
                   review_adjs,
                   reply_adjs,
                   pair_matrix,
                   mask_sent_list,
                   review_ibose_list,
                   reply_ibose_list,
                   rev_arg_2_rep_arg_tags_list,
                   rep_arg_2_rev_arg_tags_list,
                   ):

        rev2rev_weights_norm = torch.Tensor(self.non_symmetric_normalize_adj(rev2rev_weights)).to(self.config.device)
        rev2rep_weights_norm = torch.Tensor(self.non_symmetric_normalize_adj(rev2rep_weights)).to(self.config.device)
        rep2rev_weights_norm = torch.Tensor(self.non_symmetric_normalize_adj(rep2rev_weights)).to(self.config.device)
        rep2rep_weights_norm = torch.Tensor(self.non_symmetric_normalize_adj(rep2rep_weights)).to(self.config.device)

        if 'fake' in self.config.drop_type:
            if self.config.drop_type == 'fake_before_norm':
                rev2rev_weights_gcl = self.drop_edge(torch.Tensor(rev2rev_weights).to(self.config.device),
                                                     rev2rev_weights_norm, review_adjs[0].to(self.config.device), drop_edge_rate=self.drop_edge_rate)
                rev2rep_weights_gcl = self.drop_edge(torch.Tensor(rev2rep_weights).to(self.config.device),
                                                     rev2rep_weights_norm, pair_matrix[0].to(self.config.device), drop_edge_rate=self.drop_edge_rate)
                rep2rev_weights_gcl = self.drop_edge(torch.Tensor(rep2rev_weights).to(self.config.device),
                                                     rep2rev_weights_norm, pair_matrix[0].to(self.config.device).T, drop_edge_rate=self.drop_edge_rate)
                rep2rep_weights_gcl = self.drop_edge(torch.Tensor(rep2rep_weights).to(self.config.device),
                                                     rep2rep_weights_norm, reply_adjs[0].to(self.config.device), drop_edge_rate=self.drop_edge_rate)

                rev2rev_weights_gcl_norm = torch.Tensor(self.non_symmetric_normalize_adj(rev2rev_weights_gcl.cpu().numpy())).to(self.config.device)
                rev2rep_weights_gcl_norm = torch.Tensor(self.non_symmetric_normalize_adj(rev2rep_weights_gcl.cpu().numpy())).to(self.config.device)
                rep2rev_weights_gcl_norm = torch.Tensor(self.non_symmetric_normalize_adj(rep2rev_weights_gcl.cpu().numpy())).to(self.config.device)
                rep2rep_weights_gcl_norm = torch.Tensor(self.non_symmetric_normalize_adj(rep2rep_weights_gcl.cpu().numpy())).to(self.config.device)
            elif self.config.drop_type == 'fake_after_norm':
                rev2rev_weights_gcl_norm = self.drop_edge(rev2rev_weights_norm.clone(),
                                                          rev2rev_weights_norm, review_adjs[0].to(self.config.device),
                                                          drop_edge_rate=self.drop_edge_rate)
                rev2rep_weights_gcl_norm = self.drop_edge(rev2rep_weights_norm.clone(),
                                                          rev2rep_weights_norm, pair_matrix[0].to(self.config.device),
                                                          drop_edge_rate=self.drop_edge_rate)
                rep2rev_weights_gcl_norm = self.drop_edge(rep2rev_weights_norm.clone(),
                                                          rep2rev_weights_norm, pair_matrix[0].to(self.config.device).T,
                                                          drop_edge_rate=self.drop_edge_rate)
                rep2rep_weights_gcl_norm = self.drop_edge(rep2rep_weights_norm.clone(),
                                                          rep2rep_weights_norm, reply_adjs[0].to(self.config.device),
                                                          drop_edge_rate=self.drop_edge_rate)
            else:
                raise ValueError
        elif 'random' in self.config.drop_type:
            if self.config.drop_type == 'random_before_norm':
                rev2rev_weights_gcl = self.drop_random_edge(torch.Tensor(rev2rev_weights).to(self.config.device),
                                                     rev2rev_weights_norm, review_adjs[0].to(self.config.device),
                                                     drop_edge_rate=self.drop_edge_rate)
                rev2rep_weights_gcl = self.drop_random_edge(torch.Tensor(rev2rep_weights).to(self.config.device),
                                                     rev2rep_weights_norm, pair_matrix[0].to(self.config.device),
                                                     drop_edge_rate=self.drop_edge_rate)
                rep2rev_weights_gcl = self.drop_random_edge(torch.Tensor(rep2rev_weights).to(self.config.device),
                                                     rep2rev_weights_norm, pair_matrix[0].to(self.config.device).T,
                                                     drop_edge_rate=self.drop_edge_rate)
                rep2rep_weights_gcl = self.drop_random_edge(torch.Tensor(rep2rep_weights).to(self.config.device),
                                                     rep2rep_weights_norm, reply_adjs[0].to(self.config.device),
                                                     drop_edge_rate=self.drop_edge_rate)

                rev2rev_weights_gcl_norm = torch.Tensor(self.non_symmetric_normalize_adj(rev2rev_weights_gcl.cpu().numpy())).to(self.config.device)
                rev2rep_weights_gcl_norm = torch.Tensor(self.non_symmetric_normalize_adj(rev2rep_weights_gcl.cpu().numpy())).to(self.config.device)
                rep2rev_weights_gcl_norm = torch.Tensor(self.non_symmetric_normalize_adj(rep2rev_weights_gcl.cpu().numpy())).to(self.config.device)
                rep2rep_weights_gcl_norm = torch.Tensor(self.non_symmetric_normalize_adj(rep2rep_weights_gcl.cpu().numpy())).to(self.config.device)
            elif self.config.drop_type == 'random_after_norm':
                rev2rev_weights_gcl_norm = self.drop_random_edge(rev2rev_weights_norm.clone(),
                                                          rev2rev_weights_norm, review_adjs[0].to(self.config.device),
                                                          drop_edge_rate=self.drop_edge_rate)
                rev2rep_weights_gcl_norm = self.drop_random_edge(rev2rep_weights_norm.clone(),
                                                          rev2rep_weights_norm, pair_matrix[0].to(self.config.device),
                                                          drop_edge_rate=self.drop_edge_rate)
                rep2rev_weights_gcl_norm = self.drop_random_edge(rep2rev_weights_norm.clone(),
                                                          rep2rev_weights_norm, pair_matrix[0].to(self.config.device).T,
                                                          drop_edge_rate=self.drop_edge_rate)
                rep2rep_weights_gcl_norm = self.drop_random_edge(rep2rep_weights_norm.clone(),
                                                          rep2rep_weights_norm, reply_adjs[0].to(self.config.device),
                                                          drop_edge_rate=self.drop_edge_rate)
            else:
                raise ValueError
        else:
            raise ValueError


        crf_loss, pair_loss, \
        review_reps_gcl1, reply_reps_gcl1, review_reps_gcl2, reply_reps_gcl2, \
        review_gcn_rep1_gcl1, review_gcn_rep2_gcl1, reply_gcn_rep1_gcl1, reply_gcn_rep2_gcl1, \
        review_gcn_rep1_gcl2, review_gcn_rep2_gcl2, reply_gcn_rep1_gcl2, reply_gcn_rep2_gcl2 = self.tgmodel(review_feature,
                                                                                                            reply_feature,
                                                                                                            review_masks, reply_masks,
                                                                                                            review_lengths, reply_lengths,
                                                                                                            rev2rev_weights_norm,
                                                                                                            rev2rep_weights_norm,
                                                                                                            rep2rev_weights_norm,
                                                                                                            rep2rep_weights_norm,
                                                                                                            rev2rev_weights_gcl_norm,
                                                                                                            rev2rep_weights_gcl_norm,
                                                                                                            rep2rev_weights_gcl_norm,
                                                                                                            rep2rep_weights_gcl_norm,
                                                                                                            review_ibose_list, reply_ibose_list,
                                                                                                            rev_arg_2_rep_arg_tags_list,
                                                                                                            rep_arg_2_rev_arg_tags_list,
                                                                                                            )
    
        if self.config.use_cl_loss:
            pair_matrix_tensor = pair_matrix[0].to(self.config.device)
            # print("pair_matrix_tensor", pair_matrix_tensor.size(), "\n", pair_matrix_tensor)
            rev_sent_num, rep_sent_num = pair_matrix_tensor.size()
            if self.config.cl_type == "gcl":
                features_1 = torch.cat([review_reps_gcl1[0], reply_reps_gcl1[0]], 0)
                features_2 = torch.cat([review_reps_gcl2[0], reply_reps_gcl2[0]], 0)

                gcl_loss = self.cl_criterion(features_1, features_2)
            elif self.config.cl_type == "gcl2":
                review_adj_tensor = review_adjs[0].to(self.config.device)
                reply_adj_tensor = reply_adjs[0].to(self.config.device)

                mask = torch.cat([torch.cat([review_adj_tensor, pair_matrix_tensor], -1),
                                  torch.cat([pair_matrix_tensor.transpose(0, 1), reply_adj_tensor], -1)], 0)
                features_1 = torch.cat([review_reps_gcl1[0], reply_reps_gcl1[0]], 0)
                features_2 = torch.cat([review_reps_gcl2[0], reply_reps_gcl2[0]], 0)

                gcl_loss = self.cl_criterion(features_1, features_2, mask)
            else:
                raise ValueError

        else:
            gcl_loss = torch.tensor(0).to(self.config.device)

        return crf_loss, pair_loss, gcl_loss

    def val_mode(self,
                 review_feature,
                 reply_feature,
                 review_mask, reply_mask,
                 review_lengths, reply_lengths,
                 rev2rev_weights, rev2rep_weights,
                 rep2rev_weights, rep2rep_weights,
                 review_ibose_list, reply_ibose_list
                 ):
        rev2rev_weights_norm = self.non_symmetric_normalize_adj(rev2rev_weights)
        rev2rep_weights_norm = self.non_symmetric_normalize_adj(rev2rep_weights)
        rep2rev_weights_norm = self.non_symmetric_normalize_adj(rep2rev_weights)
        rep2rep_weights_norm = self.non_symmetric_normalize_adj(rep2rep_weights)

        pred_rev_args_list, pred_rep_args_list, pred_args_pair_dict_list, pred_args_pair_2_dict_list = self.tgmodel.predict(
            review_feature,
            reply_feature,
            review_mask,
            reply_mask,
            review_lengths,
            reply_lengths,
            torch.Tensor(rev2rev_weights_norm).to(self.config.device),
            torch.Tensor(rev2rep_weights_norm).to(self.config.device),
            torch.Tensor(rep2rev_weights_norm).to(self.config.device),
            torch.Tensor(rep2rep_weights_norm).to(self.config.device),
            review_ibose_list,
            reply_ibose_list
        )
        
        return pred_rev_args_list, pred_rep_args_list, pred_args_pair_dict_list, pred_args_pair_2_dict_list
    
    def load_separate_ckpt(self, context_model_weights_path=None, tg_weights_path=None):
        if context_model_weights_path:
            self.prober.load_state_dict(context_model_weights_path)

        if tg_weights_path:
            load_state_dict_with_mismatch(self.tgmodel, tg_weights_path)

    # def freeze_bert_backbone(self):
    #     for n, p in self.prober.bert.named_parameters():
    #         p.requires_grad = False

    def drop_edge(self, weights, weights_norm, label_adjs, drop_edge_rate):
        sample_vector = (torch.gt(weights, 0) * (1 - label_adjs)).view(-1)
        if sample_vector.sum().item() == 0:
            return weights
        sample_num = math.ceil(sample_vector.sum().item() * drop_edge_rate)
        weights_vector = weights.view(-1) # drop edge that has not been normalized TODO drop edge that has been normalized
        # print("label_adjs", label_adjs)
        # print("sample_num", sample_num)
        # print("weights before drop", weights)
        if self.drop_edge_type == 'uniform':
            # sample_idx = (sample_vector == 1).nonzero(as_tuple=False).view(-1)
            # choice = torch.multinomial(sample_idx.float(), sample_num)
            # sample_vector[sample_idx[choice]] = 0

            choice = torch.multinomial(sample_vector.float(), sample_num)
            weights_vector[choice] = 0
        elif self.drop_edge_type == 'bigger':
            sample_vector = (torch.gt(weights, 0) * (1 - label_adjs) * weights_norm).view(-1) # calculate the import of edge that has been normalized TODO select edge that has not been normalized
            choice = torch.multinomial(sample_vector.float(), sample_num)
            weights_vector[choice] = 0

        elif self.drop_edge_type == 'smaller':
            sample_vector = (torch.gt(weights, 0) * (1 - label_adjs) * (1-weights_norm)).view(-1)
            choice = torch.multinomial(sample_vector.float(), sample_num)
            weights_vector[choice] = 0

        else:
            raise ValueError
        # print("weights after drop", weights_vector.view(*label_adjs.size()))
        return weights_vector.view(*label_adjs.size())

    def drop_random_edge(self, weights, weights_norm, label_adjs, drop_edge_rate):
        sample_vector = (torch.gt(weights, 0)).view(-1)
        if sample_vector.sum().item() == 0:
            return weights
        sample_num = math.ceil(sample_vector.sum().item() * drop_edge_rate)
        weights_vector = weights.view(-1) # drop edge that has not been normalized TODO drop edge that has been normalized
        # print("label_adjs", label_adjs)
        # print("sample_num", sample_num)
        # print("weights before drop", weights)
        if self.drop_edge_type == 'uniform':
            # sample_idx = (sample_vector == 1).nonzero(as_tuple=False).view(-1)
            # choice = torch.multinomial(sample_idx.float(), sample_num)
            # sample_vector[sample_idx[choice]] = 0

            choice = torch.multinomial(sample_vector.float(), sample_num)
            weights_vector[choice] = 0
        elif self.drop_edge_type == 'bigger':
            sample_vector = (torch.gt(weights, 0) * weights_norm).view(-1) # calculate the import of edge that has been normalized TODO select edge that has not been normalized
            choice = torch.multinomial(sample_vector.float(), sample_num)
            weights_vector[choice] = 0

        elif self.drop_edge_type == 'smaller':
            sample_vector = (torch.gt(weights, 0) * (1-weights_norm)).view(-1)
            choice = torch.multinomial(sample_vector.float(), sample_num)
            weights_vector[choice] = 0

        else:
            raise ValueError
        # print("weights after drop", weights_vector.view(*label_adjs.size()))
        return weights_vector.view(*label_adjs.size())

    def non_symmetric_normalize_adj(self, adj):
        """Non Symmetrically normalize adjacency matrix."""
        denom = np.sum(adj, axis=-1, keepdims=True) + 1
        return adj / denom