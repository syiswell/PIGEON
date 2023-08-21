import torch
import torch.nn as nn
import torch.nn.functional as F


class SCL_Loss(nn.Module):
    """reference Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, temperature=0.07, sim_type='cos', use_only_sents_in_ac=True, base_temperature=0.07):
        super(SCL_Loss, self).__init__()
        self.temperature = temperature
        self.sim_type = sim_type
        self.base_temperature = base_temperature
        self.use_only_sents_in_ac = use_only_sents_in_ac

    def forward(self, features, mask, mask_sents=None):
        # only consider the sentences in AC or consider all sentences ???
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        batch_size = features.shape[0]

        # labels = labels.contiguous().view(-1, 1)
        # if labels.shape[0] != batch_size:
        #     raise ValueError('Num of labels does not match num of features')
        # mask = torch.eq(labels, labels.T).float().to(device)
        if mask.shape[0] != batch_size or mask.shape[1] != batch_size:
            raise ValueError('Num of mask does not match num of features')

        if self.sim_type == "cos":
            features = F.normalize(features, dim=-1)

        if self.use_only_sents_in_ac:
            # if use the mask_sents then only consider the sentences in AC
            # mask_sents = mask_sents.contiguous().view(-1, 1)
            # print("features", features.size())
            if mask_sents.shape[0] != batch_size:
                raise ValueError('Num of mask_sents does not match num of features')
            features_selected = features[mask_sents]
        else:
            features_selected = features

        contrast_count = features_selected.size(1)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = features_selected[:, 0]
        anchor_count = 1

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask_pos = mask * logits_mask
        mask_neg = (torch.ones_like(mask) - mask) * logits_mask

        if self.use_only_sents_in_ac:
            # if use the mask_sents then only consider the sentences in AC
            mask_pos = mask_pos[mask_sents]
            mask_neg = mask_neg[mask_sents]

        similarity = torch.exp(torch.mm(anchor_feature, contrast_feature.t()) / self.temperature)
        pos = torch.sum(similarity * mask_pos, 1)
        neg = torch.sum(similarity * mask_neg, 1)
        loss = -(torch.mean(torch.log((pos + 1e-8) / (pos + neg + 1e-8))))

        if torch.isinf(loss) or torch.isnan(loss):
            print("similarity", similarity.size(), similarity)
            print("pos similarity", pos.size(), pos)
            print("neg similarity", neg.size(), neg)
            print("mask_pos", mask_pos.size(), mask_pos)
            print("mask_neg", mask_neg.size(), mask_neg)
            print("mask", mask.size(), mask)
            print("mask_sents", mask_sents.size(), mask_sents)
            print("loss", loss)
            loss = torch.zeros_like(loss).to(device)

        return loss


class SCL2_Loss(nn.Module):
    """reference Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, temperature=0.07, sim_type='cos', use_only_sents_in_ac=True, base_temperature=0.07):
        super(SCL2_Loss, self).__init__()
        self.temperature = temperature
        self.sim_type = sim_type
        self.base_temperature = base_temperature
        self.use_only_sents_in_ac = use_only_sents_in_ac

    def forward(self, features, mask, mask_sents=None):
        # only consider the sentences in AC or consider all sentences ???
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        batch_size = features.shape[0]

        if mask.shape[0] != batch_size or mask.shape[1] != batch_size:
            raise ValueError('Num of mask does not match num of features')

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        if self.sim_type == "cos":
            features = F.normalize(features, dim=-1)

        if self.use_only_sents_in_ac:
            # if use the mask_sents then only consider the sentences in AC
            # mask_sents = mask_sents.contiguous().view(-1, 1)
            # print("features", features.size())
            if mask_sents.shape[0] != batch_size:
                raise ValueError('Num of mask_sents does not match num of features')
            mask_sents = mask_sents & torch.gt((mask * logits_mask).sum(-1), 0)
        else:
            mask_sents = torch.gt((mask * logits_mask).sum(-1), 0)

        anchor_feature = features[mask_sents][:, 0]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # print("anchor_feature", anchor_feature.size())
        # print("contrast_feature", contrast_feature.size())


        # mask-out self-contrast cases
        mask_pos = mask * logits_mask
        mask_neg = (torch.ones_like(mask) - mask) * logits_mask

        # if use the mask_sents then only consider the sentences in AC
        mask_pos = mask_pos[mask_sents]
        mask_neg = mask_neg[mask_sents]

        similarity = torch.exp(torch.mm(anchor_feature, contrast_feature.t()) / self.temperature)
        pos = torch.sum(similarity * mask_pos, 1)
        neg = torch.sum(similarity * mask_neg, 1)
        loss = -(torch.mean(torch.log((pos + 1e-8) / (pos + neg + 1e-8)) / mask_pos.sum(-1)))

        if torch.isinf(loss) or torch.isnan(loss):
            print("similarity", similarity.size(), similarity)
            print("pos similarity", pos.size(), pos)
            print("neg similarity", neg.size(), neg)
            print("mask_pos", mask_pos.size(), mask_pos)
            print("mask_neg", mask_neg.size(), mask_neg)
            print("mask", mask.size(), mask)
            print("mask_sents", mask_sents.size(), mask_sents)
            print("loss", loss)
            loss = torch.zeros_like(loss).to(device)

        return loss


class SUBCL_Loss(nn.Module):
    """Sub-graph Contrastive Learning"""
    def __init__(self, temperature=0.07, sim_type='cos', use_only_sents_in_ac=True, base_temperature=0.07):
        super(SUBCL_Loss, self).__init__()
        self.temperature = temperature
        self.sim_type = sim_type
        self.base_temperature = base_temperature
        self.use_only_sents_in_ac = use_only_sents_in_ac

    def forward(self, features, mask, mask_sents=None):
        '''
        :param features: [batch_size, 2, hidden_size]
        :param mask: [batch_size, batch_size]
        :param mask_sents: [batch_size, 1]
        :return:
        '''

        # only consider the sentences in AC or consider all sentences ???
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        batch_size = features.shape[0]

        # labels = labels.contiguous().view(-1, 1)
        # if labels.shape[0] != batch_size:
        #     raise ValueError('Num of labels does not match num of features')
        # mask = torch.eq(labels, labels.T).float().to(device)
        if mask.shape[0] != batch_size or mask.shape[1] != batch_size:
            raise ValueError('Num of mask does not match num of features')

        if self.sim_type == "cos":
            features = F.normalize(features, dim=-1)

        if self.use_only_sents_in_ac:
            # if use the mask_sents then only consider the sentences in AC
            # mask_sents = mask_sents.contiguous().view(-1, 1)
            if mask_sents.shape[0] != batch_size:
                raise ValueError('Num of mask_sents does not match num of features')
            features_selected = features[mask_sents]
        else:
            features_selected = features

        contrast_count = features_selected.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = torch.cat(torch.unbind(features_selected, dim=1), dim=0)
        anchor_count = contrast_count # 2

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        mask_sents = mask_sents.repeat(anchor_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask_pos = mask * logits_mask
        mask_neg = (torch.ones_like(mask) - mask) * logits_mask

        if self.use_only_sents_in_ac:
            # if use the mask_sents then only consider the sentences in AC
            mask_pos = mask_pos[mask_sents]
            mask_neg = mask_neg[mask_sents]

        similarity = torch.exp(torch.mm(anchor_feature, contrast_feature.t()) / self.temperature)
        pos = torch.sum(similarity * mask_pos, 1)
        neg = torch.sum(similarity * mask_neg, 1)
        loss = -(torch.mean(torch.log((pos + 1e-8) / (pos + neg + 1e-8))))
        if torch.isinf(loss) or torch.isnan(loss):
            print("similarity", similarity.size(), similarity)
            print("pos similarity", pos.size(), pos)
            print("neg similarity", neg.size(), neg)
            print("mask_pos", mask_pos.size(), mask_pos)
            print("mask_neg", mask_neg.size(), mask_neg)
            print("mask", mask.size(), mask)
            print("mask_sents", mask_sents.size(), mask_sents)
            print("loss", loss)
            loss = torch.zeros_like(loss).to(device)

        return loss


class SUB_Loss(nn.Module):
    """
    Subtract loss, this is similar with BPR loss
    Whether distinguish paragraph:
    1. no (all): no delete (default, beacuse the comparison between similarity of two sentence in different paragraph and
    that of two sentence in the same paragraph means strongly focus the key information.)
    2. yes (cross): delete the comparison between similarity of two sentence in different paragraph and
    that of two sentence in the same paragraph: [mask] can control this
    """
    def __init__(self, temperature=0.07, sub_type='pre', sim_type='dot', base_temperature=0.07):
        super(SUB_Loss, self).__init__()
        self.temperature = temperature
        self.sub_type = sub_type
        self.sim_type = sim_type
        self.base_temperature = base_temperature

    def forward(self, features, mask, mask_sents=None, mask_range=None):
        # only consider the sentences in AC
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        batch_size = features.shape[0]

        assert len(features.shape) == 3

        # The mask determine the pos and neg sample range: make the re2re zero to stop the interaction between sentence pair in the same paragraph
        if mask.shape[0] != batch_size or mask.shape[1] != batch_size:
            raise ValueError('Num of mask does not match num of features')

        # if use the mask_sents then only consider the sentences in AC
        # mask_sents = mask_sents.contiguous().view(-1, 1)
        if mask_sents.shape[0] != batch_size:
            raise ValueError('Num of mask_sents does not match num of features')

        if self.sim_type == "cos":
            features = F.normalize(features, dim=-1)
        features_selected = features[mask_sents] # if dot TODO numerical stability


        contrast_count = features_selected.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = features_selected[:, 0]
        anchor_count = 1

        # tile mask
        # mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask_pos = mask * logits_mask
        mask_neg = (torch.ones_like(mask) - mask) * logits_mask
        if mask_range != None:
            mask_neg = mask_neg * mask_range

        # if use the mask_sents then only consider the sentences in AC
        mask_pos = mask_pos[mask_sents]
        mask_neg = mask_neg[mask_sents]

        similarity = torch.mm(anchor_feature, contrast_feature.t()) / self.temperature

        if self.sub_type == "post":
            # first sum all subtract value for all pos-neg pair, and then compute the prob
            pos = torch.sum(similarity * mask_pos, 1) * mask_neg.sum(1)
            neg = torch.sum(similarity * mask_neg, 1) * mask_pos.sum(1)
            log_prob = F.logsigmoid(pos-neg)
            loss = torch.mean(-log_prob) # for sent_num in AC

        elif self.sub_type == "pre": # default
            # first compute the prob for all pos-neg pair, and then sum
            loss = torch.tensor(0, dtype=torch.float).to(device)
            # loss = 0
            pair_num = 0
            for i in range(similarity.size(0)):
                sent_sim = similarity[i]
                sent_mask_pos = mask_pos[i]
                sent_mask_neg = mask_neg[i]
                sent_pos_num = sent_mask_pos.sum()
                sent_neg_num = sent_mask_neg.sum()
                if sent_pos_num > 0 and sent_neg_num > 0: # sent_sim.size(-1)-1
                    pair_num += sent_pos_num * sent_neg_num
                    sent_sim_pos = torch.masked_select(sent_sim, sent_mask_pos.bool()).unsqueeze(-1)
                    sent_sim_neg = torch.masked_select(sent_sim, sent_mask_neg.bool()).unsqueeze(0)
                    loss += -F.logsigmoid(sent_sim_pos - sent_sim_neg).sum()
            # print("pair_num", pair_num)
            # print("loss", loss)
            # print("similarity", similarity.size(), similarity)
            # print("mask_pos", mask_pos.size(), mask_pos)
            # print("mask_neg", mask_neg.size(), mask_neg)
            # print("mask", mask.size(), mask)
            # print("mask_sents", mask_sents.size(), mask_sents)

            if pair_num > 0:
                loss = loss / pair_num
        else:
            raise ValueError

        if torch.isinf(loss) or torch.isnan(loss):
            print("similarity", similarity.size(), similarity)
            print("mask_pos", mask_pos.size(), mask_pos)
            print("mask_neg", mask_neg.size(), mask_neg)
            print("mask", mask.size(), mask)
            print("mask_sents", mask_sents.size(), mask_sents)
            raise ValueError
            # loss = torch.zeros_like(loss).to(device)

        return loss


class SUB_IO_Loss(nn.Module):
    """
    Subtract loss, this is similar with BPR loss
    Whether distinguish paragraph:
    1. no (all): no delete (default, beacuse the comparison between similarity of two sentence in different paragraph and
    that of two sentence in the same paragraph means strongly focus the key information.)
    2. yes (cross): delete the comparison between similarity of two sentence in different paragraph and
    that of two sentence in the same paragraph: [mask] can control this
    """
    def __init__(self, temperature=0.07, sub_type='pre', sim_type='dot', base_temperature=0.07):
        super(SUB_IO_Loss, self).__init__()
        self.temperature = temperature
        self.sub_type = sub_type
        self.sim_type = sim_type
        self.base_temperature = base_temperature

    def forward(self, features_o, features_i, mask, mask_sents=None, mask_range=None):
        # only consider the sentences in AC
        device = (torch.device('cuda') if features_o.is_cuda else torch.device('cpu'))

        batch_size = features_o.shape[0]

        assert len(features_o.shape) == 2

        # The mask determine the pos and neg sample range: make the re2re zero to stop the interaction between sentence pair in the same paragraph
        if mask.shape[0] != batch_size or mask.shape[1] != batch_size:
            raise ValueError('Num of mask does not match num of features')

        # if use the mask_sents then only consider the sentences in AC
        # mask_sents = mask_sents.contiguous().view(-1, 1)
        if mask_sents.shape[0] != batch_size:
            raise ValueError('Num of mask_sents does not match num of features')

        if self.sim_type == "cos":
            features_o = F.normalize(features_o, dim=-1)
            features_i = F.normalize(features_i, dim=-1)
        features_selected = features_o[mask_sents] # if dot TODO numerical stability

        anchor_feature = features_selected
        contrast_feature = features_i

        # tile mask
        # mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        # logits_mask = torch.scatter(
        #     torch.ones_like(mask),
        #     1,
        #     torch.arange(batch_size).view(-1, 1).to(device),
        #     0
        # )
        mask_pos = mask #* logits_mask
        mask_neg = (torch.ones_like(mask) - mask) #* logits_mask
        if mask_range != None:
            mask_neg = mask_neg * mask_range

        # if use the mask_sents then only consider the sentences in AC
        mask_pos = mask_pos[mask_sents]
        mask_neg = mask_neg[mask_sents]

        similarity = torch.mm(anchor_feature, contrast_feature.t()) / self.temperature

        if self.sub_type == "post":
            # first sum all subtract value for all pos-neg pair, and then compute the prob
            pos = torch.sum(similarity * mask_pos, 1) * mask_neg.sum(1)
            neg = torch.sum(similarity * mask_neg, 1) * mask_pos.sum(1)
            log_prob = F.logsigmoid(pos-neg)
            loss = torch.mean(-log_prob) # for sent_num in AC

        elif self.sub_type == "pre": # default
            # first compute the prob for all pos-neg pair, and then sum
            loss = torch.tensor(0, dtype=torch.float).to(device)
            # loss = 0
            pair_num = 0
            for i in range(similarity.size(0)):
                sent_sim = similarity[i]
                sent_mask_pos = mask_pos[i]
                sent_mask_neg = mask_neg[i]
                sent_pos_num = sent_mask_pos.sum()
                sent_neg_num = sent_mask_neg.sum()
                if sent_pos_num > 0 and sent_neg_num > 0: # sent_sim.size(-1)-1
                    pair_num += sent_pos_num * sent_neg_num
                    sent_sim_pos = torch.masked_select(sent_sim, sent_mask_pos.bool()).unsqueeze(-1)
                    sent_sim_neg = torch.masked_select(sent_sim, sent_mask_neg.bool()).unsqueeze(0)
                    loss += -F.logsigmoid(sent_sim_pos - sent_sim_neg).sum()
            # print("pair_num", pair_num)
            # print("loss", loss)
            # print("similarity", similarity.size(), similarity)
            # print("mask_pos", mask_pos.size(), mask_pos)
            # print("mask_neg", mask_neg.size(), mask_neg)
            # print("mask", mask.size(), mask)
            # print("mask_sents", mask_sents.size(), mask_sents)

            if pair_num > 0:
                loss = loss / pair_num
        else:
            raise ValueError

        if torch.isinf(loss) or torch.isnan(loss):
            print("similarity", similarity.size(), similarity)
            print("mask_pos", mask_pos.size(), mask_pos)
            print("mask_neg", mask_neg.size(), mask_neg)
            print("mask", mask.size(), mask)
            print("mask_sents", mask_sents.size(), mask_sents)
            raise ValueError
            # loss = torch.zeros_like(loss).to(device)

        return loss


class SUB_Cross_Loss(nn.Module):
    """
    Subtract contrastive loss
    Whether distinguish paragraph:
    1. no (all): no delete (default, beacuse the comparison between similarity of two sentence in different paragraph and
    that of two sentence in the same paragraph means strongly focus the key information.)
    2. yes (cross): delete the comparison between similarity of two sentence in different paragraph and
    that of two sentence in the same paragraph: [mask] can control this
    """
    def __init__(self, temperature=0.07, sub_type='pre', sim_type='dot', base_temperature=0.07):
        super(SUB_Cross_Loss, self).__init__()
        self.temperature = temperature
        self.sub_type = sub_type
        self.sim_type = sim_type
        self.base_temperature = base_temperature

    def forward(self, features1, features2, mask, mask_sents1=None, mask_sents2=None):
        # only consider the sentences in AC
        device = (torch.device('cuda') if features1.is_cuda else torch.device('cpu'))

        size1 = features1.shape[0]
        size2 = features2.shape[0]

        # The mask determine the pos and neg sample range: make the re2re zero to stop the interaction between sentence pair in the same paragraph
        if mask.shape[0] != size1 or mask.shape[1] != size2:
            raise ValueError('Num of mask does not match num of features')

        # if use the mask_sents then only consider the sentences in AC
        # mask_sents = mask_sents.contiguous().view(-1, 1)
        if mask_sents1.shape[0] != size1:
            raise ValueError('Num of mask_sents1 does not match num of features')
        if mask_sents2.shape[0] != size2:
            raise ValueError('Num of mask_sents2 does not match num of features')

        if self.sim_type == "cos":
            features1 = F.normalize(features1, dim=-1)
            features2 = F.normalize(features2, dim=-1)
        features1_selected = features1[mask_sents1] # if dot TODO numerical stability
        features2_selected = features2[mask_sents2]  # if dot TODO numerical stability

        loss1, pair_num1 = self.compute_loss(features1_selected, features2, mask, mask_sents1, device)
        loss2, pair_num2 = self.compute_loss(features2_selected, features1, mask.t(), mask_sents2, device)

        loss = loss1 + loss2
        pair_num = pair_num1 + pair_num2
        if pair_num > 0:
            loss = loss / pair_num

        if torch.isinf(loss) or torch.isnan(loss):
            print("mask_pos", mask_sents1)
            print("mask_neg", mask_sents2)
            print("mask", mask)
            raise ValueError
            # loss = torch.zeros_like(loss).to(device)

        return loss

    def compute_loss(self, feature1, feature2, mask, mask_sents, device):
        loss = torch.tensor(0, dtype=torch.float).to(device)

        mask_pos = mask
        mask_neg = torch.ones_like(mask) - mask

        # if use the mask_sents then only consider the sentences in AC
        mask_pos = mask_pos[mask_sents]
        mask_neg = mask_neg[mask_sents]

        similarity = torch.mm(feature1, feature2.t())

        pair_num = 0
        for i in range(similarity.size(0)):
            sent_sim = similarity[i]
            sent_mask_pos = mask_pos[i]
            sent_mask_neg = mask_neg[i]
            sent_pos_num = sent_mask_pos.sum()
            sent_neg_num = sent_mask_neg.sum()
            if sent_pos_num > 0 and sent_neg_num > 0:  # sent_sim.size(-1)-1
                pair_num += sent_pos_num * sent_neg_num
                sent_sim_pos = torch.masked_select(sent_sim, sent_mask_pos.bool()).unsqueeze(-1)
                sent_sim_neg = torch.masked_select(sent_sim, sent_mask_neg.bool()).unsqueeze(0)
                loss += -F.logsigmoid(sent_sim_pos - sent_sim_neg).sum()
        print("pair_num", pair_num)
        print("loss", loss)

        return loss, pair_num


class SUB_Independent_Loss(nn.Module):
    """
    Subtract contrastive loss
    Whether distinguish paragraph:
    1. no (all): no delete (default, beacuse the comparison between similarity of two sentence in different paragraph and
    that of two sentence in the same paragraph means strongly focus the key information.)
    2. yes (cross): delete the comparison between similarity of two sentence in different paragraph and
    that of two sentence in the same paragraph: [mask] can control this
    """
    def __init__(self, temperature=0.07, sub_type='pre', sim_type='dot', base_temperature=0.07):
        super(SUB_Independent_Loss, self).__init__()
        self.temperature = temperature
        self.sub_type = sub_type
        self.sim_type = sim_type
        self.base_temperature = base_temperature

    def forward(self, features1, features2, mask_cross, mask1, mask2, mask_sents1=None, mask_sents2=None):
        # only consider the sentences in AC
        device = (torch.device('cuda') if features1.is_cuda else torch.device('cpu'))

        size1 = features1.shape[0]
        size2 = features2.shape[0]

        # The mask determine the pos and neg sample range: make the re2re zero to stop the interaction between sentence pair in the same paragraph
        if mask_cross.shape[0] != size1 or mask_cross.shape[1] != size2:
            raise ValueError('Num of mask does not match num of features')

        if mask1.shape[0] != size1:
            raise ValueError('Num of mask1 does not match num of features')
        if mask2.shape[0] != size2:
            raise ValueError('Num of mask2 does not match num of features')

        # if use the mask_sents then only consider the sentences in AC
        # mask_sents = mask_sents.contiguous().view(-1, 1)
        if mask_sents1.shape[0] != size1:
            raise ValueError('Num of mask_sents1 does not match num of features')
        if mask_sents2.shape[0] != size2:
            raise ValueError('Num of mask_sents2 does not match num of features')

        if self.sim_type == "cos":
            features1 = F.normalize(features1, dim=-1)
            features2 = F.normalize(features2, dim=-1)
        features1_selected = features1[mask_sents1] # if dot TODO numerical stability
        features2_selected = features2[mask_sents2]  # if dot TODO numerical stability

        loss1, pair_num1 = self.compute_loss(features1_selected, features2, mask_cross, mask_sents1, device)
        loss2, pair_num2 = self.compute_loss(features2_selected, features1, mask_cross.t(), mask_sents2, device)
        loss3, pair_num3 = self.compute_loss(features1_selected, features1, mask1, mask_sents1, device)
        loss4, pair_num4 = self.compute_loss(features2_selected, features2, mask2, mask_sents2, device)

        loss = loss1 + loss2 + loss3 + loss4
        pair_num = pair_num1 + pair_num2 + pair_num3 + pair_num4
        if pair_num > 0:
            loss = loss / pair_num

        if torch.isinf(loss) or torch.isnan(loss):
            print("mask_pos", mask_sents1)
            print("mask_neg", mask_sents2)
            print("mask_cross", mask_cross)
            print("mask1", mask1)
            print("mask2", mask2)
            raise ValueError
            # loss = torch.zeros_like(loss).to(device)

        return loss

    def compute_loss(self, feature1, feature2, mask, mask_sents, device):
        loss = torch.tensor(0, dtype=torch.float).to(device)

        mask_pos = mask
        mask_neg = torch.ones_like(mask) - mask

        # if use the mask_sents then only consider the sentences in AC
        mask_pos = mask_pos[mask_sents]
        mask_neg = mask_neg[mask_sents]

        similarity = torch.mm(feature1, feature2.t())

        pair_num = 0
        for i in range(similarity.size(0)):
            sent_sim = similarity[i]
            sent_mask_pos = mask_pos[i]
            sent_mask_neg = mask_neg[i]
            sent_pos_num = sent_mask_pos.sum()
            sent_neg_num = sent_mask_neg.sum()
            if sent_pos_num > 0 and sent_neg_num > 0:  # sent_sim.size(-1)-1
                pair_num += sent_pos_num * sent_neg_num
                sent_sim_pos = torch.masked_select(sent_sim, sent_mask_pos.bool()).unsqueeze(-1)
                sent_sim_neg = torch.masked_select(sent_sim, sent_mask_neg.bool()).unsqueeze(0)
                loss += -F.logsigmoid(sent_sim_pos - sent_sim_neg).sum()
        # print("pair_num", pair_num)
        # print("loss", loss)

        return loss, pair_num


class SCL_Independent_Loss(nn.Module):
    def __init__(self, temperature=0.07, sim_type='cos', use_only_sents_in_ac=True, base_temperature=0.07):
        super(SCL_Independent_Loss, self).__init__()
        self.temperature = temperature
        self.sim_type = sim_type
        self.base_temperature = base_temperature
        self.use_only_sents_in_ac = use_only_sents_in_ac

    def forward(self, features1, features2, mask_cross, mask1, mask2, mask_sents1=None, mask_sents2=None):
        # only consider the sentences in AC
        device = (torch.device('cuda') if features1.is_cuda else torch.device('cpu'))

        size1 = features1.shape[0]
        size2 = features2.shape[0]

        # The mask determine the pos and neg sample range: make the re2re zero to stop the interaction between sentence pair in the same paragraph
        if mask_cross.shape[0] != size1 or mask_cross.shape[1] != size2:
            raise ValueError('Num of mask does not match num of features')

        if mask1.shape[0] != size1:
            raise ValueError('Num of mask1 does not match num of features')
        if mask2.shape[0] != size2:
            raise ValueError('Num of mask2 does not match num of features')

        # if use the mask_sents then only consider the sentences in AC
        # mask_sents = mask_sents.contiguous().view(-1, 1)
        if mask_sents1.shape[0] != size1:
            raise ValueError('Num of mask_sents1 does not match num of features')
        if mask_sents2.shape[0] != size2:
            raise ValueError('Num of mask_sents2 does not match num of features')

        if self.sim_type == "cos":
            features1 = F.normalize(features1, dim=-1)
            features2 = F.normalize(features2, dim=-1)

        loss1 = self.compute_loss(features1, features2, mask_cross, mask_sents1, device, 'cross')
        loss2 = self.compute_loss(features2, features1, mask_cross.t(), mask_sents2, device, 'cross')
        loss3 = self.compute_loss(features1, features1, mask1, mask_sents1, device, 'self')
        loss4 = self.compute_loss(features2, features2, mask2, mask_sents2, device, 'self')

        loss = loss1 + loss2 + loss3 + loss4

        # if torch.isinf(loss) or torch.isnan(loss):
        #     print("mask_pos", mask_sents1)
        #     print("mask_neg", mask_sents2)
        #     print("mask_cross", mask_cross)
        #     print("mask1", mask1)
        #     print("mask2", mask2)
        #     raise ValueError
            # loss = torch.zeros_like(loss).to(device)

        return loss

    def compute_loss(self, feature1, feature2, mask, mask_sents, device, mode="cross"):
        batch_size = feature1.size(0)
        # tile mask
        if mode == 'self':
            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size).view(-1, 1).to(device),
                0
            )
            if self.use_only_sents_in_ac:
                mask_sents = mask_sents & torch.gt((mask * logits_mask).sum(-1), 0)
            else:
                mask_sents = torch.gt((mask * logits_mask).sum(-1), 0)
        else:
            logits_mask = torch.ones_like(mask)
            if self.use_only_sents_in_ac:
                mask_sents = mask_sents & torch.gt(mask.sum(-1), 0)
            else:
                mask_sents = torch.gt(mask.sum(-1), 0)

        contrast_feature = feature2
        anchor_feature = feature1[mask_sents]

        mask_pos = mask * logits_mask
        mask_neg = (torch.ones_like(mask) - mask) * logits_mask

        # if use the mask_sents then only consider the sentences in AC
        mask_pos = mask_pos[mask_sents]
        mask_neg = mask_neg[mask_sents]

        similarity = torch.exp(torch.mm(anchor_feature, contrast_feature.t()) / self.temperature)
        pos = torch.sum(similarity * mask_pos, 1)
        neg = torch.sum(similarity * mask_neg, 1)
        loss = -(torch.mean(torch.log((pos + 1e-8) / (pos + neg + 1e-8)) / mask_pos.sum(1)))

        if torch.isinf(loss) or torch.isnan(loss):
            # print("similarity", similarity.size(), similarity)
            # print("pos similarity", pos.size(), pos)
            # print("neg similarity", neg.size(), neg)
            # print("mask_pos", mask_pos.size(), mask_pos)
            # print("mask_neg", mask_neg.size(), mask_neg)
            # print("mask", mask.size(), mask)
            # print("mask_sents", mask_sents.size(), mask_sents)
            # print("loss", loss)
            loss = torch.zeros_like(loss).to(device)
            # raise ValueError

        return loss


class GCL_Loss(nn.Module):

    def __init__(self, temperature=1, sim_type='cos'):
        super(GCL_Loss, self).__init__()
        self.temperature = temperature
        self.sim_type = sim_type

    def forward(self, features_1, features_2):
        # only consider the sentences in AC
        device = (torch.device('cuda') if features_1.is_cuda else torch.device('cpu'))

        batch_size = features_1.shape[0]

        assert len(features_1.shape) == 2


        if self.sim_type == "cos":
            features_1 = F.normalize(features_1, dim=-1)
            features_2 = F.normalize(features_2, dim=-1)

        logit_mask = torch.scatter(
            torch.ones([features_1.size(0), features_1.size(0)]).to(device),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        mask_pos = 1 - logit_mask

        similarity_12 = torch.exp(torch.mm(features_1, features_2.t()) / self.temperature)
        similarity_11 = torch.exp(torch.mm(features_1, features_1.t()) / self.temperature)
        similarity_22 = torch.exp(torch.mm(features_2, features_2.t()) / self.temperature)


        pos_1 = torch.sum(similarity_12 * mask_pos, 1)
        neg_2 = torch.sum(similarity_12 * logit_mask, 1) + torch.sum(similarity_11 * logit_mask, 1)
        loss_1 = -(torch.mean(torch.log((pos_1 + 1e-8) / (pos_1 + neg_2 + 1e-8)) / mask_pos.sum(-1)))

        pos_2 = torch.sum(similarity_12 * mask_pos, 1)
        neg_2 = torch.sum(similarity_12 * logit_mask, 1) + torch.sum(similarity_22 * logit_mask, 1)
        loss_2 = -(torch.mean(torch.log((pos_2 + 1e-8) / (pos_2 + neg_2 + 1e-8)) / mask_pos.sum(-1)))

        loss = (loss_1 + loss_2) * 0.5

        if torch.isinf(loss) or torch.isnan(loss):
            print("similarity_11", similarity_11.size(), similarity_11)
            print("similarity_22", similarity_22.size(), similarity_22)
            print("similarity_12", similarity_12.size(), similarity_12)
            print("mask_pos", mask_pos.size(), mask_pos)

            raise ValueError
            # loss = torch.zeros_like(loss).to(device)

        return loss


class GCL2_Loss(nn.Module):

    def __init__(self, temperature=1, sim_type='cos'):
        super(GCL2_Loss, self).__init__()
        self.temperature = temperature
        self.sim_type = sim_type

    def forward(self, features_1, features_2, mask):
        # only consider the sentences in AC
        device = (torch.device('cuda') if features_1.is_cuda else torch.device('cpu'))

        batch_size = features_1.shape[0]

        assert len(features_1.shape) == 2

        if self.sim_type == "cos":
            features_1 = F.normalize(features_1, dim=-1)
            features_2 = F.normalize(features_2, dim=-1)

        logit_mask = torch.scatter(
            torch.ones([features_1.size(0), features_1.size(0)]).to(device),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask_pos_12 = mask
        mask_pos_ = logit_mask * mask
        mask_neg_ = logit_mask * (1-mask)


        similarity_12 = torch.exp(torch.mm(features_1, features_2.t()) / self.temperature)
        similarity_11 = torch.exp(torch.mm(features_1, features_1.t()) / self.temperature)
        similarity_22 = torch.exp(torch.mm(features_2, features_2.t()) / self.temperature)

        pos_1 = torch.sum(similarity_12 * mask_pos_12, 1) + torch.sum(similarity_11 * mask_pos_, 1)
        neg_1 = torch.sum(similarity_12 * (1-mask_pos_12), 1) + torch.sum(similarity_11 * mask_neg_, 1)
        loss_1 = -(torch.mean(torch.log((pos_1 + 1e-8) / (pos_1 + neg_1 + 1e-8)) / (mask_pos_12.sum(-1) + mask_pos_.sum(-1))))

        pos_2 = torch.sum(similarity_12 * mask_pos_12, 1) + torch.sum(similarity_22 * mask_pos_, 1)
        neg_2 = torch.sum(similarity_12 * (1-mask_pos_12), 1) + torch.sum(similarity_22 * mask_neg_, 1)
        loss_2 = -(torch.mean(torch.log((pos_2 + 1e-8) / (pos_2 + neg_2 + 1e-8)) / (mask_pos_12.sum(-1)+mask_pos_.sum(-1))))

        loss = (loss_1 + loss_2) * 0.5

        if torch.isinf(loss) or torch.isnan(loss):
            print("similarity_11", similarity_11.size(), similarity_11)
            print("similarity_22", similarity_22.size(), similarity_22)
            print("similarity_12", similarity_12.size(), similarity_12)
            print("mask_pos_12", mask_pos_12.size(), mask_pos_12)
            print("mask_pos_", mask_pos_.size(), mask_pos_)
            print("mask_neg_", mask_neg_.size(), mask_neg_)


            raise ValueError
            # loss = torch.zeros_like(loss).to(device)

        return loss