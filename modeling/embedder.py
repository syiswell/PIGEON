import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from termcolor import colored


class Embedder(nn.Module):

    def __init__(self, bertModel):
        super(Embedder, self).__init__()
        self.bert = bertModel
        print(colored(f"""[Embedder INFO]: 
        Embedder: cls embedder
        """, 'yellow'))
    
    def forward(self, tokens, attn_masks):
        batch_size, num_sents, max_sent_len = tokens.size()
        input_ids = tokens.view(-1, max_sent_len)
        input_mask = attn_masks.view(-1, max_sent_len)
        bert_outputs = self.bert(input_ids, input_mask)  # bert_feature: (batch_size*max_sent_num, max_sent_len, bert_feature_dim)
        bert_token_feature = bert_outputs.last_hidden_state
        
        bert_feature = bert_token_feature[:, 0, :].view(batch_size, num_sents, -1)
        return bert_token_feature.view(batch_size, num_sents, max_sent_len, -1), bert_feature


class TokenEmbedder(nn.Module):

    def __init__(self, bertModel, cfg):
        super(TokenEmbedder, self).__init__()
        self.bert = bertModel
        if cfg.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.num_layers = cfg.num_embedding_layer
        self.lstm_token = nn.LSTM(input_size=cfg.bert_output_size,
                                  hidden_size=cfg.hidden_size//2,
                                  num_layers=cfg.num_embedding_layer,
                                  batch_first=True,
                                  bidirectional=True)
        self.drop_lstm = nn.Dropout(p=0.5)
        print(colored(f"""[Embedder INFO]: 
        Embedder: token embedder
        Freeze Parameters: {cfg.freeze_bert}
        """, 'yellow'))
    
    def forward(self, tokens, attn_masks, num_tokens):
        batch_size, num_sents, max_sent_len = tokens.size()
        input_ids = tokens.view(-1, max_sent_len) # [batch_size, max_sent_num, max_seq_len]
        input_mask = attn_masks.view(-1, max_sent_len)
        bert_outputs = self.bert(input_ids, input_mask) # bert_feature: (batch_size*max_sent_num, max_sent_len, bert_feature_dim)
        bert_token_feature = bert_outputs.last_hidden_state
        
        bert_feature_flatten = bert_token_feature[:, 1:, :]
        num_tokens_flatten = num_tokens.view(-1)
        sorted_num_tokens, tokenIdx = num_tokens_flatten.sort(0, descending=True)
        _, recover_token_idx = tokenIdx.sort(0, descending=False)
        sorted_sent_emb_tensor_flatten = bert_feature_flatten[tokenIdx]
        packed_tokens = pack_padded_sequence(sorted_sent_emb_tensor_flatten, sorted_num_tokens.cpu(), True)
        _, (h_n, _) = self.lstm_token(packed_tokens, None)
        h_n = self.drop_lstm(h_n)
        h_n = h_n.view(self.num_layers, 2, batch_size*num_sents, -1)
        lstm_embedding = torch.cat((h_n[-1, 0],h_n[-1, 1]), dim=1) # (batch_size*num_sents, bert_feature_dim)
        
        return bert_token_feature.view(batch_size, num_sents, max_sent_len, -1),\
               lstm_embedding[recover_token_idx].view(batch_size, num_sents, -1)