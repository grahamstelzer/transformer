import torch
import torch.nn as nn

import math

# embeds layer
# sentence mapped to list of vectors by embeddings layer
# 512

"""
TODO: 
    - nn.Module used a LOT, still not sure fully what it does
    - other TODO tages scattered thru program for areas which
      I am significantly confused about
"""

class InputEmbeddings(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int): # dimension of model, num words in vocab
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) 

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


# position encodings
# vector to store the position data of a word in a sentence
# 512 also

class PositionalEncoding(nn.Module):    

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None: # dropout for overfitting??
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # matrix of shape (seq_len, d_model)
        # need vectors of size d_model, but need seq_len amount of them

        pe = torch.zeros(seq_len, d_model)

        # formula for positional encoding - see png or video from that 1 dude
        position = torch.arrange(0, seq_len, dtype=torch.float).unsqueeze(1) # len=seq_len - 1
        div_term = torch.exp(torch.arrange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # apply sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) # odd terms

        pe = pe.unsqueeze(0) # becomes tensor of shape (1, seq_len,d_model)

        self.register_buffer('pe', pe) # save position encodings as a buffer in the model

    def forward(self, x):

        # add posiitonal encoding to every word in sentence
        #   NOTE: requires_grad fixes tensors since positions should be the same within a 
        #   sentence and NOT learned through model run
        x = x + (self.pe[:, :x_shape[1], :]).requires_grad_(False) 
        return self.dropout(x) # NOTE: calls built in nn.Dropout 



class LayerNormalization(nn.Module):

    def __init__(self, eps: float=10**-6) -> None:
        super().__init__()
        self.eps = eps # on denom, see visual, avoids large x_mu and division by 0
        self.alpha = nn.Parameter(torch.ones(1)) # multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # added

    def forward(self, x):
        # calc mean and std dev
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias




class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        # tensor: (batch, seq length, d_model) --> (batch, seqlen, d_ff) --> (batch, seqlen, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))






class MultiHeadAttentionBlock(nn.Module):
    # see visual for better breakdown:
    def __init__(self, d_model: int, h: int, dropout: float) -> None: # h is num of heads
        super().__init__()
        self.d_model = d_model
        self.h = h

        # divide model int h heads, check its possible first
        assert d_model % h == 0, "d_model not divisble by h"

        self.d_k = d_model // h # d_k used in visual
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv

        self.w_o = nn.Linear(d_model, d_model) # Wo

        self.dropout = nn.Dropout(dropout)

    @staticmethod # can call without an instance of the class existing
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # TODO: this is the formula from slides, we need to figure out how the code actually works
        d_k = query.shape[-1]
        
        # (batch, h, seqlen, d_k) --> (batch, h, seqlen)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # @ is matrix multiply
        if mask is not None:
            # all values we want to mask, just replace with small values
            attention_scores.masked_fill_(mask == 0, -1e9) # replace all values for which mask==0 with -1e9

        attention_scores = attention_scores.softmax(dim = -1) # (batch, h, seqlen, seqlen)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        # will give small matrices for each head
        return (attention_scores @ value), attention_scores # second attention_scores used for visualizings (score given by model at current interaction)
    
    def forward(self, q, k, v, mask): # query, key, value
    # NOTE: mask: (see diagram)
    #             we do (with other steps) (Q * K) * V from the formula,
    #             but in the Q*K step, where we multiply each word by each other word
    #             if we do not want words to interact with each other,
    #             we multiply those in the matrix by a very small number
    #             - this works if we consider the numerator in the softmax function
    #               is e^x, so if x is very small number, e goes to 0, softmax zeros the word

        query = self.w_q(q) # (bactch, seqlen, dmodel) --> (batch, seqken, dmodel)
        # (seq by dmodel) * (dmodel by dmodel) = (seq by dmodel)
        key = self.w_k(k) # (bactch, seqlen, dmodel) --> (batch, seqken, dmodel)
        value = self.w_v(v) # (bactch, seqlen, dmodel) --> (batch, seqken, dmodel)


        # divide into smaller matrices to give each to a single head

        # TODO: figure out the view function
        # (batch, seqlen, dmodel) --> (batch, seqlen, h, d_k) --> (batch, h, seqlen, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).tranpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).tranpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).tranpose(1, 2)

        

        # calc attention using formula
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (batch, h , seq, d_k) --> (batch, seq, h, d_k) --> (batch, seq, de_model)
        x = x.tranpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k) # TODO: contiguous function?? "in place"

        # (batch, seq, d_model) --> (batch, seq, d_model)
        return self.w_o(x)


# connection in encoder after multihead, to add feed forward
# with skip connection to add norm
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        # add and norm?
        # TODO: where does sublayer come from
        return x + self.dropout(sublayer(self.norm(x)))





class EncoderBlock(nn.Module):
    
    def __init__(self, 
        self_attention_block: MultiHeadAttentionBlock, 
        feed_forward_block: FeedForwardBlock,
        dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)]) # TODO: ModuleList

    def forward(self, x, src_mask): # src_mask used to hide padding words from others

        # "x watching itself"
        # in encoder (different for decoder), query key value are the same
        # calling forward method of multiheadattentionblock:
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        
        return x


class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)
