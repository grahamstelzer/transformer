import torch
import torch.nn as nn

import math

# embeds layer
# sentence mapped to list of vectors by embeddings layer
# 512

"""
TODO: multiple TODO comments scattered throughout code represent 2 things (usually overlapping)
      1. a code snippet or function I dont quite understand
      2. something I am not sure how to turn into C++ yet
      
      below are a couple vocabulary terms that are helpful to know when looking at the code
"""


"""
TODO: nn.Module:

"""



"""
TODO: d_model:
      - think of as "dimension of model"
      - usually 512
      - ex. word embeddings are each a vector of size 512 -> which obviously decides other matrix sizes when walking through the model
"""




"""
TODO: dropout:
      - main point is to randomly select some elements of intermediate data representations and set them to 0 (say in the input tensor) 
      - this prevents overfitting and over-reliance on certian patterns
      - ex. applied to output of attention mechanism
      - ex. applied to token embeddigs + position embeddings before passing to transformer layer
      - dropout in the embedding layer:
        - randomly zeroes out parts of the embedding vector for each token, not the entire token embedding.
        - embedding vector is [0.1,0.2,0.3,0.4], dropout zeroes out 50% of it result is [0.1,0,0.3,0]
        - encourages the model to distribute information across the dimensions of the embedding vector instead of relying on specific ones
"""


"""
TODO: tensors:
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
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # len=seq_len - 1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # apply sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) # odd terms

        pe = pe.unsqueeze(0) # becomes tensor of shape (1, seq_len,d_model)

        self.register_buffer('pe', pe) # save position encodings as a buffer in the model

    def forward(self, x):

        # add posiitonal encoding to every word in sentence
        #   NOTE: requires_grad fixes tensors since positions should be the same within a 
        #   sentence and NOT learned through model run
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) 

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
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)

        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        

        # calc attention using formula
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (batch, h , seq, d_k) --> (batch, seq, h, d_k) --> (batch, seq, de_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k) # TODO: contiguous function?? "in place"

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
        # NOTE: each leyer is an EncoderBLock that we call their forward method in
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


# DECODER:
# NOTE: very similar to encoder, main difference is cross-attention:
#       - second mh attention calculation takes Query and Key from encoder


class DecoderBlock(nn.Module):
    
    def __init__(self, 
        self_attention_block: MultiHeadAttentionBlock, 
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float) -> None:

        super().__init__()

        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        
        # 3 connections? TODO: check (prolly on diagram)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    # NOTE/TODO: from tutorial, src and tgt are here because of translation task
    #            this is probably not exactly what we want since we will not always be translating
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask)) # NOTE: verify

        # NOTE: difference, query comes from DECoder, key and value from ENCder   + mask of encoder
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, x, x, src_mask)) # NOTE: verify

        x = self.residual_connections[2](x, self.feed_forward_block)

        return x


class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            # NOTE: each leyer is a Decoderblock that we call their forward method in
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)




# linear layer (last before output)
# output from mha is seq by dlayer (ignoring batch dimension)
# linear layer projects the embedding back into vocabulary



class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size) # TODO: make sure we know how to run a linear layer

    def forward(self, x):
        # (batch, seqlen, dmodel) --> (batch seqlen, vocabsize)
        # TODO: logarithmic softmax
        return torch.log_softmax(self.proj(x), dim = -1)








# transformer:

class Transformer(nn.Module):

    def __init__(self, 
            encoder: Encoder, 
            decoder: Decoder,
            src_embed: InputEmbeddings,
            tgt_embed: InputEmbeddings,
            src_pos: PositionalEncoding,
            tgt_pos: PositionalEncoding,
            projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer


    # methods: encode, decode, project
    # NOTE: instead of 1 forward, during inference, can reuse output of encoder instead of recaluclating.
    #       also helps for visualizing

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)


    def project(self, x):
        return self.projection_layer(x)




# given hyperparameters, build transformer:
# NOTE: considers translation, but can use for other tasks
#       so uses translation task names for things
def build_transformer(src_vocab_size: int, 
                      tgt_vocab_size: int,
                      src_seq_len: int,
                      tgt_seq_len: int,
                      d_model: int=512,
                      N: int=6, # number of layers (blocks), paper uses 6
                      h: int=8, # number of heads, paper uses 8
                      dropout: float=0.1,
                      d_ff=2048 # hidden feedforward layer TODO: check to make sure this makes sense
                      ) -> Transformer:

    # create embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # create positional encoding layers
    # TODO: possibly optimize this portion?
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # create encoder blocks:
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention = MultiHeadAttentionBlock(d_model, h, dropout)

        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)

        encoder_block = EncoderBlock(encoder_self_attention, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # create decoder blocks:
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)

        decoder_block = DecoderBlock(decoder_self_attention, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # create encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # intialize parameters
    # NOTE: normally random, can be decided to make training a bit faster
    # NOTE: in this case using xavier_uniform_
    for p in transformer.parameters(): # TODO: where does this function come from?? possibly from imported nn.Module stuff??
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer