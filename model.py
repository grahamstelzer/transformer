import torch
import torch.nn as nn

import math

# embeds layer
# sentence mapped to list of vectors by embeddings layer
# 512

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


