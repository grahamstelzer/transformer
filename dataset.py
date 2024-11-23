# create the tensors the model will use
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        # NOTE: special methods to build tensor for SOS, EOS, PAD
        #       they build tensors with specific values referring to SOS, EOS, PAD
        # TODO: Tensor()
        # TODO: token_to_id()
        self.sos_token = torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])], dtype=torch.int64) # NOTE: dtype as long since vocab can be more than 32 bit
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id(['[EOS]'])], dtype=torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id(['[PAD]'])], dtype=torch.int64)


    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: Any) -> Any:
        # first get original pair from HF dataset
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # tokenizer splits sentence into single words then maps each word to its corresponding number in the vocabulary

        # gives input ids so numbers of each word in sentence (as an array)
        # TODO: probably print this, not quite sure how to picture it
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # NOTE: must pad sentence to reach seqlen since model works on fixed seqlen
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # NOTE: -2 spaces for SOS and EOS tokens
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # NOTE: only add EOS token for label side

        # NOTE: must make sure chosen seq_len is long enough to represent all sentences in dataset
        #       if not, raise exception (padding tokens should never become negative)
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')


        """
        BUILDING TENSORS FOR INPUT TO ENC AND DEC:
        1. input tensor to enc
        2. input tensor to dec
        3. label/target for dec (expected output for dec)
        """

        # add SOS, EOS and padding to enc input
        encoder_input = torch.cat(
            {
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            }
        )

        # add only SOS and padding to dec input
        decoder_input = torch.cat(
            {
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            }
        )

        # add only EOS and padding to label
        label = torch.cat(
            {
                torch.tensor(dec_input_tokens, dtype=torch.int64)
                self.eos_token
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            }
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            # (seq_len):
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,

            # encoder mask:
            # NOTE: consider that we are adding padding tokens to the sequences
            #       we do not want these tokens to be seen in training
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            # unsqueeze to add seq dimensions and to add batch dimension
            # TODO: unsqueeze function

            # decoder mask:
            # NOTE: causal mask: each word only should look at previous words and non-padding words
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() 
                            &  # binary and
                            causal_mask(decoder_input,size(0)), # (1, seq_len) & (1, seq_len, seq_len)
            
            # "label": label
        }


def causal_mask: