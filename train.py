# using huggingface en-it from tutorial
# will hopefully be able to change this
# TODO: figure out if there are C++ libraries that can do some of these things
#       since i personally dont want to code a Tokenizer (yet)

import torch
import torch.nn as nn

from torch.utils.data import Dataset, Dataloader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer # create vocab from list of sentences
from tokenizers.pre_tokenizers import Whitespace # split words according to whitespace


from pathlib import Path # create absolute paths given relative paths


def get_all_sentences(ds, lang):
    # NOTE: for this example, each item is a pair of sentences, english:italian
    for item in ds:
        yield item['translation'][lang] # extracts the translation of the language that we want



# build tokenizer:
# NOTE: see huggingface tokenizer tutorial, most of this is done already
# TODO: recoding this is reinventing the wheel. hopefully C++ has a better option
def get_or_build_tokenizer(config, ds, lang): # cfg defined elsewhere, dataset, language
    tokenizer_path = Path(config['tokenizer_file'].format(lang)) # file for tokenizer, ex: = "../tokenizers/tokenizer_{0}.json"
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]')) # unseen word, map to "number"(vector??) corresponding to that word
        tokenizer.pre_tokenizer = Whitespace()

        # word level trainer (splits based on whitespace as opposed to say, parts of a word)
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)

        tokenizer = train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)

        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer



# load dataset and build tokenizer:
def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config[lang_src]}-{config[lang_tgt]}', split='train') # NOTE: defines split from HF, but we will define ourselves

    # build tokenizers:
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # split 90/10 for train/validate
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    
    # TODO: random_split from pytorch splits dataset in two based on train_ds_size and val_ds_size
    train_ds_size, val_ds_size = random_split(ds_raw, [train_ds_size, val_ds_size])

