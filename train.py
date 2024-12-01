# using huggingface en-it from tutorial
# will hopefully be able to change this
# TODO: figure out if there are C++ libraries that can do some of these things
#       since i personally dont want to code a Tokenizer (yet)

"""
TODO: when running validation or even visuals, its suggested we train the model for more than a couple hours first
      we should verify that the model receives the weights from previous runs when training, since i dont know if its suggested to just run train.py multiple times.
"""


"""
TODO: when we convert this to C++ and especially when we use this for other projects, we should comment or label wherever we use target and source texts.
      that way when we change the input from text to any other form of data, we can simply find the inputs and outputs and make sure they are the same
      form for the tokenizer
      TODO: double check the tokenizer can work on other things? check its input?

"""



import warnings # should watch at least once, otherwise sort of unecessary (CUDA warnings)

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_weights_file_path, get_config

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer # create vocab from list of sentences
from tokenizers.pre_tokenizers import Whitespace # split words according to whitespace

from pathlib import Path # create absolute paths given relative paths

# TODO: probably substitute these
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm


# VALIDATION LOOP (visualization)

# NOTE: run greedy decoding on model, run encoder only once
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # precompute encoder output, reuse for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)

    # initialize decoder with SOS
    # TODO: .empty(), .fill_(), .type_as(), .to()
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device) # NOTE: two dimenions for batch and num tokens for decoder input

    # now keep asking the decoder to output the next token until EOS token or max_len reached
    # so decoder output will become next input of next step
    while True:

        if decoder_input.size(1) == max_len: # check number of tokens in second dimension (input to decoder)
            break
        
        # build mask for the target (decoder input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device) # dont want input to watch future words

        # calc output of decoder
        # reuse encoder_output each iteration
        out = model.decoder(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token:
        prob = model.project(out[:, -1])
        # get token with max probability using greedy search
        _, next_word = torch.max(prob, dim=1)

        # append
        # TODO: should maybe output each step of this just to visualize
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break

    # TODO: squeeze function
    return decoder_input.squeeze(0) # squeeze to remove batch dimension


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2):

    model.eval() # tells pytorch that we will evaluate model 
    count = 0
    
    # NOTE: uncomment next couple lines if using writer library 
    # look at two sentences and check output:
    source_texts = []
    expected = []
    predicted = []
    
    # size of control window
    console_width = 80

    # TODO: torch.no_grad()
    with torch.no_grad(): # removes gradient calculation since we do not want to train, just check
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            # compare to expected label

            # TODO: double check batch is the right datatype?
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]

            # to get out text, must use tokenizer to convert tokens into text, use tgt_tokenizer since this the target language
            # TODO: detach(), cpu(), numpy()
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # since we are using tqdm, printing with base python print might mess with the progress bar
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break


    # TODO: not critical, see github example for writer with more functionality


# END VALIDATION THINGS

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

        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)

        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer



# load dataset and build tokenizer:
def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train') # NOTE: defines split from HF, but we will define ourselves

    # build tokenizers:
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # split 90/10 for train/validate
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    
    # TODO: random_split from pytorch splits dataset in two based on train_ds_size and val_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])


    # NOTE: parts after this done following completion of dataset.py
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # also watch the max sentence length
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        # load each sentence from src and tgt, convert to ids, check the length, add a bit more than max for seqlen
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
 # TODO: double check ids (i forget where they came from)
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    
    print(f'Max len of src sentence: {max_len_src}')
    print(f'Max len of tgt sentence: {max_len_tgt}')

        
    # TODO: Dataloader from torch utils
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1) # validaiton: batch_size=1 to process each sentence seperately
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt



def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

# NOTE: reduce number of heads or layer if too much for GPU






# built after config.py was created
def train_model(config):
    # define device to put the tnesors on:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    # verify weights folder created:
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device) # TODO: check get_vocab_size() method and .to() method

    # TODO: Tensorboard: visualize loss
    writer = SummaryWriter(config['experiment_name'])

    # TODO: Adam optimizer (double check understanding on how optimizers work)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # preload if necessary (in case crash or something)
    initial_epoch = 0
    global_step = 0
    if config['preload']: 
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    # NOTE: loss function: cross entropy
    #       - ignore padding, smooth confidence to prevent overfitting ("take 0.1 percent of confidence score and give to others")
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)


    # actual training loop:
    for epoch in range(initial_epoch, config['num_epochs']):

        batch_iterator = tqdm(train_dataloader, desc=f'processing epoch {epoch:02d}') # progress bar
        # TODO batch_iterator: I'm not sure how this works
        for batch in batch_iterator:

            model.train() # change to inside this loop so each time validation run, put model back into training loop

            # get tensors
            encoder_input = batch['encoder_input'].to(device) # (batch, seqlen)
            decoder_input = batch['decoder_input'].to(device) # (batch, seqlen)
            encoder_mask = batch['encoder_mask'].to(device) # (batch, 1, 1, seqlen) NOTE: only hide padding tokens
            decoder_mask = batch['decoder_mask'].to(device) # (batch, 1, seqlen, seqlen)  NOTE: hid padding AND subsequent tokens

            # run tensors through transformer:
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch, seqlen, dmodel)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch, seqlen, dmodel)
            # must map back to vocabulary
            proj_output = model.project(decoder_output) # (batch, seqlen, tgt_vocab_size)

            # now we have output from model, must compare it to label
            label = batch['label'].to(device) # (batch, seqlen)

            # (batch, seqlen, tgt_vocab_size) -> (batch * seqlen, tgt_vocab_size)
            # because we want to compare to label
            # TODO: understand loss function inputs and how it works, may need to write own version
            #       ...nn.CrossEntropyLoss(...)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)) # TODO .view function

            # update/show loss progress bar:
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # log loss on tensorboard
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()


            # backprop loss:
            loss.backward()

            # update weights (job of optimizer)
            # TODO: write an optimizer functions
            optimizer.step()
            optimizer.zero_grad()

            # NOTE: uncomment to watch model train at every step, must make sure model is trained otherwise will give just big strings of commmas as predicted
            # run_validation(model, val_dataloader, tokenizer, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer) # NOTE writer not used unless with better writer libray (see validation/ghub)

            global_step += 1

        # run outside of each epoch so model gets trained
        # NOTE: see inference.ipynb for quick example
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer) # NOTE writer not used unless with better writer libray (see validation/ghub)

        

        # save model every epoch
        model_filename = get_weights_file_path(config, f'{epoch:02}')
        
        # extremely necessary to save snapshots with large vocab, 
        # otherwise optimzer always starts from 0 when figuring out how to move each weight, even if starting from another epoch and 
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(), # NOTE: all the weights of the model
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step      
        }, model_filename)




if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)