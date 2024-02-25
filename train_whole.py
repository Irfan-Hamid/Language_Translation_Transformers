from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path
from dataset import causal_mask_with_future

import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter


def greedy_decode_whole(model_causal_mask, model_causal_mask_with_future, source, source_mask, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Initialize the decoder input with the SOS token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    # Precompute the encoder output to reuse it for every step
    encoder_output = model_causal_mask.encode(source, source_mask)

    while decoder_input.size(1) < max_len:
        # Step 1: Extend the sequence using causal mask with model_causal_mask
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model_causal_mask.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model_causal_mask.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)

        if next_word.item() == eos_idx:
            decoder_input = torch.cat([decoder_input, next_word.unsqueeze(0)], dim=1)
            break

        # Add the new token to the sequence using your suggested concatenation method
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        # Step 2: Refine the previous token using the new token as future context with model_causal_mask_with_future
        if decoder_input.size(1) > 2:  # Ensure there's at least one token to refine
            # Use the last two tokens (previous token and the new token) for refinement
            refinement_segment = decoder_input[:, -2:]
            refinement_mask = causal_mask_with_future(refinement_segment.size(1)).type_as(source_mask).to(device)
            refinement_out = model_causal_mask_with_future.decode(encoder_output, source_mask, refinement_segment, refinement_mask)
            refinement_prob = model_causal_mask_with_future.project(refinement_out[:, -2])  # Get probabilities for the token before the last
            _, refined_word = torch.max(refinement_prob, dim=1)

            # Update the second last token with the refined prediction
            decoder_input[:, -2] = refined_word

    return decoder_input.squeeze(0)



def validate_train_model_whole(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted_whole = []  # Predictions using greedy_decode_whole

    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out_whole = greedy_decode_whole(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_whole_text = tokenizer_tgt.decode(model_out_whole.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted_whole.append(model_out_whole_text)

            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED WHOLE: ':>12}{model_out_whole_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break

    if writer:
        # Evaluate the character error rate for predictions using greedy_decode_whole
        metric = torchmetrics.CharErrorRate()
        cer_whole = metric(predicted_whole, expected)
        writer.add_scalar('validation cer whole', cer_whole, global_step)
        writer.flush()

        # Compute the word error rate for predictions using greedy_decode_whole
        metric = torchmetrics.WordErrorRate()
        wer_whole = metric(predicted_whole, expected)
        writer.add_scalar('validation wer whole', wer_whole, global_step)
        writer.flush()

        # Compute the BLEU metric for predictions using greedy_decode_whole
        metric = torchmetrics.BLEUScore()
        bleu_whole = metric(predicted_whole, expected)
        writer.add_scalar('validation BLEU whole', bleu_whole, global_step)
        writer.flush()  


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # It only has the train split, so we divide it overselves
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

def train_model_causal_mask(config,current_epoch):
    config['experiment_name'] = "runs/tmodel_causal_mask"  # Unique experiment name for this model
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"Device memory: {torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    epoch = current_epoch
    torch.cuda.empty_cache()
    model.train()

    print(f"Epoch {epoch}: Training with causal mask")

    batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch:02d} - Training with causal mask")
    for batch in batch_iterator:

        encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
        decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
        encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
        decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

        # Run the tensors through the encoder, decoder and the projection layer
        encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
        decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
        proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

        # Compare the output with the label
        label = batch['label'].to(device) # (B, seq_len)

        # Compute the loss using a simple cross entropy
        loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
        batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

        # Log the loss
        writer.add_scalar('train loss/causal_mask', loss.item(), global_step)
        writer.flush()

        # Backpropagate the loss
        loss.backward()

        # Update the weights
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1

        # # Run validation at the end of every epoch using the whole training approach
        # validate_train_model_whole(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch, indicating it's been trained with both causal masks
        model_filename = get_weights_file_path(config, f"causal_mask_epoch_{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

    return model   

def train_model_causal_mask_with_future(config, current_epoch):
    config['experiment_name'] = "runs/tmodel_causal_mask_with_future"  # Unique experiment name for this model
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # Tensorboard
    writer = SummaryWriter(f"{config['experiment_name']}_with_future")  # Modify experiment name

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    epoch = current_epoch
    print(f"Epoch {epoch}: Training with causal mask with future")
    batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch:02d} - Training with causal mask with future")
    for batch in batch_iterator:
        encoder_input = batch['encoder_input'].to(device)
        decoder_input = batch['decoder_input'].to(device)
        encoder_mask = batch['encoder_mask'].to(device)
        decoder_mask = batch['decoder_mask_with_future'].to(device)

        # Run the tensors through the encoder, decoder, and the projection layer
        encoder_output = model.encode(encoder_input, encoder_mask)
        decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        proj_output = model.project(decoder_output)

        # Compare the output with the label
        label = batch['label'].to(device)

        # Compute the loss using a simple cross entropy
        loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
        batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

        # Log the loss
        writer.add_scalar('train loss/causal_mask_with_future', loss.item(), global_step)
        writer.flush()

        # Backpropagate the loss
        loss.backward()

        # Update the weights
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1

        # Run validation after training for all epochs
        validate_train_model_whole(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model after training for all epochs
        model_filename = get_weights_file_path(config, f"causal_mask_with_future_epoch_{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

    return model

def alternate_training(config, num_epochs):
    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch+1}/{num_epochs}")

        # Train one epoch with causal mask
        print(f"Training epoch {epoch+1} with causal mask")
        train_model_causal_mask(config, epoch)

        # Train one epoch with causal mask and future context
        print(f"Training epoch {epoch+1} with causal mask and future context")
        train_model_causal_mask_with_future(config, epoch)

        print(f"Completed Epoch {epoch+1}/{num_epochs}")




if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    num_epochs = config['num_epochs']
    alternate_training(config, num_epochs)

