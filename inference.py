import torch
import muspy
import argparse

import numpy as np

from data.encode import get_encoding, extract_notes, encode_notes, decode_notes, reconstruct, chunk_music

from model.model import TransformerModel

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--file', type=str)
    parser.add_argument('--model', type=str)

    parser.add_argument('--task', type=str, default='add_voice', choices=['add_voice', 'continue'])
    parser.add_argument('--tokens', type=int, default=20)

    parser.add_argument('--temperature', type=float, default=1.0)

    parser.add_argument("--embedding_size", type=int, default=16)
    parser.add_argument("--head_size", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=16)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)

    return parser.parse_args()

def add_voice_tokens(tokens, encoding, n, new_voice = None):
    current_voices = np.max(tokens[:, encoding['dimensions'].index('instrument')])
    
    if new_voice is None:
        new_voice = current_voices + 1

    start_of_notes_index = np.where(tokens[:, encoding['dimensions'].index('type')] == encoding['type_code_map']['start-of-notes'])[0][0]
    new_voice_token = np.array([encoding['type_code_map']['instrument'], 0, 0, 0, 0, new_voice])

    tokens = np.insert(tokens, start_of_notes_index, new_voice_token, axis=0)

    new_voice_notes = [encoding['type_code_map']['note']] + encoding['n_tokens'][1:-1] + [new_voice]
    new_voice_notes = np.array([new_voice_notes] * n)

    end_of_notes_index = np.where(tokens[:, encoding['dimensions'].index('type')] == encoding['type_code_map']['end-of-song'])[0][0]
    tokens = np.insert(tokens, end_of_notes_index, new_voice_notes, axis=0)

    return tokens, n

def continue_tokens(tokens, encoding, n):
    current_voices = np.max(tokens[:, encoding['dimensions'].index('instrument')])

    for i in range(current_voices):
        tokens, _ = add_voice_tokens(tokens, encoding, n, new_voice=i)

    return tokens, n * current_voices

def convert_output(input, output, encoding, tokens):
    output = output[0].detach().numpy()

    for i in range(-tokens-1, -1):
        start = 0

        for j in range(1, len(encoding['dimensions']) - 1):
            size = encoding['n_tokens'][j] + 1

            input[i, j] = np.argmax(output[i, start:start+size])

            start += size   

    return input

def temperature_scaled_softmax(logits, temperature=1.0):
    logits = logits / temperature
    return torch.softmax(logits, dim=1)

def main():
    args = parse_args()

    encoding = get_encoding()

    midi = muspy.read_midi(args.file)
    midi.adjust_resolution(encoding['resolution'])

    input = encode_notes(extract_notes(midi, encoding['resolution']), encoding)

    if args.task == 'add_voice':
        input, tokens_added = add_voice_tokens(input, encoding, args.tokens) 
    elif args.task == 'continue':   
        input, tokens_added = continue_tokens(input, encoding, args.tokens) 

    input = torch.tensor(input).unsqueeze(0)

    transformer = TransformerModel(
        embedding_size=args.embedding_size,
        head_size=args.head_size,
        hidden_size=args.hidden_size,
        layers=args.layers,
        dropout=args.dropout,
    )

    transformer.load_state_dict(torch.load(args.model), strict=False)

    transformer.eval()

    output, _ = transformer(input, mask=None)
    output = temperature_scaled_softmax(output, args.temperature)

    output = convert_output(input[0].numpy(), output, encoding, tokens_added)

    reconstructed = reconstruct(decode_notes(output, encoding), encoding['resolution'])
    reconstructed.write_midi('reconstructed.mid')

if __name__ == '__main__':
    main()