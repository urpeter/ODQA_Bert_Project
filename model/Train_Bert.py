from pathlib import Path

from model.ODQA_Dataset import ODQA_Dataset
from transformers import AutoModelForQuestionAnswering
from torch.utils.data import DataLoader
from transformers import AdamW
import utils.preprocessing
import os
from argparse import ArgumentParser
import torch
import pickle
# Initialize wandb for logging
import wandb
wandb.init(project="Bert_ODQA")

#TODO load training encodings
#def load_pickled_glove(GLOVE_PATH):
 #   return pickle.load(open(f'../outputs/glove_dict.pkl', 'rb'))

def main(train_encodings,val_encodings):
    model = AutoModelForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad',
                                                          return_dict=True)
    train_dataset = ODQA_Dataset(train_encodings)
    val_dataset = ODQA_Dataset(val_encodings)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

    optim = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions,
                            end_positions=end_positions)
            loss = outputs[0]
            wandb.log({'training loss (extraction)': loss})
            loss.backward()
            optim.step()

    model.eval()


if __name__ == '__main__':
    parser = ArgumentParser(description='Training script')
    parser.add_argument('--out', default='/local/anasbori/outputs', type=str, help='Path to output directory')

    # Parse given arguments
    args = parser.parse_args()
    print(args.out)

    # Input paths
    dirpath = args.out
    SEARCHQA_VAL = Path("/".join([dirpath, 'searchqa_val.pkl']))
    QUASAR_DEV = Path("/".join([dirpath, 'quasar_dev_short.pkl']))

    infile = open(QUASAR_DEV, 'rb')
    trainset,valset = pickle.load(infile)
    infile.close()
    main(trainset,valset)