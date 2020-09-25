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

# Returns the model
def training():

    # Input paths
    dirpath = "../outputs"  # args.out
    SEARCHQA_VAL = Path("/".join([dirpath, 'searchqa_val.pkl']))
    QUASAR_DEV = Path("/".join([dirpath, 'quasar_dev_short.pkl']))
    # Open Pickled file
    infile = open(QUASAR_DEV, 'rb')
    encodings = pickle.load(infile)
    infile.close()
    #Init model
    model = AutoModelForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad',
                                                     return_dict=True)
    train_dataset = ODQA_Dataset(encodings)
    #val_dataset = ODQA_Dataset(encodings)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Start Training")
    # Train on Dataset
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

    #model.eval()

    return model

if __name__ == '__main__':
    training()
    print("Training Done")
    '''parser = ArgumentParser(description='Training script')
    parser.add_argument('--out', default='/local/anasbori/outputs', type=str, help='Path to output directory')

    # Parse given arguments
    args = parser.parse_args()
    print(args.out)'''

