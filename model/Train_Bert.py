from model.ODQA_Dataset import ODQA_Dataset
from transformers import AutoModelForQuestionAnswering
from torch.utils.data import DataLoader
from transformers import AdamW
import utils.preprocessing
import os
import torch
import pickle

def load_pickled_glove(GLOVE_PATH):
    return pickle.load(open(f'../outputs/glove_dict.pkl', 'rb'))

def main(train_encodings,val_encodings):
    model = AutoModelForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad',
                                                          return_dict=True)
    train_dataset = ODQA_Dataset(train_encodings)
    val_dataset = ODQA_Dataset(val_encodings)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

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
            loss.backward()
            optim.step()

    model.eval()


if __name__ == '__main__':

    infile = open(filename, 'rb')
    new_dict = pickle.load(infile)
    infile.close()
    main()