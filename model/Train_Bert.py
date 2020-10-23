# -*- coding: utf-8 -*-

#from pathlib import Path
from transformers import AutoModelForQuestionAnswering, Trainer, TrainingArguments, AdamW
import torch
from torch.utils.data import DataLoader
#from torch.utils.data import IterableDataset
from transformers import AdamW
#import utils.preprocessing
import os
#from argparse import ArgumentParser
import csv
import pickle
from pathlib import Path
# Initialize wandb for logging
import wandb
wandb.login(key="06447813c3501a681da170acfe62a6f5aca4cf35")
wandb.init(project="Bert_ODQA")


class ODQA_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings2):
        self.encodings = encodings2

    def __getitem__(self, idx):
       # print(idx)
        #print("Lange des Encodingdicts",len(self.encodings))
        #blub = (self.encodings.items())
        #print("blub:",blub)
        #for key,val in blub:
         #print("Key,val length",key,len(val))
        #bla = csv.writer(open("output.csv", "w",encoding="utf-8"),delimiter=';')
        #bla.writerows(self.encodings.items())
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()} #<----------

    def __len__(self):
        return len(self.encodings.input_ids)


def training():
    # Input paths
    #dirpath = "../batch_output"  # args.out

   #DATASET_PATH_SEARCHQA = Path("/".join([dirpath, 'searchqa_train.pkl']))
    #SEARCHQA_VAL = Path("/".join([dirpath, 'searchqa_val.pkl']))
    #SEARCHQA_TEST = Path("/".join([dirpath, 'searchqa_test.pkl']))
    #DATASET_PATH_QUASAR = Path("/".join([dirpath, 'quasar_train_short.pkl']))
    #QUASAR_TEST = Path("/".join([dirpath, 'quasar_test_short.pkl']))
    #QUASAR_DEV = Path("/".join([dirpath, 'quasar_dev_short.pkl']))
    batches = {}
    #trainset_path = os.listdir(Path("./batch_output/train"))
    #devset_path = os.listdir(Path("./batch_output/dev"))
    #valset_path = os.listdir(Path("./batch_output/val"))
    paths = ["train","test"]
    for path in paths:
        type_path = "/".join(["./batch_output",path])
        batches[type_path] = (os.listdir(type_path))

    #test_set_list = [QUASAR_DEV,QUASAR_TEST,SEARCHQA_TEST,SEARCHQA_VAL]

    #Init model
    model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-uncased',
                                                     return_dict=True)
    for tipe, files in batches.items():
        # Open Pickled file
        for batch in files:
            # Open file from the list of training files e.g. quasar_long_train_1.pkl, searchqa_train_1.pkl
            with open("/".join([tipe, batch]), 'rb') as infile:
                print("Loading File: " + str(batch))
                encodings = pickle.load(infile)
            print("Loaded File")

            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            print("Start Training: " + str(batch))
            # Train on Dataset

            model.to(device)
            model.train()
            # open list of dataset to train on
            for encoding in encodings:

                train_dataset = ODQA_Dataset(encoding)
                #train_dataset.__getitem__()
                train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

                optim = AdamW(model.parameters(), lr=5e-5)

                for epoch in range(2):
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
                        wandb.log({'optimization':optim})
                        optim.step()

                wandb.log({"Endcodedfiles":encoding})

                # model.eval()
            print("Training Batch Done")
        print("Training for "+str(tipe)+ " Done")
    print("saving model to file")
    model.save_pretrained("./saved_model")
    print("saved model")

    '''training_args = TrainingArguments(
        output_dir='./training_output',  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=1,  # batch size per device during training
        per_device_eval_batch_size=30,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
    )

        for encoding in encodings:
           # print(encoding)
            train_dataset = ODQA_Dataset(encoding)
        #val_dataset = ODQA_Dataset(encodings)
            print("Load Trainer")
            trainer = Trainer(
                model=model,  # the instantiated 🤗 Transformers model to be trained
                args=training_args,  # training arguments, defined above
                train_dataset=train_dataset,  # training dataset
                #eval_dataset=val_dataset  # evaluation dataset
            )
            print("Start Training")
            trainer.train()
    '''




'''if __name__ == '__main__':
    training()
    print("Training Done")
    parser = ArgumentParser(description='Training script')
    parser.add_argument('--out', default='/local/anasbori/outputs', type=str, help='Path to output directory')

    # Parse given arguments
    args = parser.parse_args()
    print(args.out)'''

