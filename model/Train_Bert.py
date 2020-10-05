from pathlib import Path
from model.ODQA_Dataset import ODQA_Dataset
from transformers import AutoModelForQuestionAnswering, Trainer, TrainingArguments
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

    DATASET_PATH_SEARCHQA = Path("/".join([dirpath, 'searchqa_train.pkl']))
    SEARCHQA_VAL = Path("/".join([dirpath, 'searchqa_val.pkl']))
    SEARCHQA_TEST = Path("/".join([dirpath, 'searchqa_test.pkl']))
    DATASET_PATH_QUASAR = Path("/".join([dirpath, 'quasar_train_short.pkl']))
    QUASAR_TEST = Path("/".join([dirpath, 'quasar_test_short.pkl']))
    QUASAR_DEV = Path("/".join([dirpath, 'quasar_dev_short.pkl']))

    batches = os.listdir("./batch_output")
    test_set_list = [QUASAR_DEV,QUASAR_TEST,SEARCHQA_TEST,SEARCHQA_VAL]

    #Init model
    model = AutoModelForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad',
                                                     return_dict=True)
    training_args = TrainingArguments(
        output_dir='./training_output',  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=30,  # batch size per device during training
        per_device_eval_batch_size=30,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
    )

    for batch in batches:
        # Open Pickled file
        infile = open(batch, 'rb')
        encodings = pickle.load(infile)
        infile.close()

        train_dataset = ODQA_Dataset(encodings)
        #val_dataset = ODQA_Dataset(encodings)

        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            #eval_dataset=val_dataset  # evaluation dataset
        )

        trainer.train()

        '''device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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

    #model.eval()'''
        print("Training Done")
    return model

'''if __name__ == '__main__':
    training()
    print("Training Done")
    parser = ArgumentParser(description='Training script')
    parser.add_argument('--out', default='/local/anasbori/outputs', type=str, help='Path to output directory')

    # Parse given arguments
    args = parser.parse_args()
    print(args.out)'''

