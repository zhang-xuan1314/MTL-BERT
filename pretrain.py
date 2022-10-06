import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from model import  BertModel
from dataset import Smiles_Bert_Dataset,Pretrain_Collater
import time
import os
from torch.utils.data import DataLoader
from metrics import AverageMeter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--Smiles_head', nargs='+', default=["CAN_SMILES"], type=str)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

small = {'name': 'small', 'num_layers': 4, 'num_heads': 4, 'd_model': 128, 'path': 'small_weights'}
medium = {'name': 'medium', 'num_layers': 8, 'num_heads': 8, 'd_model': 256, 'path': 'medium_weights'}
large = {'name': 'large', 'num_layers': 12, 'num_heads': 12, 'd_model': 576, 'path': 'large_weights'}

arch = medium     ## small 3 4 128   medium: 6 6  256     large:  12 8 516
num_layers = arch['num_layers']
num_heads =  arch['num_heads']
d_model =  arch['d_model']

dff = d_model*4
vocab_size = 60
dropout_rate = 0.1

model = BertModel(num_layers=num_layers,d_model=d_model,dff=dff,num_heads=num_heads,vocab_size=vocab_size)
model.to(device)

# data = pd.read_csv('data/chem.csv')

full_dataset = Smiles_Bert_Dataset('data/chem.csv',Smiles_head=args.Smiles_head)

train_size = int(0.98 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset,batch_size=512,shuffle=True,collate_fn=Pretrain_Collater())
test_dataloader = DataLoader(test_dataset,batch_size=512,shuffle=False,collate_fn=Pretrain_Collater())

optimizer = optim.Adam(model.parameters(),1e-4,betas=(0.9,0.98))

loss_func = nn.CrossEntropyLoss(ignore_index=0,reduction='none')

train_loss = AverageMeter()
train_acc = AverageMeter()
test_loss = AverageMeter()
test_acc = AverageMeter()

def train_step(x, y, weights):
    model.train()
    optimizer.zero_grad()
    predictions = model(x)
    loss = (loss_func(predictions.transpose(1,2),y)*weights).sum()/weights.sum()
    loss.backward()
    optimizer.step()

    train_loss.update(loss.detach().item(),x.shape[0])
    train_acc.update(((y==predictions.argmax(-1))*weights).detach().cpu().sum().item()/weights.cpu().sum().item(),
                     weights.cpu().sum().item())


def test_step(x,y, weights):
    model.eval()
    with torch.no_grad():
        predictions = model(x)
        loss = (loss_func(predictions.transpose(1, 2), y) * weights).sum()/weights.sum()

        test_loss.update(loss.detach(), x.shape[0])
        test_acc.update(((y == predictions.argmax(-1)) * weights).detach().cpu().sum().item()/weights.cpu().sum().item(),
                              weights.cpu().sum().item())

for epoch in range(100):
    start = time.time()

    for (batch, (x, y, weights)) in enumerate(train_dataloader):
        train_step(x, y, weights)

        if batch%500==0:
            print('Epoch {} Batch {} training Loss {:.4f}'.format(
                epoch + 1, batch, train_loss.avg))
            print('traning Accuracy: {:.4f}'.format(train_acc.avg))

        if batch % 1000 == 0:
            for x, y ,weights in test_dataloader:
                test_step(x, y , weights)
            print('Test loss: {:.4f}'.format(test_loss.avg))
            print('Test Accuracy: {:.4f}'.format(test_acc.avg))

            test_acc.reset()
            test_loss.reset()
            train_acc.reset()
            train_loss.reset()

    print('Epoch {} is Done!'.format(epoch))
    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    print('Epoch {} Training Loss {:.4f}'.format(epoch + 1, train_loss.avg))
    print('training Accuracy: {:.4f}'.format(train_acc.avg))
    print('Epoch {} Test Loss {:.4f}'.format(epoch + 1, test_loss.avg))
    print('test Accuracy: {:.4f}'.format(test_acc.avg))
    torch.save(model.state_dict(),'weights/' + arch['path']+'_bert_weights{}_{}.pt'.format(arch['name'],epoch+1) )
    torch.save(model.encoder.state_dict(), 'weights/' + arch['path'] + '_bert_encoder_weights{}_{}.pt'.format(arch['name'], epoch + 1))
    print('Successfully saving checkpoint!!!')


