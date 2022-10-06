import pandas as pd
import numpy as np
import torch
import rdkit
from rdkit import Chem
import re

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

smiles_regex_pattern = r'Si|Mg|Ca|Fe|As|Al|Cl|Br|[#%\)\(\+\-1032547698:=@CBFIHONPS\[\]icosn]|/|\\'

smiles_str2num = {'<PAD>':0 ,'Cl': 1, 'Br': 2, '#': 3, '(': 4, ')': 5, '+': 6, '-': 7, '0': 8, '1':  9,
    '2': 10,'3': 11,'4':12,'5':13,'6':14,'7':15,'8':16,'9':17,':':18,'=':19,'@':20,'C':21,
    'B':22,'F':23,'I':24,'H':25,'O':26,'N':27,'P':28,'S':29,'[':30,']':31,'c':32,'i':33,'o':34,
    'Si': 35, 'Mg': 36, 'Ca': 37, 'Fe': 38, 'As': 39, 'Al': 40,
    'n':41,'p':42,'s':43,'%':44,'/':45,'\\':46,'<MASK>':47,'<UNK>':48,'<GLOBAL>':49,'<p1>':50,
    '<p2>':51,'<p3>':52,'<p4>':53,'<p5>':54,'<p6>':55,'<p7>':56,'<p8>':57,'<p9>':58,'<p10>':59}

smiles_num2str =  {i:j for j,i in smiles_str2num.items()}

smiles_char_dict = list(smiles_str2num.keys())

def randomize_smile(sml):
    m = Chem.MolFromSmiles(sml)
    ans = list(range(m.GetNumAtoms()))
    np.random.shuffle(ans)
    nm = Chem.RenumberAtoms(m, ans)
    smiles = Chem.MolToSmiles(nm, canonical=False)
    return smiles

def canonical_smile(sml):
    m = Chem.MolFromSmiles(sml)
    smiles = Chem.MolToSmiles(m, canonical=True)
    return smiles

class Smiles_Bert_Dataset(Dataset):
    def __init__(self, path, Smiles_head):
        if path.endswith('txt'):
            self.df = pd.read_csv(path,sep='\t')
        else:
            self.df = pd.read_csv(path)

        self.data = self.df[Smiles_head].to_numpy().reshape(-1).tolist()
        self.vocab = smiles_str2num
        self.devocab = smiles_num2str

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        smiles = self.data[item]
        x, y, weights = self.numerical_smiles(smiles)
        return x, y, weights

    def numerical_smiles(self, smiles):
        nums_list = self._char_to_idx(smiles)

        choices = np.random.permutation(len(nums_list) - 1)[:int(len(nums_list) * 0.15)] + 1
        y = np.array(nums_list).astype('int64')
        weight = np.zeros(len(nums_list))
        for i in choices:
            rand = np.random.rand()
            weight[i] = 1
            if rand < 0.8:
                nums_list[i] = 48
            elif rand < 0.9:
                nums_list[i] = int(np.random.rand() * 46 + 0.1)

        x = np.array(nums_list).astype('int64')
        weights = weight.astype('float32')
        return x, y, weights

    def _char_to_idx(self, seq):
        char_list = re.findall(smiles_regex_pattern, seq)
        char_list = ['<GLOBAL>'] + char_list
        return [self.vocab.get(char_list[j],self.vocab['<UNK>']) for j in range(len(char_list))]


class Prediction_Dataset(object):
    def __init__(self, df, smiles_head='SMILES', reg_heads=[],clf_heads=[]):

        self.df = df
        self.reg_heads = reg_heads
        self.clf_heads = clf_heads

        self.smiles = self.df[smiles_head].to_numpy().reshape(-1).tolist()

        self.reg = np.array(self.df[reg_heads].fillna(-1000)).astype('float32')
        self.clf = np.array(self.df[clf_heads].fillna(-1000)).astype('int32')

        self.vocab = smiles_str2num
        self.devocab = smiles_num2str

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        smiles = self.smiles[item]

        properties = [None, None]
        if len(self.clf_heads)>0:
            clf = self.clf[item]
            properties[0] = clf

        if len(self.reg_heads)>0:
            reg = self.reg[item]
            properties[1] = reg

        nums_list = self._char_to_idx(seq=smiles)
        if len(self.reg_heads) + len(self.clf_heads) >0:
            ps = ['<p{}>'.format(i+1) for i in range(len(self.reg_heads) + len(self.clf_heads))]
            nums_list = [smiles_str2num[p] for p in ps] + nums_list
        x = np.array(nums_list).astype('int32')
        return x, properties

    def numerical_smiles(self, smiles):
        smiles = self._char_to_idx(seq=smiles)
        x = np.array(smiles).astype('int64')
        return x


    def _char_to_idx(self, seq):
        char_list = re.findall(smiles_regex_pattern, seq)
        char_list = ['GLOBAL'] + char_list
        return [self.vocab.get(char_list[j],self.vocab['<UNK>']) for j in range(len(char_list))]

class Pretrain_Collater():
    def __init__(self):
        super(Pretrain_Collater, self).__init__()
    def __call__(self,data):
        xs, ys, weights = zip(*data)

        xs = pad_sequence([torch.from_numpy(np.array(x)) for x in xs], batch_first=True).long().to(device)
        ys = pad_sequence([torch.from_numpy(np.array(y)) for y in ys], batch_first=True).long().to(device)
        weights = pad_sequence([torch.from_numpy(np.array(weight)) for weight in weights], batch_first=True).float().to(device)

        return xs, ys, weights


class Finetune_Collater():
    def __init__(self,args):
        super(Finetune_Collater, self).__init__()
        self.clf_heads = args.clf_heads
        self.reg_heads = args.reg_heads

    def __call__(self, data):
        xs, properties_list = zip(*data)
        xs = pad_sequence([torch.from_numpy(np.array(x)) for x in xs], batch_first=True).long().to(device)
        properties_dict = {'clf':None,'reg':None}

        if len(self.clf_heads) >0:
           properties_dict['clf'] = torch.from_numpy(np.concatenate([p[0].reshape(1,-1) for p in properties_list],0).astype('int32')).to(device)

        if len(self.reg_heads) >0:
           properties_dict['reg'] = torch.from_numpy(np.concatenate([p[1].reshape(1,-1) for p in properties_list],0).astype('float32')).to(device)

        return xs, properties_dict
