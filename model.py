import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class PostionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PostionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :].unsqueeze(0)

def make_src_mask(src,src_pad_idx=0):
    src_mask = (src == src_pad_idx).unsqueeze(1).unsqueeze(2).to(device)
    return src_mask

def make_trg_mask(trg,trg_pad_idx=0):
    trg_pad_mask = (trg == trg_pad_idx).unsqueeze(1).unsqueeze(3)
    trg_len = trg.shape[1]
    trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor)
    trg_mask = trg_pad_mask & trg_sub_mask
    return trg_mask.to(device)

class ScaledDotProductAttention(nn.Module):
    def __init__(self,dropout_rate=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, Q, K, V, attn_mask=None):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(K.shape[-1])  # scores : [batch_size, n_heads, len_q, len_k]

        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = F.softmax(scores,dim=-1)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads,rate):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=rate)
        self.dot_product_attention = ScaledDotProductAttention(dropout_rate=rate)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.num_heads, self.depth).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.num_heads, self.depth).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.num_heads, self.depth).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = self.dot_product_attention(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.d_model) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        output = self.dropout(output)
        return self.layernorm(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model,dff, rate):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.GELU(),
            nn.Linear(dff, d_model)
        )
        self.dropout = nn.Dropout(p=rate)
        self.layernorm = nn.LayerNorm(d_model)
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        output = self.dropout(output)
        return self.layernorm(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self,d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model,num_heads,rate)
        self.pos_ffn = PoswiseFeedForwardNet(d_model,dff,rate)

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, num_layers,d_model,num_heads,dff,input_vocab_size,maximum_position_encoding=200,rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.src_emb = nn.Embedding(input_vocab_size, d_model)
        self.pos_emb = PostionalEncoding(d_model,200,device=device)
        self.layers = nn.ModuleList([EncoderLayer(d_model,num_heads,dff,rate) for _ in range(num_layers)])
        self.dropout = nn.Dropout(rate)

    def forward(self, enc_inputs):

        # nn.TransformerEncoder
        '''
        enc_inputs: [batch_size, src_len]
        '''
        word_emb = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        pos_emb = self.pos_emb(enc_inputs) # [batch_size, src_len, d_model]
        enc_outputs = word_emb + pos_emb
        enc_self_attn_mask = make_src_mask(enc_inputs) # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class EncoderForPrediction(nn.Module):
    def __init__(self, num_layers,d_model,num_heads,dff,input_vocab_size,maximum_position_encoding=200,rate=0.1,prediction_nums=0):
        super(EncoderForPrediction, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.prediction_nums = prediction_nums
        self.src_emb = nn.Embedding(input_vocab_size, d_model)
        self.pos_emb = PostionalEncoding(d_model,300,device=device)
        self.layers = nn.ModuleList([EncoderLayer(d_model,num_heads,dff,rate) for _ in range(num_layers)])
        self.dropout = nn.Dropout(rate)

    def forward(self, enc_inputs):

        # nn.TransformerEncoder
        '''
        enc_inputs: [batch_size, src_len]
        '''
        word_emb = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        pos_emb = self.pos_emb(enc_inputs[:,self.prediction_nums:]) # [batch_size, src_len, d_model]

        enc_outputs = word_emb

        enc_outputs[:,self.prediction_nums:] += pos_emb

        enc_self_attn_mask = make_src_mask(enc_inputs) # [batch_size, src_len, src_len]

        enc_self_attn_mask =  enc_self_attn_mask.repeat(1, self.num_heads, enc_self_attn_mask.shape[-1], 1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # enc_self_attn_mask[:,:,:,:self.prediction_nums]=1
        # enc_self_attn_mask[:, :, torch.arange(self.prediction_nums), torch.arange(self.prediction_nums)] = 0

        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class BertModel(nn.Module):
    def __init__(self,num_layers = 6,d_model = 256,dff = 512,num_heads = 8,vocab_size = 50,dropout_rate = 0.1):
        super(BertModel, self).__init__()
        self.encoder = Encoder(num_layers=num_layers,d_model=d_model,
                        num_heads=num_heads,dff=dff,input_vocab_size=vocab_size,maximum_position_encoding=200,rate=dropout_rate)
        self.fc1 = nn.Linear(d_model,d_model*2)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(d_model*2,vocab_size)


    def forward(self,x):
        x,attns = self.encoder(x)
        y = self.fc1(x)
        y = self.dropout1(y)
        y = F.gelu(y)
        y = self.fc2(y)
        return y


class PredictionModel(nn.Module):
    def __init__(self,num_layers = 6,d_model = 256,dff = 512,num_heads = 8,vocab_size = 60,dropout_rate = 0.1, reg_nums=0,clf_nums=0):
        super(PredictionModel, self).__init__()

        self.reg_nums = reg_nums
        self.clf_nums = clf_nums

        self.encoder = EncoderForPrediction(num_layers=num_layers,d_model=d_model,
                        num_heads=num_heads,dff=dff,input_vocab_size=vocab_size,
                               maximum_position_encoding=200,rate=dropout_rate,
                               prediction_nums=self.reg_nums+self.clf_nums)



        self.fc_list = nn.ModuleList()

        for i in range(self.clf_nums+self.reg_nums):
            self.fc_list.append(nn.Sequential(nn.Linear(d_model,2*d_model),
                                              nn.Dropout(0.1),nn.LeakyReLU(0.1),
                                              nn.Linear(2*d_model,1)))

        # self.fc1 = nn.Linear(d_model,2*d_model)
        # self.dropout1 = nn.Dropout(0.1)
        # self.fc2 = nn.Linear(2*d_model,1)

    def forward(self,x):
        x,attns = self.encoder(x)

        ys = []

        for i in range(self.clf_nums+self.reg_nums):
            y = self.fc_list[i](x[:,i])
            ys.append(y)

        y = torch.cat(ys,dim=-1)

        # y = self.fc1(x)
        # y = self.dropout1(y)
        # y = self.fc2(y)
        # y = y.squeeze(-1)
        properties = {'clf':None,'reg':None}
        if self.clf_nums>0:
            clf = y[:,:self.clf_nums]
            properties['clf'] = clf
        if self.reg_nums>0:
            reg = y[:,self.clf_nums:]
            properties['reg'] = reg
        return properties

