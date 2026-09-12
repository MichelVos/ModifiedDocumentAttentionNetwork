#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#  This software is a computer program written in Python whose purpose is 
#  to recognize text and layout from full-page images with end-to-end deep neural networks.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.

from torch.nn.functional import log_softmax
from torch.nn import AdaptiveMaxPool2d, Conv1d, Conv2d
from torch.nn import Module, AdaptiveMaxPool2d, LogSoftmax, AdaptiveAvgPool2d
import torch.nn as nn
import torch
import math

class Decoder_Transformer(Module):
    def __init__(self, params):
        super(Decoder_Transformer, self).__init__()

        self.vocab_size = params["vocab_size"]
        self.enc_size = params["enc_size"]
        self.hidden_size = params.get("hidden_size", 256)
        self.num_layers = params.get("num_layers", 2)
        self.num_heads = params.get("num_heads", 4)
        self.dropout = params.get("dropout", 0.1)

        # (H, W) → (1, W)
        self.ada_pool = AdaptiveMaxPool2d((1, None))

        # Linear projection to model dimension (in case enc_size != hidden_size)
        self.input_proj = nn.Linear(self.enc_size, self.hidden_size)

        # Transformer encoder layers for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=self.dropout,
            batch_first=True,  # (B, T, C)
            activation="gelu",
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Final projection to vocab
        self.classifier = nn.Linear(self.hidden_size, self.vocab_size + 1)
        self.log_softmax = LogSoftmax(dim=2)

    def forward(self, x):
        """
        Args:
            x: encoder output (B, C, H, W)
        Returns:
            log_probs: (T, B, vocab_size+1) for CTC loss
        """
        # Collapse height
        x = self.ada_pool(x).squeeze(2)  # (B, C, W)
        x = x.permute(0, 2, 1)           # (B, W, C)

        # Project to model dimension
        x = self.input_proj(x)           # (B, W, hidden_size)

        pe = sinusoidal_positional_encoding(x.size(1), self.hidden_size, x.device)
        x = x + pe
        # Positional encoding
        #positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        #pos_enc = torch.sin(positions / (10000 ** (torch.arange(0, self.hidden_size, 2, device=x.device) / self.hidden_size)))
        #pos_enc = torch.stack((pos_enc, torch.cos(positions / (10000 ** (torch.arange(0, self.hidden_size, 2, device=x.device) / self.hidden_size)))), dim=-1)
        #pos_enc = pos_enc.view(1, x.size(1), -1)
        #if pos_enc.size(-1) > self.hidden_size:
        #    pos_enc = pos_enc[..., :self.hidden_size]
        #x = x + pos_enc  # add sinusoidal position encoding

        # Transformer encoding
        x = self.transformer(x)          # (B, W, hidden_size)

        # Classify each timestep
        x = self.classifier(x)           # (B, W, vocab_size+1)

        # Log-softmax for CTC
        log_probs = self.log_softmax(x)  # (B, W, vocab_size+1)

        # Transpose for nn.CTCLoss compatibility
        return log_probs.permute(0, 2, 1)

def sinusoidal_positional_encoding(seq_len, hidden_size, device):
    positions = torch.arange(seq_len, device=device).unsqueeze(1)  # [W, 1]
    div_term = torch.exp(torch.arange(0, hidden_size, 2, device=device) * (-math.log(10000.0) / hidden_size))  # [hidden_size/2]
    pe = torch.zeros(seq_len, hidden_size, device=device)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    return pe.unsqueeze(0)  # [1, W, hidden_size]

class Decoder_BiLSTM(Module):
    def __init__(self, params):
        super(Decoder_BiLSTM, self).__init__()

        self.vocab_size = params["vocab_size"]
        self.enc_size = params["enc_size"]
        self.hidden_size = params.get("hidden_size", 256)
        self.num_layers = params.get("num_layers", 2)
        self.dropout = params.get("dropout", 0.3)

        # (H, W) → (1, W)
        self.ada_pool = AdaptiveMaxPool2d((1, None))

        # Bidirectional LSTM for temporal context
        self.lstm = nn.LSTM(
            input_size=self.enc_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=True,
            batch_first=True
        )

        # Linear projection to vocab size (+1 for blank)
        self.classifier = nn.Linear(self.hidden_size * 2, self.vocab_size + 1)

        # For stability with CTC
        self.log_softmax = LogSoftmax(dim=2)

    def forward(self, x):
        """
        Args:
            x: encoder output of shape (B, C, H, W)
        Returns:
            log_probs: (T, B, vocab_size+1) for CTC loss
        """
        # Collapse height (usually 1)
        x = self.ada_pool(x).squeeze(2)  # (B, C, W)
        x = x.permute(0, 2, 1)  # (B, W, C)

        # BiLSTM temporal modeling
        x, _ = self.lstm(x)  # (B, W, 2*hidden_size)

        # Classify each timestep
        x = self.classifier(x)  # (B, W, vocab_size+1)

        # Log-softmax for CTC
        log_probs = self.log_softmax(x)  # (B, W, vocab_size+1)

        # Transpose to (T, B, vocab_size+1) as expected by nn.CTCLoss
        return log_probs.permute(0, 2, 1)

class RowwiseAttentionPool(Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.attn_conv = Conv2d(channels, 1, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0))

    def forward(self, x):
        # x: [B, C, H, W]
        attn = torch.softmax(self.attn_conv(x), dim=2)   # attention weights over height
        out = torch.sum(x * attn, dim=2)                 # weighted sum over height
        return out  # [B, C, W]

class DecoderRowWise(Module):
    def __init__(self, params):
        super(DecoderRowWise, self).__init__()

        self.vocab_size = params["vocab_size"]
        self.row_attn = RowwiseAttentionPool(params["enc_size"])
        self.end_conv = Conv1d(in_channels=params["enc_size"], out_channels=self.vocab_size + 1, kernel_size=1)

    def forward(self, x):
        x = self.row_attn(x)
        x = self.end_conv(x)
        return log_softmax(x, dim=1)    

class Decoder(Module):
    def __init__(self, params):
        super(Decoder, self).__init__()

        self.vocab_size = params["vocab_size"]
        #self.ada_pool = AdaptiveMaxPool2d((1, None))
        self.ada_pool = AdaptiveAvgPool2d((1, None))
        self.end_conv = Conv1d(in_channels=params["enc_size"], out_channels=self.vocab_size+1, kernel_size=1)

    def forward(self, x):
        x = self.ada_pool(x).squeeze(2)
        x = self.end_conv(x)
        return log_softmax(x, dim=1)
'''
class Decoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        C = params["enc_size"]
        V = params["vocab_size"] + 1  # blank at V-1
        self.temporal = nn.Sequential(
            nn.Conv1d(V, 256, 3, padding=1), # V was C
            nn.GELU(),
            nn.GroupNorm(32, 256),
            nn.Dropout(0.1),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(32, 256),
            nn.Dropout(0.1),
        )
        self.out = nn.Conv1d(256, V, 1)

    def forward(self, x):             # x: [B,C,H,W]
        x = x.mean(dim=2)             # [B,C,W] ; T=W
        x = self.temporal(x)          # [B,256,W]
        x = self.out(x)               # [B,V,W]
        x = x.permute(2,0,1)          # [T,B,V]
        return x.log_softmax(dim=2)
'''
class CTCtopR(nn.Module):
    def __init__(self, params): #, input_size, rnn_cfg, nclasses, rnn_type='gru'):
        super(CTCtopR, self).__init__()
        self.params = params
        nclasses = params["vocab_size"]  + 1  # blank at nclasses-1
        input_size = params["enc_size"]
        #input_size = (self.params["config"]["max_size"]["max_height"], self.params["config"]["max_size"]["max_width"]) 
        hidden, num_layers = params["rnn_hidden_size"], params["rnn_layers"]

        if params["rnn_type"] == 'gru':

            self.rec = nn.GRU(input_size, hidden, num_layers=num_layers, bidirectional=True, dropout=.2)
        elif params["rnn_type"] == 'lstm':
            self.rec = nn.LSTM(input_size, hidden, num_layers=num_layers, bidirectional=True, dropout=.2)
        else:
            print('problem! - no such rnn type is defined')
            exit()
        
        self.fnl = nn.Sequential(nn.Dropout(.2), nn.Linear(2 * hidden, nclasses))
    '''
    def forward(self, x):

        y = x #.permute(2, 3, 0, 1)[0]
        y = self.rec(y)[0]
        y = self.fnl(y)

        return log_softmax(y, dim=1)
    '''
    def forward(self, x):
        # x: (B, C, H, W)
        
        # Collapse height and channels into one feature dimension,
        # and treat width as sequence length
        y = x.permute(3, 0, 1, 2)   # (W, B, C, H)
        y = y.flatten(2)             # (W, B, C*H)

        # Pass through RNN
        y, _ = self.rec(y)           # (W, B, 2*hidden)
        
        # Classification layer
        y = self.fnl(y)              # (W, B, nclasses)
        r = log_softmax(y, dim=2)    # expects T, B, C  receives  B  C  T
        return r.permute(1, 2, 0)
