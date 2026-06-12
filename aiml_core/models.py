import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for unsupervised reconstruction of sensor signals.
    The reconstruction error represents abnormal deviation (degradation)
    which is inverted to compute the Hybrid Health Index.
    """
    def __init__(self, input_dim=14, hidden_dim=8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        _, (h_n, _) = self.encoder(x)
        # h_n shape: (1, batch, hidden_dim)
        seq_len = x.size(1)
        # Repeat hidden state for decoder
        h_repeated = h_n.repeat(seq_len, 1, 1).transpose(0, 1)
        decoded, _ = self.decoder(h_repeated)
        return decoded


class BayesianLSTM(nn.Module):
    """
    LSTM Prognostics model incorporating Monte Carlo Dropout for
    predicting remaining useful life (RUL) with uncertainty quantification.
    """
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1, dropout_rate=0.25):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, mc_dropout=True):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Extract last timestep state
        last_out = lstm_out[:, -1, :]
        
        # Apply dropout. If mc_dropout is True, force dropout to be active
        # regardless of whether the model is in eval() mode.
        if mc_dropout:
            out = F.dropout(last_out, p=self.dropout_rate, training=True)
        else:
            out = self.dropout(last_out)
            
        return self.linear(out)
