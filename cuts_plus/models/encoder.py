from cuts_plus.utils.imports import *


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        num_vars: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        batch_first: bool = True
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_vars,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=batch_first,
        )

        return

    def forward(self, x):
        batch_size = x.size(0)
        o, (h, c) = self.lstm(x)  # o: (batch_size, time_step, hidden_dim), h: (num_layers, batch_size, hidden_dim), c: (num_layers, batch_size, hidden_dim)
        h = h.permute((1, 0, 2))  # h: (batch_size, num_layers, hidden_dim)
        enc_out = torch.reshape(h, (batch_size, -1))  # h: (batch_size, num_layers * hidden_dim)

        return enc_out
    

class GRUEncoder(nn.Module):
    def __init__(
        self,
        num_vars: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        batch_first: bool = True,
    ) -> None:
        super().__init__()

        self.gru = nn.GRU(
            input_size=num_vars,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=batch_first,
        )

    def forward(self, x):
        batch_size = x.size(0)
        o, h = self.gru(x)  # o: (batch_size, time_step, hidden_dim), h: (num_layers, batch_size, hidden_dim), c: (num_layers, batch_size, hidden_dim)
        h = h.permute((1, 0, 2))  # h: (batch_size, num_layers, hidden_dim)
        enc_out = torch.reshape(h, (batch_size, -1))  # h: (batch_size, num_layers * hidden_dim)

        return enc_out
    