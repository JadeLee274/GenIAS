from einops import rearrange
from exp.utils.common_import import *


class GRUCell(nn.Module):
    def __init__(
        self,
        d_in: int,
        num_units: int,
        n_nodes: int,
        concat_h: bool = False,
    ) -> None:
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        mpnn_channel = d_in*n_nodes+num_units if concat_h else d_in*n_nodes
        
        self.forget_gate = MPNN(
            c_in=mpnn_channel,
            c_out=num_units,
            concat_h=concat_h,
        )
        self.update_gate = MPNN(
            c_in=mpnn_channel,
            c_out=num_units,
            concat_h=concat_h
        )
        self.c_gate = MPNN(
            c_in=mpnn_channel,
            c_out=num_units,
            concat_h=concat_h
        )

    def forward(self, x: Tensor, h: Tensor, adj: Tensor) -> Tensor:
        r = self.sigmoid.forward(self.forget_gate.forward(x, h, adj))
        u = self.sigmoid.forward(self.update_gate.forward(x, h, adj))
        c = self.c_gate.forward(x, r * h, adj)
        c = self.tanh.forward(c)
        return u * h + (1.0 - u) * c


class MPNN(nn.Module):
    def __init__(self, c_in: int, c_out: int, concat_h: bool = True) -> None:
        super().__init__()
        self.concat_h = concat_h
        self.mlp = nn.Conv1d(c_in, c_out, kernel_size=1)
        
    def forward(self, x: Tensor, h: Tensor, graph: Tensor) -> Tensor:
        _, _, n = x.shape
        
        x_repeat = x[:, :, :, None].expand(-1, -1, -1, n)
        x_messages = torch.einsum('bcmn,bmn->bcmn', (x_repeat, graph))
        x_messages = rearrange(x_messages, 'b c m n -> b (c m) n')

        if self.concat_h:
            out = self.mlp(torch.cat([x_messages, h], dim=1))
        else:
            out = self.mlp(x_messages)
        
        return out


class LocalConv1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        n_nodes: int,
    ) -> None:
        super().__init__()
        self.out_channel = out_channels
        self.conv_list = nn.ModuleList([
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size
            ) for _ in range(n_nodes)
        ])
    
    def forward(self, x: Tensor) -> Tensor:
        b, _, n = x.shape
        out = torch.zeros((b, self.out_channel, n)).to(x.device)
        for i in range(n):
            x_local_in = x[..., i].unsqueeze(-1)
            x_local_out: Tensor = self.conv_list[i](x_local_in)
            out[..., i] = x_local_out.squeeze(-1)
        return out


class CUTS_Plus_Net(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        data_dim: int = 1,
        mlp_hid: int = 32,
        n_layers: int = 1,
        shared_weights_decoder: bool = False,
        concat_h: bool = True,
    ) -> None:
        super().__init__()
        n_nodes = n_nodes
        hidden_ch = mlp_hid

        self.in_ch = data_dim
        self.hidden_ch = mlp_hid
        self.n_layers = n_layers
        
        self.conv_encoder1 = nn.Conv1d(
            in_channels=hidden_ch,
            out_channels=hidden_ch,
            kernel_size=1,
        )
        self.conv_encoder2 = nn.Conv1d(
            in_channels=2*hidden_ch,
            out_channels=hidden_ch,
            kernel_size=1,
        )
        
        if shared_weights_decoder:
            self.decoder = nn.Conv1d(
                in_channels=2*hidden_ch,
                out_channels=data_dim,
                kernel_size=1,
            )
        else:
            self.decoder = LocalConv1D(
                in_channels=2*hidden_ch,
                out_channels=data_dim,
                kernel_size=1,
                n_nodes=n_nodes,
            )
        
        self.activation = nn.LeakyReLU()
        
        self.cells = nn.ModuleList()
        for i in range(self.n_layers):
            self.cells.append(
                GRUCell(
                    d_in=data_dim if i==0 else hidden_ch, 
                    num_units=hidden_ch, 
                    n_nodes=n_nodes,
                    concat_h=concat_h
                )
            )
                
        self.h0 = self.init_state(n_nodes=n_nodes)
        
        self.gt = nn.Parameter(torch.zeros(n_nodes, n_nodes))

    @property
    def causality_mtx(self):
        return torch.sigmoid(self.gt)
    
    def init_state(self, n_nodes: int) -> nn.ParameterList:
        h = []
        for _ in range(self.n_layers):
            h.append(
                nn.parameter.Parameter(
                    torch.zeros([self.hidden_ch, n_nodes])
                )
            )
        return nn.ParameterList(h)
    
    def update_state(
        self,
        x: Tensor,
        h: List[nn.Parameter],
        graph: Tensor,
    ) -> List[nn.Parameter]:
        rnn_in = x
        for layer in range(self.n_layers):
            rnn_in = h[layer] = self.cells[layer](rnn_in, h[layer], graph)
        return h
    
    def forward(self, x: Tensor, fwd_graph: Tensor) -> Tensor:
        x = rearrange(x, 'b n t -> b t n')
        bs, _, steps = x.shape
        h = [h_.expand(bs, -1, -1) for h_ in self.h0]
        pred = []

        for step in range(steps):
            x_now = x[..., step].unsqueeze(1)
            
            """Update state"""
            h = self.update_state(x_now, h, fwd_graph)
            h_now = h[-1]
            
            """Prediction"""
            x_repr = self.activation.forward(self.conv_encoder1.forward(h_now))
            x_repr = self.activation.forward(
                self.conv_encoder2.forward(torch.cat([x_repr, h_now], dim=1)))
            x_repr = torch.cat([x_repr, h_now], dim=1)
            x_hat2 = self.decoder(x_repr)
            pred.append(x_hat2)
        
        pred = torch.stack(pred, dim=-1)
        pred = rearrange(pred, 'b c n s -> b n s c')
        pred = pred.squeeze(-1)
        
        return pred[:, :, -1:]
    
    def init_cuts_plus(
        self,
        time: str,
        data:str,
        subdata: Optional[str],
    ) -> None:
        cuts_plus_dir = os.path.join(
            os.getcwd(), 'cuts_plus', 'checkpoints', data
        )

        if subdata is not None:
            cuts_plus_dir = os.path.join(cuts_plus_dir, subdata)
            
        cuts_plus_dir = os.path.join(cuts_plus_dir, time, 'best_model.pt')

        cuts_plus_ckpt = torch.load(cuts_plus_dir)
        self.load_state_dict(cuts_plus_ckpt['model'])

        return
