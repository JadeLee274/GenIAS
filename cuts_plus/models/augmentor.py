from cuts_plus.utils.imports import *
from cuts_plus.models.cuts_plus import CUTS_Plus_Net


class PositiveAugmentor(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        data_dim: int,
        input_step: int,
        pred_step: int,
        noise_level: float,
    ) -> None:
        super().__init__()
        self.input_step = input_step
        self.pred_step = pred_step
        self.causal_discoverer = CUTS_Plus_Net(
            n_nodes=n_nodes,
            data_dim=data_dim,
        )
        self.noise_level = noise_level

        return
    
    def forward(self, x: Tensor, causality_mtx: Tensor) -> Tensor:
        x_pos = x.clone()
        causality_mtx = (causality_mtx > 0.5).int()
        start_node = random.choice(range(len(causality_mtx)))
        effects = [
            i for i, val in enumerate(causality_mtx[start_node]) if val == 1
        ]
        self.start_node = start_node
        self.effects = effects
		
        x_pos[:, self.input_step, start_node] \
        += torch.randn(x.size(0)) * self.noise_level

        with torch.no_grad():
            graph = (self.causal_discoverer.causality_mtx > 0.5).float()
            graph = graph[None].expand(x.size(0), -1, -1)
            x_out = self.causal_discoverer.forward(
                x=x_pos[:, :self.input_step],
                fwd_graph=graph,
            ).transpose(1, 2)
        
        x_pos[:, self.pred_step, effects] = x_out[:, self.pred_step, :]

        return x_pos
	

class NegativeAugmentor(nn.Module):
    def __init__(
        self,
        cutoff_probability: float = 0.1,
        disturb_all: bool = False,
    ) -> None:
        super().__init__()
        self.cutoff_probability = cutoff_probability
        self.disturb_all = disturb_all
        self.bias_candidates = (
            -0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5
        )
        self.percent = 0.5

    @torch.no_grad()
    def get_indices_to_disturb(self, causality_mtx: Tensor) -> List[int]:
        if self.disturb_all:
            indices = list(range(len(causality_mtx)))
            return indices
        
        def dfs(node, visited, subgraph):
            visited[node] = True
            subgraph.append(node)
            for neighbor, is_causal in enumerate(causality_mtx[node]):
                if is_causal and not visited[neighbor]:
                    if random.random() > self.cutoff_probability:
                        dfs(neighbor, visited, subgraph)
                    else:
                        break
        
        visited = [False] * len(causality_mtx)
        subgraph = []
        
        start_node = random.choice(range(len(causality_mtx)))
        dfs(start_node, visited, subgraph)
        
        indices = subgraph

        return indices
    
    @torch.no_grad()
    def disturb(self, x: Tensor, indices: List[int]) -> Tensor:
        transform_instance = AddBias(
            bias_candidates=self.bias_candidates,
            percent=self.percent,
        )
        disturbed_x = transform_instance.forward(x, indices)

        return disturbed_x

    def forward(self, x: Tensor, causality_mtx: Tensor) -> Tensor:
        assert causality_mtx.dtype == torch.float, \
        "causality_mtx elements must be of type float"
        causality_mtx = (causality_mtx > 0.5).int()
        indices_to_disturb = self.get_indices_to_disturb(causality_mtx)
        disturbed_x = self.disturb(x, indices_to_disturb)
        self.indices_to_disturb = indices_to_disturb

        assert not disturbed_x.requires_grad, \
        "disturbed_x should not require gradients"
        return disturbed_x.clone().detach()
    

class AddBias(nn.Module):
    def __init__(
        self,
        bias_candidates: Tuple[float],
        percent: float,
    ) -> None:
        super(AddBias, self).__init__()
        self.bias_candidates = bias_candidates
        self.percent = percent

    def forward(self, ts: Tensor, indices: List[int]):
        n_steps = int((ts.shape[1] * self.percent))
        ts_bias = ts.clone().detach()
        
        steps = np.random.choice(ts.shape[1], n_steps, replace=False)
        bias = torch.tensor(
            random.choices(
                data=self.bias_candidates,
                k=len(ts)*n_steps*len(indices)
            ),
            device=ts.device,
        ).view(len(ts), n_steps, len(indices))
        
        for i, step in enumerate(steps):
            ts_bias[:, step, indices] += bias[:, i, :]
        
        return ts_bias
