import warnings
from cuts_plus.utils.imports import *


def gumbel_softmax(
    logits: Tensor,
    tau: float = 1.0,
    hard: bool = False,
    eps: float = 1e-10,
    dim: int = -1
) -> Tensor:
    if eps != 1e-10:
        warnings.warn("'eps' parameter is deprecated and has no effect.")

    gumbels = (
        -torch.empty_like(
            input=logits,
            memory_format=torch.legacy_contiguous_format,
        ).exponential_().log()
    )  # ~ Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~ Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            input=logits,
            memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft

    return ret


# def gumbel_sigmoid_sample(
#     graph: Tensor,
#     batch_size: int,
#     tau: float = 1.0
# ) -> Tensor:
#     probabilty = graph[None, :, :, None].expand(batch_size, -1, -1, -1)
#     logits = torch.concat([probabilty, (1 - probabilty)], axis=-1)
#     samples = gumbel_softmax(logits=logits, tau=tau, hard=True)[:, :, :, 0]

#     return samples


# def sample_bernoulli(sample_matrix: Tensor, batch_size: int) -> Tensor:
#     sample_matrix = sample_matrix[None].expand(batch_size, -1, -1)
#     return torch.bernoulli(sample_matrix).float()
