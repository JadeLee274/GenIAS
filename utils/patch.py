from utils.common_import import *
from utils.alias import *


def patch(
    x: Matrix,
    x_tilde: Matrix,
    tau: float,
) -> Matrix:
    _, data_dim = x.shape
    x_tilde_patched = torch.empty_like(x) # torch.emty_like(x_tilde)

    for d in data_dim:
        x_d = x[:, d]
        x_tilde_d = x_tilde[:, d]
        deviation_d = torch.sum((x_d - x_tilde_d) ** 2)
        amplitude_d = torch.max(x_d) - torch.min(x_d)
        if deviation_d > tau * amplitude_d:
            x_tilde_patched[:, d] = x_tilde[:, d]
        else:
            x_tilde_patched[:, d] = x[:, d]

    return x_tilde_patched
