from math import log, sqrt
from .common_import import *
"""
Codes for loss function of TCN-VAE. This code follows the paper
GenIAS: Generator for Instantiating Anomalies in Time Series,
Darban et al. 2025.

Paper link: https://arxiv.org/pdf/2502.08262

Default values of arguments follow the paper.
"""

################################ TCN-VAE loss ################################

mseloss = nn.MSELoss()


def mse_loss(x: Tensor, x_hat: Tensor) -> Tensor:
    return torch.mean((x - x_hat)**2, dim=[1, 2])


def recon_loss(x: Tensor, x_hat: Tensor) -> Tensor:
    """
    Reconstruction loss.

    Parameters:
        x:     Original data.
        x_hat: Reconstructed data.

    Returns:
        MSE loss between x and x_hat.
    """
    return mse_loss(x, x_hat).mean(dim=0)


def pert_loss(
    x: Tensor,
    x_hat: Tensor,
    x_tilde: Tensor,
    delta_min: float = 0.1,
    delta_max: float = 0.2,
) -> Tensor:
    """
    Perturbation loss of VAE.

    Parameters:
        x:         Original data, i.e., anchor.
        x_hat:     Reconstrucded data, without anomalies injected.
        x_tilde:   Reconstructed data, with anomalies injected.

        delta_min: Minmum threshold of recon(x, x_hat) - recon(x, x_tilde). 
                   Default 0.1.

        delta_max: Maxmum threshold of recon(x, x_tilde). Default 0.2.

    Returns:
        Perturbation loss.
    """
    return (F.relu(mse_loss(x, x_hat) - mse_loss(x, x_tilde) + delta_min) \
    + F.relu(mse_loss(x, x_tilde) - delta_max)).mean(dim=0)


def zero_pert_loss(x: Tensor, x_tilde: Tensor) -> Tensor:
    """
    Zero perturbatino loss of vAE.
    Follows the equation (4) of the paper.

    Parameters:
        x:       Input of VAE.
        x_tilde: Perturbed output of VAE.

    Returns:
        Zero perturbation loss.
    """
    assert x.shape[0] == x_tilde.shape[0] # batch size

    ind = torch.where(torch.isclose(x, torch.zeros_like(x)).all(dim=1))
    loss = 0.0
    count = 0

    for i in torch.unique(ind[0]):
        sub_count = (ind[0] == i).sum().item()
        sub_loss = mseloss(
            x[i, :, ind[1][count:count+sub_count]],
            x_tilde[i, :, ind[1][count:count+sub_count]]
        )
        loss += (sub_loss + 1) ** -1
        count += sub_count
    
    loss /= x.shape[0]

    return loss


def kld_loss(
    mu: Tensor,
    logvar: Tensor,
    prior_var: float,
) -> Tensor:
    """
    Modified KL-Divergence loss of VAE.
    Follows the equation (5) of the paper.
    The existence of logvar_prior is justified in the theorem that is proved 
    at the appendix of the paper.

    Paramters:
        mu:           Mean of latent space, from the encodr.
        logvar:       Log variance of the latent space, from the encoder.
        logvar_prior: Prior log variance of the latent space.

    Returns:
        KL-Divergence loss.
    """
    return torch.mean(
        -0.5 * torch.mean(
            input=(1 + logvar - (mu/prior_var)**2 \
                   - torch.exp(logvar)/prior_var - 2*log(sqrt(prior_var))
            ),
            dim=2,
        ),
    dim=[0, 1],
    )


def vae_loss(
    x: Tensor,
    x_hat: Tensor,
    x_tilde: Tensor,
    mu: Tensor,
    logvar: Tensor,
    prior_var: float = 0.5,
    recon_weight: float = 1.0,
    pert_weight: float = 0.1,
    zero_pert_weight: float = 0.01,
    kld_weight: float = 0.1,
) -> Tensor:
    """
    Total loss function of VAE.

    Parameters:
        x:                Input of VAE.
        x_hat:            Reconstructed output, from the VAE.
        x_tilde:          Perturbed output, from the VAE.
        mu:               Mean of latent space, from the encoder.
        logvar:           Log variance of latent space, from the encoder.
        prior_var:        Prior variance prior of the latent space. 
                          Default 0.5.
        recon_weight:     Weight of the reconstruction loss. Default 1.0.
        pert_weight:      Weight of the perturbation loss. Default 0.1.
        zero_pert_weight: Weight of the zero perturbation loss. 
                          Default 0.01.
                          It the dataset if univariate dataset, set it to 0.0.
        kld_weight:       Weight of the kld loss. Default 0.1.

    Returns:
        Reconstruction loss, Perturbatino loss, Zero perturbation loss, 
        KL-Divergence loss, and Total loss of VAE.
        
        All losses except the total loss are detached. 
        They are only for recording loss during training.
    """
    recon = recon_loss(x, x_hat)
    pert = pert_loss(x, x_hat, x_tilde)
    zero_pert = zero_pert_loss(x, x_tilde)
    kld = kld_loss(mu, logvar, prior_var)
    total = recon_weight * recon + pert_weight * pert \
            + zero_pert_weight * zero_pert + kld_weight * kld
    
    return recon.detach().item(), pert.detach().item(), \
           zero_pert.detach().item(), kld.detach().item(), total


def svdd_loss(
    objective: str,
    radius: Tensor,
    center: Tensor,
    nu: float,
    x: Tensor,
) -> Tensor:
    """
    Loss function of Deep SVDD.

    Parameters:
        objective: Either 'one-class' or 'soft-boundary';
                   
                   If 'one-class', then model assumes that all data are normal
                   and the model optimizes only the mse loss between the 
                   model output and the center;
                   
                   If 'soft-boundary', then the model optimizes the 
                   sphere radius.

        radius:    Initial Radius of the sphere.
        center:    Center of the sphere.
        nu:        Trade-off between penalties and the sphere volume.
        x:         Output of model.
    """
    assert objective in ('one-class', 'soft-boundary'), \
    "objective must be either 'one-class' or 'soft-boundary'"

    dist = torch.sum((x - center) ** 2, dim=1)

    if objective == 'soft-boundary':
        scores = dist - radius ** 2
        loss = radius ** 2 \
               + torch.mean(torch.max(torch.zeros_like(scores), scores)) / nu
    elif objective == 'one-class':
        loss = torch.mean(dist)
    
    return dist, loss

################################# CARLA loss #################################

class pretextloss():
    """
    Loss function for pretext stage of CARLA.
    
    Parameters:
        batch_size:     Batch size.
        temperature:    The cardinality of the set of all triplets (a, p, n)
        initial_margin: Initial margin that controlls the minimum distance
                        between positive and negative pairs.
        adjust_factor:  Adjustment factor when updating the margin.

    Optimizing this loss is for decreasing the distance between the anchor and
    its corresponding positive sample, while simultaneously increasing the
    distance between the anchor and its corresponding negative sample, in the
    representation space.

    Such approach encourages the model to learn a representation that can
    differentiate between normal and abnormal windows.
    """
    def __init__(
        self,
        batch_size: int,
        temperature: float = 0.4,
        initial_margin: float = 1.0,
        adjust_factor: float = 0.1,
    ) -> None:
        self.temperature = temperature
        self.batch_size = batch_size
        self.margin = initial_margin
        self.adjust_factor = adjust_factor

    def __call__(
        self,
        representations: Tensor,
        current_loss: Optional[float] = None,
    ) -> Tensor:
        anchor, positive_pair, negative_pair = torch.split(
            tensor=representations,
            split_size_or_sections=representations.shape[0]//3,
            dim=0,
        )
        anchor = F.normalize(anchor, dim=-1)
        positive_pair = F.normalize(positive_pair, dim=-1)
        negative_pair = F.normalize(negative_pair, dim=-1)

        # update margin
        if current_loss is not None:
            self.margin = max(
                0.01,
                self.margin - self.adjust_factor * current_loss
            )
        
        positive_dist = torch.sum(
            input=(anchor - positive_pair) ** 2,
            dim=-1
        ) / self.temperature

        negative_dist = torch.sum(
            input=torch.pow(anchor.unsqueeze(1) - negative_pair, 2),
            dim=-1
        ) / self.temperature

        hard_negetive_dist = torch.min(
            input=negative_dist,
            dim=-1,
        )[0]

        loss = torch.clamp(
            input=self.margin + positive_dist - hard_negetive_dist,
            min=0.0,
        )
        loss = torch.mean(loss)

        return loss 
 

class classificationloss():
    """
    Classification loss for CARLA's self-supervised classification stage.

    Optimizing this loss is to maximize the similarity between logits of window
    and nearest neighbor, and minimize the similarity between logits of window
    and furthest neighbor. By doing so, the model can classify the normal data
    and anomalous data more clearly.

    Parameters:
        window_logit:   The logit of window. This can be both anchor and
                        negative pair.
        nearest_logit:  The logit of nearest neighbor.
        furthest_logit: The logit of furthest neighbor.
    """
    def __init__(self) -> None:
        self.bceloss = nn.BCELoss()

    def __call__(
        self,
        window_logit: Tensor,
        nearest_logit: Tensor,
        furthest_logit: Tensor,
    ) -> Tuple[Tensor, float, float]:
        B, N = window_logit.shape
        positive_similarity = torch.bmm(
            window_logit.view(B, 1, N),
            nearest_logit.view(B, N, 1),
        ).squeeze() # (B,)

        ones = torch.ones_like(positive_similarity) # (B,)
        consistency = self.bceloss.forward(
            positive_similarity,
            ones
        )

        negative_similarity = torch.bmm(
            window_logit.view(B, 1, N),
            furthest_logit.view(B, N, 1),
        ).squeeze() # (B,)

        zeros = torch.zeros_like(negative_similarity) # (B,)
        inconsistency = self.bceloss.forward(
            negative_similarity,
            zeros
        )

        consistency_sum = consistency + inconsistency

        return consistency_sum, consistency.item(), inconsistency.item()


def entropy(
    x: Tensor,
    input_as_logit: bool = True,
    entropy_weight: float = 5.0,
) -> Tensor:
    """
    Customized entropy loss. In order to prevent overfitting and class 
    diversity, this loss must be maximized.

    Parameters:
        x:              Logit.
        input_as_logit: Whether the input is given as tensor or logit.
                        Default True, since the ClassificationModel outputs
                        the output state as logit.
        entropy_weight: The weight of the entropy loss term. Default 5.0.
    """
    if input_as_logit:
        x_ = torch.clamp(x, min=1e-8)
        b = x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    
    if len(b.size()) == 2:
        return -b.sum(dim=1).mean() * entropy_weight
    elif len(b.size()) == 1:
        return -b.sum() * entropy_weight
    else:
        raise ValueError(
            f'Expected input size to be 1 or 2, but got {b.size()}')
