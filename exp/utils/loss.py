import warnings
from math import log
import torch.nn.functional as F
from exp.utils.common_import import *
warnings.filterwarnings('ignore')


class PLADLoss:
    def __init__(
        self,
        window_size: int,
        data_dim: int,
    ) -> None:
        ones = torch.ones(window_size, data_dim)
        zeros = torch.ones(window_size, data_dim)
        self.e = torch.cat([ones, zeros], dim=1)

        self.mse = nn.MSELoss()

        return
    
    def __call__(self, perturbations: Tensor) -> Tuple[Tensor, float]:
        plad_loss = self.mse.forward(
            input=perturbations,
            target=self.e.to(perturbations.device),
        )
        return plad_loss, plad_loss.item()


class DiscriminatorLoss:
    def __init__(self) -> None:
        self.bce = nn.BCELoss()
        return
    
    def __call__(
        self,
        positive_pseudo_label: Tensor,
        negative_pseudo_label: Tensor,
    ) -> Tuple[Tensor, float, float]:
        batch_size = positive_pseudo_label.shape[0]
        negative_label = torch.zeros(batch_size, 1).to(positive_pseudo_label.device)
        positive_label = torch.ones(batch_size, 1).to(negative_pseudo_label.device)
        negative_bce = self.bce.forward(negative_pseudo_label, negative_label)
        positive_bce = self.bce.forward(positive_pseudo_label, positive_label)
        total_bce = negative_bce + positive_bce
        return total_bce, negative_bce.item(), positive_bce.item()



class TCNPerturbatorLoss:
    def __init__(
        self,
        batch_size: int,
        window_size: int,
        data_dim: int,
        recon_pert_mse_delta_min: float,
        pert_mse_delta_min: float,
        prior_var: float,
        recon_loss_weight: float,
        pert_loss_weight: float,
        zero_pert_loss_weight: float,
        kld_loss_weight: float,
    ) -> None:
        # Data shapes
        self.batch_size = batch_size
        self.window_size = window_size
        self.data_dim = data_dim

        # Perturbator loss parameters
        self.recon_pert_mse_delta_min = recon_pert_mse_delta_min
        self.pert_mse_delta_min = pert_mse_delta_min
        self.prior_var = prior_var

        # Loss weights
        self.recon_loss_weight = recon_loss_weight
        self.pert_loss_weight = pert_loss_weight
        self.zero_pert_loss_weight = zero_pert_loss_weight
        self.kld_loss_weight = kld_loss_weight

        return
    
    def mse_loss(self, x: Tensor, x_recon: Tensor) -> Tensor:
        return torch.mean((x - x_recon) ** 2, dim=[1, 2])
    
    def recon_loss(self, x: Tensor, x_recon: Tensor) -> Tensor:
        mse = self.mse_loss(x=x, x_recon=x_recon)
        reconloss = mse.mean(dim=0)
        return reconloss
    
    def pert_loss(self, x: Tensor, x_recon: Tensor, x_pert: Tensor) -> Tensor:
        triplet = F.relu(
            self.mse_loss(x, x_recon) - self.mse_loss(x, x_pert) \
            + self.recon_pert_mse_delta_min
        )
        regularization_term = F.relu(
            self.mse_loss(x, x_pert) - self.pert_mse_delta_min
        )
        pert_loss = (triplet + regularization_term).mean(dim=0)
        return pert_loss
    
    def zero_pert_loss(self, x: Tensor, x_pert: Tensor) -> Tensor:
        zero_ind = torch.where(
            torch.isclose(x, torch.zeros_like(x)).all(dim=1)
        )
        loss = 0.0
        count = 0

        for i in torch.unique(zero_ind[0]):
            sub_count = (zero_ind[0] == i).sum().item()
            sub_loss = self.mse_loss(
                x=x[i:i+1, :, zero_ind[1][count:count+sub_count]],
                x_recon=x_pert[i:i+1, :, zero_ind[1][count:count+sub_count]],
            )
            loss += (sub_loss + 1.0) ** -1
            count += sub_count
        
        loss /= x.shape[0]
        return loss
    
    def kld_loss(self, mu: Tensor, logvar: Tensor) -> Tensor:
        return torch.mean(
            input=-0.5 * torch.mean(
                input=(1 + logvar - (mu / self.prior_var) ** 2 \
                       - torch.exp(logvar) / (self.prior_var ** 2) \
                       -2 * log(self.prior_var)
                ),
                dim=2,
            ),
            dim=[0, 1],
        )
    
    def __call__(
        self,
        x: Tensor,
        x_recon: Tensor,
        x_pert: Tensor,
        mu: Tensor,
        logvar: Tensor,
    ) -> Tuple[Tensor, float, float, float, float]:
        recon_loss = self.recon_loss(x=x, x_recon=x_recon)
        pert_loss = self.pert_loss(x=x, x_recon=x_recon, x_pert=x_pert)
        zero_pert_loss = self.zero_pert_loss(x=x, x_pert=x_pert)
        kld_loss = self.kld_loss(mu=mu, logvar=logvar)

        total_loss = self.recon_loss_weight * recon_loss \
                     + self.pert_loss_weight * pert_loss \
                     + self.zero_pert_loss_weight * zero_pert_loss \
                     + self.kld_loss_weight * kld_loss \

        return total_loss, recon_loss.item(), pert_loss.item(), \
               zero_pert_loss.item(), kld_loss.item()


class TCNDiscriminatorLoss:
    def __init__(self) -> None:
        self.bce = nn.BCELoss()
        
    def __call__(
        self,
        x_label: Tensor,
        x_pert_label: Tensor,
    ) -> Tuple[Tensor, float, float]:
        batch_size = x_label.shape[0]
        negative_label = torch.zeros(batch_size, 1).to(x_label.device)
        positive_label = torch.ones(batch_size, 1).to(x_label.device)
        negative_bce = self.bce.forward(x_label, negative_label)
        positive_bce = self.bce.forward(x_pert_label, positive_label)
        total_bce = negative_bce + positive_bce
        return total_bce, negative_bce.item(), positive_bce.item()
    

class PretextLoss:
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
        temperature: float = 0.4,
        initial_margin: float = 1.0,
        adjust_factor: float = 0.1,
    ) -> None:
        self.temperature = temperature
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
 

class ClassificationLoss:
    """
    Classification loss for CARLA's self-supervised classification stage.
    """
    def __init__(self) -> None:
        self.bceloss = nn.BCELoss()

    def __call__(
        self,
        window_logit: Tensor,
        nearest_logit: Tensor,
        furthest_logit: Tensor,
    ) -> Tuple[Tensor, float, float]:
        """
        Optimizing this loss is to maximize the similarity between logits of 
        window and nearest neighbor, and minimize the similarity between logits
        of window and furthest neighbor. By doing so, the model can classify 
        the normal data and anomalous data more clearly.

        Parameters:
            window_logit:   The logit of window. This can be both anchor and
                            negative pair.
            nearest_logit:  The logit of nearest neighbor.
            furthest_logit: The logit of furthest neighbor.
        """
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
