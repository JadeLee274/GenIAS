import warnings
from math import log, sqrt
import torch.nn.functional as F
from exp.utils.common_import import *
warnings.filterwarnings('ignore')


# class PerturbatorLoss:
#     def __init__(
#         self,
#         batch_size: int,
#         window_size: int,
#         data_dim: int,
#         pert_factors_weight: float,
#     ) -> None:
#         self.positive_label = torch.ones(batch_size,)
#         self.negative_label = torch.zeros(batch_size,)
#         self.ones = torch.ones(window_size, data_dim)
#         self.zeros = torch.zeros(window_size, data_dim)
#         self.bce = nn.BCELoss()
#         self.mse = nn.MSELoss()
#         self.pert_factors_weight = pert_factors_weight
#         return
    
#     def __call__(
#         self,
#         pert_mult_factor: Tensor,
#         pert_add_factor: Tensor,
#         x_class: Tensor,
#         x_perturbed_class: Tensor,
#         mu: Tensor,
#         logvar: Tensor,
#     ) -> Tuple[Tensor, float, float, float, float]:
#         device = pert_mult_factor.device
#         x_class = x_class.reshape(-1)
#         x_perturbed_class = x_perturbed_class.reshape(-1)
#         bce_anchor = self.bce.forward(
#             input=x_class,
#             target=self.negative_label.to(device),
#         )
#         bce_negative = self.bce.forward(
#             input=x_perturbed_class,
#             target=self.positive_label.to(device),
#         )
#         kld_loss = torch.mean(
#             input= -0.5 * torch.mean(
#                 input=1 + logvar - mu ** 2 - torch.exp(logvar),
#                 dim=[1, 2],
#             ),
#             dim=0,
#         )
#         tensor_concat = torch.cat([pert_mult_factor, pert_add_factor], dim=1)
#         tensor_concat = tensor_concat.to(device)
#         epsilon = torch.cat([self.ones, self.zeros], dim=0)
#         epsilon = epsilon.to(device)
#         pert_factor_loss = self.mse.forward(tensor_concat, epsilon)
#         pert_factor_loss = self.pert_factors_weight * pert_factor_loss
        
#         total_loss = bce_anchor + bce_negative + kld_loss + pert_factor_loss

#         return total_loss, bce_anchor.item(), bce_negative.item(), \
#                kld_loss.item(), pert_factor_loss.item()


class LinearPerturbatorLoss:
    def __init__(
        self,
        batch_size: int,
        window_size: int,
        data_dim: int,
        delta_min: float,
        delta_max: float,
        prior_var: float,
        recon_loss_weight: float,
        perturbation_loss_weight: float,
        zero_perturbation_loss_weight: float,
        nonzero_perturbation_loss_bound: float,
        zero_perturbation_loss_bound: float,
        kld_loss_weight: float,
    ) -> None:
        self.batch_size = batch_size
        self.window_size = window_size
        self.data_dim = data_dim
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.prior_var = prior_var
        self.recon_loss_weight = recon_loss_weight
        self.perturbation_loss_weight = perturbation_loss_weight
        self.zero_perturbation_loss_weight = zero_perturbation_loss_weight
        self.nonzero_perturbation_loss_bound = nonzero_perturbation_loss_bound
        self.zero_perturbation_loss_bound = zero_perturbation_loss_bound

        self.kld_loss_weight = kld_loss_weight
        self.mse = nn.MSELoss()
        return
    
    def mse_loss(self, x: Tensor, x_recon: Tensor) -> Tensor:
        return torch.mean((x - x_recon)**2, dim=[1, 2])
    
    def recon_loss(self, x: Tensor, x_recon: Tensor) -> Tensor:
        mse = self.mse_loss(x=x, x_recon=x_recon)
        reconloss = mse.mean(dim=0)
        return reconloss
    
    def perturbation_loss(
        self,
        x: Tensor,
        x_recon: Tensor,
        x_pert: Tensor,
    ) -> Tensor:
        triplet = F.relu(
            self.mse_loss(x, x_recon) - self.mse_loss(x, x_pert) + self.delta_min
        )
        regularization_term = F.relu(self.mse_loss(x, x_pert) - self.delta_max)
        pert_loss = (triplet + regularization_term).mean(dim=0)
        return pert_loss

    def zero_pert_loss(self, x: Tensor, x_pert: Tensor) -> Tensor:
        zero_ind = torch.where(torch.isclose(x, torch.zeros_like(x)).all(dim=1))
        loss = 0.0
        count = 0

        for i in torch.unique(zero_ind[0]):
            sub_count = (zero_ind[0] == i).sum().item()
            sub_loss = self.mse_loss(
                x=x[i:i+1, :, zero_ind[1][count:count+sub_count]],
                x_recon=x_pert[i:i+1, :, zero_ind[1][count:count+sub_count]],
            )
            loss += (sub_loss + self.zero_perturbation_loss_bound) ** -1
            count += sub_count
        
        loss /= x.shape[0]
        return loss
    
    def kld_loss(self, mu: Tensor, logvar: Tensor) -> Tensor:
        return torch.mean(
            input=-0.5 * torch.mean(
                input=(1 + logvar - (mu / self.prior_var) ** 2 \
                       - torch.exp(logvar) / (self.prior_var ** 2) \
                       - 2 * log(self.prior_var)
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
        pert_loss = self.perturbation_loss(x=x, x_recon=x_recon, x_pert=x_pert)
        zero_pert_loss = self.zero_pert_loss(x=x, x_pert=x_pert)
        kld_loss = self.kld_loss(mu=mu, logvar=logvar)
        
        # total_loss = self.recon_loss_weight * recon_loss \
        #              + self.perturbation_loss_weight * pert_loss \
        #              + self.zero_perturbation_loss_weight * zero_pert_loss \
        #              + self.kld_loss_weight * kld_loss

        total_loss = self.recon_loss_weight * recon_loss \
                     + self.perturbation_loss_weight * pert_loss \
                     + self.zero_perturbation_loss_weight * zero_pert_loss \
                     + self.kld_loss_weight * kld_loss


        # return total_loss, recon_loss.item(), pert_loss.item(), \
        #        zero_pert_loss.item(), kld_loss.item()

        return total_loss, recon_loss.item(), pert_loss.item(), \
               zero_pert_loss.item(), kld_loss.item()


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
        # Perturbator loss tensors
        self.ones = torch.ones(window_size, data_dim)
        self.zeros = torch.zeros(window_size, data_dim)
        # Loss weights
        self.recon_loss_weight = recon_loss_weight
        self.pert_loss_weight = pert_loss_weight
        self.zero_pert_loss_weight = zero_pert_loss_weight
        self.kld_loss_weight = kld_loss_weight
        # MSE loss
        self.mse = nn.MSELoss()
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
        zero_ind = torch.where(torch.isclose(x, torch.zeros_like(x)).all(dim=1))
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


class TCNPerturbatorLoss_:
    def __init__(
        self,
        window_size: int,
        data_dim: int,
        recon_loss_weight: float,
    ) -> None:
        self.ones = torch.ones(window_size, data_dim)
        self.zeros = torch.zeros(window_size, data_dim)
        self.window_size = window_size
        self.data_dim = data_dim
        self.mse = nn.MSELoss()
        self.recon_loss_weight = recon_loss_weight
        
        return
    
    def kld_loss(self, mu: Tensor, logvar: Tensor) -> Tensor:
        return torch.mean(
            input=-0.5 * torch.mean(
                input=(1 + logvar - mu ** 2 - torch.exp(logvar)),
                dim=2,
            ),
            dim=[0, 1],
        )
    
    def recon_loss(self, mult_factor: Tensor, add_factor: Tensor) -> Tensor:
        assert len(mult_factor) == len(add_factor), \
        f"mult_factor length and add_factor length mismatch"
        mult_factor_loss = 0.0
        add_factor_loss = 0.0

        for idx in range(len(mult_factor)):
            ones = self.ones.to(mult_factor.device)
            zeros = self.zeros.to(mult_factor.device)
            mult = mult_factor[idx]
            add = add_factor[idx]
            mult_loss = self.mse.forward(mult, ones)
            add_loss = self.mse.forward(add, zeros)
            mult_factor_loss += mult_loss
            add_factor_loss += add_loss
        
        mult_factor_loss = mult_factor_loss / len(mult_factor)
        add_factor_loss = add_factor_loss / len(add_factor)
        recon_loss = mult_factor_loss + add_factor_loss
            
        return recon_loss
    
    def __call__(
        self,
        mu: Tensor,
        logvar: Tensor,
        mult_factor: Tensor,
        add_factor: Tensor,
    ) -> Tuple[Tensor, float, float]:
        kld_loss = self.kld_loss(mu=mu, logvar=logvar)
        recon_loss = self.recon_loss(
            mult_factor=mult_factor,
            add_factor=add_factor,
        )
        perturbator_loss = kld_loss + self.recon_loss_weight * recon_loss
        return perturbator_loss, kld_loss.item(), recon_loss.item()


class TCNPerturbatorLoss__:
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
        # factor_loss_weight: float,
    ) -> None:
        # Data shapes
        self.batch_size = batch_size
        self.window_size = window_size
        self.data_dim = data_dim
        # Perturbator loss parameters
        self.recon_pert_mse_delta_min = recon_pert_mse_delta_min
        self.pert_mse_delta_min = pert_mse_delta_min
        self.prior_var = prior_var
        # Perturbator loss tensors
        self.ones = torch.ones(window_size, data_dim)
        self.zeros = torch.zeros(window_size, data_dim)
        # Loss weights
        self.recon_loss_weight = recon_loss_weight
        self.pert_loss_weight = pert_loss_weight
        self.zero_pert_loss_weight = zero_pert_loss_weight
        self.kld_loss_weight = kld_loss_weight
        # self.factor_loss_weight = factor_loss_weight
        # MSE loss
        self.mse = nn.MSELoss()
        # Newley added tensors for experimental purpose
        self.ones = torch.ones(window_size, data_dim)
        self.zeros = torch.zeros(window_size, data_dim)
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
        zero_ind = torch.where(torch.isclose(x, torch.zeros_like(x)).all(dim=1))
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
    
    # Newely added metric for experimental purpose
    # def factor_loss(self, recon_factor: Tensor, pert_factor: Tensor) -> Tensor:
    #     mse = self.mse.forward(recon_factor, pert_factor)
    #     contrastive_loss = (mse + 1.0) ** -1
    #     return contrastive_loss

    # def factor_loss(self, mult_factor: Tensor, add_factor: Tensor) -> Tensor:
    #     mult_factor_mse = torch.zeros((1, ), requires_grad=True)
    #     add_factor_mse = torch.zeros((1, ), requires_grad=True)

    #     for idx in range(len(mult_factor)):
    #         mult = mult_factor[idx]
    #         add = add_factor[idx]
    #         mult_mse = self.mse.forward(mult, self.ones)
    #         add_mse = self.mse.forward(add, self.zeros)
    #         mult_factor_mse += mult_mse
    #         add_factor_mse += add_mse
        
    #     mult_factor_mse /= mult_factor.shape[0]
    #     add_factor_mse /= add_factor.shape[0]
    #     factor_loss = mult_factor_mse + add_factor_mse

    #     return factor_loss
    
    def __call__(
        self,
        x: Tensor,
        x_recon: Tensor,
        x_pert: Tensor,
        mu: Tensor,
        logvar: Tensor,
        # recon_mult: Tensor,
        # recon_add: Tensor,
        # pert_mult: Tensor,
        # pert_add: Tensor,
    ) -> Tuple[Tensor, float, float, float, float]:
        recon_loss = self.recon_loss(x=x, x_recon=x_recon)
        pert_loss = self.pert_loss(x=x, x_recon=x_recon, x_pert=x_pert)
        zero_pert_loss = self.zero_pert_loss(x=x, x_pert=x_pert)
        kld_loss = self.kld_loss(mu=mu, logvar=logvar)

        # Newley added loss for experimental purpose
        # mult_factor_loss = self.factor_loss(
        #     recon_factor=recon_mult,
        #     pert_factor=pert_mult,
        # )
        # add_factor_loss = self.factor_loss(
        #     recon_factor=recon_add,
        #     pert_factor=pert_add,
        # )
        # factor_loss = mult_factor_loss + add_factor_loss

        total_loss = self.recon_loss_weight * recon_loss \
                     + self.pert_loss_weight * pert_loss \
                     + self.zero_pert_loss_weight * zero_pert_loss \
                     + self.kld_loss_weight * kld_loss \
                    #  + self.factor_loss_weight * factor_loss

        return total_loss, recon_loss.item(), pert_loss.item(), \
               zero_pert_loss.item(), kld_loss.item(), \
            #    factor_loss.item()


class BidirectionalPerturbatorLoss:
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
        # MSE loss
        self.mse = nn.MSELoss()
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
        zero_ind = torch.where(torch.isclose(x, torch.zeros_like(x)).all(dim=1))
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
        # temporal_mu: Tensor,
        # temporal_logvar: Tensor,
        feature_mu: Tensor,
        feature_logvar: Tensor,
    ) -> Tuple[Tensor, float, float, float, float]:
        recon_loss = self.recon_loss(x=x, x_recon=x_recon)
        pert_loss = self.pert_loss(x=x, x_recon=x_recon, x_pert=x_pert)
        zero_pert_loss = self.zero_pert_loss(x=x, x_pert=x_pert)
        # temporal_kld_loss = self.kld_loss(
            # mu=temporal_mu,
            # logvar=temporal_logvar,
        # )
        feature_kld_loss = self.kld_loss(
            mu=feature_mu,
            logvar=feature_logvar,
        )
        # kld_loss = temporal_kld_loss + feature_kld_loss
        kld_loss = feature_kld_loss

        total_loss = self.recon_loss_weight * recon_loss \
                     + self.pert_loss_weight * pert_loss \
                     + self.zero_pert_loss_weight * zero_pert_loss \
                     + self.kld_loss_weight * kld_loss \

        return total_loss, recon_loss.item(), pert_loss.item(), \
               zero_pert_loss.item(), kld_loss.item()


class TCNDiscriminatorLoss:
    def __init__(self) -> None:
        self.bce = nn.BCEWithLogitsLoss()
        
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
