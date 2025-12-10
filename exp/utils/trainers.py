import logging
from math import cos, pi
from faiss import IndexFlatL2
from sklearn.metrics import precision_recall_curve, auc, f1_score
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
from exp.models.augmentor import *
from exp.data_factory.loader import *
from exp.utils.common_import import *
from exp.utils.loss import *
from genias.models.carla import PretextModel, ClassificationModel
from genias.utils.loss import pretextloss, classificationloss, entropy
from genias.utils.metric import *


# class LinearPerturbatorTrainer(object):
#     def __init__(
#         self,
#         time: str,
#         data: str,
#         subdata: Optional[str],
#         batch_size: int,
#         window_size: int,
#         delta_min: float,
#         delta_max: float,
#         prior_var: float,
#         recon_loss_weight: float,
#         perturbation_loss_weight: float,
#         zero_perturbation_loss_weight: float,
#         zero_perturbation_loss_bound: float,
#         nonzero_perturbation_loss_bound: float,
#         kld_loss_weight: float,
#         gpu_num: int,
#         learning_rate: float,
#         epochs: int,
#         save_interval: int,
#         plot_interval: int,
#     ) -> None:
#         self.time = time
#         self.data = data
#         self.subdata = subdata

#         # Set data and device
#         self.train_dataset = PerturbationDataset(
#             data=data,
#             subdata=subdata,
#             mode='train',
#             window_size=window_size,
#         )
#         self.train_loader = DataLoader(
#             dataset=self.train_dataset,
#             batch_size=batch_size,
#             shuffle=True,
#             drop_last=True,
#         )
#         self.test_dataset = PerturbationDataset(
#             data=data,
#             subdata=subdata,
#             mode='test',
#             window_size=window_size,
#         )
#         data_dim = self.train_dataset.data_dim
#         self.device = torch.device(f'cuda:{gpu_num}')
        
#         # Set perturbator
#         latent_perturbator_factor = 1 / prior_var

#         self.perturbator = LinearPerturbator(
#             data_dim=data_dim,
#             window_size=window_size,
#             latent_perturbator_factor=latent_perturbator_factor,
#         )
#         self.perturbator = self.perturbator.to(self.device)

#         # Set binary classifier
#         self.discriminator = TCNDiscriminator(
#             window_size=window_size,
#             tcn_mid_dim=50,
#             data_dim=data_dim,
#             mlp_mid_dim=20,
#         )
#         self.discriminator = self.discriminator.to(self.device)

#         # Set optimizer
#         self.learning_rate = learning_rate
#         self.optimizer = Adam(
#             params=list(self.perturbator.parameters()) \
#                    + list(self.discriminator.parameters()),
#                 lr=learning_rate,
#         )
#         self.perturbator_loss = LinearPerturbatorLoss(
#             batch_size=batch_size,
#             window_size=window_size,
#             data_dim=data_dim,
#             delta_min=delta_min,
#             delta_max=delta_max,
#             prior_var=prior_var,
#             recon_loss_weight=recon_loss_weight,
#             perturbation_loss_weight=perturbation_loss_weight,
#             zero_perturbation_loss_weight=zero_perturbation_loss_weight,
#             nonzero_perturbation_loss_bound=nonzero_perturbation_loss_bound,
#             zero_perturbation_loss_bound=zero_perturbation_loss_bound,
#             kld_loss_weight=kld_loss_weight,
#         )
#         self.discriminator_loss = TCNDiscriminatorLoss(
#             batch_size=batch_size
#         )
#         self.epochs = epochs
#         self.save_interval = save_interval

#         self.ckpt_dir = os.path.join('exp', 'checkpoints', data)
        
#         if subdata is not None:
#             self.ckpt_dir = os.path.join(self.ckpt_dir, subdata)

#         self.ckpt_dir = os.path.join(self.ckpt_dir, 'perturbator', time)
#         os.makedirs(self.ckpt_dir, exist_ok=True)

#         self.plot_interval = plot_interval
        
#         return
    
#     def train(self) -> None:
#         for epoch in range(self.epochs):
#             epoch_total_loss = 0.0
#             epoch_recon_loss = 0.0
#             epoch_pert_loss = 0.0
#             epoch_zero_pert_loss = 0.0
#             epoch_kld_loss = 0.0
#             epoch_negative_bce_loss = 0.0
#             epoch_positive_bce_loss = 0.0

#             for x in self.train_loader:
#                 self.optimizer.zero_grad()

#                 x: Tensor = x.to(self.device).float()
#                 x_recon, x_pert, _, _, _, _, mu, logvar \
#                 = self.perturbator.forward(x)

#                 x_label = self.discriminator.forward(x)
#                 x_pert_label = self.discriminator.forward(x_pert)

#                 total_pert_loss, recon_loss, pert_loss, \
#                 zero_pert_loss, kld_loss = self.perturbator_loss(
#                     x=x,
#                     x_recon=x_recon,
#                     x_pert=x_pert,
#                     mu=mu,
#                     logvar=logvar,
#                 )
#                 total_bce, negative_bce, positive_bce \
#                 = self.discriminator_loss(
#                     x_label=x_label,
#                     x_pert_label=x_pert_label,
#                 )

#                 total_loss = total_pert_loss + total_bce
#                 total_loss.backward()
#                 self.optimizer.step()
#                 epoch_total_loss += total_loss.item()
#                 epoch_recon_loss += recon_loss
#                 epoch_pert_loss += pert_loss
#                 epoch_zero_pert_loss += zero_pert_loss
#                 epoch_kld_loss += kld_loss
#                 epoch_negative_bce_loss += negative_bce
#                 epoch_positive_bce_loss += positive_bce
            
#             epoch_total_loss /= len(self.train_loader)
#             epoch_recon_loss /= len(self.train_loader)
#             epoch_pert_loss /= len(self.train_loader)
#             epoch_zero_pert_loss /= len(self.train_loader)
#             epoch_kld_loss /= len(self.train_loader)
#             epoch_negative_bce_loss /= len(self.train_loader)
#             epoch_positive_bce_loss /= len(self.train_loader)

#             logging.info(f'Epoch {epoch + 1} results:')
#             logging.info(f'- Total loss: {epoch_total_loss:.4e}')
#             logging.info(f'- Recon loss: {epoch_recon_loss:.4e}')
#             logging.info(f'- Pert loss: {epoch_pert_loss:.4e}')
#             logging.info(f'- Zero pert loss: {epoch_zero_pert_loss:.4e}')
#             logging.info(f'- KLD loss: {epoch_kld_loss:.4e}')
#             logging.info(f'- Negative BCE loss: {epoch_negative_bce_loss:.4e}')
#             logging.info(f'- Positive BCE loss: {epoch_positive_bce_loss:.4e}\n')


#             if (epoch + 1) % self.save_interval == 0:
#                 torch.save(
#                     obj={
#                         'perturbator': self.perturbator.state_dict(),
#                         'discriminator': self.discriminator.state_dict(),
#                         'optimizer': self.optimizer.state_dict(),
#                         'epoch': epoch + 1,
#                     },
#                     f=os.path.join(self.ckpt_dir, f'epoch_{epoch + 1}.pt'),
#                 )
        
#         logging.info('Perturbator training finished.\n')

#         return
    
#     def eval(self) -> None:
#         self.discriminator.eval()
#         x_label_list = []

#         for x in self.test_dataset:
#             x = torch.tensor(x).unsqueeze(0).float().to(self.device)
#             x_label = self.discriminator.forward(x)
#             x_label = (x_label > 0.5).int().item()
#             x_label_list.append(x_label)
            
#         x_label_list = np.array(x_label_list, dtype=np.int32)

#         f1 = f1_score(
#             y_true=self.test_dataset.test_labels,
#             y_pred=x_label_list
#         )

#         if self.data in ['MSL', 'SMAP', 'SMD']:
#             logging.info(
#                 f'F1 score on {self.data} {self.subdata}: {round(f1, 4)}\n')

#         return

#     def inference(self, epoch: int) -> None:
#         self.discriminator.init_discriminator(
#             time=self.time,
#             data=self.data,
#             subdata=self.subdata,
#             epoch=epoch,
#         )
#         self.discriminator.eval()
#         x_label_list = []

#         for x in self.test_dataset:
#             x = torch.tensor(x).unsqueeze(0).float().to(self.device)
#             x_label = self.discriminator.forward(x)
#             x_label = (x_label > 0.5).int().item()
#             x_label_list.append(x_label)
            
#         x_label_list = np.array(x_label_list, dtype=np.int32)

#         f1 = f1_score(
#             y_true=self.test_dataset.test_labels,
#             y_pred=x_label_list
#         )

#         if self.data in ['MSL', 'SMAP', 'SMD']:
#             logging.info(
#                 f'F1 score on {self.data} {self.subdata}: {round(f1, 4)}\n')

#         return

    

class TCNPerturbatorTrainer(object):
    def __init__(
        self,
        time: str,
        data: str,
        subdata: Optional[str],
        batch_size: int,
        window_size: int,
        recon_pert_mse_delta_min: float,
        pert_mse_delta_min: float,
        prior_var: float,
        recon_loss_weight: float,
        pert_loss_weight: float,
        zero_pert_loss_weight: float,
        kld_loss_weight: float,
        discriminator_loss_weight: float,
        gpu_num: int,
        learning_rate: int,
        epochs: int,
        save_interval: int,
    ) -> None:
        # Time and data information
        self.time = time
        self.data = data
        self.subdata = subdata
        self.batch_size = batch_size
        self.window_size = window_size
        
        # Training infos
        self.epochs = epochs
        self.save_interval = save_interval
        
        # Train and test dataset/dataloader
        self.train_dataset = PerturbationDataset(
            data=data,
            subdata=subdata,
            mode='train',
            window_size=window_size,
        )
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        self.test_dataset = PerturbationDataset(
            data=data,
            subdata=subdata,
            mode='test',
            window_size=window_size,
        )
        
        # Set device
        self.device = torch.device(f'cuda:{gpu_num}')

        # Perturbator and discriminator
        self.data_dim = self.train_dataset.data_dim
        assert self.data_dim > 1, "Data dimension should be > 1"

        latent_perturbator_factor = 1 / prior_var
        # self.perturbator = TCNPerturbator_(
        #     window_size=self.window_size,
        #     data_dim=self.data_dim,
        #     latent_perturbator_factor=latent_perturbator_factor,
        # )
        self.perturbator = TCNPerturbator(
            window_size=self.window_size,
            data_dim=self.data_dim,
            sigma_pert_factor=2.0,
        )
        self.perturbator = self.perturbator.to(self.device)
        self.discriminator = TCNDiscriminator(
            data_dim=self.data_dim,
        )
        self.discriminator = self.discriminator.to(self.device)

        # Set optimizer
        self.optimizer = Adam(
            params=list(self.perturbator.parameters()) \
                   + list(self.discriminator.parameters()),
            lr=learning_rate,
        )

        # Loss
        self.perturbator_loss = TCNPerturbatorLoss(
            batch_size=batch_size,
            window_size=self.window_size,
            data_dim=self.data_dim,
            recon_pert_mse_delta_min=recon_pert_mse_delta_min,
            pert_mse_delta_min=pert_mse_delta_min,
            prior_var=prior_var,
            recon_loss_weight=recon_loss_weight,
            pert_loss_weight=pert_loss_weight,
            zero_pert_loss_weight=zero_pert_loss_weight,
            kld_loss_weight=kld_loss_weight,
            # factor_loss_weight=factor_loss_weight,
        )
        # self.perturbator_loss = TCNPerturbatorLoss(
        #     window_size=self.window_size,
        #     data_dim=self.data_dim,
        #     recon_loss_weight=recon_loss_weight,
        # )
        self.discriminator_loss = TCNDiscriminatorLoss()
        self.discriminator_loss_weight = discriminator_loss_weight
        
        # Checkpoint directory
        self.ckpt_dir = os.path.join('exp', 'checkpoints', data)

        if subdata is not None:
            self.ckpt_dir = os.path.join(self.ckpt_dir, subdata)
        
        self.ckpt_dir = os.path.join(self.ckpt_dir, 'perturbator', time)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        return
    
    def save_model(self, epoch: int) -> None:
        torch.save(
            obj={
                'perturbator': self.perturbator.state_dict(),
                'discriminator': self.discriminator.state_dict(),
                'optimzier': self.optimizer.state_dict(),
                'epoch': epoch,
            },
            f=os.path.join(self.ckpt_dir, f'epoch_{epoch}.pt'),
        )

        return
    
    def plot_perturbation(
        self,
        time: str,
        epoch: int,
        figsize: Tuple[int, int] = (15, 20),
        ylim_upper: float = 10.0,
    ) -> None:
        var_num = self.data_dim // 5

        save_path = os.path.join('exp', 'figs', 'perturbator', time)

        if self.subdata is not None:
            save_path = os.path.join(save_path, self.subdata)

        save_path = os.path.join(save_path, f'epoch_{epoch}')
        os.makedirs(save_path, exist_ok=True)

        for i in range(len(self.train_dataset) // self.window_size):
            idx = i * self.window_size
            data = self.train_dataset[idx]
            data: Tensor = torch.tensor(data).float().unsqueeze(0)
            data = data.to(self.device)
            # _, pert, _, _, _, _, _, _ = self.perturbator.forward(data)
            _, pert, _, _ = self.perturbator.forward(data)

            data = data.squeeze(0).detach().cpu().numpy()
            pert=  pert.squeeze(0).detach().cpu().numpy()

            fig, axes = plt.subplots(var_num, 5, figsize=figsize)
            axes = axes.flatten()

            for i in range(self.data_dim):
                axes[i].plot(data[:, i])
                axes[i].set_title(f"Dim {i+1}")
                axes[i].set_xticks([])
                axes[i].set_yticks([])
                axes[i].set_ylim(-ylim_upper, ylim_upper)
                axes[i].set_yticks([-ylim_upper, ylim_upper])

            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'data_{idx}.png'))
            plt.close()

            fig, axes = plt.subplots(var_num, 5, figsize=figsize)
            axes = axes.flatten()

            for i in range(self.data_dim):
                axes[i].plot(pert[:, i])
                axes[i].set_title(f"Dim {i+1}")
                axes[i].set_xticks([])
                axes[i].set_yticks([])
                axes[i].set_ylim(-ylim_upper, ylim_upper)
                axes[i].set_yticks([-ylim_upper, ylim_upper])

            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'neg_pair_{idx}.png'))
            plt.close()
        
        return
    
    def train(self) -> None:
        self.perturbator.train()
        self.discriminator.train()

        for epoch in range(self.epochs):
            epoch_total_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_pert_loss = 0.0
            epoch_zero_pert_loss = 0.0
            epoch_kld_loss = 0.0
            # epoch_factor_loss = 0.0
            epoch_negative_bce_loss = 0.0
            epoch_positive_bce_loss = 0.0

            for x in self.train_loader:
                self.optimizer.zero_grad()

                x: Tensor = x.to(self.device).float()
                x_recon, x_pert, mu, logvar = self.perturbator.forward(x)

                x_label = self.discriminator.forward(x)
                x_pert_label = self.discriminator.forward(x_pert)

                total_pert_loss, recon_loss, pert_loss, \
                zero_pert_loss, kld_loss = self.perturbator_loss(
                    x=x,
                    x_recon=x_recon,
                    x_pert=x_pert,
                    mu=mu,
                    logvar=logvar,
                )
                total_bce_loss, negaitve_bce_loss, positive_bce_loss \
                = self.discriminator_loss(
                    x_label=x_label,
                    x_pert_label=x_pert_label,
                )

                total_loss = total_pert_loss \
                             + self.discriminator_loss_weight * total_bce_loss

                total_loss.backward()
                self.optimizer.step()
                epoch_total_loss += total_loss.item()
                epoch_recon_loss += recon_loss
                epoch_pert_loss += pert_loss
                epoch_zero_pert_loss += zero_pert_loss
                epoch_kld_loss += kld_loss
                # epoch_factor_loss += factor_loss
                epoch_negative_bce_loss += negaitve_bce_loss
                epoch_positive_bce_loss += positive_bce_loss
            
            epoch_total_loss /= len(self.train_loader)
            epoch_recon_loss /= len(self.train_loader)
            epoch_pert_loss /= len(self.train_loader)
            epoch_zero_pert_loss /= len(self.train_loader)
            epoch_kld_loss /= len(self.train_loader)
            # epoch_factor_loss /= len(self.train_loader)
            epoch_negative_bce_loss /= len(self.train_loader)
            epoch_positive_bce_loss /= len(self.train_loader)

            logging.info(f'Epoch {epoch + 1} results:')
            logging.info(f'- Total loss: {epoch_total_loss:.4e}')
            logging.info(f'- Recon loss: {epoch_recon_loss:.4e}')
            logging.info(f'- Pert loss: {epoch_pert_loss:.4e}')
            logging.info(f'- Zero pert loss: {epoch_zero_pert_loss:.4e}')
            logging.info(f'- KLD loss: {epoch_kld_loss:.4e}')
            # logging.info(f'- Factor loss: {epoch_factor_loss:.4e}')
            logging.info(f'- Negative BCE loss: {epoch_negative_bce_loss:.4e}')
            logging.info(f'- Positive BCE loss: {epoch_positive_bce_loss:.4e}\n')

            if epoch == 0 or (epoch + 1) % self.save_interval == 0:
                self.save_model(epoch=epoch+1)
                
            if epoch == self.epochs:
                self.plot_perturbation(time=self.time, epoch=epoch+1)

        return

    def train_(self) -> None:
        self.perturbator.train()
        self.discriminator.train()

        for epoch in range(self.epochs):
            epoch_total_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_kld_loss = 0.0
            epoch_negative_bce_loss = 0.0
            epoch_positive_bce_loss = 0.0

            for x in self.train_loader:
                x: Tensor = x.to(self.device).float()
                x_recon, x_pert, mu, logvar \
                = self.perturbator.forward(x=x)
                pert_loss, recon_loss, kld_loss \
                = self.perturbator_loss(
                    x=x,
                    x_recon=x_recon,
                    x_pert=x_pert,
                    mu=mu,
                    logvar=logvar,
                )
                x_label = self.discriminator.forward(x=x)
                x_pert_label = self.discriminator.forward(x=x_pert)
                bce, negaitve_bce, positive_bce = self.discriminator_loss(
                    x_label=x_label,
                    x_pert_label=x_pert_label,
                )
                total_loss = pert_loss + bce

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                epoch_total_loss += total_loss.item()
                epoch_recon_loss += recon_loss
                epoch_kld_loss += kld_loss
                epoch_negative_bce_loss += negaitve_bce
                epoch_positive_bce_loss += positive_bce
            
            epoch_total_loss /= len(self.train_loader)
            epoch_recon_loss /= len(self.train_loader)
            epoch_kld_loss /= len(self.train_loader)
            epoch_negative_bce_loss /= len(self.train_loader)
            epoch_positive_bce_loss /= len(self.train_loader)

            logging.info(f'Epoch {epoch + 1} results:')
            logging.info(f'- Total loss: {epoch_total_loss:.4e}')
            logging.info(f'- Recon loss: {epoch_recon_loss:.4e}')
            logging.info(f'- KLD loss: {epoch_kld_loss:.4e}')
            logging.info(f'- Positive BCE loss: {epoch_negative_bce_loss:.4e}')
            logging.info(f'- Negative BCE loss: {epoch_positive_bce_loss:.4e}\n')

            if epoch == 0 or (epoch + 1) % self.save_interval == 0:
                self.save_model(epoch=epoch+1)
            
            if epoch + 1 == self.epochs:
                self.plot_perturbation(time=self.time, epoch=epoch+1)

        return

    def eval(self) -> None:
        self.discriminator.eval()
        pred_list = []

        for idx in range(len(self.test_dataset)):
            x = self.test_dataset[idx]
            x = torch.tensor(x).float().unsqueeze(0).to(self.device)
            pred = self.discriminator.forward(x)
            pred = (pred > 0.5).int().item()
            pred_list.append(pred)
        
        pred_list = np.array(pred_list, dtype=np.int32)

        f1 = f1_score(
            y_true=self.test_dataset.test_labels,
            y_pred=pred_list,
        )

        if self.data in ['MSL', 'SMAP', 'SMD']:
            logging.info(
                f'F1 score on {self.data} {self.subdata}: {round(f1, 4)}\n'
            )
        else:
            logging.info(f'F1 socre on {self.data}: {round(f1, 4)}\n')
        
        return
    
    def inference(self, epoch: int) -> None:
        self.discriminator.init_discriminator(
            time=self.time,
            data=self.data,
            subdata=self.subdata,
            epoch=epoch,
        )
        self.discriminator.eval()
        x_label_list = []

        for x in self.test_dataset:
            x = torch.tensor(x).unsqueeze(0).float().to(self.device)
            x_label = self.discriminator.forward(x)
            x_label = (x_label > 0.5).int().item()
            x_label_list.append(x_label)
            
        x_label_list = np.array(x_label_list, dtype=np.int32)

        f1 = f1_score(
            y_true=self.test_dataset.test_labels,
            y_pred=x_label_list
        )

        if self.data in ['MSL', 'SMAP', 'SMD']:
            logging.info(
                f'F1 score on {self.data} {self.subdata}: {round(f1, 4)}\n')

        return
    

class BidirectionalPerturbatorTrainer(object):
    def __init__(
        self,
        time: str,
        data: str,
        subdata: Optional[str],
        batch_size: int,
        window_size: int,
        recon_pert_mse_delta_min: float,
        pert_mse_delta_min: float,
        prior_var: float,
        sigma_pert_factor: float,
        recon_loss_weight: float,
        pert_loss_weight: float,
        zero_pert_loss_weight: float,
        kld_loss_weight: float,
        discriminator_loss_weight: float,
        gpu_num: int,
        learning_rate: int,
        epochs: int,
        save_interval: int,
    ) -> None:
        # Time and data information
        self.time = time
        self.data = data
        self.subdata = subdata
        self.batch_size = batch_size
        self.window_size = window_size
        
        # Training infos
        self.epochs = epochs
        self.save_interval = save_interval
        
        # Train and test dataset/dataloader
        self.train_dataset = PerturbationDataset(
            data=data,
            subdata=subdata,
            mode='train',
            window_size=window_size,
        )
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.test_dataset = PerturbationDataset(
            data=data,
            subdata=subdata,
            mode='test',
            window_size=window_size,
        )
        
        # Set device
        self.device = torch.device(f'cuda:{gpu_num}')

        # Perturbator and discriminator
        self.data_dim = self.train_dataset.data_dim
        assert self.data_dim > 1, "Data dimension should be > 1"

        self.perturbator = BidirectionalPerturbator(
            window_size=self.window_size,
            data_dim=self.data_dim,
            sigma_pert_factor=sigma_pert_factor,
        )
        self.perturbator = self.perturbator.to(self.device)
        # self.discriminator = TCNDiscriminator(
        #     window_size=self.window_size,
        #     data_dim=self.data_dim,
        # )
        self.discriminator = Discriminator(
            data_dim=self.data_dim,
            conv_hidden_dim=self.window_size//2,
            mlp_hidden_dim=self.data_dim//2,
        )
        self.discriminator = self.discriminator.to(self.device)

        # Set optimizer
        self.optimizer = Adam(
            params=list(self.perturbator.parameters()) \
                   + list(self.discriminator.parameters()),
            lr=learning_rate,
            # weight_decay=1e-3,
        )

        # Loss
        self.perturbator_loss = BidirectionalPerturbatorLoss(
            batch_size=batch_size,
            window_size=self.window_size,
            data_dim=self.data_dim,
            recon_pert_mse_delta_min=recon_pert_mse_delta_min,
            pert_mse_delta_min=pert_mse_delta_min,
            prior_var=prior_var,
            recon_loss_weight=recon_loss_weight,
            pert_loss_weight=pert_loss_weight,
            zero_pert_loss_weight=zero_pert_loss_weight,
            kld_loss_weight=kld_loss_weight,
        )
        self.discriminator_loss = TCNDiscriminatorLoss(batch_size=batch_size)
        self.discriminator_loss_weight = discriminator_loss_weight
        
        # Checkpoint directory
        self.ckpt_dir = os.path.join('exp', 'checkpoints', data)

        if subdata is not None:
            self.ckpt_dir = os.path.join(self.ckpt_dir, subdata)
        
        self.ckpt_dir = os.path.join(self.ckpt_dir, 'perturbator', time)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        return
    
    def save_model(self, epoch: int) -> None:
        torch.save(
            obj={
                'perturbator': self.perturbator.state_dict(),
                'discriminator': self.discriminator.state_dict(),
                'optimzier': self.optimizer.state_dict(),
                'epoch': epoch,
            },
            f=os.path.join(self.ckpt_dir, f'epoch_{epoch}.pt'),
        )

        return
    
    def plot_perturbation(
        self,
        time: str,
        epoch: int,
        figsize: Tuple[int, int] = (15, 20),
        ylim_upper: float = 10.0,
    ) -> None:
        var_num = self.data_dim // 5

        save_path = os.path.join('exp', 'figs', 'perturbator', time)

        if self.subdata is not None:
            save_path = os.path.join(save_path, self.subdata)

        save_path = os.path.join(save_path, f'epoch_{epoch}')
        os.makedirs(save_path, exist_ok=True)

        for i in range(len(self.train_dataset) // self.window_size):
            idx = i * self.window_size
            data = self.train_dataset[idx]
            data: Tensor = torch.tensor(data).float().unsqueeze(0)
            data = data.to(self.device)
            _, pert, _, _ = self.perturbator.forward(data)

            data = data.squeeze(0).detach().cpu().numpy()
            pert=  pert.squeeze(0).detach().cpu().numpy()

            fig, axes = plt.subplots(var_num, 5, figsize=figsize)
            axes = axes.flatten()

            for i in range(self.data_dim):
                axes[i].plot(data[:, i])
                axes[i].set_title(f"Dim {i+1}")
                axes[i].set_xticks([])
                axes[i].set_yticks([])
                axes[i].set_ylim(-ylim_upper, ylim_upper)
                axes[i].set_yticks([-ylim_upper, ylim_upper])

            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'data_{idx}.png'))
            plt.close()

            fig, axes = plt.subplots(var_num, 5, figsize=figsize)
            axes = axes.flatten()

            for i in range(self.data_dim):
                axes[i].plot(pert[:, i])
                axes[i].set_title(f"Dim {i+1}")
                axes[i].set_xticks([])
                axes[i].set_yticks([])
                axes[i].set_ylim(-ylim_upper, ylim_upper)
                axes[i].set_yticks([-ylim_upper, ylim_upper])

            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'neg_pair_{idx}.png'))
            plt.close()
        
        return
    
    def train(self) -> None:
        self.perturbator.train()
        self.discriminator.train()

        for epoch in range(self.epochs):
            epoch_total_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_pert_loss = 0.0
            epoch_zero_pert_loss = 0.0
            epoch_kld_loss = 0.0
            epoch_negative_bce_loss = 0.0
            epoch_positive_bce_loss = 0.0

            for x in self.train_loader:
                self.optimizer.zero_grad()

                x: Tensor = x.to(self.device).float()
                # x_recon, x_pert, temporal_mu, temporal_logvar, \
                # feature_mu, feature_logvar = self.perturbator.forward(x)
                x_recon, x_pert, \
                feature_mu, feature_logvar = self.perturbator.forward(x)

                x_label = self.discriminator.forward(x)
                x_pert_label = self.discriminator.forward(x_pert)

                total_pert_loss, recon_loss, pert_loss, \
                zero_pert_loss, kld_loss = self.perturbator_loss(
                    x=x,
                    x_recon=x_recon,
                    x_pert=x_pert,
                    # temporal_mu=temporal_mu,
                    # temporal_logvar=temporal_logvar,
                    feature_mu=feature_mu,
                    feature_logvar=feature_logvar,
                )
                total_bce_loss, negaitve_bce_loss, positive_bce_loss \
                = self.discriminator_loss(
                    x_label=x_label,
                    x_pert_label=x_pert_label,
                )

                total_loss = total_pert_loss \
                             + self.discriminator_loss_weight * total_bce_loss

                total_loss.backward()
                self.optimizer.step()
                epoch_total_loss += total_loss.item()
                epoch_recon_loss += recon_loss
                epoch_pert_loss += pert_loss
                epoch_zero_pert_loss += zero_pert_loss
                epoch_kld_loss += kld_loss
                epoch_negative_bce_loss += negaitve_bce_loss
                epoch_positive_bce_loss += positive_bce_loss
            
            epoch_total_loss /= len(self.train_loader)
            epoch_recon_loss /= len(self.train_loader)
            epoch_pert_loss /= len(self.train_loader)
            epoch_zero_pert_loss /= len(self.train_loader)
            epoch_kld_loss /= len(self.train_loader)
            epoch_negative_bce_loss /= len(self.train_loader)
            epoch_positive_bce_loss /= len(self.train_loader)

            logging.info(f'Epoch {epoch + 1} results:')
            logging.info(f'- Total loss: {epoch_total_loss:.4e}')
            logging.info(f'- Recon loss: {epoch_recon_loss:.4e}')
            logging.info(f'- Pert loss: {epoch_pert_loss:.4e}')
            logging.info(f'- Zero pert loss: {epoch_zero_pert_loss:.4e}')
            logging.info(f'- KLD loss: {epoch_kld_loss:.4e}')
            logging.info(f'- Negative BCE loss: {epoch_negative_bce_loss:.4e}')
            logging.info(f'- Positive BCE loss: {epoch_positive_bce_loss:.4e}\n')

            if (epoch + 1) % self.save_interval == 0 or epoch == 0:
                self.save_model(epoch=epoch+1)
                self.plot_perturbation(time=self.time, epoch=epoch+1)

        return

    def eval(self) -> None:
        self.discriminator.eval()
        pred_list = []

        for idx in range(len(self.test_dataset)):
            x = self.test_dataset[idx]
            x = torch.tensor(x).float().unsqueeze(0).to(self.device)
            pred = self.discriminator.forward(x)
            pred = (pred > 0.5).int().item()
            pred_list.append(pred)
        
        pred_list = np.array(pred_list, dtype=np.int32)

        f1 = f1_score(
            y_true=self.test_dataset.test_labels,
            y_pred=pred_list,
        )

        if self.data in ['MSL', 'SMAP', 'SMD']:
            logging.info(
                f'F1 score on {self.data} {self.subdata}: {round(f1, 4)}\n'
            )
        else:
            logging.info(f'F1 socre on {self.data}: {round(f1, 4)}\n')
        
        return
    

class PretextTrainer(object):
    def __init__(
        self,
        time: str,
        data: str,
        subdata: Optional[str],
        window_size: int,
        positive_augementor_time: str,
        perturbator_time: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        gpu_num: int,
        num_neighborhoods: int,
    ) -> None:
        if subdata is not None:
            logging.info(f'Pretext training on {data} {subdata} start...')
        else:
            logging.info(f'Pretext training on {data} start...')
        
        self.time = time
        
        train_dataset = PretextDataset(
            time=time,
            data=data,
            subdata=subdata,
            window_size=window_size,
            positive_augmentor_time=positive_augementor_time,
            perturbator_time=perturbator_time,
            processor_num=gpu_num,
        )
        data_dim = train_dataset.data_dim

        model = PretextModel(in_channels=data_dim, mid_channels=4)
        self.device = torch.device(f'cuda:{gpu_num}')
        self.model = model.to(self.device)
        self.criterion = pretextloss()

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        self.timeseries_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        self.learning_rate = learning_rate
        self.optimizer = Adam(params=self.model.parameters(), lr=learning_rate)
        self.epochs = epochs

        self.ckpt_dir = os.path.join('exp', 'checkpoints', 'pretext', data)
        self.classification_data_dir = os.path.join(
            'exp', 'data', 'classification_data', data
        )
        if data in ['MSL', 'SMAP', 'SMD']:
            self.ckpt_dir = os.path.join(self.ckpt_dir, subdata)
            self.classification_data_dir = os.path.join(
                self.classification_data_dir, subdata
            )
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.classification_data_dir, exist_ok=True)

        self.num_neighborhoods = num_neighborhoods

        return
    
    def _cosine_schedule(
        self,
        optimizer: Adam,
        current_epoch: int,
        total_epochs: int,
        initial_learning_rate: float,
        lr_decay_rate: float = 0.01,
    ) -> None:
        eta_min = initial_learning_rate * (lr_decay_rate ** 3)
        scheduled_learning_rate = eta_min \
        + (initial_learning_rate - eta_min) \
        * (1 + cos(pi * current_epoch / total_epochs)) / 2

        for param_group in optimizer.param_groups:
            param_group['lr'] = scheduled_learning_rate

        return

    def train(self) -> None:
        self.model.train()

        for epoch in range(self.epochs):
            self._cosine_schedule(
                optimizer=self.optimizer,
                current_epoch=epoch,
                total_epochs=self.epochs,
                initial_learning_rate=self.learning_rate,
            )
            epoch_loss = 0.0
            prev_loss = None

            for batch in self.train_loader:
                self.optimizer.zero_grad()
                anchor, positive_pair, negative_pair = batch
                anchor: Tensor = anchor.to(self.device)
                positive_pair: Tensor = positive_pair.to(self.device)
                negative_pair: Tensor = negative_pair.to(self.device)
                B, W, F = anchor.shape
                triplets = torch.cat(
                    tensors=[anchor, positive_pair, negative_pair],
                    dim=0,
                )
                triplets = triplets.view(3*B, F, W)
                representations = self.model.forward(triplets)
                loss = self.criterion(
                    representations=representations,
                    current_loss=prev_loss,
                )
                loss.backward()
                self.optimizer.step()
                prev_loss = loss.item()
                epoch_loss += prev_loss

            epoch_loss /= len(self.train_loader)
            logging.info(f'Epoch {epoch + 1} train loss: {epoch_loss:.4e}')
        
        # After training, save pretext model
        torch.save(
            obj={
                'resnet': self.model.resnet.state_dict(),
                'contrastive_head': self.model.contrastive_head.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            },
            f=os.path.join(self.ckpt_dir, f'{self.time}.pt')
        )

        return
    
    def select_neighbors(self) -> None:
        anchor_reps = []
        negative_reps = []

        for batch in self.timeseries_loader:
            anchor, _, negative_pair = batch
            anchor: Tensor = anchor.to(self.device)
            anchor = anchor.transpose(-2, -1)
            anchor_rep = self.model.forward(x=anchor).detach().cpu()
            anchor_reps.append(anchor_rep)

            negative_pair: Tensor = negative_pair.to(self.device)
            negative_pair = negative_pair.transpose(-2, -1)
            negative_rep = self.model.forward(x=negative_pair).detach().cpu()
            negative_reps.append(negative_rep)
        
        anchor_reps = torch.cat(anchor_reps, dim=0).numpy()
        negative_reps = torch.cat(negative_reps, dim=0).numpy()

        reps = np.concatenate([anchor_reps, negative_reps], axis=0)
        index_searcher = IndexFlatL2(reps.shape[1])
        
        assert index_searcher.d == reps.shape[1], \
        f'{index_searcher.d} != {reps.shape[1]}'

        index_searcher.add(reps)
        
        # Select nearest/furthest indicees of the anchor.
        nearest_indices_list = []
        furthest_indices_list = []

        for anchor_rep in anchor_reps:
            anchor_query = anchor_rep.reshape(1, -1)
            _, indices = index_searcher.search(anchor_query, reps.shape[0])
            indices = indices.reshape(-1)
            nearest_indices = indices[1: self.num_neighborhoods+1]
            furthest_indices = indices[-self.num_neighborhoods:]
            nearest_indices_list.append(nearest_indices)
            furthest_indices_list.append(furthest_indices)

        # Saving nearest/furthest indeces of the anchor.
        nearest_indices_list = np.array(nearest_indices_list)
        furthest_indices_list = np.array(furthest_indices_list)
        np.save(
            file=os.path.join(
                self.classification_data_dir,
                f'anchor_nn_indices_{self.time}.npy'
            ),
            arr=nearest_indices_list,
        )
        np.save(
            file=os.path.join(
                self.classification_data_dir,
                f'anchor_fn_indices_{self.time}.npy'
            ),
            arr=furthest_indices_list,
        )

        # Select nearest/furthest indicees of the negative pair.
        nearest_indices_list = []
        furthest_indices_list = []

        for negative_rep in negative_reps:
            negative_query = negative_rep.reshape(1, -1)
            _, indices = index_searcher.search(negative_query, reps.shape[0])
            indices = indices.reshape(-1)
            nearest_indices = indices[1: self.num_neighborhoods+1]
            furthest_indices = indices[-self.num_neighborhoods:]
            nearest_indices_list.append(nearest_indices)
            furthest_indices_list.append(furthest_indices)

        # Saving nearest/furthest indeces of the negaitve pair.
        nearest_indices_list = np.array(nearest_indices_list)
        furthest_indices_list = np.array(furthest_indices_list)
        np.save(
            file=os.path.join(
                self.classification_data_dir,
                f'negative_nn_indices_{self.time}.npy'
            ),
            arr=nearest_indices_list,
        )
        np.save(
            file=os.path.join(
                self.classification_data_dir,
                f'negative_fn_indices_{self.time}.npy'
            ),
            arr=furthest_indices_list,
        )

        print('\nPretext stage done. Moving on to classification stage.\n')

        return


class ClassificationTrainer(object):
    def __init__(
        self,
        time: str,
        data: str,
        subdata: Optional[str],
        window_size: int,
        gpu_num: int,
        epochs: int,
        batch_size: int,
        learning_rate: float
    ) -> None:
        self.time = time
        self.data = data
        self.subdata = subdata
        self.device = torch.device(f'cuda:{gpu_num}')
        self.epochs = epochs
        train_dataset = ClassificationDatasaet(
            time=time,
            data=data,
            subdata=subdata,
            window_size=window_size,
            mode='train',
        )
        self.test_dataset = ClassificationDatasaet(
            time=time,
            data=data,
            subdata=subdata,
            window_size=window_size,
            mode='test',
        )
        data_dim = train_dataset.data_dim
        model = ClassificationModel(in_channels=data_dim)

        resnet_dir = os.path.join('exp', 'checkpoints', 'pretext', data)
        classification_data_dir = os.path.join(
            'exp', 'data', 'classification_data', data)
        self.ckpt_dir = os.path.join('exp', 'checkpoints', 'classification', data)

        if data in ['MSL', 'SMAP', 'SMD']:
            resnet_dir = os.path.join(resnet_dir, subdata)
            classification_data_dir = os.path.join(
                classification_data_dir, subdata
            )
            self.ckpt_dir = os.path.join(self.ckpt_dir, subdata)
        
        os.makedirs(self.ckpt_dir, exist_ok=True)

        resnet_ckpt = torch.load(os.path.join(resnet_dir, f'{time}.pt'))
        model.resnet.load_state_dict(resnet_ckpt['resnet'])
        self.model = model.to(self.device)

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=learning_rate,
        )
        self.criterion = classificationloss()

        logging.info(f'Classification training on {data} {subdata} start...\n')

        return
    
    def train(self) -> None:
        self.model.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_consistency_loss = 0.0
            epoch_inconsistency_loss = 0.0
            epoch_entropy_loss = 0.0

            for batch in self.train_loader:
                self.optimizer.zero_grad()
                batch_loss = torch.zeros(1, device=self.device)
                batch_consistency_sum = torch.zeros(1, device=self.device)
                batch_consistency = 0.0
                batch_inconsistency = 0.0
                
                window, nearest_neighbor, furthest_neighbor = batch
                window: Tensor = window.to(self.device)
                window = window.transpose(-2, -1)
                window_logit = self.model.forward(window)
                entropy_loss = entropy(torch.mean(window_logit, dim=0))
                batch_loss -= entropy_loss
                epoch_entropy_loss += entropy_loss.item()
                
                nearest_neighbor: Tensor = nearest_neighbor.to(self.device)
                nearest_neighbor = nearest_neighbor.transpose(-2, -1)
                nearest_logit = self.model.forward(nearest_neighbor)
                
                furthest_neighbor: Tensor = furthest_neighbor.to(self.device)
                furthest_neighbor = furthest_neighbor.transpose(-2, -1)
                furthest_logit = self.model.forward(furthest_neighbor)

                consistency_sum, consistency, inconsistency = \
                self.criterion(window_logit, nearest_logit, furthest_logit)

                batch_consistency_sum += consistency_sum
                batch_consistency += consistency
                batch_inconsistency += inconsistency

                batch_loss += batch_consistency_sum
                epoch_loss += batch_loss.item()
                epoch_consistency_loss += batch_consistency
                epoch_inconsistency_loss += batch_inconsistency

                batch_loss.backward()
                self.optimizer.step()
            
            epoch_consistency_loss /= len(self.train_loader)
            epoch_inconsistency_loss /= len(self.train_loader)
            epoch_entropy_loss /= len(self.train_loader)
            epoch_loss /= len(self.train_loader)

            logging.info(f'Epoch {epoch + 1} loss:')
            logging.info(
                f'- Consistency loss: {round(epoch_consistency_loss, 4)}'
            )
            logging.info(
                f'- Inconsistency loss: {round(epoch_inconsistency_loss, 4)}'
            )
            logging.info(
                f'- Entropy loss: {round(epoch_entropy_loss, 4)}'
            )
            logging.info(f'- Total loss: {round(epoch_loss, 4)}\n')

        torch.save(
            obj={
                'model': self.model.state_dict(),
                'optim': self.optimizer.state_dict(),
            },
            f=os.path.join(self.ckpt_dir, f'{self.time}.pt')
        )

        if self.data in ['MSL', 'SMAP', 'SMD']:
            logging.info(f'Starting inference on {self.data} {self.subdata}\n')
        else:
            logging.info(f'Starting inference on {self.data}\n')
        
        return
    
    def inference(self) -> None:
        self.model.eval()
        inference_logits = []

        for idx in range(len(self.test_dataset)):
            test_data = torch.tensor(self.test_dataset[idx])
            test_data = test_data.to(self.device)
            test_data = test_data.unsqueeze(0).transpose(-2, -1)
            inference_logit = self.model.forward(test_data)
            inference_logit = inference_logit.squeeze(0).detach().cpu().numpy()
            inference_logits.append(inference_logit)
        
        classes = [0 for _ in range(10)]

        for inference_logit in inference_logits:
            max_idx = np.argmax(inference_logit)
            classes[max_idx] += 1
        
        major_class = classes.index(max(classes))

        anomaly_scores = []

        for inference_logit in inference_logits:
            major_probability = inference_logit[major_class]
            anomaly_scores.append(1 - major_probability)
        
        anomaly_scores = np.array(anomaly_scores)

        precision, recall, thresholds = precision_recall_curve(
            y_true=self.test_dataset.test_labels,
            y_score=anomaly_scores,
        )

        auc_pr = auc(recall, precision)

        best_threshold = 0
        best_precision = 0
        best_recall = 0
        best_f1 = 0

        for i in range(len(thresholds)):
            f1_score = f1score(precision[i], recall[i])
            if f1_score > best_f1:
                best_f1 = f1_score
                best_precision = precision[i]
                best_recall = recall[i]
                best_threshold = thresholds[i]

        logging.info(f'- Best F1 score: {round(best_f1, 4)}')
        logging.info(f'- Best Precision: {round(best_precision, 4)}')
        logging.info(f'- Best Recall: {round(best_recall, 4)}')
        logging.info(f'- Best Threshold: {round(best_threshold, 4)}')
        logging.info(f'- AUC-PR: {round(auc_pr, 4)}')

        best_anomaly_prediction \
        = np.where(anomaly_scores >= best_threshold, 1, 0)
        
        best_f1_score, best_tp, best_fp, best_fn = f1_stat(
            prediction=best_anomaly_prediction,
            gt=self.test_dataset.test_labels
        )
        logging.info(f'- Best True Positive: {best_tp}')
        logging.info(f'- Best False Positive: {best_fp}')
        logging.info(f'- Best False Negative: {best_fn}')

        return best_f1_score, best_tp, best_fp, best_fn, auc_pr      
