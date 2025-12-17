import logging
from tqdm import tqdm
from math import cos, pi
from faiss import IndexFlatL2
from sklearn.metrics import precision_recall_curve, auc, f1_score
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from exp.models import *
from exp.data_factory import *
from exp.utils.common_import import *
from exp.utils.loss import *
from exp.utils.metric import *
from exp.utils.utils import *


class CUTSplusTrainer(object):
    def __init__(
        self,
        time: str,
        data: str,
        subdata: Optional[str],
        batch_size: int,
        window_size: int,
        seed: int,
        predict_data: bool = True,
        discover_graph: bool = True,
        gpu_num: int = 0,
        epochs: int = 50,
        save_interval: int = 5,
        pred_start_lr: float = 1e-2,
        pred_end_lr: float = 1e-3,
        gumbel_start_tau: float = 1.0,
        gumbel_end_tau: float = 0.1,
        graph_start_lambda: float = 0.1,
        graph_end_lambda: float = 0.01,
        graph_start_lr: float = 1e-3,
        graph_end_lr: float = 1e-4,
        graph_plot_step: int = 10,
    ) -> None:
        # Training infos
        self.time = time
        self.data = data
        self.subdata = subdata
        self.batch_size = batch_size
        self.window_size = window_size
        self.seed = seed
        self.predict_data = predict_data
        self.discover_graph = discover_graph
        self.gpu_num = gpu_num
        self.epochs = epochs
        self.save_interval = save_interval
        self.pred_start_lr = pred_start_lr
        self.pred_end_lr = pred_end_lr
        self.gumbel_start_tau = gumbel_start_tau
        self.gumbel_end_tau = gumbel_end_tau
        self.graph_start_lambda = graph_start_lambda
        self.graph_end_lambda = graph_end_lambda
        self.graph_start_lr = graph_start_lr
        self.graph_end_lr = graph_end_lr
        self.graph_plot_step = graph_plot_step

        self.log_dir = os.path.join('log', 'cuts_plus', data)
        os.makedirs(self.log_dir, exist_ok=True)

        self.plot_dir = self.log_dir

        if subdata is not None:
            self.plot_dir = os.path.join(self.plot_dir, subdata)
        
        self.plot_dir = os.path.join(self.plot_dir, time)
        os.makedirs(self.plot_dir, exist_ok=True)

        self.ckpt_dir = os.path.join('exp', 'checkpoints', data)
        
        if subdata is not None:
            self.ckpt_dir = os.path.join(self.ckpt_dir, subdata)

        self.ckpt_dir = os.path.join(self.ckpt_dir, 'cuts_plus', time)

        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.batch_size = batch_size

        train_dataset = CUTSplusDataset(
            dataset=data,
            subdata=subdata,
            mode='train',
            window_size=window_size,
            train_ratio=0.8
        )
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        val_dataset = CUTSplusDataset(
            dataset=data,
            subdata=subdata,
            mode='val',
            window_size=window_size,
            train_ratio=0.8,
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        self.predict_data = predict_data
        self.discover_graph = discover_graph
        self.n_nodes = train_dataset.data_dim
        self.window_size = window_size
        self.prediction_step = 1
        self.input_step = self.window_size - self.prediction_step
        self.graph_start_lr = graph_start_lr
        self.graph_end_lr = graph_end_lr
        self.graph_plot_step = graph_plot_step
        self.epochs = epochs
        self.save_interval = save_interval

        model = CUTS_Plus_Net(n_nodes=self.n_nodes, data_dim=1)
        self.device = torch.device(f'cuda:{gpu_num}')
        self.model = model.to(self.device)

        self.criterion = nn.MSELoss()
        self.pred_optimizer = Adam(
            params=[p for n, p in model.named_parameters() if n != 'gt'],
            lr=pred_start_lr,
        )

        pred_scheduler_gamma = (pred_end_lr / pred_start_lr) ** (1 / epochs)
        self.pred_scheduler = StepLR(
            optimizer=self.pred_optimizer,
            step_size=1,
            gamma=pred_scheduler_gamma
        )

        self.gumbel_tau_gamma \
        = (gumbel_end_tau / gumbel_start_tau) ** (1 / epochs)
        self.gumbel_tau = gumbel_start_tau

        self.graph_lambda_gamma \
        = (graph_end_lambda / graph_start_lambda) ** (1 / epochs)
        self.graph_lambda = graph_start_lambda

        return
    
    def prediction(
        self,
        previous_values: Tensor,
        prediction_gt: Tensor,
    ) -> Tensor:
        # Define bernouli sampling for each batch
        def sample_bernoulli(sample_matrix: Tensor, batch_size: int) -> Tensor:
            sample_matrix = sample_matrix[None].expand(batch_size, -1, -1)
            return torch.bernoulli(sample_matrix).float()
        
        self.model.train()
        self.pred_optimizer.zero_grad()

        gt_prob = self.model.gt
        g_prob = self.g

        graph = torch.einsum("nm,ml->nl", g_prob, torch.sigmoid(gt_prob))

        graph_sampled = sample_bernoulli(
            sample_matrix=graph,
            batch_size=self.batch_size
        )

        prediction = self.model.forward(
            x=previous_values,
            fwd_graph=graph_sampled
        )
        prediction = prediction.transpose(1, 2)
        assert prediction_gt.shape == prediction.shape

        pred_loss = self.criterion.forward(prediction, prediction_gt)
        pred_loss.backward()
        self.pred_optimizer.step()

        return pred_loss
    
    def update_optim_sched(self, epoch: Optional[int]) -> None:
        if epoch == None:
            epoch = 0
        
        gamma = (self.graph_end_lr / self.graph_start_lr) ** (1 / self.epochs)
        self.graph_optimizer = Adam(
            params=[self.model.gt],
            lr=self.graph_start_lr * gamma ** epoch,
        )
        self.graph_scheduler = StepLR(
            optimizer=self.graph_optimizer,
            step_size=1,
            gamma=gamma,
        )

        return

    def graph_discovery(
        self,
        previous_values: Tensor,
        prediction_gt: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # Define Gumbel sigmoid sampling for each batch
        def gumbel_sigmoid_sample(
            graph: Tensor,
            batch_size: int,
            tau: float = 1.0,
        ) -> Tensor:
            prob = graph[None, :, :, None].expand(batch_size, -1, -1, -1)
            logits = torch.concat([prob, (1-prob)], axis=-1)
            samples = gumbel_softmax(logits, tau=tau, hard=True)[:, :, :, 0]
            return samples
        
        _, n = self.model.gt.shape
        self.graph_optimizer.zero_grad()

        gt_prob = self.model.gt
        g_prob = self.g

        graph = torch.einsum("nm,ml->nl", g_prob, torch.sigmoid(gt_prob))
        graph_sampled = gumbel_sigmoid_sample(
            graph=graph,
            batch_size=self.batch_size,
        )
        graph_loss_sparsity \
        = torch.linalg.norm(graph.flatten(), ord=1) / (n * n)

        prediction = self.model.forward(
            x=previous_values,
            fwd_graph=graph_sampled
        )
        prediction = prediction.transpose(1, 2)
        assert prediction_gt.shape == prediction.shape

        graph_loss_data = self.criterion.forward(prediction, prediction_gt)
        
        graph_loss: Tensor \
        = graph_loss_sparsity * self.graph_lambda + graph_loss_data

        graph_loss.backward()
        self.graph_optimizer.step()

        return graph_loss, graph_loss_sparsity, graph_loss_data
    
    def train(self) -> None:
        pbar = tqdm(total=self.epochs)
        metric_best = float('inf')
        graph_discover_step = 0

        for epoch in range(self.epochs):
            self.g = torch.eye(self.n_nodes).to(self.device)
            self.update_optim_sched(epoch=epoch)
            epoch_prediction_loss = 0.0
            epoch_graph_loss = 0.0

            if self.predict_data:
                for batch in self.train_loader:
                    batch: Tensor = batch.to(self.device)
                    previous_values = batch[:, :self.input_step]
                    prediction_gt = batch[:, self.input_step:]
                    prediction_loss = self.prediction(
                        previous_values=previous_values,
                        prediction_gt=prediction_gt,
                    )
                    epoch_prediction_loss += prediction_loss.item()
                    pbar.set_postfix_str(
                        f"S1 loss={prediction_loss.item():.2f}, spr=IDLE"
                    )
            
                self.pred_scheduler.step()

            if self.discover_graph:
                for batch in self.train_loader:
                    batch: Tensor = batch.to(self.device)
                    previous_values = batch[:, :self.input_step]
                    prediction_gt = batch[:, self.input_step:]
                    graph_discover_step += self.batch_size
                    _, graph_loss_sparsity, graph_loss_data = \
                    graph_loss, graph_loss_sparsity, graph_loss_data = \
                        self.graph_discovery(
                        previous_values=previous_values,
                        prediction_gt=prediction_gt,
                    )
                    epoch_graph_loss += graph_loss.item()

                    pbar.set_postfix_str(
                        f"S2 loss={graph_loss_data.item():.2f}, spr={graph_loss_sparsity.item():.2f}"
                    )
                
                self.graph_scheduler.step()
                self.gumbel_tau *= self.gumbel_tau_gamma
                self.graph_lambda *= self.graph_lambda_gamma
            
            pbar.update(1)

            graph = torch.einsum(
                "nm,ml->nl", self.g, torch.sigmoid(self.model.gt)
            ).detach().cpu().numpy()

            # Plot graph onnve in graph_plot_step epochs
            if epoch == 0 or (epoch + 1) % self.graph_plot_step == 0:
                plot_matrix(
                    name='Graph',
                    matrix=graph,
                    plot_dir=self.plot_dir,
                    log_step=graph_discover_step,
                )
                np.save(
                    file=os.path.join(
                        self.plot_dir, f'graph_epoch_{epoch + 1}.npy'
                    ),
                    arr=graph,
                )
            epoch_prediction_loss /= len(self.train_loader)
            epoch_graph_loss /= len(self.train_loader)

            logging.info(f'Epoch {epoch + 1} train results:')
            logging.info(f'- Prediction loss: {epoch_prediction_loss:.4e}')
            logging.info(f'- Graph loss: {epoch_graph_loss:.4e}')

            # Evaluation
            self.model.eval()
            total_loss = 0
            total_samples = 0

            with torch.no_grad():
                for batches in self.val_loader:
                    batches: Tensor = batches.to(self.device).float()
                    previous_values = batches[:, :self.input_step]
                    prediction_gt = batches[:, self.input_step:]
                    graph = torch.einsum(
                        "nm,ml->nl", self.g, torch.sigmoid(self.model.gt)
                    )
                    graph = graph[None].expand(previous_values.size(0), -1, -1)
                    prediction = self.model.forward(
                        x=previous_values,
                        fwd_graph=graph
                    )
                    prediction = prediction.transpose(1, 2)
                    assert prediction_gt.shape == prediction.shape

                    loss = self.criterion.forward(
                        prediction,
                        prediction_gt,
                    )
                    total_loss += loss.item() * previous_values.size(0)
                    total_samples += previous_values.size(0)
            
            average_loss = total_loss / total_samples
            is_best = average_loss < metric_best

            logging_info = f'- Evaluation prediction loss: {average_loss:.4e}\n'

            if is_best:
                logging_info = f'- Evaluation prediction loss: {average_loss:.4e} [New best]\n'
                checkpoint_best = {
                    "epoch": epoch + 1,
                    "model": self.model.state_dict(),
                    "pred_optim": self.pred_optimizer.state_dict(),
                    "graph_optim": self.graph_optimizer.state_dict(),
                    "cuasal_matrix": self.model.gt,
                    "seed": self.seed,
                }
                torch.save(
                    obj=checkpoint_best,
                    f=os.path.join(self.ckpt_dir, 'best_model.pt'),
                )
                metric_best = average_loss

            logging.info(logging_info)            

            if (epoch + 1) % self.save_interval == 0:
                checkpoint = {
                    "epoch": epoch + 1,
                    "model": self.model.state_dict(),
                    "pred_optim": self.pred_optimizer.state_dict(),
                    "graph_optim": self.graph_optimizer.state_dict(),
                    "cuasal_matrix": self.model.gt,
                    "seed": self.seed,
                }
                torch.save(
                    obj=checkpoint,
                    f=os.path.join(
                        self.ckpt_dir, f'epoch_{epoch + 1}.pt'
                    ),
                )
        
        return


class TCNPerturbatorTrainer(object):
    def __init__(
        self,
        time: str,
        data: str,
        subdata: Optional[str],
        batch_size: int,
        window_size: int,
        seed: int,
        recon_pert_mse_delta_min: float = 0.1,
        pert_mse_delta_min: float = 0.2,
        prior_var: float = 0.5,
        recon_loss_weight: float = 1.0,
        pert_loss_weight: float = 0.1,
        zero_pert_loss_weight: float = 0.01,
        kld_loss_weight: float = 0.1,
        discriminator_loss_weight: float = 5.0,
        gpu_num: int = 0,
        learning_rate: float = 1e-4,
        epochs: int = 50,
        save_interval: int = 5,
    ) -> None:
        # Training infos
        self.time = time
        self.data = data
        self.subdata = subdata
        self.batch_size = batch_size
        self.window_size = window_size
        self.recon_pert_mse_delta_min = recon_pert_mse_delta_min
        self.pert_mse_delta_min = pert_mse_delta_min
        self.prior_var = prior_var
        self.recon_loss_weight = recon_loss_weight
        self.pert_loss_weight = pert_loss_weight
        self.zero_pert_loss_weight = zero_pert_loss_weight
        self.kld_loss_weight = kld_loss_weight
        self.discriminator_loss_weight = discriminator_loss_weight
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.save_interval = save_interval
        self.seed = seed
        
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

        self.perturbator = TCNPerturbator(
            window_size=self.window_size,
            data_dim=self.data_dim,
            latent_perturbator_factor=2.0,
        )
        self.perturbator = self.perturbator.to(self.device)
        self.discriminator = TCNDiscriminator(
            window_size=self.window_size,
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
        )
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
                'epoch': epoch,
                'perturbator': self.perturbator.state_dict(),
                'discriminator': self.discriminator.state_dict(),
                'optimzier': self.optimizer.state_dict(),
                'epoch': epoch,
                'seed': self.seed,
            },
            f=os.path.join(self.ckpt_dir, f'epoch_{epoch}.pt'),
        )

        return
    
    def plot_perturbation(
        self,
        epoch: int,
        figsize: Tuple[int, int] = (15, 20),
        ylim_upper: float = 10.0,
    ) -> None:
        var_num = self.data_dim // 5

        save_path = os.path.join('exp', 'figs', 'perturbator', self.data)

        if self.subdata is not None:
            save_path = os.path.join(save_path, self.subdata)
        
        save_path = os.path.join(save_path, self.time, f'epoch_{epoch}')
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

                x: Tensor = x.to(self.device)
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

            if epoch == 0 or (epoch + 1) % self.save_interval == 0:
                self.save_model(epoch=epoch+1)
                
            if (epoch + 1) == self.epochs:
                self.plot_perturbation(epoch=epoch+1)

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
        seed: int,
        window_size: int,
        positive_augementor_time: str,
        perturbator_time: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        gpu_num: int,
        num_neighborhoods: int,
        apply_patch: bool = False,
    ) -> None:
        if subdata is not None:
            logging.info(f'Pretext training on {data} {subdata} start...\n')
        else:
            logging.info(f'Pretext training on {data} start...\n')
        
        self.time = time
        
        train_dataset = PretextDataset(
            time=time,
            data=data,
            subdata=subdata,
            window_size=window_size,
            seed=seed,
            positive_augmentor_time=positive_augementor_time,
            perturbator_time=perturbator_time,
            processor_num=gpu_num,
            apply_patch=apply_patch,
        )
        data_dim = train_dataset.data_dim

        model = PretextModel(in_channels=data_dim, mid_channels=4)
        self.device = torch.device(f'cuda:{gpu_num}')
        self.model = model.to(self.device)
        self.criterion = PretextLoss()

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

        self.ckpt_dir = os.path.join('exp', 'checkpoints', data)
        self.classification_data_dir = os.path.join(
            'exp', 'data', 'classification_data', data
        )
        if data in ['MSL', 'SMAP', 'SMD']:
            self.ckpt_dir = os.path.join(self.ckpt_dir, subdata)
            self.classification_data_dir = os.path.join(
                self.classification_data_dir, subdata
            )
        self.ckpt_dir = os.path.join(self.ckpt_dir, 'pretext')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.classification_data_dir, exist_ok=True)

        self.num_neighborhoods = num_neighborhoods
        self.apply_patch = apply_patch
        self.seed = seed

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
                anchor: Tensor = anchor.to(self.device).float()
                positive_pair: Tensor = positive_pair.to(self.device).float()
                negative_pair: Tensor = negative_pair.to(self.device).float()
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
            
            if (epoch + 1) == self.epochs:
                logging.info(
                    f'Epoch {epoch +1} train loss: {epoch_loss:.4e}\n'
                )
        
        # After training, save pretext model
        torch.save(
            obj={
                'resnet': self.model.resnet.state_dict(),
                'contrastive_head': self.model.contrastive_head.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'seed': self.seed
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

        print('Pretext stage done. Moving on to classification stage.\n')

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

        resnet_dir = os.path.join('exp', 'checkpoints', data)
        classification_data_dir = os.path.join(
            'exp', 'data', 'classification_data', data)
        self.ckpt_dir = os.path.join('exp', 'checkpoints', data)

        if data in ['MSL', 'SMAP', 'SMD']:
            resnet_dir = os.path.join(resnet_dir, subdata)
            classification_data_dir = os.path.join(
                classification_data_dir, subdata
            )
            self.ckpt_dir = os.path.join(self.ckpt_dir, subdata)
        
        resnet_dir = os.path.join(resnet_dir, 'pretext')
        self.ckpt_dir = os.path.join(self.ckpt_dir, 'classification')
        
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
        self.criterion = ClassificationLoss()

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
                window = window.transpose(-2, -1).float()
                window_logit = self.model.forward(window)
                entropy_loss = entropy(torch.mean(window_logit, dim=0))
                batch_loss -= entropy_loss
                epoch_entropy_loss += entropy_loss.item()
                
                nearest_neighbor: Tensor = nearest_neighbor.to(self.device)
                nearest_neighbor = nearest_neighbor.transpose(-2, -1).float()
                nearest_logit = self.model.forward(nearest_neighbor)
                
                furthest_neighbor: Tensor = furthest_neighbor.to(self.device)
                furthest_neighbor = furthest_neighbor.transpose(-2, -1).float()
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
            test_data = test_data.to(self.device).float()
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
        logging.info(f'- Best False Negative: {best_fn}\n')

        return best_f1_score, best_tp, best_fp, best_fn, auc_pr      
