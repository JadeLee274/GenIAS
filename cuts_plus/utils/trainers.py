import logging
from tqdm import tqdm
from cuts_plus.data_factory.loader import Loader
from cuts_plus.models.cuts_plus import CUTS_Plus_Net
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from cuts_plus.utils.imports import *
from cuts_plus.utils.functions import gumbel_softmax
from cuts_plus.utils.utils import plot_matrix


class CUTS_PLUS_Trainer(object):
    def __init__(
        self,
        time: str,
        data: str,
        subdata: Optional[str],
        window_size: int,
        batch_size: int,
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
        self.log_dir = os.path.join('log', 'cuts_plus', data)
        os.makedirs(self.log_dir, exist_ok=True)

        self.plot_dir = self.log_dir
        if subdata is not None:
            self.plot_dir = os.path.join(self.plot_dir, subdata)
        self.plot_dir = os.path.join(self.plot_dir, time)
        os.makedirs(self.plot_dir, exist_ok=True)

        self.checkpoint_dir = os.path.join('cuts_plus', 'checkpoints', data)
        if subdata is not None:
            self.checkpoint_dir = os.path.join(self.checkpoint_dir, subdata)
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, time)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.batch_size = batch_size

        train_dataset = Loader(
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
        val_dataset = Loader(
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

            logging.info(f'\nEpoch {epoch + 1} train results:')
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
                }
                torch.save(
                    obj=checkpoint_best,
                    f=os.path.join(self.checkpoint_dir, 'best_model.pt'),
                )
                metric_best = average_loss

            logging.info(logging_info)            

            if (epoch + 1) % self.save_interval == 0:
                checkpoint = {
                    "model": self.model.state_dict(),
                    "pred_optim": self.pred_optimizer.state_dict(),
                    "graph_optim": self.graph_optimizer.state_dict(),
                    "cuasal_matrix": self.model.gt,
                }
                torch.save(
                    obj=checkpoint,
                    f=os.path.join(
                        self.checkpoint_dir, f'epoch_{epoch + 1}.pt'
                    ),
                )
        
        return
