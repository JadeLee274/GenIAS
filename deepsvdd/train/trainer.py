import os, logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from data_factory import DeepSVDDDataset
from utils.common_import import *
from deepsvdd.model import SVDD

class Trainer():
    
    def __init__(
        self, model = SVDD, nu: float = 0.1, c : Optional[Tensor] = None,
        device: str = 'cuda:0',
        ) -> None:
        
        self.model = model
        self.nu = nu
        self.device = device
        self.c = c
        self.R = 0.0

    def train(
        self, dataset: DeepSVDDDataset, total_epoch: int,
        batch_size: int = 300, lr: float = 1e-2, weight_decay: float = 1e-2,
        type_obj: str = 'oneclass', load_epoch: int = 0
        ) -> None:

        ckpt_path = os.path.join('chekpoints/deepsvdd')
        self.model = self.model.to(self.device)
        train_loader = DataLoader(dataset=dataset, batch_size=batch_size)
        optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay 
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=total_epoch // 25, gamma=0.9
            )
        
        if load_epoch > 0:
            _loaded_dict = self._load_model(target={
                'R': self.R, 'c': self.c, 'model': self.model,
                'optim': optimizer, 'sched': scheduler
                }, path=ckpt_path, epoch=load_epoch) 
            breakpoint()
            optimizer.state_dict = _loaded_dict['optim'].state_dict
            scheduler.state_dict = _loaded_dict['sched'].state_dict
        if self.c is None:
            self.init_center_c(train_loader)

        self.model.train()

        for epoch in range(load_epoch, total_epoch):
            loss = 0.0
            for batch in train_loader:
                batch = batch.to(self.device).to(torch.float32)
                output = self.model.forward(batch)
                if type_obj == 'oneclass':
                    loss_batch = F.mse_loss(output, self.c)
                elif type_obj == 'softbdry':
                    dist = ((output - self.c) ** 2).sum(dim=(1, 2))
                    ball_loss = dist - self.R ** 2
                    loss_batch = F.relu(ball_loss).mean(dim=0) / self.nu + self.R ** 2
                else:
                    NotImplementedError
                optimizer.zero_grad()
                loss_batch.backward()
                optimizer.step()
                if type_obj == 'softbdry':
                    self.update_radius(dist)
                loss += loss_batch.item()
            loss /= len(train_loader)
            scheduler.step()
            logging.info(f'EPOCH {epoch:04d} | loss {loss:.6f}')
            if (epoch + 1) % 100 == 0:
                self._save_model(target={
                    'R': self.R, 'c': self.c, 'model': self.model,
                    'optim': optimizer, 'sched': scheduler
                    }, path=ckpt_path, epoch=epoch + 1) 

    def test(
        self, dataset: DeepSVDDDataset, load_model: Optional[nn.Module] = None,
        type_obj: str = 'oneclass'
        ) -> ...:
        if load_model is not None:
            self.model = load_model
            self.model.to(self.device)
        self.model.eval()
        test_loader = DataLoader(dataset=dataset, batch_size=1)
        score = []
        with torch.no_grad():
            for batch, label in test_loader:
                batch = batch.to(self.device).to(torch.float32)
                output = self.model.forward(batch)
                if type_obj == 'oneclass':
                    dist = F.mse_loss(output, self.c)
                elif type_obj == 'softbdry':
                    dist = F.mse_loss(output - self.c) - self.R ** 2
                else:
                    NotImplementedError
                score.append([label.item(), dist.item()])
        self.test_score = np.array(score)
        self.test_auc = roc_auc_score(
            self.test_score[0].astype(np.int16), self.test_score[1]
            )
        logging.info(
            f'Anomaly score {self.test_score[1].mean().item():.4f} | ' \
            f'ROC-AUC score {self.test_auc:.4f}'
            )
        
    def init_center_c(self, train_loader: DataLoader, eps: float = 0.1) -> Tensor:
        window_size = next(iter(train_loader)).shape[1]
        c = torch.zeros(
            (window_size, self.model.representation_dim), device=self.device
            )
        self.model.eval()
        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(self.device).to(torch.float32)
                output = self.model.forward(batch)
                c += torch.sum(output, dim=0)
        c /= len(train_loader)
        self.c = torch.where(abs(c) < eps, torch.sign(c) * eps, c)
    
    def _save_model(
        self, target: Dict[str, Tuple[
            Tensor, nn.Module, optim.Optimizer, optim.lr_scheduler.StepLR
            ]], 
        path: str, epoch: int
        ) -> None:
        for name, obj in target.items():
            if hasattr(obj, 'state_dict'):
                torch.save(obj.state_dict(), os.path.join(
                    path, f'{name}_{epoch}.pt'
                    ))
            else:
                torch.save(obj, os.path.join(path, f'{name}_{epoch}.pt'))

    def _load_model(
        self, target: Dict[str, Tuple[
            Tensor, nn.Module, optim.Optimizer, optim.lr_scheduler.StepLR
            ]], 
        path: str, epoch: int
        ) -> None:
        for name, obj in target.items():
            if hasattr(obj, 'state_dict'):
                breakpoint()
                obj.load_state_dict(torch.load(os.path.join(
                    path, f'{name}_{epoch}.pt',
                    ), weights_only=True))
            else:
                obj = torch.load(os.path.join(
                    path, f'{name}_{epoch}.pt'
                    ))
        return target
        
    def update_radius(self, dist: Tensor) -> float:
        self.R = torch.quantile(torch.sqrt(dist), 1 - self.nu)