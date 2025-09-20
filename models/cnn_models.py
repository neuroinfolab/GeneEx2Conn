# models/cnn_models.py
from env.imports import *
from data.data_utils import create_data_loader  
from models.train_val import train_model
from models.base_models import BaseModel  
from typing import Tuple
from torch import optim

class ConvNet1D(nn.Module):
    """
    1d conv：
    - input:(batch, features), reshape  (batch, 1, features)
    -  Conv1d + ReLU + (optinal) BN + Dropout
    - After flattening, connect two layers of fully connected output scalars (regression); If binarize=True, it will be trained in binary classification

    """
    def __init__(
        self,
        input_dim: int,
        binarize = None,
        # conv hparameter
        channels: Tuple[int, int, int] = (32, 64, 128),
        kernel_sizes: Tuple[int, int, int] = (7, 5, 3),
        strides: Tuple[int, int, int] = (1, 1, 1),
        paddings: Tuple[int, int, int] = (3, 2, 1),
        use_bn: bool = True,
        dropout_rate: float = 0.1,
        # fc width
        fc_hidden: int = 256,
        # training hpara
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        epochs: int = 100,
        # early stop / lr 
        patience: int = 50,
        lr_factor: float = 0.3,
        lr_patience: int = 20,
        lr_threshold: float = 0.05,
        lr_cooldown: int = 1,
        min_lr: float = 1e-6,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.binarize = bool(binarize) if binarize is not None else False
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout_rate = dropout_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ▼▼▼  START ▼▼▼
        # 1. BatchNorm1d 
        #    参数 1 是因为我们的输入在 unsqueeze 后有 1 个通道 (channel)
        self.input_batch_norm = nn.BatchNorm1d(1)
        # ▲▲▲  END ▲▲▲
        
        c1, c2, c3 = channels
        k1, k2, k3 = kernel_sizes
        s1, s2, s3 = strides
        p1, p2, p3 = paddings

        def block(in_ch, out_ch, k, s, p):
            layers = [nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)]
            if use_bn:
                layers.append(nn.BatchNorm1d(out_ch))
            layers += [nn.ReLU(inplace=True), nn.Dropout(self.dropout_rate)]
            return nn.Sequential(*layers)

        # conv
        conv_layers = [
            block(1,  c1, k1, s1, p1),
            block(c1, c2, k2, s2, p2),
            block(c2, c3, k3, s3, p3),
        ]
        self.conv = nn.Sequential(*conv_layers)

        # Calculate the length after convolution
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.input_dim)
            out = self.conv(dummy)
            conv_out_dim = out.numel()

        # fc part
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(fc_hidden, 1)  ）
        )

        # 
        self.model = nn.Sequential(self.conv, self.head).to(self.device)

        # loss: mse
        if self.binarize:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.MSELoss()

        # optimizer
        #self.optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay) #may get stuck in a sharp local minimum
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=lr_factor,
            patience=lr_patience,
            threshold=lr_threshold,
            cooldown=lr_cooldown,
            min_lr=min_lr
        )
        self.patience = patience

    # ------- forword -------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features)
        x = x.unsqueeze(1)  # → (batch, 1, features)
         # ▼▼▼ START ▼▼▼
        # 2. Before the data enters the convolutional layer，go through BatchNorm1d 
        x = self.input_batch_norm(x)
        # ▲▲▲  END ▲▲▲
        return self.model(x).squeeze(-1)  # → (batch,)

    @torch.no_grad()
    def predict(self, dataset, indices=None):
        """
        return (preds, targets)
        - if binarize=True：preds -> 0/1（thereshod 0.5）
        - else: preds is continuous
        """
        self.eval()
        if indices is None:
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        else:
            loader = DataLoader(Subset(dataset, indices), batch_size=self.batch_size, shuffle=False, pin_memory=True)

        preds, targs = [], []
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            else:
                x, y = batch['x'], batch['y']
            x = x.to(self.device, non_blocking=True).float()
            logits = self.forward(x)
            preds.append(torch.sigmoid(logits).cpu().numpy() if self.binarize else logits.cpu().numpy())
            targs.append(y.numpy() if torch.is_tensor(y) else np.asarray(y))

        preds = np.concatenate(preds)
        targs = np.concatenate(targs)
        if self.binarize:
            preds = (preds > 0.5).astype(np.int32)
        return preds, targs

    # ------- training  -------
    def fit(self, dataset, train_indices, test_indices, verbose: bool = True):
        """
        - DataLoader
        - train_model(self, train_loader, test_loader, ...)
        - train_model ：model.train()/eval()、optimizer、scheduler、criterion 
        """
        train_ds = Subset(dataset, train_indices)
        test_ds  = Subset(dataset, test_indices)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,  pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=self.batch_size, shuffle=False, pin_memory=True)

       \
        return train_model(
            model=self,
            train_loader=train_loader,
            val_loader=test_loader,
            criterion=self.criterion,
            optimizer=self.optimizer,
            epochs=self.epochs,          
            patience=self.patience,
            scheduler=self.scheduler,
            verbose=verbose
            )