# models/cnn_models.py
from env.imports import *
from data.data_utils import create_data_loader
from models.train_val import train_model
from models.base_models import BaseModel
from typing import Tuple, Union
from torch import optim

class ConvNet1D(nn.Module):
    """
    1d conv：
    - input:(batch, features), reshape  (batch, 1, features)
    -  Conv1d + ReLU + MaxPool + (optional) BN + Dropout
    - After flattening, connect two layers of fully connected output scalars (regression); If binarize=True, it will be trained in binary classification
    """
    def __init__(
        self,
        input_dim: int,
        binarize = None,
        channels: Union[int, Tuple[int, ...]] = (32,),
        kernel_sizes: Union[int, Tuple[int, ...]] = 7,
        strides: Union[int, Tuple[int, ...]] = 1,
        paddings: Union[int, Tuple[int, ...]] = 3,
        use_bn: bool = True,
        dropout_rate: float = 0.1,
        fc_hidden: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        epochs: int = 100,
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

        self.input_batch_norm = nn.BatchNorm1d(1)

        channels_param = channels
        kernel_sizes_param = kernel_sizes

        channels = (channels,) if isinstance(channels, int) else tuple(channels)
        n_layers = len(channels)

        kernel_sizes = (kernel_sizes,) * n_layers if isinstance(kernel_sizes, int) else tuple(kernel_sizes)
        strides = (strides,) * n_layers if isinstance(strides, int) else tuple(strides)
        paddings = (paddings,) * n_layers if isinstance(paddings, int) else tuple(paddings)

        if len(kernel_sizes) != n_layers or len(strides) != n_layers or len(paddings) != n_layers:
            raise ValueError(
                "The length of channels, kernel_sizes, strides, and paddings must be the same."
            )

        # ▼▼▼ DEBUG BLOCK ▼▼▼
        print("--- SWEEP DEBUG ---")
        print(f"Received channels: {channels_param} (type: {type(channels_param)})")
        print(f"Received kernel_sizes: {kernel_sizes_param} (type: {type(kernel_sizes_param)})")
        print(f"Number of layers to build: {n_layers}")
        print(f"Final kernel_sizes tuple: {kernel_sizes} (len: {len(kernel_sizes)})")
        print("--------------------")
        # ▲▲▲ END DEBUG BLOCK ▲▲▲

        def block(in_ch, out_ch, k, s, p):
            layers = [nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)]
            if use_bn:
                layers.append(nn.BatchNorm1d(out_ch))
            layers += [
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(self.dropout_rate)
            ]
            return nn.Sequential(*layers)

        conv_layers = []
        in_channels = 1
        for i in range(n_layers):
            conv_layers.append(
                block(in_channels, channels[i], kernel_sizes[i], strides[i], paddings[i])
            )
            in_channels = channels[i]
        self.conv = nn.Sequential(*conv_layers)
        final_conv_channels = channels[-1]
        # ▼▼▼ MODIFICATION: Replace Flatten with GAP ▼▼▼
        # We no longer need the dummy forward pass to calculate conv_out_dim
        # with torch.no_grad():
        #     dummy = torch.zeros(1, 1, self.input_dim)
        #     out = self.conv(dummy)
        #     conv_out_dim = out.numel()

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(final_conv_channels, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(fc_hidden, 1)
        )

        self.model = nn.Sequential(self.conv, self.head).to(self.device)

        if self.binarize:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.MSELoss()

        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=lr_factor, patience=lr_patience,
            threshold=lr_threshold, cooldown=lr_cooldown, min_lr=min_lr
        )
        self.patience = patience

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.input_batch_norm(x)
        return self.model(x).squeeze(-1)

    @torch.no_grad()
    def predict(self, loader):
        self.eval()
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

    def fit(self, dataset, train_indices, test_indices, verbose: bool = True):
        train_ds = Subset(dataset, train_indices)
        test_ds  = Subset(dataset, test_indices)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,  pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=self.batch_size, shuffle=False, pin_memory=True)

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




class SimpleConvNet1D(nn.Module):
    """
    conv1d with downsampling
    """
    def __init__(
        self,
        input_dim: int,
        # --- training parameters ---
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 30,
        patience: int = 10,
        
        # --- for sweep ---
        conv_out_channels: int = 64,    # the number of output channels of the convolutional layer
        conv_kernel_size: int = 15,     
        conv_stride: int = 1,           
        
        pool_kernel_size: int = 4,      
        pool_stride: int = 4,            
        fc_hidden: int = 128,           
        dropout_rate: float = 0.5,
 
        binarize = None,
        optimizer_type: str = 'sgd', 
        lr_patience: int = 7,
        lr_factor: float = 0.5,

    ):
        super().__init__()
        # save for sweep
        self.input_dim = input_dim
        self.binarize = bool(binarize) if binarize is not None else False
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.dropout_rate = dropout_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # model
        
        # The input with batchNorm1d layer
        #self.input_batch_norm = nn.BatchNorm1d(1)

        # single conv layer
        self.conv_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=1, 
                out_channels=conv_out_channels, 
                kernel_size=conv_kernel_size, 
                stride=conv_stride,
                padding=(conv_kernel_size - 1) // 2 
            ),
            nn.BatchNorm1d(conv_out_channels),
            nn.LeakyReLU(inplace=True) 
        )

        # Max Pooling
        self.pool_layer = nn.MaxPool1d(
            kernel_size=pool_kernel_size, 
            stride=pool_stride
        )

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # use GAP to compress dimen
        fc_input_dim = conv_out_channels
        '''
        # Dynamically calculate the dimensions entering the FC layer
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.input_dim)
            dummy = self.input_batch_norm(dummy)
            dummy = self.conv_layer(dummy)
            dummy = self.pool_layer(dummy)
            fc_input_dim = dummy.numel()
        '''
        #  head part
        self.head = nn.Sequential(
            nn.Linear(fc_input_dim, fc_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(fc_hidden, 1)
        )

        # loss function
        self.criterion = nn.MSELoss()

        # optimizer
        if optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        elif optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")
            
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=lr_factor, patience=lr_patience)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        #x = self.input_batch_norm(x)
        x = self.conv_layer(x)
        x = self.pool_layer(x)
        x = self.global_avg_pool(x)  # output dimen: (batch, channels, 1)
        x = x.squeeze(-1) 
        x = self.head(x)
        return x.squeeze(-1)
        

    def fit(self, dataset, train_indices, test_indices, verbose: bool = True):
        train_ds = Subset(dataset, train_indices)
        test_ds  = Subset(dataset, test_indices)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,  pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=self.batch_size, shuffle=False, pin_memory=True)
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
    def predict(self, loader):
        self.eval()
        preds, targs = [], []
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            else:
                x, y = batch['x'], batch['y']
            x = x.to(self.device, non_blocking=True).float()
            logits = self.forward(x)
            preds.append(torch.sigmoid(logits).cpu().detach().numpy() if self.binarize else logits.cpu().detach().numpy())
            targs.append(y.numpy() if torch.is_tensor(y) else np.asarray(y))

        preds = np.concatenate(preds)
        targs = np.concatenate(targs)
        if self.binarize:
            preds = (preds > 0.5).astype(np.int32)
        return preds, targs