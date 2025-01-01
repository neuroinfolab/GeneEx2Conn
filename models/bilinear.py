# Gene2Conn/models/bilinear.py

from imports import *
from data.data_utils import create_data_loader
from models.train_val import train_model


class BilinearModel(nn.Module):
    def __init__(self, input_dim, reduced_dim, activation='none', learning_rate=0.01, epochs=100, batch_size=128, lambda_reg=1.0):
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.linear = nn.Linear(input_dim//2, reduced_dim, bias=False)
        self.linear2 = nn.Linear(input_dim//2, reduced_dim, bias=False)

        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'softplus':
            self.activation = nn.Softplus()
        else:  # 'none'
            self.activation = nn.Identity()

        self.criterion = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.optimizer = Adam(self.parameters(), lr=learning_rate)


    def get_params(self): # for local model saving
        return {
            'input_dim': self.linear.in_features,
            'reduced_dim': self.linear.out_features,
            'activation': self.activation.__class__.__name__.lower(),
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'lambda_reg': self.lambda_reg,
            'device': str(self.device)
        }
    
    def forward(self, x): 
        mid = x.size(1) // 2 # define split point
        out1 = self.activation(self.linear(x[:, :mid]))
        out2 = self.activation(self.linear2(x[:, mid:]))
        return torch.sum(out1 * out2, dim=1) # dot product for paired samples

    def predict(self, X):
        self.eval()
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self(X).cpu().numpy()
        return predictions

    def fit(self, X_train, y_train, X_test, y_test, verbose=True):
        train_loader = create_data_loader(X_train, y_train, self.batch_size, self.device, shuffle=False)
        val_loader = create_data_loader(X_test, y_test, self.batch_size, self.device, shuffle=False)
        return train_model(self, train_loader, val_loader, self.epochs, self.criterion, self.optimizer, verbose=verbose)