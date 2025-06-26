import torch 
import torch.nn as nn
import torch.optim as optim

class RecurrentNeuronalNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # poids pour l'entrée 
        self.W_xh = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)

        # poids pour l'état caché
        self.W_hh = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)

        # biais
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))

        # couche de sortie
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        for t in range(seq_len):
            x_t = x[:, t, :] 
            h_t = torch.tanh(x_t @ self.W_xh + h_t @ self.W_hh + self.b_h)
        
        output = self.fc(h_t) 
        return output