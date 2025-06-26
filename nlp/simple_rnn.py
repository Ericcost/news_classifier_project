import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleRNN(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Poids de l'entrée vers l'état caché
        self.W_xh = nn.Parameter(torch.randn(embedding_dim, hidden_dim) * 0.1)

        # Poids de l'état caché récurrent
        self.W_hh = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)

        # Biais de l'état caché
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))

        # Couche de sortie : transforme h_t -> logits
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):  
        """
        x: (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, embedding_dim)
            h_t = torch.tanh(x_t @ self.W_xh + h_t @ self.W_hh + self.b_h)

        output = self.fc(h_t)  # (batch_size, output_dim)
        return output
