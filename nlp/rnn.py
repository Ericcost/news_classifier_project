import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNFromScratch(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Matrices de poids
        self.W_ih = nn.Linear(embedding_dim, hidden_dim, bias=True)
        self.W_hh = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """
        x: LongTensor (batch_size, seq_len)
        """
        batch_size, seq_len = x.shape
        
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = self.layer_norm(embedded)
        
        # Initialisation état caché h_0 = 0
        h_t = torch.zeros(batch_size, self.W_hh.out_features, device=x.device)
        
        # Boucle temporelle
        for t in range(seq_len):
            x_t = embedded[:, t, :]  # (batch_size, embedding_dim)
            h_t = torch.tanh(self.W_ih(x_t) + self.W_hh(h_t))
        
        # Prédiction à partir du dernier état caché
        out = self.output_layer(h_t)  # (batch_size, output_dim)
        
        return out

