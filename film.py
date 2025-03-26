import torch
import torch.nn as nn

class FiLMGenerator(nn.Module):
    """FiLM Generator that produces gamma and beta from conditioning input.
       The same gamma and beta is shared between every layer."""
    def __init__(self, c_dim, d_model, n_layers):
        super(FiLMGenerator, self).__init__()
        self.n_layers = n_layers
        self.fc = nn.Sequential(
            nn.Linear(c_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2 * d_model)  # Output gamma and beta
        )

    def forward(self, c):
        """
        Args:
            condition: Conditioning input tensor (batch_size, condition_dim)
        Returns:
            gamma: Scaling factor (batch_size, d_model)
            beta: Shifting factor (batch_size, d_model)
        """
        gamma_beta = self.fc(c)  # (batch_size, 2 * feature_dim)
        gamma_betas = torch.chunk(gamma_beta, 2, dim=-1)  # Split into gamma_beta tuples
        return gamma_betas
