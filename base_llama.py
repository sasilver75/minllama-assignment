from dataclasses import dataclass

import re
from torch import dtype
from config import LlamaConfig
from utils import *

class LlamaPreTrainedModel(nn.Module):
  config_class = LlamaConfig
  base_model_prefix = "llama"

  def __init__(self, config: LlamaConfig):
      super().__init__()
      self.config = config # Given a LlamaConfig object, which has model hyperparameters
      self.vocab_size = config.vocab_size # Pulls the vocab size from the config
      self.n_layers = config.n_layers # Pulls the number of layers from the config

  def init_weights(self):
    # Initialize weights
    self.apply(self._init_weights)

  def _init_weights(self, module):
    """ 
    Initialize the weights of the model. 
    If the module is a nn.Linear module:
      Initializes the module's weights to a normal distribution with mean 0 and std 0.02.
      Initializes the module's bias to 0, if the module has a bias.
    If the module is a nn.Embedding module:
      Initializes the module's weights to a normal distribution with mean 0 and std 0.02.
    """
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  @property
  def dtype(self) -> dtype:
    return get_parameter_dtype(self)
