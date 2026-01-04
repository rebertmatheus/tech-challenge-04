import pytest
import sys
from unittest.mock import MagicMock

# Mock torch and pytorch_lightning globally to avoid DLL issues on Windows
torch_mock = MagicMock()
pl_mock = MagicMock()

# Mock all torch submodules
torch_submodules = [
    'torch', 'torch.nn', 'torch.nn.functional', 'torch.optim', 'torch.utils',
    'torch.utils.data', 'torchvision', 'torchvision.transforms'
]

for module in torch_submodules:
    sys.modules[module] = MagicMock()

# Mock pytorch_lightning and its submodules
pl_submodules = [
    'pytorch_lightning', 'pytorch_lightning.callbacks', 'pytorch_lightning.loggers',
    'pytorch_lightning.utilities', 'pytorch_lightning.utilities.rank_zero'
]

for module in pl_submodules:
    sys.modules[module] = MagicMock()

# Set torch availability flag
pytest.TORCH_AVAILABLE = True