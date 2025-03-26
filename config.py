import math
from dataclasses import dataclass

@dataclass
class Hyperparams:
    name = "8_layers_4_blocks_boss_od-3"
    
    train_input_dir = "data/train/ht1-input.wav"
    train_target_dir = "data/train/ht1-target.wav"
    val_input_dir = "data/val/ht1-input.wav"
    val_target_dir = "data/val/ht1-target.wav"
    test_input_dir = "data/test/la2a-input.wav"
    test_target_dir = "data/test/la2a-target.wav"

    overlap: int = 2048 # Number of samples used for the model to "warm-up" before starting the TBPTT on each epoch
    proc_len: int = 4096 # The number of samples used for one backward iteration during the TBPTT. Use only power of two for efficient parallel scan
    sequence_length: int = 61440 + overlap # Number of samples to split audio
    
    epochs: int = 100
    batch_size: int = 32
    alpha: float = 0.001 # weight on the second loss_fn
    retrain: bool = False
    

    assert math.log2(proc_len) - int(math.log2(proc_len)) == 0, "proc_len must be a power of two"
    assert math.log2(overlap) - int(math.log2(overlap)) == 0, "overlap must be a power of two"
    assert overlap + proc_len <= sequence_length, "sequence_length must be higher or the same as (overlap + proc_len)"
    

@dataclass
class ModelParams:
    # AUDIO CHANNELS, 1 for mono
    input_size: int = 1
    output_size: int = 1

    # S5/MAMBA BLOCK
    n_layers: int = 8 # Number of layers
    d_model: int = 16 # Number of input/output features, H in the S5 paper
    d_state: int = 32 # Latent size, P in the S5 paper
    blocks: int = 4 # Number of blocks used for the initialization of A, J in the S5 paper
    step_rescale: int = 1 # modify this to change sampling rate (e.g. 44100 to 48000 -> 48000/44100)
    conj_sym: bool = False # Effectively cut state-space matrix sizes by half
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    expand_factor: int = 2 # Expansion factor in the Mamba architecture, E in the Mamba paper
    d_inner: int = int(expand_factor * d_model)
    bias: bool = False # Whether to use bias in the Mamba projection layers

    # FiLM/conditioning
    c_dim: int = 2 # number of conditioning vectors

    # CAUSAL CONV (not used)
    kernel_size: int = 4
    # don't change ↓
    in_channels: int = d_inner
    out_channels: int = d_inner
    stride: int = 1
    padding: int = 0
    dilation: int = 1
    groups: int = d_inner
    conv_bias: bool = False
    padding_mode: str = 'zeros'