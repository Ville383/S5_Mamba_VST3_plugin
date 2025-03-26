import numpy as np
import torch
import torch.nn.functional as F
from s5 import binary_operator, associative_scan
#from pscan import pscan
from typing import Tuple


def make_HiPPO(N):
    """ Create a HiPPO-LegS matrix.
        From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
        Args:
            N (int32): state size
        Returns:
            N x N HiPPO LegS matrix
    """
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A


def make_NPLR_HiPPO(N):
    """
    Makes components needed for NPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        N (int32): state size

    Returns:
        N x N HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B

    """
    # Make -HiPPO
    hippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = np.sqrt(np.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)
    return hippo, P, B


def make_DPLR_HiPPO(N):
    """
    Makes components needed for DPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Note, we will only use the diagonal part
    Args:
        N:

    Returns:
        eigenvalues Lambda, low-rank term P, conjugated HiPPO input matrix B,
        eigenvectors V, HiPPO B pre-conjugation

    """
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, np.newaxis] * P[np.newaxis, :]

    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = np.linalg.eigh(S * -1j)
    
    P = V.conj().T @ P
    B_orig = B
    B = V.conj().T @ B

    return torch.from_numpy(Lambda_real).type(torch.float32) + 1j * torch.from_numpy(Lambda_imag).type(torch.float32), torch.from_numpy(V).type(torch.complex64)


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    """ Initialize the learnable timescale Delta by sampling
         uniformly between dt_min and dt_max.
         Args:
             dt_min (float32): minimum value
             dt_max (float32): maximum value
         Returns:
             init function
     """
    def init(shape):
        """ Init function
             Args:
                 key: jax random key
                 shape tuple: desired shape
             Returns:
                 sampled log_step (float32)
         """
        return np.random.uniform(size=shape) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)
    
    return init


def init_log_steps(input):
    """ Initialize an array of learnable timescale parameters
         Args:
             input: tuple containing the array shape H and
                    dt_min and dt_max
         Returns:
             initialized array of timescales (float32): (H, 1)
     """
    H, dt_min, dt_max = input
    log_steps = []
    for _ in range(H):
        log_step = log_step_initializer(dt_min=dt_min, dt_max=dt_max)(shape=(1,))
        log_steps.append(log_step)

    return np.array(log_steps)[:,0]


def discretize_bilinear(Lambda, B_tilde, Delta):
    """Discretize a diagonalized, continuous-time linear SSM
    using bilinear transform method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B_tilde (complex64): input matrix                      (P,H)
        Delta (float32): discretization step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = torch.ones(Lambda.shape[0], device=Lambda.device)
    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar


def apply_ssm(dA, dB, C, x, h, conj_sym):
    """ Compute the BxLxH output of discretized SSM given an BxLxH input.
        Args:
            dA       (complex64): discretized diagonal state matrix (P)
            dB       (complex64): discretized input matrix          (P,H)
            C        (complex64): output matrix                     (H,P)
            x        (float32):   input sequence of features        (L,H)
            conj_sym (bool):      enforce conjugate symmetry
        Returns:
            hs (float32):             the SSM outputs (S5 layer preactivations) (L,P)
            hidden_state (complex64): last hidden state from parallel scan      (P)
    """
    x = x.to(dB.dtype) # float32 -> complex64

    Bu_elements = torch.vmap(lambda u: dB @ u)(x)
    dA = dA.tile(x.shape[0], 1)

    if h is not None:
        Bu_elements[0] = h * dA[0] + Bu_elements[0]

    #hs = pscan(dA, Bu_elements)
    _, hs = associative_scan(binary_operator, (dA, Bu_elements))

    if conj_sym:
        return torch.vmap(lambda x: 2*(C @ x).real)(hs), hs[-1]
    else:
        return torch.vmap(lambda x: (C @ x).real)(hs), hs[-1]
