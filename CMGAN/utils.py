#共用函式區
import torch
import torch.nn as nn
import numpy as np
from pesq import pesq
from pystoi import stoi
from joblib import Parallel, delayed


def power_compress(x):
    real = x[..., 0]
    imag = x[..., 1]
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**0.3
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], 1)


def power_uncompress(real, imag):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag ** (1.0 / 0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], -1)


#計算PESQ, STOI，一次進來一個batch
def est_score(est_sig, ref_sig, sample_rate):
    est_sig = est_sig.cpu()
    ref_sig = ref_sig.cpu()
    
    #torch轉numpy array 
    est_sig = est_sig.numpy()     #[:, 0]
    ref_sig = ref_sig.numpy()     #[:, 0]
    
    pesq_val = 0
    stoi_val = 0
    
    for i in range(est_sig.shape[0]):
        try:
            pesq_val += pesq(sample_rate, ref_sig[i], est_sig[i], 'wb')
            stoi_val += stoi(ref_sig[i], est_sig[i], sample_rate, extended=False)
        except:
            print(ref_sig)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(est_sig)
            pesq_val = 0
            stoi_val = 0
        
    #batch平均pesq, stoi分數
    pesq_val = pesq_val/est_sig.shape[0]
    stoi_val = stoi_val/est_sig.shape[0]
    return pesq_val, stoi_val



#用於discriminator loss
def pesq_loss(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, "wb")
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score

def batch_pesq(clean, noisy):
    pesq_score = Parallel(n_jobs=-1)(
        delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy)
    )
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score).to("cuda")