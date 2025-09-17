from models import generator
from utils import *
from config import get_config_inference
import os
import numpy as np
import torchaudio
import torch
import random
import soundfile as sf
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi



args = get_config_inference()

@torch.no_grad()
def Inference(model_path, output_path, noisy_file):
    n_fft = 400
    hop = 100
    cut_len = 16000 * 2

    #建立model並載入train好之參數
    model = generator.Generator(channels = 64).cuda()
    model.load_state_dict((torch.load(model_path)))

    #由路徑載入音檔並進行前處理
    noisy, sr = torchaudio.load(noisy_file) 
    if (sr != 16000):
        print("Resampling...")
        noisy = torchaudio.functional.resample(noisy, orig_freq=sr, new_freq=16000)
    noisy = noisy.cuda()
                                        
    ######################音擋長度調整，可省略#########################                                                             
    '''
    signal_length = noisy.size(-1)
    #訊號長度不到cut length(2 * 16k point/sec)，需要padding，方式為循環播放
    noisy = noisy.squeeze()
    if signal_length < cut_len:
        units = cut_len // signal_length    #取整數
        noisy_ds_final = []
        for i in range(units):
            noisy_ds_final.append(noisy)
        #循環播放以補足cut_length 
        noisy_ds_final.append(noisy[: cut_len % signal_length])
        noisy = torch.cat(noisy_ds_final, dim=-1)
    #若訊號長度大於cut length(2s)，直接隨機切2秒的訊號下來
    else:
        starting_pt = random.randint(0, signal_length-cut_len)   #從0到cut length範圍內隨機選一點開始    
        #無論選到哪一個起點都會補到cut_length的長度 (畫圖可驗證)                                 
        noisy = noisy[starting_pt: starting_pt+cut_len]
    noisy = noisy.unsqueeze(0)
    '''
    ##################此段可省略###########################
    

    #Normalization ???(待釐清原因)
    c = torch.sqrt(noisy.size(-1)/torch.sum((noisy**2.0), dim=-1))  #方均根倒數
    noisy = torch.transpose(noisy, 0, 1)
    noisy = torch.transpose(noisy*c, 0 ,1)
    
    #STFT       
    noisy_spec = torch.stft(noisy, n_fft, hop, 
                            window=torch.hamming_window(n_fft).cuda(), onesided=True, return_complex=False)
    noisy_spec = power_compress(noisy_spec).permute(0,1,3,2)
    
    #給model的輸入有三個channel，real, imag, mag
    noisy_mag = torch.sqrt(noisy_spec[:,0,:,:]**2 + noisy_spec[:,1,:,:]**2).unsqueeze(1)
    noisy_input = torch.cat([noisy_mag,noisy_spec], dim=1)
    est_real, est_imag = model(noisy_input)     ############### model feed forward
    est_real, est_imag = est_real.permute(0,1,3,2), est_imag.permute(0,1,3,2)
    
    #iSTFT
    est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
    est_audio = torch.istft(est_spec_uncompress, n_fft, hop,
            window=torch.hamming_window(n_fft).cuda(), onesided = True, return_complex = False)

    #save estimated audio
    est_audio = est_audio / c
    est_audio = torch.flatten(est_audio).cpu().numpy()

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_name = os.path.split(noisy_file)[-1]
    saved_path = os.path.join(output_path, output_name)
    sf.write(saved_path, est_audio, sr)

    return est_audio

#處理單個檔案，需要為1D訊號。sample rate若非16000會自行調整
#Inference(args.load_model_path, args.output_path, noisy_file="D:\DL_practice\VoiceBank_CMGAN\\testset\\noisy\\p232_001.wav")
#Inference(args.load_model_path, args.output_path, noisy_file="D:\DL_practice\\raw_input\\test1.wav")



#--------------------------------------------------------------------------------------------------------------------------------------------
#執行voicebank testset的測試
def Inference_Voicebank(args):
    noisy_dir = os.path.join(args.data_path, "noisy")
    clean_dir = os.path.join(args.data_path, "clean")
    noisy_list = os.listdir(noisy_dir)

    pesq_total = 0.0
    stoi_total = 0.0
    num = len(noisy_list)
    for file in tqdm(noisy_list, desc="Progress bar", colour="#FF0000"):
        noisy_path = os.path.join(noisy_dir, file)
        clean_path = os.path.join(clean_dir, file)
        #Inference
        est_audio = Inference(args.load_model_path, args.output_path, noisy_path)
        
        #載入clean以計算pesq, stoi
        clean_audio, sr = torchaudio.load(clean_path)
        cut_len = 16000 *2
        #clean長度調整
        signal_length = clean_audio.size(-1)
        #訊號長度不到cut length(2 * 16k point/sec)，需要padding，方式為循環播放
        clean_audio = clean_audio.squeeze()
        if signal_length < cut_len:
            units = cut_len // signal_length    #取整數
            clean_ds_final = []
            for i in range(units):
                clean_ds_final.append(clean_audio)
            #循環播放以補足cut_length 
            clean_ds_final.append(clean_audio[: cut_len % signal_length])
            clean_audio = torch.cat(clean_ds_final, dim=-1)
        #若訊號長度大於cut length(2s)，直接隨機切2秒的訊號下來
        else:
            starting_pt = random.randint(0, signal_length-cut_len)   #從0到cut length範圍內隨機選一點開始    
            #無論選到哪一個起點都會補到cut_length的長度 (畫圖可驗證)                                 
            clean_audio = clean_audio[starting_pt: starting_pt+cut_len]
        clean_audio = clean_audio.numpy()  #tensor to ndarray

        #compute metrics
        #print(clean_audio.shape, est_audio.shape)
        temp_pesq = pesq(sr, clean_audio, est_audio, 'wb')
        temp_stoi = stoi(clean_audio, est_audio, sr, extended=False)

        pesq_total += temp_pesq
        stoi_total += temp_stoi
    
    pesq_avg = pesq_total / num
    stoi_avg = stoi_total / num

    print("\npesq: ", pesq_avg, "stoi: ", stoi_avg)

#對Voicebank testset進行測試
#Inference_Voicebank(args)

#--------------------------------------------------------------------------------------------------------------------------------------------
'''
#載入model參數
model_path = "./saved_model/CMGAN_epoch0_PESQ1.2"
model = generator.Encoder(in_channel=2, channels=2).cuda()
model.load_state_dict((torch.load(model_path)))

for name, param in model.state_dict().items():
    print(name, param)
'''