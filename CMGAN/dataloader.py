import os
import torchaudio
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
from utils import *
import random
#import natsort


#trainset共11572筆、testset共824筆
class VoiceBank_dataset(torch.utils.data.Dataset):
    #繼承: 要依照寫法以下為三個必要function(init, len, getitem)
    def __init__(self, data_dir, cut_len=16000*2):      
        self.cut_len = cut_len
        self.clean_dir = os.path.join(data_dir, "clean")  #拼接dataset路徑和"/clean/"
        self.noisy_dir = os.path.join(data_dir, "noisy")  #拼接dataset路徑和"/noisy/"
        self.clean_wavlist = os.listdir(self.clean_dir)
        #self.clean_dir = natsort.natsorted(self.clean_dir_org)   #依照名字排序(這個dataset應該不需要)

    #整個set的資料數目 
    def __len__(self):
        return len(self.clean_wavlist)
    
    def __getitem__(self, idx):
        #路徑弄好(此為idx這個音檔，單個)
        clean_file_dir = os.path.join(self.clean_dir, self.clean_wavlist[idx])
        noisy_file_dir = os.path.join(self.noisy_dir, self.clean_wavlist[idx])
        
        #讀取檔案 => 變成tensor
        clean_ds,_ = torchaudio.load(clean_file_dir)
        noisy_ds,_ = torchaudio.load(noisy_file_dir)
        #將大小為1的維度移除: torch.size([1,N]) --> torch.size([N])
        clean_ds = clean_ds.squeeze()
        noisy_ds = noisy_ds.squeeze()

        assert len(clean_ds) == len(noisy_ds)  #檢查clean和noisy長度是否一樣
        signal_length = len(clean_ds)     #訊號長度
        
        #訊號長度不到cut length(2 * 16k point/sec)，需要padding，方式為循環播放
        if signal_length < self.cut_len:
            units = self.cut_len // signal_length    #取整數
            clean_ds_final = []
            noisy_ds_final = []
            for i in range(units):
                clean_ds_final.append(clean_ds)
                noisy_ds_final.append(noisy_ds)
            #循環播放以補足cut_length
            clean_ds_final.append(clean_ds[: self.cut_len % signal_length])   
            noisy_ds_final.append(noisy_ds[: self.cut_len % signal_length])
            clean_ds = torch.cat(clean_ds_final, dim=-1)
            noisy_ds = torch.cat(noisy_ds_final, dim=-1)
        #若訊號長度大於cut length(2s)，直接隨機切2秒的訊號下來
        else:
            starting_pt = random.randint(0, signal_length-self.cut_len)   #從0到cut length範圍內隨機選一點開始    
            #無論選到哪一個起點都會補到cut_length的長度 (畫圖可驗證)                                 
            clean_ds = clean_ds[starting_pt: starting_pt+self.cut_len]
            noisy_ds = noisy_ds[starting_pt: starting_pt+self.cut_len]

        return clean_ds, noisy_ds, signal_length


def load_data(datapath, batch_size, n_cpu, cut_len):
    torchaudio.set_audio_backend("soundfile")            #windows: soundfile, linux: sox_io
    #trainset跟testset分別的路徑
    train_dir = os.path.join(datapath, "trainset")
    test_dir = os.path.join(datapath, "testset")

    #由上面寫好的class建立Dataset object
    train_ds = VoiceBank_dataset(train_dir, cut_len)
    test_ds = VoiceBank_dataset(test_dir, cut_len)

    #建立Dataloader
    train_dataset = torch.utils.data.DataLoader(dataset=train_ds, 
                                                batch_size=batch_size,
                                                pin_memory=True,   #依照電腦性能，開了比較快，默認False
                                                shuffle=False,     #資料照順序餵
                                                sampler=DistributedSampler(train_ds),     #不確定!!!!
                                                drop_last=True,    #多於batch size的要不要丟掉
                                                num_workers=n_cpu)  #一般根據cpu核心數來填，越多應該越快
    
    test_dataset = torch.utils.data.DataLoader(dataset=test_ds,
                                               batch_size=batch_size,
                                               pin_memory=True,
                                               shuffle=False,
                                               sampler=DistributedSampler(test_ds),
                                               drop_last=False,
                                               num_workers=n_cpu)
    
    
    #print(np.shape(batch[0]), np.shape(batch[1]), np.shape(batch[2])  #每個batch四筆資料，內容為:取樣後的clean,noisy和signal length
    return train_dataset, test_dataset

#load_data("D:\DL_practice\VoiceBank_CMGAN", 1, 1, 32000)