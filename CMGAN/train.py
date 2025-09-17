from models import generator
from models import discriminator
from utils import *
from config import get_config_train

import os 
import time
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import dataloader
import torch
import numpy as np
import torch.nn.functional as F
from torchinfo import summary
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP  
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm


args = get_config_train()

class Trainer():
    def __init__(self, train_ds, test_ds, gpu_id: int):
        #stft設定
        self.n_fft = 400
        self.hop = 100

        self.train_ds = train_ds
        self.test_ds = test_ds

        #建立model區
        self.generator = generator.Generator(channels=64).cuda()   #in_channels=3, channels=64 (default) #將Model加載到gpu       
        summary(self.generator, [(1, 3, args.cut_len // self.hop + 1, int(self.n_fft / 2) + 1)])
        self.discriminator = discriminator.Discriminator(in_channel=2, channels=16).cuda()
        summary(self.discriminator, [(1, 1, args.cut_len // self.hop + 1, int(self.n_fft / 2) + 1), (1, 1, args.cut_len // self.hop + 1, int(self.n_fft / 2) + 1)])

        #優化器
        self.gen_optimizer = torch.optim.AdamW(self.generator.parameters(), lr=args.init_lr)
        self.disc_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr= 2 * args.init_lr) 

        self.generator = DDP(self.generator, device_ids=[gpu_id], find_unused_parameters=True)    #有多張時顯卡適用，分散訓練
        self.discriminator = DDP(self.discriminator, device_ids=[gpu_id])
        self.gpu_id = gpu_id

    def generator_forward(self, clean, noisy):
        
        #Normalization ???(待釐清原因)
        c = torch.sqrt(noisy.size(-1)/torch.sum((noisy**2.0), dim=-1))  #方均根倒數
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
        noisy, clean = torch.transpose(noisy*c, 0 ,1), torch.transpose(clean*c, 0, 1)

        #STFT       
        noisy_spec = torch.stft(noisy, self.n_fft, self.hop, 
                                window=torch.hamming_window(self.n_fft).to(self.gpu_id), onesided=True, return_complex=False)
        clean_spec = torch.stft(clean, self.n_fft, self.hop, 
                                window=torch.hamming_window(self.n_fft).to(self.gpu_id), onesided=True, return_complex=False)
        noisy_spec = power_compress(noisy_spec).permute(0,1,3,2)
        clean_spec = power_compress(clean_spec)
        
        #clean拆成real part和imagine part
        clean_real = clean_spec[:,0,:,:].unsqueeze(1)
        clean_imag = clean_spec[:,1,:,:].unsqueeze(1)
        clean_mag = torch.sqrt(clean_real**2 + clean_imag**2)
        
        #給model的輸入有三個channel，real, imag, mag
        noisy_mag = torch.sqrt(noisy_spec[:,0,:,:]**2 + noisy_spec[:,1,:,:]**2).unsqueeze(1)
        noisy_input = torch.cat([noisy_mag,noisy_spec], dim=1)
        est_real, est_imag = self.generator(noisy_input)
        est_real, est_imag = est_real.permute(0,1,3,2), est_imag.permute(0,1,3,2)
        est_mag = torch.sqrt(est_real**2 + est_imag**2)
        
        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
        est_audio = torch.istft(
            est_spec_uncompress,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
            onesided = True,
            return_complex = False
        )
        
        return {
            "est_mag": est_mag,
            "est_real": est_real,
            "est_imag": est_imag,
            "est_audio": est_audio,
            "clean_mag": clean_mag,
            "clean_real": clean_real,
            "clean_imag": clean_imag
        }

    def generator_loss(self, generator_outputs):
        
        #Loss_TF
        loss_mag = F.mse_loss(generator_outputs["est_mag"], generator_outputs["clean_mag"])
        loss_ri = F.mse_loss(generator_outputs["est_real"], generator_outputs["clean_real"]) + F.mse_loss(generator_outputs["est_imag"], generator_outputs["clean_imag"])

        #Loss_GAN
        #先Discriminator所計算出的clean & enhance差距
        discriminator_outputs = self.discriminator(generator_outputs["clean_mag"], generator_outputs["est_mag"])
        #再透過計算Discriminator評估能力以達到adversarial
        standard_metric = torch.ones(args.batch_size).to(self.gpu_id)   #clean signal的pesq分數(Normalized後)，應該要都是1 
        loss_GAN = F.mse_loss(discriminator_outputs.flatten(), standard_metric)

        #Loss_Time
        loss_time = F.l1_loss(generator_outputs["est_audio"], generator_outputs["clean_audio"])  #Mean L1 loss of the batch


        #Weighted Sum
        gen_loss = args.loss_weight[0]*loss_mag + (1-args.loss_weight[0])*loss_ri + args.loss_weight[1]*loss_GAN + args.loss_weight[2]*loss_time 
        return gen_loss


    def discriminator_loss(self, generator_outputs):
        length = generator_outputs["est_audio"].size(-1)
        est_audio_list = list(generator_outputs["est_audio"].detach().cpu().numpy())  #.detach()在backward時只更新Discriminator參數不影響Generator  
        clean_audio_list = list(generator_outputs["clean_audio"].cpu().numpy())    #.numpy()轉為ndarray
        pesq_score = batch_pesq(clean_audio_list, est_audio_list)
        
        #If silent frame exist => PESQ is none
        if pesq_score != None:
            standard_metric = torch.ones(args.batch_size).to(self.gpu_id)   #clean signal的pesq分數(Normalized後)，應該要都是1 
            disc_standard_score = self.discriminator(generator_outputs["clean_mag"], generator_outputs["clean_mag"])
            disc_est_score = self.discriminator(generator_outputs["clean_mag"], generator_outputs["est_mag"].detach())
            disc_loss = F.mse_loss(disc_standard_score.flatten(), standard_metric) + F.mse_loss(disc_est_score.flatten(), pesq_score)
        else:
            #print("########## Encounter Silent Part  ############")
            disc_loss = None
        
        return disc_loss

    #每個bacth中怎麼train
    def train_per_batch(self, batch):
        #當筆batch之資料clean: 4*32000, noisy: 4*32000, signal length
        clean_signal = batch[0].to(self.gpu_id)
        noisy_siganl = batch[1].to(self.gpu_id)
        signal_length = batch[2]


        #Generator train step
        generator_outputs = self.generator_forward(clean_signal, noisy_siganl)
        generator_outputs["clean_audio"] = clean_signal

        gen_loss = self.generator_loss(generator_outputs)    #forward propogation
        self.gen_optimizer.zero_grad()      #clear gradient
        gen_loss.backward()                 #calculate gradient
        self.gen_optimizer.step()           #update parameters
        
        #Discriminator train step
        disc_loss = self.discriminator_loss(generator_outputs)
        if disc_loss != None:
            self.disc_optimizer.zero_grad()
            disc_loss.backward()
            self.disc_optimizer.step()
        else: 
            disc_loss = torch.tensor([0.0])
        
        return 

    
    #每個bacth怎麼test
    @torch.no_grad()
    def test_per_batch(self,batch):
        clean_signal = batch[0].to(self.gpu_id)
        noisy_signal = batch[1].to(self.gpu_id)
        signal_length = batch[2]

        #Generator evaluation
        generator_outputs = self.generator_forward(clean_signal, noisy_signal)
        generator_outputs["clean_audio"] = clean_signal

        gen_loss = self.generator_loss(generator_outputs)
        
        #Discriminator evaluation
        disc_loss = self.discriminator_loss(generator_outputs)
        if disc_loss == None:
            disc_loss = torch.tensor([0.0])
        
        return gen_loss.item(), disc_loss.item(), generator_outputs["est_audio"], clean_signal

    ##整個test過程
    def test_total(self):
        #.eval()關閉batch norm & dropout
        self.generator.eval()
        
        gen_loss_total = 0.0
        disc_loss_total = 0.0
        pesq_total = 0.0
        stoi_total = 0.0
        for idx, batch in enumerate(tqdm(self.test_ds, desc="Eval bar", colour="#9F35FF")):
            step = idx + 1
            gen_loss, disc_loss, est_sig, ref_sig = self.test_per_batch(batch)
            gen_loss_total += gen_loss
            disc_loss_total += disc_loss
            temp_pesq, temp_stoi = est_score(est_sig, ref_sig, args.cut_len//2)
            pesq_total += temp_pesq
            stoi_total += temp_stoi 
            

        #取整個testset的 [loss、分數] 平均
        gen_loss_avg = gen_loss_total / step
        disc_loss_avg = disc_loss_total / step
        pesq_avg = pesq_total / step 
        stoi_avg = stoi_total / step
        print("\nPESQ: ",pesq_avg, " | STOI: ",stoi_avg," | Gen Loss: ", gen_loss_avg, " | Disc Loss: ", disc_loss_avg)

        return pesq_avg

    #整個train過程，每個epoch test一次
    def train_total(self):
        #漸進式調整learning rate
        scheduler_G = torch.optim.lr_scheduler.StepLR(self.gen_optimizer, step_size=args.decay_epoch, gamma=0.5)
        scheduler_D = torch.optim.lr_scheduler.StepLR(self.disc_optimizer, step_size=args.decay_epoch, gamma=0.5)

        for epoch in tqdm(range(args.epochs), desc="Epoch", colour="#0080FF"):
            #.train()啟用batch norm & dropout
            self.generator.train()       
            self.discriminator.train()
            for idx, batch in enumerate(tqdm(self.train_ds, desc="Train bar",colour="#ff7300")):
                #print(np.shape(batch[0]), np.shape(batch[1]), np.shape(batch[2])  #每個batch四筆資料，每筆內容為:取樣後的clean,noisy和signal length
                step = idx + 1
                self.train_per_batch(batch)

            #test
            self.test_total()
            #儲存模型參數
            saved_model_path = os.path.join(args.model_save_path, "CMGAN_epoch"+ str(epoch))
            if self.gpu_id == 0:
                print("Saving  model.........")
                torch.save(self.generator.module.state_dict(), saved_model_path)

            scheduler_G.step()
            scheduler_D.step()

#---------------------------------------------------------------------------------------------------------
#執行Train大架構(平行分散設定、train等)
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    #init_process_group(backend="nccl", rank=rank, world_size=world_size)  #windows系統沒有nccl要換成gloo
    init_process_group(backend="gloo", rank=rank, world_size=world_size)

def main(rank: int, world_size: int, args):
    ddp_setup(rank, world_size)
    if rank == 0:
        available_gpus = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        print(available_gpus)
    train_ds, test_ds = dataloader.load_data(
        args.data_path, args.batch_size, 2, args.cut_len
    )
    
    trainer = Trainer(train_ds,test_ds,0)    #rank=0
    trainer.train_total()
    destroy_process_group()


if __name__ == "__main__":

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size)