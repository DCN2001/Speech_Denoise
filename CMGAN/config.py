import argparse


#調組態區(training)    
def get_config_train():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=2)   #原版batch_size = 4
    parser.add_argument("--init_lr", type=float, default=5e-4, help="inital learning rate")
    parser.add_argument("--decay_epoch", type=int, default=30, help="epoch from which to start lr decay")
    #parser.add_argument("--n_cpu", type=int, default=1)
    parser.add_argument("--cut_len", type=int, default=16000*2)
    parser.add_argument("--data_path", type=str, default= 'D:\DL\VoiceBank_CMGAN',help='dir to VoiceBank dataset')
    parser.add_argument("--model_save_path", type=str, default='./saved_model')
    parser.add_argument("--loss_weight", type=list, default=[0.9,0.05,0.2])
    
    args= parser.parse_args()

    return args 

#調組態區(inference)
def get_config_inference():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='D:\DL\VoiceBank_CMGAN\\testset')
    parser.add_argument("--load_model_path", type=str, default='./saved_model/CMGAN_epoch2')    #切換要載入的模型參數路徑
    parser.add_argument("--output_path", type=str, default='D:\DL\CMGAN\\raw_output')         #輸出音檔存放處
    
    args = parser.parse_args()

    return args