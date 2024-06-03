import torch.cuda

from trainer.modelTrain import  train_model
from argParser import  parse_arg
from utils.dataloader import GeneralData
from utils.tools import setup_seed
import numpy as np
from utils.enmTypes import LossType,InputType

if  __name__ == '__main__':
    config = parse_arg()
    config.Device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'current device is {config.Device}...............')
    for data in ['ml-100k']:  # 'ml-1m','ml-100k','coat','ml-10m','yahoo','lastfm','kuai'
        config.data_name = data
        for model in ['lightgcn']:
            config.model_name = model

            # for loss in [LossType.BPR,LossType.UBPR,LossType.EBPR,LossType.PDA,LossType.RELMF,LossType.UPL,LossType.MFDU,LossType.DPR]:
            for alpha in [0.5,1,2,3,4,5]:
                #[LossType.BPR,LossType.UBPR,LossType.EBPR,LossType.RELMF,LossType.PDA,LossType.UPL,LossType.MFDU]:
                loss = LossType.DPR
                config.alpha = alpha
                if loss in [LossType.RELMF,LossType.MFDU]:
                    config.data_type = InputType.POINTWISE
                elif loss in [LossType.CPR]:
                    config.data_type = InputType.LISTWISE
                else:
                    config.data_type = InputType.PAIRWISE


                setup_seed(2023)
                dataGeneral = GeneralData(config)
                config.loss_name = loss
                config.file_name = '_'.join([model, data, loss.name])
                # config.file_name = '_'.join([model,data,loss.name,str(config.alpha)])
                train_model(config,dataGeneral)