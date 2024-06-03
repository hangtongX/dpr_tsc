from utils.getters import get_model,get_lossF
from trainer.Trainer import  Trainer
from utils.dataloader import GeneralData
from evaluator import evaluator,ranker
from torch.optim import  Adam


def train_model(config,datagenerator):
    dataGenerator = datagenerator
    Model = get_model(model_name= config.model_name)
    if config.model_name == 'lightgcn':
        model = Model(config,dataGenerator.datainfo,dataGenerator.createSparseGraph()).to(config.Device)
    else:
        model = Model(config,dataGenerator.datainfo).to(config.Device)
    rank = ranker.Ranker(config,dataGenerator.item_count)
    stopper = evaluator.Evaluator(config)
    if config.loss_name.name == 'DPR':
        lossF = get_lossF(config.loss_name)(dataGenerator)
        lossF.alpha = config.alpha
        lossF.cal_propensity()
    else:
        lossF = get_lossF(config.loss_name)(dataGenerator)
    trainer = Trainer(model= model,ranker= rank,evaluator= stopper, optimizer= Adam,loss_func= lossF,dataGenerater= dataGenerator,config= config,DEVICE= config.Device)
    trainer.fit()
    trainer.evaluate()

