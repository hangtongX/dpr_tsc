from tqdm import  tqdm
from utils.tools import set_color
import  torch

class Trainer(object):
    def __init__(self,model, ranker, evaluator, optimizer, loss_func,dataGenerater, config,DEVICE='cpu'):
        self.model = model
        self.ranker = ranker
        self.evaluator = evaluator
        self.optimizer = optimizer(self.model.parameters(),lr = config.lr,weight_decay = config.weight_decay)
        self.lossF = loss_func
        self.lossname = config.loss_name.name
        self.trainloader = dataGenerater.trainDataloader()
        self.validateloader = dataGenerater.validateDataloader()
        self.testloader = dataGenerater.testDataloader()
        self.epochs = config.epoch
        self.Device = DEVICE
        self.modelname = config.model_name
        self.data_name = config.data_name
        self.filename = '_'.join([config.data_name, config.model_name, config.loss_name.name])
        # self.filename = '_'.join([config.data_name,config.model_name,config.loss_name.name,str(config.alpha)])
        self.validate_metric = config.validate_metric
        self.validate_k = config.validate_k
        self.validate_str = 0.0
        self.loss_result = 0.0
        self.save_result = config.save_result

    def train_one_epoch(self,epoch):
        traindata = (
            tqdm(
                self.trainloader,
                total=len(self.trainloader),
                ncols=150,
                desc=set_color(f"{self.modelname}_{self.lossname}--{self.data_name}--current epoch: {epoch}--loss: {self.loss_result}--validate result: {self.validate_str}", 'green')
            )
        )

        self.model.train()
        loss_all = []

        for data in traindata:

            loss = self.lossF(self.model, data,self.Device)
            self.check_nan(loss)
            self.optimizer.zero_grad()
            loss.backward()
            loss_all.append(loss.detach().item())
            self.optimizer.step()
            self.loss_result = format(torch.mean(torch.Tensor(loss_all)),'.6f')


        if epoch > 10:
            validate = self.ranker.validate(self.model,self.testloader,self.validate_metric,self.validate_k)
            self.validate_str = validate
            # traindata.set_postfix_str(set_color(f' validate result : {validate}', 'blue'))
            stop = self.evaluator.record_val(validate,epoch,self.model.state_dict())
            return stop
        else:
            return False

    def test_performance(self):
        self.ranker.evaluate(self.model,self.testloader)

    def fit(self):
        stop = False
        for epoch in range(self.epochs):
            stop = self.train_one_epoch(epoch)
            if stop:
                self.evaluator.show_log()
                break
        self.evaluator.save_model(self.filename)
        if not stop:
            self.evaluator.show_log(earltStop= False)

        self.test_performance()

    def check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def evaluate(self):
        self.model.load_state_dict(self.evaluator.get_best_model(modelname= self.filename))
        self.ranker.evaluate(self.model,self.testloader)
        self.ranker.output_result()
        if self.save_result:
            self.ranker.save_result()