import  argparse
import  os
from utils.enmTypes import InputType,LossType,EvalType

dir_prefix = os.getcwd()
print(f'current file path is {dir_prefix}')
def parse_arg():
    parser = argparse.ArgumentParser()

    #General Set
    parser.add_argument('--path', type = str, default=dir_prefix)
    parser.add_argument('--Device',type= str, default= 'cpu')

    #data Set
    parser.add_argument('--data_type', type= InputType,default= InputType.PAIRWISE)
    parser.add_argument('--eval_type', type= EvalType, default= EvalType.FULLSORT)
    parser.add_argument('--data_name',type= str,default='coat')
    parser.add_argument('--sample',type= bool,default= True)
    parser.add_argument('--mask',type= bool,default= False)
    parser.add_argument('--graph',type= bool,default= False)
    parser.add_argument('--batchsize',type= int,default= 2048)
    parser.add_argument('--num_worker', type= int, default= 4)

    #Ranker Set
    parser.add_argument('--topk',type= list, default=[1,5,10,20,30])
    parser.add_argument('--metrics',type= list, default=['hit','mrr','map','recall','ndcg','precision','avg_pop','tail_percent'])
    parser.add_argument('--tail_ratio',type = float, default= 0.9)

    #evaluator
    parser.add_argument('--sortType',type = str, default= 'descending')
    parser.add_argument('--patience_max', type = int, default= 50)

    #Trainer Set
    parser.add_argument('--model_name',type= str,default='MF')
    parser.add_argument('--lr',type= float,default=1e-3)
    parser.add_argument('--weight-decay',type= float,default=1e-4)
    parser.add_argument('--loss_name',type= LossType,default= LossType.BPR)
    parser.add_argument('--file_name',type =str,default='MF')
    parser.add_argument('--embedding_size', type= int, default= 16)
    parser.add_argument('--save_result',type = bool, default= True)
    parser.add_argument('--epoch',type = int,default= 300)
    parser.add_argument("--validate_metric",type = str,default='recall')
    parser.add_argument('--validate_k',type = int ,default= 10)
    parser.add_argument('-alpha',type = int,default= 1)


    args = parser.parse_args()

    return  args


