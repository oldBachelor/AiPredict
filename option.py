import argparse
import template
parser = argparse.ArgumentParser()
parser.add_argument('--template',default='.',help='You can set various templates in option.py')



#Hardware specifications

#Data specifications
parser.add_argument('--dataset',default='elect',help = 'Name of the dataset')
parser.add_argument('--train_data_raw',  default='train.csv',
                    help='original train data')
parser.add_argument('--test_data_raw',  default='test.csv',
                    help='original test data')
parser.add_argument('--dir_data', type=str, default='data/elect',
                    help='dataset directory')
parser.add_argument('--predict_column', type=str, default='Sunspots',
                    help='dataset directory')
parser.add_argument('--path_result', type = str,default='data/elect/result',
                    help ="predict result directory")


#Model specifications
parser.add_argument('--modle',default='Mymodel4',help ='model name')

#Training specifications
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=1,
                    help='input batch size for training')

parser.add_argument('--train_source_len',type=int,default=270,
                    help='train series source window length ')
parser.add_argument('--train_target_len',type=int,default=1,
                    help='train series target window length ')
parser.add_argument('--stride',type=int,default=1,
                    help='train series window stride ')

parser.add_argument('--test_source_len',type=int,default=270,
                    help='test series source window length ')
parser.add_argument('--test_target_len',type=int,default=120,
                    help='test series source window length ')


#Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
# Loss specifications

# Log specifications

args = parser.parse_args()
template.set_template(args)