'''
Validate trained mode with validation datasets.
'''
import os.path


from model.mymodel import *
from dataset import *

def seq_for_validation(series,predict_result,validation_source_length):
    # concate series and predict_value, and transform them into
    series = series.reshape(-1)
    if predict_result is not None:
        for elem in range(len(predict_result)):
            if elem <= 0:
                predict_result[elem] = 1e-8
            if elem >= 1:
                predict_result[elem] = 1-1e-8
        series = np.concatenate([series, predict_result], axis=0)
    seq = series[-validation_source_length:]
    seq = seq.reshape(1,-1)
    return torch.as_tensor(seq),series

class Tester():
    def __init__(self, args):
        #forcast result
        self.path_result = 'data/elect/result'
        self.filename_result = "result.npy"
        # prepare device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # prepare data
        self.path_validation = "data/elect/series"
        self.filename_validation_s = "test_s.npy"
        self.series = np.load(os.path.join(self.path_validation, self.filename_validation_s))

        # create network and load modle
        self.path = "models/mymodel_3.pth"
        self.mymodel = torch.load(self.path)

        # loss function
        self.loss_fn = nn.MSELoss()
        self.loss_fn = self.loss_fn.to(self.device)

        self.validation_target_length = 120
        self.validation_source_length = args.test_source_len


    def set(self, *, test_target_len):
        self.validation_target_length = test_target_len

    def test(self):
        # record
        # writer = SummaryWriter("Datasets/logs_test")

        predict_result = None
        outputs = np.empty((0))
        series = self.series
        self.mymodel.eval()
        total_test_loss = 0
        with torch.no_grad():
            for i in range(self.validation_target_length):
                seq, series = seq_for_validation(series, predict_result, self.validation_source_length)
                seq = seq.to(self.device).float()
                predict_result = self.mymodel(seq)
                predict_result = predict_result.to(self.device).float()
                outputs = np.concatenate((outputs, predict_result), axis=0)
        predict_series = np.resize(outputs, (1, outputs.shape[0]))

        np.save(os.path.join(self.path_result,self.filename_result),predict_series)

