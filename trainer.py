'''
Train network with train dataset.
'''
from torch.utils.tensorboard import SummaryWriter
from model.mymodel import *
from dataset import *
from option import args


class Trainer():
    def __init__(self, args):
        # prepare device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # prepare data
        self.train_base_dir = "data/elect/series"
        self.train_data = MyDataset(path_s=self.train_base_dir ,path_t=self.train_base_dir ,test=False)
        self.train_data_size = len(self.train_data)
        self.batch_size = args.batch_size

        # use DataLoader load data
        self.train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size)
        # self.test_dataloader = DataLoader(test_data, batch_size=32)

        # create network
        self.mymodel = Mymodel4()
        self.mymodel = self.mymodel.to(self.device)

        # loss function
        self.loss_fn = nn.MSELoss()
        self.loss_fn = self.loss_fn.to(self.device)

        # optimizer
        self.learning_rate = 0.01
        self.optimizer = torch.optim.SGD(self.mymodel.parameters(), self.learning_rate)
        self.epoch = 4


    def train(self):

        total_train_step = 0  # record train step
        total_test_step = 0  # record test step
        # add tensorboard

        writer = SummaryWriter("./logs_train")
        # tensorboard --logdir logs_train

        for i in range(self.epoch):
            print("--------------epoch:{}----------------".format(i))
            # begin train
            self.mymodel.train()
            for data in self.train_dataloader:
                seq, target = data
                seq = seq.to(self.device).float()
                target = target.to(self.device).float()
                output = self.mymodel(seq)
                output = output.reshape(1, 1, -1)
                loss = self.loss_fn(output, target)
                # optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_train_step = total_train_step + 1
                if total_train_step % 100 == 0:
                    print("total_train_step:{},loss:{}".format(total_train_step, loss.item()))
                    writer.add_scalar("train_loss", loss.item(), total_train_step)

            if self.epoch % 1 == 0:
                torch.save(self.mymodel, "models/mymodel_{}.pth".format(i))
                print("model is saved")

        writer.close()


if __name__ == '__main__':
    trainer = Trainer(args)
    trainer.train()