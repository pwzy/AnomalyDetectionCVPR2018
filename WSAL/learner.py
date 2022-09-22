import torch
import torch.nn as nn
from torch.nn import functional as F



# class Learner(nn.Module):
    # def __init__(self, input_dim=2048, drop_p=0.0):
        # super(Learner, self).__init__()
        # self.classifier = nn.Sequential(
            # nn.Linear(input_dim, 512),
            # nn.ReLU(),
            # nn.Dropout(0.6),
            # nn.Linear(512, 32),
            # nn.ReLU(),
            # nn.Dropout(0.6),
            # nn.Linear(32, 1),
            # nn.Sigmoid()
        # )
        # self.drop_p = 0.6
        # self.weight_init()
        # self.vars = nn.ParameterList()

        # for i, param in enumerate(self.classifier.parameters()):
            # self.vars.append(param)

    # def weight_init(self):
        # for layer in self.classifier:
            # if type(layer) == nn.Linear:
                # nn.init.xavier_normal_(layer.weight)

    # def forward(self, x, vars=None):
        # if vars is None:
            # vars = self.vars
        # x = F.linear(x, vars[0], vars[1])
        # x = F.relu(x)
        # x = F.dropout(x, self.drop_p, training=self.training)
        # x = F.linear(x, vars[2], vars[3])
        # x = F.dropout(x, self.drop_p, training=self.training)
        # x = F.linear(x, vars[4], vars[5])
        # return torch.sigmoid(x)

    # def parameters(self):
        # """
        # override this function since initial parameters will return with a generator.
        # :return:
        # """
        # return self.vars

class Learner(nn.Module):
    def __init__(self, nfeat=2048, nclass=1, dropout_rate=0.8, k_size=5):
        super(Learner, self).__init__()

        self.fc0 = nn.Linear(nfeat, 512)

        self.conv = nn.Conv1d(in_channels=512,out_channels=512,kernel_size=k_size)

        self.fc1_1 = nn.Linear(512, 128)
        self.fc1_2 = nn.Linear(128, 1)

        
        self.fc2_1 = nn.Linear(512, 128)
        self.fc2_2 = nn.Linear(128, 128)

        self.dropout_rate = dropout_rate
        self.k_size = k_size
        # if self.training:
        #     self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.fill_(1)
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, input):

        fea = F.relu(self.fc0(input))
        fea = F.dropout(fea, self.dropout_rate, training=self.training)
        # temporal
        old_fea = fea.permute(0,2,1) 
        old_fea = torch.nn.functional.pad(old_fea, (self.k_size//2,self.k_size//2), mode='replicate')
        new_fea = self.conv(old_fea)
        new_fea = new_fea.permute(0,2,1)

        # semantic
        x = F.relu(self.fc1_1(new_fea))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = torch.sigmoid(self.fc1_2(x))
        # dynamic
        new_fea = F.relu(self.fc2_1(new_fea))
        new_fea = F.dropout(new_fea, self.dropout_rate, training=self.training)
        new_fea = self.fc2_2(new_fea)

        return x, new_fea

    def val(self, input):
        # assert (x.shape[0] == 1)
        # original layers
        # import pdb;pdb.set_trace()
        input_new = input.unsqueeze(0)
        at_1 = self.__self_module__(input_new,1)#1,1,1,32,1
        input_view = input.view((1,1,1,32,-1))
        x_1 = torch.sum(input_view*at_1,-2)#1,1,1,1024
        
        x =  torch.cat([input_new,x_1,],2)
        x = self.__Score_pred__(x)

        return x, at_1, at_1

