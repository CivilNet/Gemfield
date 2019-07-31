import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import numpy as np

# 定义超参数
batch_size = 100
learning_rate = 1e-2
num_epoches = 1000
time_step = 6

def loadData(filename):
    with open(filename,'r') as file:
        lines = []
        for line in file.readlines():
            if line != '\n':
                lines.append(line)
        dataset = [ [] for i in range(len(lines)-1)]
        x = []
        y = []
        category = []
        net_input = []
        for i in range(len(dataset)):
            dataset[i][:] = (item for item in lines[i].strip().split('='))   # 逐行读取数据
            tmp = []
            x_s = eval(dataset[i][0])
            #print('x_s', x_s)
            a = np.zeros([6, 3])
            #print(a)
            for j in range(6):
                for k in range(3):
                    if k == 0:
                        a[j][k] = j
                    else:
                        a[j][k] = x_s[j][k-1]
                if a[j][1] == -1:
                    a[j][1] = 0
            #a = a.reshape(-1, 18)
            #a = np.squeeze(a)
            net_input.append(a)
            x.append(dataset[i][0])
            tmp = float(dataset[i][1])
            y.append(tmp)
            cls = dataset[i][2].strip().split('-')
            category.append(int(cls[0])*2 + int(cls[1]))
        #print('net_input', net_input)
        #print("dateset:", eval(dataset[0][0]))
        #y = [[] for i in range(len(lines)-1)]
        #print('x',type(x[0]))
        #print('y',y)
      # print('category', category)
        #print('cls1', cls1)
        #print('category', category)
        x_new = []

        for i in range(len(x)):
            x_temp = []
            tmp = eval(x[i])
            #print('tmp', tmp[0][0])
            for j in tmp:
                x_temp.append(j[0]*j[1])
            x_new.append(x_temp)
        #print(x_new)
    return np.array(net_input), np.array(category)

[x, category] = loadData('test1.dataset')

x = torch.FloatTensor(x)
category = torch.LongTensor(category)

train_x = x[:-100]
train_y = category[:-100]
test_x = x[-100:]
test_y = category[-100:]

train_set = torch.utils.data.TensorDataset(train_x, train_y)
test_set = torch.utils.data.TensorDataset(test_x, test_y)

train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                         shuffle=True)
test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                        shuffle=False)
#print('test_iter', np.shape(train_iter))
# 定义 Recurrent Network 模型
class Rnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(Rnn, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        # h0 = Variable(torch.zeros(self.n_layer, x.size(1),
        #   self.hidden_dim)).cuda()
        # c0 = Variable(torch.zeros(self.n_layer, x.size(1),
        #   self.hidden_dim)).cuda()
        #x = x.permute(0, 2, 1)
        #print('xs', np.shape(x))
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.classifier(out)
        return out


model = Rnn(3, 64, 3, 12)  # 图片大小是28x28
use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
if use_gpu:
    model = model.cuda()
# 定义loss和optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = optim.SGD(model.parameters(), lr= 0.01)

# 开始训练
for epoch in range(num_epoches):
    print('epoch {}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0.0
    running_acc = 0.0
    i = 1
    for train_x, train_y in train_iter:
    #     img, label = data
    #     b, c, h, w = img.size()
    #     assert c == 1, 'channel must be 1'
    #     img = img.squeeze(1)
    #
    #     print('img', img.size())
    #     print('label', label.size())
        #print('tx', np.shape(train_x))
        #print('ty', np.shape(train_y))

        # img = img.view(b*h, w)
        # img = torch.transpose(img, 1, 0)
        # img = img.contiguous().view(w, b, -1)
        if use_gpu:
            img = Variable(train_x).cuda()
            label = Variable(train_y).cuda()
        else:
            img = Variable(train_x)
            label = Variable(train_y)
        # 向前传播
        out = model(img)
        #print(np.shape(out))
        #print(np.shape(label))
        loss = criterion(out, label)
        #print('loss', loss.item())
        #print('label', label.size(0))
        running_loss += (loss.item()* label.size(0))/(len(train_x))
        _, pred = torch.max(out, 1)
        #print(pred)
        num_correct = (pred == label).sum()
        #print('num_c', num_correct)
        #print('num_corr', num_correct.item())
        running_acc += (num_correct.item())/(len(train_x))
        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 300 == 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, num_epoches, running_loss / (batch_size),
                running_acc / (batch_size)))
        i += 1
    print('gemfield running_acc', running_acc)
    #print('len', len(train_x))
    #print('i', i)
    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (i), running_acc / (i)))
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    # for data in test_loader:
    #     img, label = data
    #     b, c, h, w = img.size()
    #     assert c == 1, 'channel must be 1'
    #     img = img.squeeze(1)
    #     print('img', img)
    #     print('label', label)
        # img = img.view(b*h, w)
        # img = torch.transpose(img, 1, 0)
        # img = img.contiguous().view(w, b, h)
    for test_x, test_y in test_iter:
        if use_gpu:
            img = Variable(test_x, volatile=True).cuda()
            label = Variable(test_y, volatile=True).cuda()
        else:
            img = Variable(test_x, volatile=True)
            label = Variable(test_y, volatile=True)
        out = model(img)
        #print('out', out)
        loss = criterion(out, label)
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)

        index = np.arange(0,100)
        a = np.array(pred)
        b = np.array(label)
        c = np.array(img)
        index = index[a != b]
        #print('index', index)
        data_err = []
        pred_err = []
        label_err = []
        for inx in range(len(index)):
            data_err.append(img[index[inx]])
            label_err.append(b[index[inx]])
            pred_err.append(a[index[inx]])
        #print('data_err', data_err)
        #print('pred_err', pred_err)
        #print('label_err', label_err)

        index_right = np.arange(0,100)
        index_right = index_right[a == b]
        data_right = []
        pred_right = []
        label_right =[]
        for k in range(10):
            data_right.append(img[index_right[k]])
            pred_right.append(b[index_right[k]])
            label_right.append(a[index_right[k]])
        #print('data_right', data_right)
        #print('pred_right', pred_right)
        #print('label_right', label_right)

        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print('gemfield eval_acc', eval_acc)
    print('gemfield Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_x)), eval_acc / (len(test_x))))
torch.save(model.state_dict(), './rnn.pth')