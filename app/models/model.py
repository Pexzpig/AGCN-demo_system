import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize(A, symmetric=True):
    # A = A+I
    A = A + torch.eye(A.size(0))
    # 所有节点的度
    d = A.sum(1)
    if symmetric:
        # D = D^-1/2
        D = torch.diag(torch.pow(d, -0.5))
        return D.mm(A).mm(D)
    else:
        # D = D^-1
        D = torch.diag(torch.pow(d, -1))
        return D.mm(A)


class GCN(nn.Module):

    def __init__(self, dim_in, dim_out, dim_hidden):
        super(GCN, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden, bias=False)
        self.fc2 = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.fc3 = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.fc4 = nn.Linear(dim_hidden, dim_hidden, bias=False)

        self.final_fc = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_out)
        )

    def forward(self, x, A):
        x = F.relu(self.fc1(A.bmm(x)))
        x = F.relu(self.fc2(A.bmm(x)))
        x = F.relu(self.fc3(A.bmm(x)))
        x = self.fc4(A.bmm(x))

        x = torch.mean(x, dim=1)
        x = self.final_fc(x)

        return x

class AttentionGCN(nn.Module):

    def __init__(self, dim_in, dim_out, dim_hidden, attention_num):
        super(AttentionGCN, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden, bias=False)
        self.fc2 = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.fc3 = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.fc4 = nn.Linear(dim_hidden, dim_hidden, bias=False)

        self.query_fc = nn.Linear(dim_hidden, dim_hidden)
        self.key_fc = nn.Linear(dim_hidden, dim_hidden)
        self.value_fc = nn.Linear(dim_hidden, dim_hidden)

        self.B = nn.Parameter(torch.ones(attention_num, attention_num) * 0.1)
        self.attention_alpha = nn.Parameter(torch.ones(attention_num, attention_num) * 1.0)

        self.final_fc = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_out)
        )

    def forward(self, x, A):
        x = F.relu(self.fc1((A + self.B).bmm(x)))
        x = F.relu(self.fc2((A + self.B).bmm(x)))

        #attention
        query = self.query_fc(x)
        key = self.key_fc(x).permute(0, 2, 1)
        b, l, c = query.shape
        energy = torch.bmm(query, key).reshape(b, l * l)
        attention = torch.softmax(energy, dim=1).reshape(b, l, l) * self.attention_alpha

        x = F.relu(self.fc3((A + self.B + attention).bmm(x)))
        x = self.fc4((A + self.B).bmm(x))

        x = torch.mean(x, dim=1)
        x = self.final_fc(x)

        return x

if __name__ == "__main__":
    x = torch.rand(10, 3)
    A = torch.rand(10, 10)
    gate = GCN(3, 5, 3)
    out = gate(x, A)
    print(out.shape)