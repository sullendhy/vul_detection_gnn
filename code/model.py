# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from modelGNN_updates import *
from utils import preprocess_features, preprocess_adj
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
           
    def forward(self, input_ids=None,labels=None): 
        outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
        logits=outputs
        prob=F.sigmoid(logits)
        if labels is not None:
            labels=labels.float()
            loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss=-loss.mean()
            return loss,prob
        else:
            return prob

# class PredictionClassification(nn.Module):
#     """Head for sentence-level classification tasks."""

#     def __init__(self, config, args, input_size=None):
#         super().__init__()
#         # self.dense = nn.Linear(args.hidden_size * 2, args.hidden_size)
#         if input_size is None:
#             input_size = args.hidden_size #input_size为256

#         self.dense = nn.Linear(input_size, args.hidden_size)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.out_proj = nn.Linear(args.hidden_size, args.num_classes)
#         print(self.out_proj)

#     def forward(self, features):  #
#         x = features
#         x = self.dropout(x)  #x[4,256], 即此时的一个源程序已经池化为一个256维的向量
#         x = self.dense(x.float())  #x[4,256]
#         x = torch.tanh(x)
#         x = self.dropout(x)
#         x = self.out_proj(x)
#         return x

# 新增的预测类
class PredictionClassification(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, args, input_size=None):
        super().__init__()
        if input_size is None:
            input_size = args.hidden_size

        self.dense1 = nn.Linear(input_size, args.hidden_size)
        self.bn1 = nn.BatchNorm1d(args.hidden_size)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)

        # 添加自注意力机制的相关参数
        self.attention = nn.MultiheadAttention(embed_dim=args.hidden_size, num_heads=4)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

        self.dense2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.bn2 = nn.BatchNorm1d(args.hidden_size)
        self.dropout3 = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(args.hidden_size, args.num_classes)

    def forward(self, features):
        x = features

        # 线性层 1
        x = self.dense1(x.float())
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)       # x [16, 256]

        # 自注意力机制
        x, _ = self.attention(x, x, x)  # Q, K, V 都是 x
        x = self.dropout2(x)

        # 线性层 2
        x = self.dense2(x)           # x [16, 256]
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout3(x)

        # 输出层
        x = self.out_proj(x)
        return x

# 原方案的类
class vul_detection_gnn(nn.Module):
    def __init__(self, encoder, config, tokenizer, args, reg_lambda=1e-4):
        super(vul_detection_gnn, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        self.reg_lambda = reg_lambda

        self.w_embeddings = self.encoder.roberta.embeddings.word_embeddings.weight.data.cpu().detach().clone().numpy()
        # 此处 embedding大小为（50265，768）
        self.tokenizer = tokenizer
        
        self.gnn = GCN_GGNN(feature_dim_size=args.feature_dim_size,
                            hidden_size=args.hidden_size,
                            num_GNN_layers=args.num_GNN_layers,
                            dropout=config.hidden_dropout_prob,
                            residual=not args.remove_residual,
                            att_op=args.att_op)
        
        gnn_out_dim = self.gnn.out_dim   # gnn_out_dim为256，此时每个程序已被处理为一个256维向量
        self.classifier = PredictionClassification(config, args, input_size=gnn_out_dim)

    def forward(self, input_ids=None, labels=None):
        # construct graph
        
        adj, x_feature = build_graph(input_ids.cpu().detach().numpy(), self.w_embeddings, window_size=self.args.window_size)
        # initilizatioin
        adj, adj_mask = preprocess_adj(adj)
        adj_feature = preprocess_features(x_feature)
        adj = torch.from_numpy(adj)
        adj_mask = torch.from_numpy(adj_mask)
        adj_feature = torch.from_numpy(adj_feature)
        # run over GNNs
        outputs = self.gnn(adj_feature.to(device).double(), adj.to(device).double(), adj_mask.to(device).double())
        logits = self.classifier(outputs)
        prob = F.sigmoid(logits)

        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            
            # Flooding technique
            loss = abs(loss - 0.40) + 0.40
            return loss, prob
        else:
            return prob


# modified from https://github.com/saikat107/Devign/blob/master/modules/model.py
class DevignModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(DevignModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        self.w_embeddings = self.encoder.roberta.embeddings.word_embeddings.weight.data.cpu().detach().clone().numpy()
        self.tokenizer = tokenizer

        self.gnn = GGGNN(feature_dim_size=args.feature_dim_size, hidden_size=args.hidden_size,
                         num_GNN_layers=args.num_GNN_layers, num_classes=args.num_classes, dropout=config.hidden_dropout_prob)

        self.conv_l1 = torch.nn.Conv1d(args.hidden_size, args.hidden_size, 3).double()
        self.maxpool1 = torch.nn.MaxPool1d(3, stride=2).double()
        self.conv_l2 = torch.nn.Conv1d(args.hidden_size, args.hidden_size, 1).double()
        self.maxpool2 = torch.nn.MaxPool1d(2, stride=2).double()

        self.concat_dim = args.feature_dim_size + args.hidden_size
        self.conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 3).double()
        self.maxpool1_for_concat = torch.nn.MaxPool1d(3, stride=2).double()
        self.conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 1).double()
        self.maxpool2_for_concat = torch.nn.MaxPool1d(2, stride=2).double()

        self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=args.num_classes).double()
        self.mlp_y = nn.Linear(in_features=args.hidden_size, out_features=args.num_classes).double()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids=None, labels=None):
        # construct graph
        if self.args.format == "uni":
            adj, x_feature = build_graph(input_ids.cpu().detach().numpy(), self.w_embeddings)
        else:
            adj, x_feature = build_graph_text(input_ids.cpu().detach().numpy(), self.w_embeddings)
        # initilization
        adj, adj_mask = preprocess_adj(adj)
        adj_feature = preprocess_features(x_feature)
        adj = torch.from_numpy(adj)
        adj_mask = torch.from_numpy(adj_mask)
        adj_feature = torch.from_numpy(adj_feature).to(device).double()
        # run over GGGN
        outputs = self.gnn(adj_feature.to(device).double(), adj.to(device).double(), adj_mask.to(device).double()).double()
        #
        c_i = torch.cat((outputs, adj_feature), dim=-1)
        batch_size, num_node, _ = c_i.size()
        Y_1 = self.maxpool1(nn.functional.relu(self.conv_l1(outputs.transpose(1, 2))))
        Y_2 = self.maxpool2(nn.functional.relu(self.conv_l2(Y_1))).transpose(1, 2)
        Z_1 = self.maxpool1_for_concat(nn.functional.relu(self.conv_l1_for_concat(c_i.transpose(1, 2))))
        Z_2 = self.maxpool2_for_concat(nn.functional.relu(self.conv_l2_for_concat(Z_1))).transpose(1, 2)
        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        avg = before_avg.mean(dim=1)
        prob = self.sigmoid(avg)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob

