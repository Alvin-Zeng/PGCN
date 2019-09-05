import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        #output = SparseMM(adj)(support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # x = F.relu(self.gc2(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        return x


class PGCN(torch.nn.Module):
    def __init__(self, model_configs, graph_configs, test_mode=False):
        super(PGCN, self).__init__()

        self.num_class = model_configs['num_class']
        self.adj_num = graph_configs['adj_num']
        self.child_num = graph_configs['child_num']
        self.child_iou_num = graph_configs['iou_num']
        self.child_dis_num = graph_configs['dis_num']
        self.dropout = model_configs['dropout']
        self.test_mode = test_mode
        self.act_feat_dim = model_configs['act_feat_dim']
        self.comp_feat_dim = model_configs['comp_feat_dim']

        self._prepare_pgcn()
        self.Act_GCN = GCN(self.act_feat_dim, 512, self.act_feat_dim, dropout=model_configs['gcn_dropout'])
        self.Comp_GCN = GCN(self.comp_feat_dim, 512, self.comp_feat_dim, dropout=model_configs['gcn_dropout'])
        self.dropout_layer = nn.Dropout(p=self.dropout)

    def _prepare_pgcn(self):

        self.activity_fc = nn.Linear(self.act_feat_dim * 2, self.num_class + 1)
        self.completeness_fc = nn.Linear(self.comp_feat_dim * 2, self.num_class)
        self.regressor_fc = nn.Linear(self.comp_feat_dim * 2, 2 * self.num_class)

        nn.init.normal_(self.activity_fc.weight.data, 0, 0.001)
        nn.init.constant_(self.activity_fc.bias.data, 0)
        nn.init.normal_(self.completeness_fc.weight.data, 0, 0.001)
        nn.init.constant_(self.completeness_fc.bias.data, 0)
        nn.init.normal_(self.regressor_fc.weight.data, 0, 0.001)
        nn.init.constant_(self.regressor_fc.bias.data, 0)


    def train(self, mode=True):

        super(PGCN, self).train(mode)


    def get_optim_policies(self):

        normal_weight = []
        normal_bias = []

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif isinstance(m, GraphConvolution):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
        ]

    def forward(self, input, target, reg_target, prop_type):
        if not self.test_mode:
            return self.train_forward(input, target, reg_target, prop_type)
        else:
            return self.test_forward(input)


    def train_forward(self, input, target, reg_target, prop_type):

        activity_fts = input[0]
        completeness_fts = input[1]
        batch_size = activity_fts.size()[0]

        # construct feature matrix
        act_ft_mat = activity_fts.view(-1, self.act_feat_dim).contiguous()
        comp_ft_mat = completeness_fts.view(-1, self.comp_feat_dim).contiguous()

        # act cosine similarity
        dot_product_mat = torch.mm(act_ft_mat, torch.transpose(act_ft_mat, 0, 1))
        len_vec = torch.unsqueeze(torch.sqrt(torch.sum(act_ft_mat * act_ft_mat, dim=1)), dim=0)
        len_mat = torch.mm(torch.transpose(len_vec, 0, 1), len_vec)
        act_cos_sim_mat = dot_product_mat / len_mat

        # comp cosine similarity
        dot_product_mat = torch.mm(comp_ft_mat, torch.transpose(comp_ft_mat, 0, 1))
        len_vec = torch.unsqueeze(torch.sqrt(torch.sum(comp_ft_mat * comp_ft_mat, dim=1)), dim=0)
        len_mat = torch.mm(torch.transpose(len_vec, 0, 1), len_vec)
        comp_cos_sim_mat = dot_product_mat / len_mat

        mask = act_ft_mat.new_zeros(self.adj_num, self.adj_num)
        for stage_cnt in range(self.child_num + 1):
            ind_list = list(range(1 + stage_cnt * self.child_num, 1 + (stage_cnt + 1) * self.child_num))
            for i, ind in enumerate(ind_list):
                mask[stage_cnt, ind] = 1 / self.child_num
            mask[stage_cnt, stage_cnt] = 1

        mask_mat_var = act_ft_mat.new_zeros(act_ft_mat.size()[0], act_ft_mat.size()[0])
        for row in range(int(act_ft_mat.size(0)/ self.adj_num)):
            mask_mat_var[row * self.adj_num : (row + 1) * self.adj_num, row * self.adj_num : (row + 1) * self.adj_num] \
                = mask

        act_adj_mat = mask_mat_var * act_cos_sim_mat
        comp_adj_mat = mask_mat_var * comp_cos_sim_mat

        # normalized by the number of nodes
        act_adj_mat = F.relu(act_adj_mat)
        comp_adj_mat = F.relu(comp_adj_mat)

        act_gcn_ft = self.Act_GCN(act_ft_mat, act_adj_mat)
        comp_gcn_ft = self.Comp_GCN(comp_ft_mat, comp_adj_mat)

        out_act_fts = torch.cat((act_gcn_ft, act_ft_mat), dim=-1)
        act_fts = out_act_fts[:-1: self.adj_num, :]
        act_fts = self.dropout_layer(act_fts)

        out_comp_fts = torch.cat((comp_gcn_ft, comp_ft_mat), dim=-1)
        comp_fts = out_comp_fts[:-1: self.adj_num, :]

        raw_act_fc = self.activity_fc(act_fts)
        raw_comp_fc = self.completeness_fc(comp_fts)

        # keep 7 proposal to calculate completeness
        raw_comp_fc = raw_comp_fc.view(batch_size, -1, raw_comp_fc.size()[-1])[:, :-1, :].contiguous()
        raw_comp_fc = raw_comp_fc.view(-1, raw_comp_fc.size()[-1])
        comp_target = target.view(batch_size, -1, self.adj_num)[:, :-1, :].contiguous().view(-1).data
        comp_target = comp_target[0: -1: self.adj_num].contiguous()

        # keep the target proposal
        type_data = prop_type.view(-1).data
        type_data = type_data[0: -1: self.adj_num]
        target = target.view(-1)
        target = target[0: -1: self.adj_num]

        act_indexer = ((type_data == 0) + (type_data == 2)).nonzero().squeeze()

        reg_target = reg_target.view(-1, 2)
        reg_target = reg_target[0: -1: self.adj_num]
        reg_indexer = (type_data == 0).nonzero().squeeze()
        raw_regress_fc = self.regressor_fc(comp_fts).view(-1, self.completeness_fc.out_features, 2).contiguous()

        return raw_act_fc[act_indexer, :], target[act_indexer], type_data[act_indexer], \
               raw_comp_fc, comp_target, \
              raw_regress_fc[reg_indexer, :, :], target[reg_indexer], reg_target[reg_indexer, :]

    def test_forward(self, input):

        activity_fts = input[0]
        completeness_fts = input[1]
        batch_size = activity_fts.size()[0]

        # construct feature matrix
        act_ft_mat = activity_fts.view(-1, self.act_feat_dim).contiguous()
        comp_ft_mat = completeness_fts.view(-1, self.comp_feat_dim).contiguous()

        # act cosine similarity
        dot_product_mat = torch.mm(act_ft_mat, torch.transpose(act_ft_mat, 0, 1))
        len_vec = torch.unsqueeze(torch.sqrt(torch.sum(act_ft_mat * act_ft_mat, dim=1)), dim=0)
        len_mat = torch.mm(torch.transpose(len_vec, 0, 1), len_vec)
        act_cos_sim_mat = dot_product_mat / len_mat

        # comp cosine similarity
        dot_product_mat = torch.mm(comp_ft_mat, torch.transpose(comp_ft_mat, 0, 1))
        len_vec = torch.unsqueeze(torch.sqrt(torch.sum(comp_ft_mat * comp_ft_mat, dim=1)), dim=0)
        len_mat = torch.mm(torch.transpose(len_vec, 0, 1), len_vec)
        comp_cos_sim_mat = dot_product_mat / len_mat

        mask = act_ft_mat.new_zeros(self.adj_num, self.adj_num)
        for stage_cnt in range(self.child_num + 1):
            ind_list = list(range(1 + stage_cnt * self.child_num, 1 + (stage_cnt + 1) * self.child_num))
            for i, ind in enumerate(ind_list):
                mask[stage_cnt, ind] = 1 / self.child_num
            mask[stage_cnt, stage_cnt] = 1

        mask_mat_var = act_ft_mat.new_zeros(act_ft_mat.size()[0], act_ft_mat.size()[0])
        for row in range(int(act_ft_mat.size(0)/ self.adj_num)):
            mask_mat_var[row * self.adj_num: (row + 1) * self.adj_num, row * self.adj_num: (row + 1) * self.adj_num] \
                = mask

        act_adj_mat = mask_mat_var * act_cos_sim_mat
        comp_adj_mat = mask_mat_var * comp_cos_sim_mat

        # normalized by the number of nodes
        act_adj_mat = F.relu(act_adj_mat)
        comp_adj_mat = F.relu(comp_adj_mat)

        act_gcn_ft = self.Act_GCN(act_ft_mat, act_adj_mat)
        comp_gcn_ft = self.Comp_GCN(comp_ft_mat, comp_adj_mat)

        out_act_fts = torch.cat((act_gcn_ft, act_ft_mat), dim=-1)
        act_fts = out_act_fts[:-1: self.adj_num, :]

        out_comp_fts = torch.cat((comp_gcn_ft, comp_ft_mat), dim=-1)
        comp_fts = out_comp_fts[:-1: self.adj_num, :]

        raw_act_fc = self.activity_fc(act_fts)
        raw_comp_fc = self.completeness_fc(comp_fts)

        raw_regress_fc = self.regressor_fc(comp_fts).view(-1, self.completeness_fc.out_features * 2).contiguous()

        return raw_act_fc, raw_comp_fc, raw_regress_fc


