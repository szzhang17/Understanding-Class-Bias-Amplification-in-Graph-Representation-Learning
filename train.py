import argparse
from classification import structure_imbalance_eval
from networks import Lin
from torch_geometric.utils import degree
import torch
import random
from utils import load_adj_neg, coarsening, GC
from torch_geometric.datasets import Planetoid

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='cora',
                    help='dataset')
parser.add_argument('--seed', type=int, default=123,
                    help='seed')
parser.add_argument('--output', type=int, default=512,
                    help='output size')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='weight decay')
parser.add_argument('--epochs', type=int, default=25,
                    help='maximum number of epochs')
parser.add_argument('--sample', type=int, default=5,
                    help='    ')
parser.add_argument('--alpha', type=int, default=15000,
                    help='    ')
parser.add_argument('--beta', type=int, default=500,
                    help='    ')
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = Planetoid(root='./dataset', name=args.dataset)
data = dataset[0]

deg = degree(data.edge_index[0, :], num_nodes=data.num_nodes)

N, E, args.num_features = data.num_nodes, data.num_edges, data.num_features
args.num_nodes = N

F_whole = GC(data.edge_index, data.x.size(0), data.x)
F_whole = F_whole.to(device)

model = Lin(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
model.train()

for epoch in range(1, args.epochs+1):

    optimizer.zero_grad()
    if epoch==1 or epoch%5==0:
        x_a, edge_index_a, cluster_a, _, super_node_list = coarsening(data.x, data.edge_index, reduce_size=0.3)
        F_a = GC(edge_index_a, x_a.size(0), x_a)
        F_a = F_a.to(device)

    neg_sample_a = torch.from_numpy(load_adj_neg(F_a.size(0), args.sample, super_node_list)).float().to(device)
    out_a = model(F_a)
    loss_neg_a = args.alpha/torch.trace(torch.mm(torch.mm(torch.transpose(out_a, 0, 1), neg_sample_a), out_a))
    z_a = out_a[cluster_a]
    out_ori = model(F_whole)
    loss_pos = args.beta / torch.trace(torch.mm(torch.transpose(z_a, 0, 1), out_ori))
    loss = loss_neg_a+loss_pos
    loss.backward()
    optimizer.step()

    if epoch%25==0:
        with torch.no_grad():
            emb = model(F_whole)
        structure_imbalance_eval(args.dataset, emb.cpu(), data.y.cpu(), 50)

