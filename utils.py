import torch.utils.data
import scipy.sparse as sp
from torch_geometric.utils import subgraph, dense_to_sparse
from torch_geometric.utils import add_self_loops, degree, to_scipy_sparse_matrix
from torch_scatter import scatter_add
from torch_geometric.utils import add_self_loops, remove_self_loops, remove_isolated_nodes
from torch_geometric.utils.num_nodes import maybe_num_nodes
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch_scatter import scatter
import numpy as np

def coalesce(
    edge_index: Tensor,
    edge_attr: Optional[Union[Tensor, List[Tensor]]] = None,
    num_nodes: Optional[int] = None,
    reduce: str = "add",
    is_sorted: bool = False,
    sort_by_row: bool = True,
) -> Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, List[Tensor]]]:

    nnz = edge_index.size(1)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    idx = edge_index.new_empty(nnz + 1)
    idx[0] = -1
    idx[1:] = edge_index[1 - int(sort_by_row)]
    idx[1:].mul_(num_nodes).add_(edge_index[int(sort_by_row)])

    if not is_sorted:
        idx[1:], perm = idx[1:].sort()
        edge_index = edge_index[:, perm]
        if edge_attr is not None and isinstance(edge_attr, Tensor):
            edge_attr = edge_attr[perm]
        elif edge_attr is not None:
            edge_attr = [e[perm] for e in edge_attr]

    mask = idx[1:] > idx[:-1]

    # Only perform expensive merging in case there exists duplicates:
    if mask.all():
        return edge_index if edge_attr is None else (edge_index, edge_attr)

    edge_index = edge_index[:, mask]
    if edge_attr is None:
        return edge_index

    dim_size = edge_index.size(1)
    idx = torch.arange(0, nnz, device=edge_index.device)
    idx.sub_(mask.logical_not_().cumsum(dim=0))

    if isinstance(edge_attr, Tensor):
        edge_attr = scatter(edge_attr, idx, 0, None, dim_size, reduce)
    else:
        edge_attr = [
            scatter(e, idx, 0, None, dim_size, reduce) for e in edge_attr
        ]

    return edge_index, edge_attr


def get_sym(edge_index, self_loops = True, num_nodes: Optional[int] = None,
            edge_weight: Optional[torch.Tensor] = None):

    if self_loops==True:
        edge_index = add_self_loops(edge_index)[0]

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight


def normalize_adj(edge_index, N):
    idx, val = get_sym(edge_index, self_loops=True, num_nodes=N)
    sparse_A_hat = torch.sparse.FloatTensor(idx, val, torch.Size([N, N]))
    return sparse_A_hat



def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_adj_neg(num_nodes, sample, super_node_list):

    super_node_list = np.array(super_node_list)

    row = np.repeat(range(num_nodes), sample)
    col = np.random.randint(0, num_nodes, size=num_nodes * sample)
    new_col = np.concatenate((col, row), axis=0)
    new_row = np.concatenate((row, col), axis=0)

    data = super_node_list[new_row] * super_node_list[new_col]
    adj_neg = sp.coo_matrix((data, (new_row, new_col)), shape=(num_nodes, num_nodes))
    adj = np.array(adj_neg.sum(1)).flatten()
    adj_neg = sp.diags(adj) - adj_neg

    return adj_neg.toarray()



def coarsening(x, edge_index, reduce_size=0.5, threshold=10, device='cpu'):

    N = x.size(0)
    deg = degree(edge_index[0, :], num_nodes=N)+1

    deg2 = degree(edge_index[0, :], num_nodes=N)
    edge_mask = (edge_index[0, :] < edge_index[1, :])
    u = edge_index[0, edge_mask]
    v = edge_index[1, edge_mask]
    w = 1 / (deg2[u] + deg2[v])
    p = w / torch.sum(w)

    indices = np.random.choice(range(u.size(0)), round(reduce_size * N), replace=False, p=p.numpy())

    cluster = torch.arange(N).to(device)

    length = u.size(0)
    for i in range(round(reduce_size * N)):
        if i == length:
            break
        cluster1 = cluster[u[indices[i]]]
        cluster2 = cluster[v[indices[i]]]
        if cluster1 != cluster2:
            if torch.sum(cluster == cluster1) + torch.sum(cluster == cluster2) <= threshold:
                if cluster1 > cluster2:
                    cluster = torch.where(cluster == cluster1, cluster2, cluster)
                else:
                    cluster = torch.where(cluster == cluster2, cluster1, cluster)

    subset = torch.unique(cluster)
    feature = torch.zeros([subset.size(0), x.size(1)]).to(device)
    edge_index = add_self_loops(edge_index)[0]
    x = x.to(device)
    edge_index = edge_index.to(device)
    deg = deg.to(device)
    t = 0
    cluster2 = torch.zeros(N)

    ori_node_list =[]
    coarsen_node_list = []
    super_node_list=[]
    for i in range(N):
        mix_node = torch.nonzero(cluster == i).view(-1)
        if mix_node.size(0) > 0:
            cluster2[mix_node] = t
            if mix_node.size(0)==1:
                feature[t, :] = x[mix_node,:]
                ori_node_list+=[t]
            else:
                node_feature = x[mix_node,:]
                node_degree = deg[mix_node].view(1, -1)
                feature[t, :] = torch.mm(node_degree, node_feature) / torch.sum(node_degree)
            super_node_list+=[mix_node.size(0)]
            t+=1
    x = feature

    edge_index = torch.cat((cluster[edge_index[0]].view(1, -1), cluster[edge_index[1]].view(1, -1)), dim=0)

    edge_weight = torch.ones(edge_index.size(1)).to(device)
    edge_index, edge_weight = coalesce(edge_index, edge_weight, reduce="sum")
    edge_index, edge_weight = subgraph(subset, edge_index, edge_attr=edge_weight, num_nodes=N, relabel_nodes=True)

    return x, edge_index, cluster2.long(), ori_node_list, super_node_list



def GC(edge_index, N, F):
    adj = normalize_adj(edge_index, N)
    x = F
    for i in range(2):
        x = torch.spmm(adj, x)
    return x

