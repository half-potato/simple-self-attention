import torch
import torch.nn.functional as F

eps = 1e-5

def apply(A, x):
    # A.shape: m, n
    # x.shape: b, k, n
    # out.shape: b, k, m
    return torch.einsum('ij,bmj->bmi', A, x)

class SimpleAttention(torch.nn.Module):
    def __init__(self, n, m):
        super().__init__()
        A = torch.nn.Parameter(torch.rand(m, n))
        B = torch.nn.Parameter(torch.rand(m, n))
        self.register_parameter('A', A)
        self.register_parameter('B', B)
        
    def forward(self, x):
        # x.shape: b, k, n
        query = apply(self.A, x).sum(1)
        query.sum(1)
        query = query / (eps+torch.linalg.norm(query, dim=1, keepdims=True))
        keys = apply(self.B, x)
        keys = keys / (eps+torch.linalg.norm(keys, dim=2, keepdims=True))
        # keys.shape: (b, k, m)
        # query.shape: (b, k)
        # similarity.shape: (b, k)
        sim = torch.einsum('bi,bji->bj', query, keys)
        out = torch.einsum('bk,bkn->bn', sim, x)
        
        """
        # test on single batch
        xt = x[0].T
        # m, k -> m
        queryt = (self.A @ xt).sum(1)
        queryt /= torch.linalg.norm(queryt)
        print(query[0], queryt)
        keyst = (self.B @ xt)
        keyst /= torch.linalg.norm(keyst, dim=0)
        print(keys[0], keyst)
        simt = queryt @ keyst
        print(sim[0], simt)
        print(out[0], xt @ simt)
        """
        return out, sim