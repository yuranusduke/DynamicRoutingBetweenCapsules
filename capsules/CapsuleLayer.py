"""
This file completes routing-by-agreement capsule from paper:
<Dynamic routing between capsules> http://www.cs.utoronto.ca/~hinton/absps/DynamicRouting.pdf
Code inspired from : https://github.com/timomernick/pytorch-capsule/blob/master/capsule_layer.py

Created by Kunhong Yu
Date: 2021/07/05
"""
import torch as t
from torch.nn import functional as F

class CapsuleLayer(t.nn.Module):
    """Define Capsule Layer module"""

    def __init__(self, in_dim, out_dim, num_in_caps, num_out_caps, routing_iter = 3):
        """
        Args :
            --in_dim: input dimension
            --out_dim: output dimension
            --num_in_caps: number of input capsules
            --num_out_caps: number of output capsules
            --routing_iter: routing iteration, default is 3
        """
        super(CapsuleLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_in_caps = num_in_caps
        self.num_out_caps = num_out_caps
        self.routing_iter = routing_iter

        self.W = t.nn.Parameter(t.rand(1, self.num_out_caps, self.num_in_caps, self.in_dim, self.out_dim))

    @staticmethod
    def squash(x):
        """Define squash activation function
        Args :
            --x: input tensor
        return :
            --v: output squash activation
        """
        # x has shape : [m, d]
        norm = t.norm(x, 2, dim = -1, keepdim = True)
        v = (x / norm) * (norm ** 2 / (1 + norm ** 2))

        return v

    def routing_by_agreement(self, u):
        """Routing-by-agreement algorithm
        Args :
            --u: capsule from layer below, has shape [m, num_in_caps, in_dim]
        return :
            --v
        """
        # 1. At first, we need to compute u\hat
        u = t.stack([u] * self.num_out_caps, dim = 1).unsqueeze(3) # [m, num_out_caps, num_in_caps, 1, in_dim]
        W = t.cat([self.W] * u.size(0), dim = 0) # [m, num_out_caps, num_in_caps, in_dim, out_dim]
        u_hat = t.matmul(u, W) # [m, num_out_caps, num_in_caps, 1, out_dim]

        # 2. Define initial logits b
        b = t.zeros(1, self.num_out_caps, self.num_in_caps, 1).cuda()

        # 3. Routing-by-agreement
        for _  in range(self.routing_iter):
            c = F.softmax(b, dim = -2) # [1, num_out_caps, num_in_caps, 1]
            c = t.cat([c] * u.size(0), dim = 0).unsqueeze(4) # [m, num_out_caps, num_in_caps, 1, 1]
            s = t.sum(u_hat * c, dim = 2, keepdim = True) # [m, num_out_caps, 1, out_dim, 1]
            v = CapsuleLayer.squash(s)
            v_ = v
            v = t.cat([v] * self.num_in_caps, dim = 2) # [m, num_out_caps, num_in_caps, out_dim, 1]
            agreement = t.mean(t.matmul(u_hat, v.permute(0, 1, 2, 4, 3)), dim = 0) # [1, num_out_caps, num_in_caps, 1, 1]
            b = b + agreement.squeeze(-1)

        return v_.squeeze()


    def forward(self, x):

        x = self.routing_by_agreement(x)

        return x
