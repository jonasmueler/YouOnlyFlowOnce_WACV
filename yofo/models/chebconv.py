"""Chebyshev graph convolution (ChebNet)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import init


class ChebConv(nn.Module):
    """Chebyshev spectral graph convolution on a fixed skeleton graph.

    Args:
        in_c: Input channels per node.
        out_c: Output channels per node.
        K: Order of the Chebyshev polynomial (``K + 1`` filter taps).
        bias: Add a learnable bias.
        normalize: Use the symmetrically normalised Laplacian.
    """

    def __init__(self, in_c: int, out_c: int, K: int, bias: bool = True, normalize: bool = True) -> None:
        super().__init__()
        self.normalize = normalize
        self.weight = nn.Parameter(torch.Tensor(K + 1, 1, in_c, out_c))
        init.xavier_normal_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))
            init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)
        self.K = K + 1

    def forward(self, inputs: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        """Apply the convolution.

        Args:
            inputs: Node features ``(B, N, C)``.
            graph: Adjacency matrix ``(N, N)`` without self loops.

        Returns:
            Node features ``(B, N, out_c)``.
        """
        laplacian = ChebConv.get_laplacian(graph, self.normalize)
        mul_l = self.cheb_polynomial(laplacian).unsqueeze(1)
        result = torch.matmul(mul_l, inputs)
        result = torch.matmul(result, self.weight)
        result = torch.sum(result, dim=0)
        if self.bias is not None:
            result = result + self.bias
        return result

    def cheb_polynomial(self, laplacian: torch.Tensor) -> torch.Tensor:
        """Stack the Chebyshev polynomials of the Laplacian up to the filter order.

        Args:
            laplacian: Graph Laplacian ``(N, N)``.

        Returns:
            Polynomials ``(K + 1, N, N)``.
        """
        num_nodes = laplacian.size(0)
        multi_order = torch.zeros([self.K, num_nodes, num_nodes], device=laplacian.device, dtype=torch.float)
        multi_order[0] = torch.eye(num_nodes, device=laplacian.device, dtype=torch.float)
        if self.K == 1:
            return multi_order
        multi_order[1] = laplacian
        for k in range(2, self.K):
            multi_order[k] = 2 * torch.mm(laplacian, multi_order[k - 1]) - multi_order[k - 2]
        return multi_order

    @staticmethod
    def get_laplacian(graph: torch.Tensor, normalize: bool) -> torch.Tensor:
        """Graph Laplacian of an adjacency matrix.

        Args:
            graph: Adjacency matrix ``(N, N)``.
            normalize: Return ``I - D^-1/2 A D^-1/2`` instead of ``D - A``.

        Returns:
            Laplacian ``(N, N)``.
        """
        if normalize:
            degree = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            return torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(
                torch.mm(degree, graph), degree
            )
        degree = torch.diag(torch.sum(graph, dim=-1))
        return degree - graph
