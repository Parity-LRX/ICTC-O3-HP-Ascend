"""Standalone ICTC-O3 tensor product with live gates on an allocated NPU."""
import argparse

import torch
import torch_npu
from ictc_o3 import EdgeTensorProduct, direction_harmonics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--lmax', type=int, default=2)
    parser.add_argument('--edges', type=int, default=1024)
    args = parser.parse_args()
    torch.npu.set_device(0)
    op = EdgeTensorProduct.create(channels=args.channels, input_lmax=args.lmax,
                                   edge_lmax=args.lmax)
    x = torch.randn(args.edges, args.channels*(args.lmax+1)**2,
                    device='npu', requires_grad=True)
    directions = torch.nn.functional.normalize(torch.randn(args.edges, 3, device='npu'), dim=-1)
    harmonics = torch.cat(direction_harmonics(directions, args.lmax, policy='generated'), -1)
    gates = torch.randn(args.edges, op.gate_width, device='npu', requires_grad=True)
    output = op(x, harmonics, gates)
    output.square().mean().backward()
    print('Output:', tuple(output.shape), 'Feature gradient:', tuple(x.grad.shape))


if __name__ == '__main__':
    main()
