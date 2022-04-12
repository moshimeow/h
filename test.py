import torch
import numpy

from KeyNet import KeyNet

side = 128
batch_size = 1

k = KeyNet(side)


print(k)

e = k.forward(torch.randn(batch_size, 1, side, side), torch.randn(batch_size, 22*5))

print(e.shape)

# exit(0)

import os
import logging

import torch
import torch.nn as nn


def export_dec3():
  # Seems to be correct - 1, 2, 64, 64 don't work, neither does 1, 3, 128, 128 for example
  inp = torch.randn(1, 1, 128, 128)
  inp2 = torch.randn(1, 22*5)
  net = k

  print("Big - num parameters:", sum(p.numel() for p in net.parameters() if p.requires_grad))

  torch.onnx.export(net,         # model being run
                    (inp, inp2),       # model input (or a tuple for multiple inputs)
                    "trained-simdr-w16.onnx",       # where to save the model
                    export_params=True,  # store the trained parameter weights inside the model file
                    opset_version=13,    # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names=['inputImg'],   # the model's input names
                    output_names=['x_axis_hmap'],  # the model's output names
                    # export_raw_ir=True,

                    #   dynamic_axes={'inputImg': {0: 'batch_size'},    # variable length axes
                    #                 'x_axis_hmap': {0: 'batch_size'},
                    #                 'y_axis_hmap': {0: 'batch_size'}}
                    )
  print(" ")
  print('Model has been converted to ONNX')

export_dec3()