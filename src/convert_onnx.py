import os
import argparse

import torch
from torch import nn
import torch.onnx

import utils
from model import *

#Function to Convert to ONNX
def Convert_ONNX(name):

    # set the model to inference mode
    model.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)

    # Export the model
    torch.onnx.export(model,         # model being run
         dummy_input,       # model input (or a tuple for multiple inputs)
         os.path.join(os.getcwd().split('src')[0], 'models', name),       # where to save the model
         export_params=True,  # store the trained parameter weights inside the model file
         opset_version=10,    # the ONNX version to export the model to
         do_constant_folding=True,  # whether to execute constant folding for optimization
         input_names = ['input'],   # the model's input names
         output_names = ['output'], # the model's output names
         dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
    print(" ")
    print('Model has been converted to ONNX')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='6DRepNet to ONNX')
    parser.add_argument('--gpu',
                        dest='gpu_id', help='GPU device id to use [0]',
                        default=-1, type=int)
    parser.add_argument('--name',
                        dest='name', help='Name to save onnx',
                        default='sixdrepnet.onnx', type=str)
    args = parser.parse_args()

    # Load Model(gpu_id -1 = cpu)
    model = SixDRepNet(backbone_name='RepVGG-B1g2',
                       backbone_file='',
                       deploy=True,
                       pretrained=False,
                       gpu_id=args.gpu_id)

    # Load Weight & Update Weight
    weight = torch.load(os.path.join(os.getcwd().split('src')[0], 'models', '6DRepNet_300W_LP_BIWI.pth'))
    model.load_state_dict(weight)

    # Convert Onnx
    Convert_ONNX(args.name)
    print(f">>> torch model to onnx files, path is {args.name}")
    print(">>> Done.")