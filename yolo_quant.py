import torch
import torch.nn as nn

from models.common import Conv


class Added_Quant_Network(nn.Module):
    def __init__(self, origin_model):
        super(Added_Quant_Network, self).__init__()
        self.quant = torch.quantization.QuantStub()     # 输入激活值量化模块
        self.origin_model = origin_model
        # self.dequant = torch.quantization.DeQuantStub() # 输出激活值解量化模块

    def forward(self, x):
        x = self.quant(x)
        out = self.origin_model(x)
        # out = self.dequant(out)
        return out

def convert(model): # 细粒度转换：对yolo中每个Conv模块（包含一个conv2d和一个silu）插入激活值量化/解量化模块
    reassign = {}
    for name, module in model.named_children():
        print(name, type(module))

        module = convert(module)

        # if isinstance(module, nn.SiLU):
        #     print("ReLu, ", name)
        #     reassign[name] = nn.ReLU(inplace=True)
        
    for key, value in reassign.items():
        model._modules[key] = value
    
    if isinstance(model, Conv):
        model._modules['quant'] = torch.quantization.QuantStub()
        model._modules['dequant'] = torch.quantization.DeQuantStub()

    # if isinstance(model, Bottleneck):
    #     print("Bottleneck: ")
    #     model._modules['ff'] = nn.quantized.FloatFunctional()
    
    # if isinstance(model, Model):
    #     print("Model + dequant: ")
    #     model._modules['dequant'] = torch.quantization.DeQuantStub()
        
    #     print("Model + quant: ")
    #     model._modules['quant'] = torch.quantization.QuantStub()

    return model

def prepare(model):

    print(type(model.model))
    
    torch.quantization.prepare(model.model, inplace=True)
    # torch.quantization.prepare(model.model.dequant, inplace=True)

    for name, module in model.model.model.named_children():
        print(name, type(module))

    # if isinstance(model, Detect):
    #     return

    # for name, module in model.named_children():
    #     print(name, type(module))
    #     prepare(module)


def fuse_module(model):
    # reassign = {}
    for name, module in model.named_children():
        print(name, type(module))
        # if isinstance(module, nn.Sequential) or isinstance(module, nn.ModuleList):

        module = fuse_module(module)

        if isinstance(module, Conv):
            print("Conv, ", name)
            torch.quantization.fuse_modules(module, [['conv','act']], inplace=True)
            # reassign[name] = nn.ReLU(inplace=True)
            # reassign[name] = module

    # for key, value in reassign.items():
    #     model._modules[key] = value
    
    return model