import torch
import time
from thop import profile, clever_format


# model_names = torch.hub.list("pytorch/vision")
model_names = ['alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 
                'googlenet', 'inception_v3', 'mobilenet_v2', 'resnet101', 'resnet152',
                'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext50_32x4d', 
                'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'squeezenet1_0', 'squeezenet1_1', 
                'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']
with open("model_spped.txt", "w") as f:
    for model_name in model_names:
        model = torch.hub.load('pytorch/vision', model_name, pretrained=False)

        model.eval()
        model.to('cuda')

        input_tensor = torch.rand(1, 3, 256, 1024).cuda()
        flops, params = profile(model, inputs=(input_tensor, ), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        with torch.no_grad():
            model(input_tensor)
            total_cost = 0.
            for i in range(100):
                torch.cuda.synchronize()
                start = time.perf_counter()
                model(input_tensor)
                torch.cuda.synchronize()
                end = time.perf_counter()
                total_cost += (end - start) * 1000
            print(
                f"Model: {model_name}  Average time spent on one forward: {total_cost / 100 :0.3f}ms  FLOPs: {flops}  Params: {params}")
            f.write(model_name + "\t" + str(total_cost / 100) + "\t" + str(flops) + "\t" + str(params) + "\n")
    