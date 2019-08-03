import torch
import time
from ptflops import get_model_complexity_info


model_names = torch.hub.list("pytorch/vision")
with open("model_spped.txt", "w") as f:
    for model_name in model_names:
        model = torch.hub.load('pytorch/vision', model_name, pretrained=False)

        model.eval()
        model.to('cuda')

        flops, params = get_model_complexity_info(
            model, (3, 256, 1024), as_strings=True, print_per_layer_stat=False)

        input_tensor = torch.rand(1, 3, 256, 1024).cuda()
        with torch.no_grad():
            model(input_tensor)
            total_cost = 0.
            for i in range(100):
                torch.cuda.synchronize()
                start = time.perf_counter()
                model(input_tensor)
                torch.cuda.synchronize()
                end = time.perf_counter()
                total_cost += (end - start)
            print(
                f"Model: {model_name}  Average time spent on one forward: {total_cost / 100}  FLOPs: {flops}  Params: {params}")
            f.write(model_name + "\t" + str(total_cost / 100) + "\t" + flops + "\t" + params + "\n")
    