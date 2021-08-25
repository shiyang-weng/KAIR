import torch
from models.network_usrnet_v1 import USRNet as net
from time import time
import os
import intel_pytorch_extension as ipex
import torch.autograd.profiler as profiler
#ipex.enable_auto_mixed_precision(mixed_dtype = torch.bfloat16)
# os.environ["LD_PRELOAD"]="/root/miniconda3/envs/torch_intel/lib/libjemalloc.so"
print("start test ipex+jemalloc: LD_PRELOAD={} , MALLOC_CONF={} ".format(os.environ.get("LD_PRELOAD"), os.environ.get("MALLOC_CONF")))

# ----------------------------------------
# Preparation
# ----------------------------------------
channel = 3
image_shape = [1, channel, 1920, 1080]
kernel_shape = [1, 1, 25, 25]
loop = 10
warmup_num = 10

model_name = 'usrnet_tiny'      # 'usrgan' | 'usrnet' | 'usrgan_tiny' | 'usrnet_tiny'

device = ipex.DEVICE

test_sf = torch.tensor([2]).to(device)  # scale factor, from {1,2,3,4}
sigma = torch.tensor([[[[0.]]]]).to(device)

# ----------------------------------------
# setup model
# ----------------------------------------
model = net(n_iter=6, h_nc=32, in_nc=4, out_nc=3, nc=[16, 32, 64, 64],
            nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
model.eval()

for key, v in model.named_parameters():
    v.requires_grad = False

model = model.to(device)

# ----------------------------------------
# test
# ----------------------------------------
with torch.no_grad():
    cnt = 0
    k = torch.rand(kernel_shape)
    x = torch.rand(image_shape)
    
    model = torch.jit.trace(model, (x, k, test_sf, sigma))

    for i in range(warmup_num):
        k = torch.rand(kernel_shape).to(device)
        x = torch.rand(image_shape).to(device)
        y = model(x, k, test_sf, sigma)

    with profiler.profile(record_shapes=False) as prof:
        with profiler.record_function("model_inference"):
            x = model(x, k, test_sf, sigma)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=1000))
    st = time()
    for i in range(loop):
        k = torch.rand(kernel_shape).to(device)
        x = torch.rand(image_shape).to(device)
        y = model(x, k, test_sf, sigma)
    et = time()
    cnt += et - st

print("usingtime: ", cnt/loop)