import torch
print(torch.__version__)
print(torch.cuda.is_available())



x = torch.randn(1).cuda()
print(x)


import transformers
import accelerate
import trl

print("PyTorch version:", torch.__version__)
print("Transformers version:", transformers.__version__)
print("Accelerate version:", accelerate.__version__)
print("TRL version:", trl.__version__)
