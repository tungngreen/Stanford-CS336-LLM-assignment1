import torch
from torch import nn


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        print(f"Output of fc1: {x.dtype}, shape: {x.shape}")
        x = self.ln(x)
        print(f"Output of LayerNorm: {x.dtype}, shape: {x.shape}")
        x = self.fc2(x)
        return x
    
model = ToyModel(10, 5).cuda()
# with torch.autocast(device_type='cuda', dtype=torch.float16):
#     for name, param in model.named_parameters():
#         if param.requires_grad:
#             # Ensure parameters are in float16
#             param.data = param.data.to(dtype=torch.float16)
#             print(f"Parameter: {name}, Type: {param.dtype}")

#     x = torch.randn(32, 10, device='cuda', dtype=torch.float32)
#     output = model(x)
    
#     print(f"Output: {output.dtype}, shape: {output.shape}")

#     loss = output.sum()
#     print(f"Loss: {loss.dtype}, value: {loss.item()}")
    
#     loss.backward()
    
#     for name, param in model.named_parameters():
#         if param.grad is not None:
#             print(f"Parameter: {name}, Gradient: {param.grad.dtype}, Value: {param.dtype}")
#         else:
#             print(f"Parameter: {name} has no gradient.")
            
            
model = ToyModel(10, 5).cuda()
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(dtype=torch.bfloat16)
            print(f"Parameter: {name}, Type: {param.dtype}")

    x = torch.randn(32, 10, device='cuda', dtype=torch.float32)
    output = model(x)
    
    print(f"Output: {output.dtype}, shape: {output.shape}")

    loss = output.sum()
    print(f"Loss: {loss.dtype}, value: {loss.item()}")
    
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Parameter: {name}, Gradient: {param.grad.dtype}, Value: {param.dtype}")
        else:
            print(f"Parameter: {name} has no gradient.")