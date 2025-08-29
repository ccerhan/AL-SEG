import sys
import platform
import time
import torch
import torchvision
from tqdm import tqdm


has_gpu = torch.cuda.is_available()
has_mps = getattr(torch, 'has_mps', False)

device = "mps" if getattr(torch, 'has_mps', False) \
    else "cuda" if torch.cuda.is_available() else "cpu"

print(f"Python Platform: {platform.platform()}")
print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")
print()
print(f"Python {sys.version}")
print("GPU is", "AVAILABLE" if has_gpu else "NOT AVAILABLE")
print("MPS (Apple Metal) is", "AVAILABLE" if has_mps else "NOT AVAILABLE")
print(f"Target device is '{device}'")

print()
print("-" * 60)
print()


def test_device(device, num_tests, batch_size):
    model = torchvision.models.resnet50().to(device)
    start = time.time()
    for _ in tqdm(range(num_tests)):
        x = torch.rand(batch_size, 3, 224, 224, device=device)
        with torch.no_grad():
            model(x)
    stop = time.time()
    elapsed_ms = (stop - start) / num_tests * 1000
    print(f"device '{device}' | batch_size {batch_size} | elapsed_time {elapsed_ms:.4f} ms")


print("Inference tests with ResNet-50:\n")

num_tests = 32
for b in range(0, 5, 2):
    for d in [device, "cpu"]:
        test_device(d, num_tests, batch_size=2 ** b)
    print()
