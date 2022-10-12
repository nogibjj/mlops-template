import torch

if torch.cuda.is_available():
    print("CUDA is available")
    print("CUDA version: {}".format(torch.version.cuda))
    print("PyTorch version: {}".format(torch.__version__))
    print("cuDNN version: {}".format(torch.backends.cudnn.version()))
    print("Number of CUDA devices: {}".format(torch.cuda.device_count()))
    print("Current CUDA device: {}".format(torch.cuda.current_device()))
    print("Device name: {}".format(torch.cuda.get_device_name(torch.cuda.current_device())))
else:
    print("CUDA is not available")