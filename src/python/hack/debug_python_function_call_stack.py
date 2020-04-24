import os
import sys
import torch

def tracefunc(frame, event, arg, indent=[0]):
      if event == "call":
          indent[0] += 2
          print "-" * indent[0] + "> call function {}:{}".format(frame.f_code.co_filename, frame.f_code.co_name)
      elif event == "return":
          print "<" + "-" * indent[0], "exit function {}:{}".format(frame.f_code.co_filename, frame.f_code.co_name)
          indent[0] -= 2
      return tracefunc

sys.stdout.flush()
sys.settrace(tracefunc)

gemfield = torch.Tensor([[1,2],[3,4]])
gemfield.requires_grad = True

civilnet = gemfield * gemfield

gemfieldout = civilnet.mean()

gemfieldout.backward()

print(gemfield.grad)
