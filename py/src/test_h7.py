import numpy as np
import torch

input_box = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8]
])

ret = input_box.reshape(-1, 2, 2)
print("ret: ", ret)
coords = ret.astype(float)
coords[..., 0] = coords[..., 0] * 1
coords[..., 1] = coords[..., 1] * 10

print("ret: ", ret)

ret = coords.reshape(-1, 4)
print("ret: ", ret)

numpy_array = np.array([1,2,3,4,5,6,7,8])
numpy_array.reshape([2,2,2])
ten = torch.from_numpy(numpy_array)
ten[..., 0] = ten[..., 0] * 1
ten[..., 1] = ten[..., 1] * 10

print("ten: ", ret)
print("--------------------")

head_dim = 64 // 8 # /=
print("head_dim: ", head_dim)
#scale = head_dim**-0.5
#print("scale: ", scale)
scale = 2**-2
print("scale(2**-2): ", scale) # = 1/(2 ^ 2)

val1 = torch.arange(8)
val2 = torch.arange(8)[:, None]
val3 = torch.arange(8)[None,:]
print("val1: ", val1)
print("val2: ", val2)
print("val3: ", val3)
print("val3_l: ", val3.long())

# torch tensor reshape
v4 = val1.view(1, 8)
print("v4: ", v4)

v4 = val1.view(8, 1)
print("v4: ", v4)

# t1 = torch.ones(2, 5, 3)
# t2 = torch.ones(1, 3, 4)
# print("tr1: ", torch.matmul(t1, t2).shape)
# print("tr1_1: ", (t1 @ t2).shape)

t1 = torch.ones(5, 3, 4)
t2 = torch.ones(4, 2)
print("tr1: ", torch.matmul(t1, t2).shape)
print("tr1_1: ", (t1 @ t2).shape)