import numpy as np
import torch
#创建元素都为0
dpz = np.zeros((5,4),dtype=int)
print(dpz.shape)
print(dpz)
#都为1
dpo = np.ones((5,4),dtype=int)
#都为2，可以设置
dpf = np.full((5,4),2,dtype=int)


arr = np.arange(12).reshape((3, 4))
print(arr)

slice_arr = arr[0:2, 1:3].copy()
slice_arr[0][0] = 1000
print(np.sum(slice_arr))
print(slice_arr)
print(arr)


slice_arr[slice_arr < 127] = 0
slice_arr[slice_arr > 127] = 1


print(slice_arr)


x = torch.range(1, 960).reshape(2,20,3,4,2)
x = x.flatten(0, 1)
print(x.shape)
x = x.unflatten(0, (2, 20))
print(x.shape)