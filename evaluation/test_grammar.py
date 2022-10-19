import numpy as np
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