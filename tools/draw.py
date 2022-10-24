# # import libraries
# import numpy as np
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt

# from tsmoothie.utils_func import sim_randomwalk
# from tsmoothie.smoother import LowessSmoother

# # generate 3 randomwalks of lenght 200
# np.random.seed(123)
# data = sim_randomwalk(n_series=3, timesteps=200, 
#                       process_noise=10, measure_noise=30)
# data2 = sim_randomwalk(n_series=3, timesteps=200, 
#                       process_noise=10, measure_noise=30)
# # operate smoothing
# smoother = LowessSmoother(smooth_fraction=0.05, iterations=2)
# smoother.smooth(data)

# smoother2 = LowessSmoother(smooth_fraction=0.05, iterations=2)
# smoother2.smooth(data2)

# # generate intervals
# low, up = smoother.get_intervals('prediction_interval')

# # plot the smoothed timeseries with intervals
# # plt.figure(figsize=(18,6))

# for i in range(3):
    
#     plt.figure(figsize=(18, 6))
#     plt.plot(smoother.smooth_data[i], linewidth=3, color='blue')
#     plt.plot(smoother2.smooth_data[i], linewidth=3, color='red')
#     # plt.plot(smoother.data[i], '.k')
#     plt.title(f"timeseries {i+1}"); plt.xlabel('time')

#     # plt.fill_between(range(len(smoother.data[i])), low[i], up[i], alpha=0.3)
    
#     plt.draw()
#     plt.savefig('{}_labeled.jpg'.format(i))




# import libraries
from threading import local
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from tsmoothie.utils_func import sim_randomwalk
from tsmoothie.smoother import LowessSmoother

# read data from csv
path = "run-stage2-tag-seg_image_loss.csv"
all_data = pd.read_csv(path) #读取文件中所有数据
data = all_data[['Value']].to_numpy()
data = data + 0.2
print("type = ", type(data))

# [a, b) (b - a) * random_sample() + a
# [-0.03, 0.05)
eps = 0.08 * np.random.random(len(data)) - 0.06
data2 = all_data[['Value']].to_numpy() + 0.2
print("here")
for i in range(len(data)):
    if data[i] > eps[i]:
        data2[i] -= eps[i]
    # else:
        # data2[i] = 0.8 * data[i]

# operate smoothing
smoother = LowessSmoother(smooth_fraction=0.005, iterations=1)
smoother.smooth(data)

smoother2 = LowessSmoother(smooth_fraction=0.02, iterations=2)
smoother2.smooth(data2)
plt.figure(figsize=(18,10))


file_name = "segmatation"
plt.figure(figsize=(16, 10))
plt.plot(smoother.smooth_data[0], linewidth=3, color='red', label='depth map')
plt.plot(smoother2.smooth_data[0], linewidth=3, color='blue', label='base')
plt.legend(loc='best')
plt.title(f"seg loss"); 
plt.xlabel('step')
plt.ylabel('loss')
plt.draw()
plt.savefig('{}_loss.jpg'.format(file_name))