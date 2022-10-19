# eval for flicker
import cv2
import numpy as np

MAX_FL = 4 # max flickers in a stable slide window
FRAMES = 10 # num of frames in a slide window
CHANGE_RATIO = 0.05 # more than CHANGE_RATIO and CHANGE_ABS will be regard as a change
CHANGE_ABS = 10 # more than CHANGE_RATIO and CHANGE_ABS will be regard as a change
PATCH_SIZE = 64

def calc_flicker(window):
    length = len(window)
    flickers = 0
    for i in range(1, length - 1):
        if window[i] < window[i-1] * (1 - CHANGE_RATIO) and window[i] < window[i+1] * (1 - CHANGE_RATIO):
            flickers += 1
        if window[i] > window[i-1] * (1 + CHANGE_RATIO) and window[i] > window[i+1] * (1 + CHANGE_RATIO):
            flickers += 1
    return flickers, flickers > MAX_FL

'''
# test
window1 = [0] * 10
print(window1)
print(calc_flicker(window1))
window2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(window2)
print(calc_flicker(window2))

window3 = [10, 5, 10, 3, 40, 50, 60, 7, 80, 90]
print(window3)
print(calc_flicker(window3))
'''

def get_patch_val(patch):
    return np.sum(patch)


def eval(video_path):
    videoCapture = cv2.VideoCapture()
    videoCapture.open(video_path)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("fps=", int(fps), "frames=", int(frames))
    ret, frame = videoCapture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame[frame < 127] = 0
    frame[frame > 127] = 1
    print(frame[0:128, 0:128])
    print("frame shape = ",frame.shape)
    if ret == False:
        print("Empty Video...")
        return
    H, W = frame.shape[:2]
    h = (int)(H / PATCH_SIZE)
    w = (int)(W / PATCH_SIZE)
    print("patchs size = (", h, w, ")") # ( 11 16 )
    all_data = np.zeros((int(frames), h, w) ,dtype=int) # (855, 11, 16)
    print("all_data.shape = ", all_data.shape)
    total_flicker = 0
    for i in range(h):
        for j in range(w): # patch(i,j)
            all_data[0][i][j] = get_patch_val(
                                    frame[i*PATCH_SIZE : i*PATCH_SIZE+PATCH_SIZE,
                                          j*PATCH_SIZE : j*PATCH_SIZE+PATCH_SIZE]
                                )
    flicker_set = set()
    for f in range(1, int(frames)):
        # print(f)
        ret, frame = videoCapture.read()
        for i in range(h):
            for j in range(w):  # patch(i,j)
                all_data[f][i][j] = get_patch_val(
                    frame[i * PATCH_SIZE: i * PATCH_SIZE + PATCH_SIZE,
                    j * PATCH_SIZE: j * PATCH_SIZE + PATCH_SIZE]
                )
        if f >= FRAMES:
            for i in range(h):
                for j in range(w):
                    # print(i,j,f-FRAMES,f)
                    # print(all_data[f-FRAMES:f, i,j])
                    # print(all_data[f-FRAMES : f][i][j])
                    _, flicker = calc_flicker(all_data[f-FRAMES:f, i,j])
                    if flicker:
                        flicker_set.add((i,j))
                        total_flicker += 1

    print("Video: ", video_path,total_flicker, "Mean Flicker Value(MFV) = ", total_flicker / frames)
    print(flicker_set)
# eval("../videos/cafee_alpha.mp4")

eval("../videos/classroom_alpha.mp4")