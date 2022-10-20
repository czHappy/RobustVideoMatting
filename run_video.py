import torch
from model import MattingNetwork

model = MattingNetwork('mobilenetv3').eval().cuda(1)  # or "resnet50"
model.load_state_dict(torch.load('checkpoints/rvm_mobilenetv3.pth'))
from inference import convert_video

convert_video(
    model,                           # The model, can be on any device (cpu or cuda).
    input_source='video/origin/caffe.mov',        # A video file or an image sequence directory.
    output_type='video',             # Choose "video" or "png_sequence"
    output_alpha="cafee_alpha.mp4",
    output_composition='out_caffe.mp4', # File path if video; directory path if png sequence.
    output_video_mbps=4,             # Output video mbps. Not needed for png sequence.
    downsample_ratio=None,           # A hyperparameter to adjust or use None for auto.
    seq_chunk=8,                    # Process n frames at once for better parallelism.
)

