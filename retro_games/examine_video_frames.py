import matplotlib.pyplot as plt
import numpy as np
import glob

video_frames = glob.glob("playback/*video*npy")

for i in range(10000):
    plt.imshow(np.load(video_frames[0])[i][:, :, 0])
    plt.savefig(f"playback/video_{i}.png")
    plt.close()
