import numpy as np
from array2gif import write_gif
import glob

frames = glob.glob("frames_*.npy")
for frames_path in frames:
    frame_array = np.load(frames_path)

    write_gif(frame_array, frames_path.replace(".npy", ".gif"), fps=5)
