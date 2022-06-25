from typing import List
import clip
import patchify

class CLIPBG:
    MODEL_STR = "ViT-B/32"

    def __init__(self):
        self.model, self.preprocess = clip.load(self.MODEL_STR, "cuda", jit=False)

    def encode_frames(self, frames: List):




    def _make_patches(self, frame, patch_size=360):
        # To get more information out of images, we divide the image
        # into smaller patches that are closer to the input size of the model
        step = int(patch_size / 2)
        patches_np = patchify(frame, (patch_size, patch_size, 3),
                              step=step)
        patches = []
        for i in range(patches_np.shape[0]):
            for j in range(patches_np.shape[1]):
                patches.append(patches_np[i, j, 0])
        return patches