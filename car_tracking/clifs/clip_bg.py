from typing import List, Dict
import clip
import patchify
import numpy as np
import torch
from ._base import BaseCLIFS
import cv2
from PIL import Image

class CLIPBG(BaseCLIFS):
    MODEL_STR = "ViT-B/32"
    PATCH_SIZE = 360

    def __init__(self):
        self.model, self.preprocess = clip.load(self.MODEL_STR, "cpu", jit=True)
        self.model.eval()

    def encode_frames(self, frames: List, ids: List, cam_id: str):
        assert len(frames) == len(ids)

        encoded_frames = []
        to_encode = []
        feature_data = []
        for i,frame in enumerate(frames):
            if frame.shape[0] < self.PATCH_SIZE or frame.shape[1] < self.PATCH_SIZE:
                patches = [frame]
            else:
                patches = self._make_patches(frame, self.PATCH_SIZE) + [frame]

            for p in patches:
                to_encode.append(p)
                feature_data.append({"id": ids[i], "cam_id": cam_id})
        
    def _calculate_images_features(self, images):
        # Preprocess an image, send it to the computation device and perform
        # inference
        #logging.info(f'Calculating features for batch of {len(images)} frames')
        for i in range(len(images)):
            images[i] = self._preprocess_image(images[i])

        image_stack = torch.stack(images, dim=0)
        image_t = image_stack.to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_t)
        return image_features

    def _preprocess_image(self, image):
        # cv2 image to PIL image to the model's preprocess function
        # which makes sure the image is ok to ingest and makes it a tensor
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        return self.preprocess(image)
    
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

if __name__ == "__main__":
    cli = CLIPBG()