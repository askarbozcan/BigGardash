from typing import List, Dict
import clip
from patchify import patchify
import numpy as np
import torch
from ._base import BaseCLIFS
import cv2
from PIL import Image

class CLIPBG(BaseCLIFS):
    MODEL_STR = "ViT-B/32"
    PATCH_SIZE = 99999
    DEVICE = "cpu"
    TOPK = 10

    def __init__(self):
        self.model, self.preprocess = clip.load(self.MODEL_STR, self.DEVICE, jit=True)
        self.model.eval()

    def encode_frames(self, frames: List, ids: List, cam_ids: List):
        assert len(frames) == len(ids) == len(cam_ids)

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
                feature_data.append({"id": ids[i], "camera": cam_ids[i], "frame":frame})
        
        if len(to_encode) > 0:
            image_features = self._calculate_images_features(to_encode)
            encoded_frames.append(image_features)

        feature_t = torch.cat(encoded_frames, dim=0)

        return feature_t, feature_data
        
    def search(self, prompt: str, feature_t: torch.Tensor, feature_data: List, n=9, threshold=200):
        text_inputs = torch.cat([clip.tokenize(prompt)]).to(self.DEVICE)
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * text_features @ feature_t.T)

        values, indices = similarity[0].topk(min(self.TOPK, similarity.shape[1]))

        used_images = set()
        response_matches = []
        for indices_idx, similarity_idx in enumerate(indices):
            if len(response_matches) >= n:
                break
            initial_match_data = feature_data[similarity_idx]
            score = float(values[indices_idx].cpu().numpy())
            initial_match_data["score"] = score**2/100_000
            img_hash = '{}-{}'.format(initial_match_data["id"],
                                      initial_match_data["camera"])
            if img_hash in used_images:
                continue

            if score < threshold:
                # We've reached the point in the sorted list
                # where scores are too low
                if len(response_matches) == 0:
                    print('No matches with score >= threshold found')
                break
            
            used_images.add(img_hash)
            response_matches.append(initial_match_data)
        
        return response_matches

    def match(self, frames: Dict[str, np.ndarray], boxes: Dict[str, np.ndarray], \
                    ids: Dict[str, np.ndarray], labels: Dict[str, np.ndarray], \
                    prompt: str):
        cutouts = []
        res_ids = []
        cam_ids = []
        for cam_id, frame in frames.items():
            for i,box in enumerate(boxes[cam_id]):
                box = box.astype(int)
                cutout = frame[box[1]:box[3], box[0]:box[2]]
                if cutout.shape[0] < 5 or cutout.shape[1] < 5:
                    continue

                cutouts.append(cutout)
                res_ids.append(ids[cam_id][i])
                cam_ids.append(cam_id)
        
        feature_t, feature_data = self.encode_frames(cutouts, res_ids, cam_ids)
        response_matches = self.search(prompt, feature_t, feature_data)

        return response_matches

        

    def _calculate_images_features(self, images):
        # Preprocess an image, send it to the computation device and perform
        # inference
        #logging.info(f'Calculating features for batch of {len(images)} frames')
        for i in range(len(images)):
            images[i] = self._preprocess_image(images[i])

        image_stack = torch.stack(images, dim=0)
        image_t = image_stack.to(self.DEVICE)
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
    import os
    cli = CLIPBG()
    path = "~/Documents/testdata"
    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    

    black_img = os.path.join(path, "black.jpeg")
    white_img = os.path.join(path, "white.jpeg")
    green_img = os.path.join(path, "green.jpeg")

    black_img = cv2.imread(black_img)
    white_img = cv2.imread(white_img)
    green_img = cv2.imread(green_img)

    frames = [black_img, white_img, green_img]
    ids = ["black", "white", "green"]

    print(cli.match({"black": black_img, "white": white_img, "green": green_img},
                    {"black": np.array([[0, 0, 5, 5]]), "white": np.array([[0, 0, 5, 5]]), "green": np.array([[0, 0, 5, 5]])},
                    {"black": np.array([0]), "white": np.array([1]), "green": np.array([2])},
                    {"black": np.array([0]), "white": np.array([1]), "green": np.array([2])},
                    "buyuk dassak"))