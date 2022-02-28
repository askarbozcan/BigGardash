import random
import cv2
import numpy as np
from ._base import BaseArtist

class TrackingArtist(BaseArtist):
    """
        
    """
    
    def __init__(self, n_colors=20):
        self.rand_colors = []
        
        for i in range(n_colors):
            c = (random.randint(20,255), random.randint(20,250), random.randint(20,250))
            self.rand_colors.append(c)

    def draw(self, frame, boxes, ids, labels):
        assert boxes.shape[0] == ids.shape[0] == labels.shape[0], "Mismatching boxes, ids or labels."
        
        ## draw detections
        blk = np.zeros(frame.shape, np.uint8)


        for i,b in enumerate(boxes):
            _id = int(ids[i])

            color_i = _id % len(self.rand_colors)
            color = self.rand_colors[color_i]

            cv2.putText(frame, str(_id), (int(b[0]+10), int(b[1]+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,0,0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 3)
            cv2.rectangle(blk, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, cv2.FILLED)


        frame = cv2.addWeighted(frame, 1.0, blk, 0.25, 1)

        return frame