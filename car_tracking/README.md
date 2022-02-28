# Car Tracking Python API

# Detectors 

Given a frame, detectors output bounding boxes, labels and confidence scores of the boxes of objects present in the frame

---

* ## YOLODetector
    [car_tracking.detectors.YOLODetector]

    ```python 
    class YOLODetector(config_file, data_file, weights_file, thresh=0.25, allowed_classes=[])
    ```

    Initialize a Detector with YOLO (Darknet) configs and weights.

    **NOTE**: (For now) need to define environment variable of the path of where Darknet was compiled.

    ex: `export DARKNET_PATH=/path/to/darknet`

    ---
    ### Parameters
    * **config_file:** Path to .cfg file defining the model. (See Darknet documentation for more information.)
    * **data_file:** Path to .data file as required by YOLO models.
    * **weights_file:** Path to .weights file which contains Darknet formatted weights for the YOLO model.
    * **thresh:** Threshold for YOLO's internal NMS.
    
        Default: 0.25, however if external NMS is used, set it low (0.01).
    * **allowed_classes:** Whitelist of class names for inference. Objects detected which do not belong to any of classes defined in this list are ignored. Useful when using pretrained models when you do not need all the classes. If empty, assumes all objects are to be detected.

        Default: []
    
    ### Methods
    * ### **detect_frame(** frame **)**
        
        #### Args
        * **frame:** RGB image represented by uint8 np.ndarray.
        
            Shape: (H,W,3)
        
        #### Output
        * **boxes:** Object bounding boxes in the format of (x1,y1,x2,y2)

            Shape: (N,4)

        * **scores:** Confidence scores of the boxes detected.
            
            Shape: (N,)

        * **labels:** Integer mappings of the object labels detected. 

            Shape: (N,)
    
    ### Usage example
    ```python
    ## Set DARKNET_PATH environment variable before running
    from os.path import join as pjoin
    from car_tracking.detector import YOLODetector

    model_path = "/path/to/yolo/config&weights"
    detector = YOLODetector(pjoin(model_path,"yolov4-tiny.cfg")
                            pjoin(model_path, "coco.data"), 
                            pjoin(model_path, "yolov4-tiny.weights"),
                            thresh=0.05)

    frame = ... # read a frame [H*W*3] np.array
    boxes, scores, labels = detector.detect_frame(frame) 
    ```

<br><br>

# NMS

Non-maxima supression algorithms to filter out erroneus detections. Given **N** detections (from Detector), returns **M** detections in the same format as Detector where **N => M**.

---
* ## SeqNMS
    [car_tracking.nms.seqnms]

    ```python 
    class SeqNMS(self, frames_N, thresh=0.2, nms_thresh=0.5, linkage_thresh=0.5)
    ```
    SeqNMS rescores boxes using high confidence, temporally close boxes, boosting low confidence boxes' scores.

    **Note:**: SeqNMS is to be used on sequential frames, so if first time **filter_boxes()** is called on a frame from a video at time **t**, the next time it is called, it should be called on the frame at time **t+1**.

    **Note 2:** On any significant scene change, call **reset_buffer()** to avoid history of frames of producing erroneous results.

    ---

     ### Parameters
    * **frames_N:** Size of frame buffer to keep for SeqNMS. Bigger buffer might produce better results, but recommended to keep this value at around 5.

    * **thresh:** Filtering threshold. Any detections with confidence score lower than **thresh** (after rescoring by SeqNMS) are filtered out from output.

        Input range: [0,1]

    * **nms_thresh:** Threshold for the IoU value to determine when a box should be suppressed with regards to a best sequence. (Treat as a hyperparameter with default being a good value.)

        Input range: [0,1]

    * **linkage_thresh:**  Threshold used to link two boxes in adjacent frames.
        
        Input range: [0,1]

    ### Methods
    * ### **filter_boxes(** in_boxes, in_scores, in_labels **)**
    	Filters detections, leaving them in the same format as before, only reducing number of detections.
	
        #### Args
        * **in_boxes:** Detections in the format of (x1,y1,x2,y2)
        
            Shape: (N, 4)

        * **in_scores:** Confidence scores of detections in range [0,1] as determined by the detector.
        
            Shape: (N, )

        * **in_labels:** Integer labels of detections.
        
            Shape: (N, )
        
        #### Output
        * **boxes:** Object bounding boxes in the format of (x1,y1,x2,y2)

            Shape: (M,4)

        * **scores:** Confidence scores of the rescored & filtered detections.
            
            Shape: (M,)

        * **labels:** Integer mappings of the rescored objects.

            Shape: (M,)
    
    ---
    
    * ### **reset_buffer()**
    	Resets frame history, in case video changes or there is a sudden (discontinuous) change in the scene.
        #### Args
         
         None.
    
        #### Output
        
        None.
    
    ### Usage example
    ```python
    ## Set DARKNET_PATH environment variable before running
    from car_tracking.nms import SeqNMS
    nms = SeqNMS(frames_N=5)

    boxes, scores, labels = detector.detect_frame(frame) 
    # filtered boxes from detector
    boxes, scores, labels = nms.filter_boxes(boxes, scores, labels) 
    ```

<br><br>

# Tracker
Tracking algorithms which utilize detections (possibly filtered by NMS) to assign unique ids to objects and track temporally.

---

* # SORT

	[car_tracking.tracker.SORT]
    
    ```python
    class SORT(max_age=1, min_hits=3, iou_threshold=0.3)
    ```
    
    Initialize a SORT tracker.
    
    ---

    ### Parameters
    * **max_age:** Maximum number of frames to keep alive a track without associated detections.
    
    * **min_hits:** Minimum number of associated detections before track is initialised.

    * **iou_threshold:** Minimum IOU for match.

    ### Methods
    * ### **update(** in_boxes, in_scores, in_labels **)**
	
        Given detections, update the internal state of tracker and return tracked objects at time **t**.

        #### Args
        * **in_boxes:** Detections in the format of (x1,y1,x2,y2)
        
            Shape: (N, 4)

        * **in_scores:** Confidence scores of detections in range [0,1] as determined by the detector.
        
            Shape: (N, )

        * **in_labels:** Integer labels of detections.
        
            Shape: (N, )
        
        #### Output
        * **boxes:** Object bounding boxes in the format of (x1,y1,x2,y2)

            Shape: (M,4)

        * **ids:** IDs of tracked objects.
            
            Shape: (M,)

        * **labels:** Integer mappings of the rescored objects.

            Shape: (M,)
    
        # Test for Berk
    
    
   
   
    
    
    
    








