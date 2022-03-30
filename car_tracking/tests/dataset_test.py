# import os
# import pytest
# from ..datasets.aicity import AICity


# def AICity_dataset_test():

#     if os.environ["DATASET_PATH"] is None:
#         raise ValueError("DATASET_PATH environment variable is not set")

#     with pytest.raises(Exception):
#         dataset_path = os.environ["DATASET_PATH"]
#         dataset = AICity(dataset_path, split_type="train")
#         dataset[0]
#         dataset[0][0].video_path

