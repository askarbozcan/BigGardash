from ._base import BaseMTMC

class DummyMTMC(BaseMTMC):
    def update(self, boxes_list, ids_list, labels_list, camera_id_list):
        assert len(boxes_list) == len(ids_list) == len(labels_list) == len(camera_id_list), \
            "The length of boxes_list, ids_list, labels_list and camera_id_list must be equal."

        all_ids = []
        for i,cid in enumerate(camera_id_list):
            camera_ids = [f"{cid}_{_id}" for _id in ids_list[i]]
            all_ids.append(camera_ids)

        return all_ids