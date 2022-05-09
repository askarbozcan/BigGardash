from ._base import BaseMTMC

class DummyMTMC(BaseMTMC):
    def update(self, boxes_dict, ids_dict, labels_dict):
        assert len(boxes_dict) == len(ids_dict) == len(labels_dict), \
            "boxes_dict, ids_dict, labels_dict must have the same length"

        all_ids = {}
        for cid, id_list in ids_dict.items():
            camera_ids = [f"{cid}_{_id}" for _id in id_list]
            all_ids[cid] = camera_ids

        return all_ids

