import os
from os.path import join as pjoin
import cv2
from xml_to_dict import XMLtoDict # pip install xml-to-dict
import time

class DETRACDataset:
    def __init__(self, data_folder, sequence_name):
        self.images_folder = pjoin(data_folder, "images", sequence_name)
        self.images_list = sorted(os.listdir(self.images_folder), key=lambda x: x.split("."))
        self.labels_file = pjoin(data_folder, "labels", sequence_name+".xml")
        _labels_dict = self.xml_to_dict(self.labels_file)

        self.ignored_regions = _labels_dict["sequence"]["ignored_region"]["box"]
        self.labels = _labels_dict["sequence"]["frame"]


    def __getitem__(self, idx):
        img_path = pjoin(self.images_folder, self.images_list[idx])
        img = cv2.imread(img_path)

        
        raw_labels = self.labels[idx]["target_list"]["target"]
        boxes = []
        ids = []
        if type(raw_labels) == dict:
            raw_labels = [raw_labels]
            
        for l in raw_labels:
            # print(l)
            _x,_y = l["box"]["left"], l["box"]["top"]
            _w,_h = l["box"]["width"], l["box"]["height"]
            boxes.append((_x,_y,_w,_h))
            ids.append(l["id"])

        return img, boxes, ids

    def visualize(self):
        for img, boxes, labels in self:
            for b in boxes:
                x,y = int(b[0]), int(b[1])
                x2,y2 = x+int(b[2]), y+int(b[3])

                cv2.rectangle(img, (x,y), (x2,y2), (255,0,0), 1)
            

            cv2.imshow('frame',img)
            time.sleep(0.05)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break        

    @staticmethod
    def xml_to_dict(filename, numeric=True):
        parser = XMLtoDict()

        with open(filename, 'r') as xml_file:
            to_parse = ''.join(xml_file.readlines())
            
        parsed = parser.parse(to_parse)
        parsed = str(parsed).replace('@', '')

        if numeric:
            import re
            locs = [(m.start(0), m.end(0)-1) for m in re.finditer(r"\'(?P<num>\d+\.?\d*)\'", parsed)]
            a_str = list(str(parsed))
            for s,e in locs:
                a_str[s] = ' '
                a_str[e] = ' '
            parsed = ''.join(a_str)

        parsed = eval(parsed)
            
        return parsed

    def in_ignored(self, pt):
        for b in self.ignored_regions:
            box = [b["left"], b["top"], b["width"], b["height"]]
            if self.bbox_check(box, pt):
                return True

        return False

    @staticmethod
    def bbox_check(box, pt):
        """
            Return whether the point is in box
            box in format x,y,w,h
        """

        return pt[0] >= box[0] and pt[0] <= box[0]+box[2] \
            and pt[1] >= box[1] and pt[1] <= box[1]+box[3]

if __name__ == "__main__":
    p = DETRACDataset("./detrac_data", "MVI_20011")
    print(p.in_ignored([10,10]))