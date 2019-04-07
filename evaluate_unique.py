import json
import pprint
import argparse
import numpy as np
import sys

def compute_IOU(bbox_gt, bbox_pre):
    # w, h = bbox_pre[2], bbox_pre[3]
    # x2, y2 = bbox_pre[0] + w, bbox_pre[1]+h
    # bbox_pre[2], bbox_pre[3] = x2, y2

    # computing area of each rectangles
    S_rec1 = (bbox_gt[2] - bbox_gt[0]) * (bbox_gt[3] - bbox_gt[1])
    S_rec2 = (bbox_pre[2] - bbox_pre[0]) * (bbox_pre[3] - bbox_pre[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(bbox_gt[1], bbox_pre[1])
    right_line = min(bbox_gt[3], bbox_pre[3])
    top_line = max(bbox_gt[0], bbox_pre[0])
    bottom_line = min(bbox_gt[2], bbox_pre[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect / (sum_area - intersect)


def compute_dis(bbox_gt, bbox_pre):
    # w, h = bbox_pre[2], bbox_pre[3]
    # x2, y2 = bbox_pre[0] + w, bbox_pre[1] + h
    # bbox_pre[2], bbox_pre[3] = x2, y2
    center_gt = [(bbox_gt[0]+bbox_gt[2])/2, (bbox_gt[1]+bbox_gt[3])/2]
    center_pre = [(bbox_pre[0]+bbox_pre[2])/2, (bbox_pre[1]+bbox_pre[3])/2]
    vec1, vec2 = np.array(center_gt), np.array(center_pre)
    # print(vec1, vec2)
    return np.sqrt(np.sum(np.square(vec1-vec2)))




def unified_evaluation_fovea(pred_path=None, ann_path=None):
    """

    :param pred_path:
    :param ann_path:
    :return:
    """
    pred_path = "/home/zcj/Documents/results/relation_model/unique_model/fovea/fovea100_test_1.json"
    ann_path = "/home/zcj/github/od_fovea_location/data/fovea/annotations/instances_test2018.json"

    count = 0
    sum_dis = 0.
    preds = {}
    with open(pred_path, 'r') as pred_file:
        pred_bboxs = json.load(pred_file)
        for pred_bbox in pred_bboxs:
            # print(pred_bbox['image_id'], pred_bbox['bbox'])
            preds[str(pred_bbox['image_id'])] = pred_bbox['bbox']

    anns = {}
    with open(ann_path, 'r') as ann_file:
        pred_bboxs = json.load(ann_file)
        # print(pred_bboxs.keys())
        for pred_bbox in pred_bboxs["annotations"]:
            # print(pred_bbox)
            anns[str(pred_bbox["image_id"])] = pred_bbox[str("bbox")]

    for key_ in preds.keys():
        pred_box = preds[key_]
        w, h = pred_box[2], pred_box[3]
        pred_box[2], pred_box[3] = pred_box[0]+w, pred_box[1]+h
        ann_box = anns[key_]
        w, h = ann_box[2], ann_box[3]
        ann_box[2], ann_box[3] = ann_box[0] + w, ann_box[1] + h
        sum_dis += compute_dis(pred_box, ann_box)
    # print(preds)
    # print(anns)
    print(sum_dis/len(preds))

def unified_evaluation_od(pred_path = "/home/zcj/Documents/results/relation_model/unique_model/od/unique200_test_83.9.json", ann_path=None):

    ann_path = "/home/zcj/github/od_fovea_location/data/od/annotations/image_info_test2018.json"
    count = 0
    sum_iou = 0.
    preds = {}
    with open(pred_path, 'r') as pred_file:
        pred_bboxs = json.load(pred_file)
        for pred_bbox in pred_bboxs:
            # print(pred_bbox['image_id'], pred_bbox['bbox'])
            preds[str(pred_bbox['image_id'])] = pred_bbox['bbox']

    anns = {}
    with open(ann_path, 'r') as ann_file:
        pred_bboxs = json.load(ann_file)
        # print(pred_bboxs.keys())
        for pred_bbox in pred_bboxs["annotations"]:
            # print(pred_bbox)
            anns[str(pred_bbox["image_id"])] = pred_bbox[str("bbox")]

    for key_ in preds.keys():
        pred_box = preds[key_]
        w, h = pred_box[2], pred_box[3]
        pred_box[2], pred_box[3] = pred_box[0]+w, pred_box[1]+h
        ann_box = anns[key_]
        w, h = ann_box[2], ann_box[3]
        ann_box[2], ann_box[3] = ann_box[0] + w, ann_box[1] + h
        sum_iou += compute_IOU(pred_box, ann_box)
    # print(preds)
    # print(anns)
    print(sum_iou/len(preds))

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate result fenerated by a model')
    parser.add_argument('--od', dest='od', default="/home/zcj/github/od_fovea_location/data/od/annotations.txt", type=str)
    parser.add_argument('--fovea',dest='fovea', default="/home/zcj/github/od_fovea_location/data/fovea/anns.txt", type=str)
    parser.add_argument('--pred',dest='pred', help="res of model generated json file path", required=True, type=str)
    parser.add_argument('--ann',dest='ann', help="the test json data, default for test196",
                        default="/home/zcj/github/od_fovea_location/data/unified/annotations/instances_minval2018.json",
                        type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

# if __name__ == '__main__':
#
#     args = parse_args()
#     print(args.pred)
#     unified_evaluation_od()

unified_evaluation_fovea()
# unified_evaluation_od()