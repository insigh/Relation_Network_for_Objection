import json
import pprint
import argparse
import numpy as np
import sys

def compute_IOU(bbox_gt, bbox_pre):
    w, h = bbox_pre[2], bbox_pre[3]
    x2, y2 = bbox_pre[0] + w, bbox_pre[1]+h
    bbox_pre[2], bbox_pre[3] = x2, y2
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
    w, h = bbox_pre[2], bbox_pre[3]
    x2, y2 = bbox_pre[0] + w, bbox_pre[1] + h
    bbox_pre[2], bbox_pre[3] = x2, y2
    center_gt = [(bbox_gt[0]+bbox_gt[2])/2, (bbox_gt[1]+bbox_gt[3])/2]
    center_pre = [(bbox_pre[0]+bbox_pre[2])/2, (bbox_pre[1]+bbox_pre[3])/2]
    vec1, vec2 = np.array(center_gt), np.array(center_pre)
    # print(vec1, vec2)
    return np.sqrt(np.sum(np.square(vec1-vec2)))




def unified_evaluation(gt_od_file=None, gt_fovea_file=None, pred_file=None, test_ann_file=None):


    image_id_to_filename = {}
    with open(test_ann_file, 'r') as test_ann_file:
        test_ann = json.load(test_ann_file)
        # print((test_ann.keys()))
        for image in test_ann['images']:
            filename = image['file_name']
            id = image['id']
            # print(type(id))
            image_id_to_filename[id] = filename
            # print(image_id_to_filename)

    gt_ods = {}
    gt_foveas = {}
    with open(gt_od_file, 'r') as gt_od_file:
        gt_od_lines = gt_od_file.readlines()
        for gt_od_line in gt_od_lines:
            gt_od = gt_od_line.strip().split()
            filename = gt_od[0]
            x1, y1, x2, y2 = gt_od[2], gt_od[3], gt_od[4], gt_od[5]
            gt_ods[filename] = list(map(int, [x1, y1, x2, y2]))
            # print(gt_ods[filename])
    with open(gt_fovea_file, "r") as gt_fovea_file:
        gt_fovea_lines = gt_fovea_file.readlines()
        for gt_fovea_line in gt_fovea_lines:
            gt_fovea = gt_fovea_line.strip().split()
            filename = gt_fovea[0]
            x1, y1, x2, y2 = gt_fovea[2], gt_fovea[3], gt_fovea[4], gt_fovea[5]
            gt_foveas[filename] = list(map(int, [x1, y1, x2, y2]))
            # print(gt_foveas[filename])

    count_od = 0
    od_iou_sum = 0.
    count_fovea = 0
    fovea_dis_sum = 0.
    with open(pred_file, 'r') as pred_file:
        predicts = json.load(pred_file)
        for predict in predicts:
            # print(predict)
            filename = image_id_to_filename[predict['image_id']]
            if predict['category_id'] == 1:
                if filename in gt_ods:
                    count_od += 1
                    predict_bbox = predict['bbox']
                    gt_bbox = gt_ods[filename]
                    # print(gt_bbox)
                    od_iou_sum += compute_IOU(gt_bbox, predict_bbox)
            else:
                if filename in gt_foveas:
                    count_fovea += 1
                    predict_bbox = predict['bbox']
                    gt_bbox = gt_foveas[filename]
                    # print(predict_bbox, gt_bbox)
                    fovea_dis_sum += compute_dis(gt_bbox, predict_bbox)
    print(od_iou_sum/count_od, fovea_dis_sum/count_fovea)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate result fenerated by a model')
    parser.add_argument('--od', dest='od', default="/home/chaojie/github/Relation-Networks-for-Object-Detection/annotations/anns.txt", type=str)
    parser.add_argument('--fovea',dest='fovea', default="/home/chaojie/github/Relation-Networks-for-Object-Detection/annotations/annotations.txt", type=str)
    parser.add_argument('--pred',dest='pred', help="res of model generated json file path", required=True, type=str)
    parser.add_argument('--ann',dest='ann', help="the test json data, default for test196",
                        default="/home/chaojie/github/Relation-Networks-for-Object-Detection/data/unified/annotations/image_info_test2018.json",
                        type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    print(args.pred)
    unified_evaluation(gt_od_file=args.od, pred_file=args.pred, gt_fovea_file=args.fovea, test_ann_file=args.ann)
