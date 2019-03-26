import pickle
import pprint
with open('/home/chaojie/github/tf-faster-rcnn/output/res101/unified_2018_minval/default/res101_faster_rcnn_iter_100000/detection_results.pkl', 'rb') as p:
    res = pickle.load(p)
    pprint.pprint(res)
