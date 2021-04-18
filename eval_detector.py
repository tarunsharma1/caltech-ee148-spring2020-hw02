import os
import json
import numpy as np
import matplotlib.pyplot as plt
import copy

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    # compare rows first, find which box is below the other
    if box_1[0] >= box_2[2] or box_2[0] >= box_1[2]:
        iou = 0
        return iou

    # compare cols to see if one box to the left or right or other box
    if box_1[1] >= box_2[3] or box_2[1] >= box_1[3]:
        iou = 0
        return iou

    # calculate intersection
    
    intersection_row = abs(max(box_1[0], box_2[0]) - min(box_1[2], box_2[2]))
    intersection_col = abs(max(box_1[1], box_2[1]) - min(box_1[3], box_2[3]))
    
    
    intersection = intersection_row * intersection_col

    # when calculating union dont include intersection area twice
    union = ((box_1[2] - box_1[0]) * (box_1[3] - box_1[1])) + ((box_2[2] - box_2[0]) * (box_2[3] - box_2[1])) - intersection
    iou = intersection/union

    
    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    local_preds = copy.deepcopy(preds)
    '''
    BEGIN YOUR CODE
    '''
    for pred_file, pred in local_preds.items():
        gt = gts[pred_file]

        for i in range(len(gt)):
            # convert the gt to integers (some of them are floats)
            gt[i] = [int(item) for item in gt[i]]

            TP_flag = 0
            for j in range(len(pred)):
                
                if pred[j] == -1:
                    continue

                # take only those predictions which are above the confidence threshold
                if pred[j][-1] < conf_thr:
                    continue
                
                iou = compute_iou(pred[j][:4], gt[i])
                
                if iou > iou_thr:

                    # if we already associated a prediction with this gt then do not recount it as TP..instead it will be counted as a FP
                    if TP_flag == 1:
                        continue

                    # if no prediction has been associated with this gt, then count it as TP and remove it from list (so its not counted as a FP for a different gt)
                    TP += 1
                    TP_flag = 1
                    # remove it from our list of preds (set it to -1 for now)
                    pred[j] = -1
            
            # if none of the predictions had an overlap with this particular gt, then we know it is a miss
            if TP_flag == 0:
                FN += 1

        # remove the -1s and whatever remains now in pred are all false positives
        pred_clean = [y for y in pred if y!= -1]
        FP += len(pred_clean)



    '''
    END YOUR CODE
    '''

    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = '/home/tarun/Downloads/caltech-ee148-hw-data/hw2/hw02_preds'
gts_path = '/home/tarun/Downloads/caltech-ee148-hw-data/hw2/hw02_annotations'

# load splits:
split_path = '/home/tarun/Downloads/caltech-ee148-hw-data/hw2/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data. 
'''
with open(os.path.join(preds_path,'preds_train_weakened.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'formatted_annotations_students.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test_weakened.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'formatted_annotations_students.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold. 

# go through the list of all bounding boxes for each image, and get their confidence scores

confidence_thrs = []
for fname in preds_train:
    bboxs = preds_train[fname]
    for box in bboxs:
        confidence_thrs.append(box[4])

confidence_thrs = np.sort(np.array(confidence_thrs))

plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('precision recall curve - training set weakened')

for iou_thresh in np.array([0.25,0.5, 0.75]):
    tp_train = np.zeros(len(confidence_thrs))
    fp_train = np.zeros(len(confidence_thrs))
    fn_train = np.zeros(len(confidence_thrs))

    for i, conf_thr in enumerate(confidence_thrs):
        tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=iou_thresh, conf_thr=conf_thr)

    # Plot training set PR curves
    precision_list = []
    recall_list = []

    for i in range(0,confidence_thrs.shape[0]):
        precision = tp_train[i]/(tp_train[i] + fp_train[i])
        recall = tp_train[i]/(tp_train[i] + fn_train[i])
        if precision==0 and recall==0:
            continue
        precision_list.append(precision)
        recall_list.append(recall)

    #print (iou_thresh)
    #print (recall_list, precision_list)
    #import ipdb;ipdb.set_trace()
    plt.plot(np.array(recall_list), np.array(precision_list), alpha=0.7, label='iou_thresh: '+str(iou_thresh))

plt.legend()
plt.show()



if done_tweaking:
    print('Code for plotting test set PR curves.')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('precision recall curve - test set weakened')

    for iou_thresh in [0.25,0.5,0.75]:
        tp_test = np.zeros(len(confidence_thrs))
        fp_test = np.zeros(len(confidence_thrs))
        fn_test = np.zeros(len(confidence_thrs))

        for i, conf_thr in enumerate(confidence_thrs):
            tp_test[i], fp_test[i], fn_test[i] = compute_counts(preds_test, gts_test, iou_thr=iou_thresh, conf_thr=conf_thr)

        # Plot training set PR curves
        precision_list = []
        recall_list = []

        for i in range(0,confidence_thrs.shape[0]):
            precision = tp_test[i]/(tp_test[i] + fp_test[i])
            recall = tp_test[i]/(tp_test[i] + fn_test[i])
            precision_list.append(precision)
            recall_list.append(recall)

        import ipdb;ipdb.set_trace()

        plt.plot(np.array(recall_list), np.array(precision_list), alpha=0.7, label='iou_thresh: '+str(iou_thresh))

    plt.legend()
    plt.show()

