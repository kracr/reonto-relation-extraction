from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
import pandas as pd
from matplotlib import pyplot
import numpy as np

def true_positive(y_true, y_pred):
    
    tp = 0
    
    for yt, yp in zip(y_true, y_pred):
        
        if yt == 1 and yp == 1:
            tp += 1
    
    return tp

def true_negative(y_true, y_pred):
    
    tn = 0
    
    for yt, yp in zip(y_true, y_pred):
        
        if yt == 0 and yp == 0:
            tn += 1
            
    return tn

def false_positive(y_true, y_pred):
    
    fp = 0
    
    for yt, yp in zip(y_true, y_pred):
        
        if yt == 0 and yp == 1:
            fp += 1
            
    return fp

def false_negative(y_true, y_pred):
    
    fn = 0
    
    for yt, yp in zip(y_true, y_pred):
        
        if yt == 1 and yp == 0:
            fn += 1
            
    return fn


def micro_precision(y_true, y_pred):


    # find the number of classes 
    num_classes = len(np.unique(y_true))
    
    # initialize tp and fp to 0
    tp = 0
    fp = 0
    
    # loop over all classes
    for class_ in y_true:
        
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        
        # calculate true positive for current class
        # and update overall tp
        tp += true_positive(temp_true, temp_pred)
        
        # calculate false positive for current class
        # and update overall tp
        fp += false_positive(temp_true, temp_pred)
        
    # calculate and return overall precision
    precision = tp / (tp + fp)
    return precision


def micro_recall(y_true, y_pred):


    # find the number of classes 
    num_classes = len(np.unique(y_true))
    
    # initialize tp and fp to 0
    tp = 0
    fn = 0
    
    # loop over all classes
    for class_ in y_true:
        
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        
        # calculate true positive for current class
        # and update overall tp
        tp += true_positive(temp_true, temp_pred)
        
        # calculate false negative for current class
        # and update overall tp
        fn += false_negative(temp_true, temp_pred)
        
    # calculate and return overall recall
    recall = tp / (tp + fn)
    return recall


def measure(y_actual, y_pred):
    class_id = set(y_actual).union(set(y_pred))
    TP = []
    FP = []
    TN = []
    FN = []
    precision=0
    recall=0

    for index ,_id in enumerate(class_id):
        TP.append(0)
        FP.append(0)
        TN.append(0)
        FN.append(0)
        for i in range(len(y_pred)):
            if y_actual[i] == y_pred[i] == _id:
                TP[index] += 1
            if y_pred[i] == _id and y_actual[i] != y_pred[i]:
                FP[index] += 1
            if y_actual[i] == y_pred[i] != _id:
                TN[index] += 1
            if y_pred[i] != _id and y_actual[i] != y_pred[i]:
                FN[index] += 1
            #precision[index]=TP[index]/(TP[index]+FP[index])
            #recall[index]=FP[index]/(TN[index]+FP[index])
        #print(TP,FP,TN,FN)
    return precision, recall

def evaluate_batch_based(predicted_batch, gold_batch, threshold = 1.0, idx2label=None, empty_label = None):
    if len(predicted_batch) != len(gold_batch):
        raise TypeError("predicted_idx and gold_idx should be of the same length.")

    correct = 0
    for i in range(len(gold_batch)):
        rec_batch = micro_avg_precision(predicted_batch[i], gold_batch[i], empty_label)
        if rec_batch >= threshold:
            correct += 1

    acc_batch = correct / float(len(gold_batch))

    return acc_batch


def evaluate_instance_based(predicted_idx, gold_idx, idx2label=None, empty_label = None):
    if len(predicted_idx) != len(gold_idx):
        raise TypeError("predicted_idx and gold_idx should be of the same length.")
    if idx2label:
        label_y = [idx2label[element] for element in gold_idx]
        pred_labels = [idx2label[element] for element in predicted_idx]
    else:
        label_y = gold_idx
        pred_labels = predicted_idx
    
    prec = micro_avg_precision(pred_labels, label_y, empty_label)
    rec = micro_avg_precision(label_y, pred_labels, empty_label)

    f1 = 0
    if (rec+prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)
    return prec, rec, f1    
    

def micro_avg_precision(guessed, correct, empty = None):
    """
    Tests:
    >>> micro_avg_precision(['A', 'A', 'B', 'C'],['A', 'C', 'C', 'C'])
    0.5
    >>> round(micro_avg_precision([0,0,0,1,1,1],[1,0,1,0,1,0]), 6)
    0.333333
    """
    correctCount = 0
    count = 0
    idx = 0
    while idx < len(guessed):
        if guessed[idx] != empty:
            count += 1
            if guessed[idx] == correct[idx]:
                correctCount +=1 
            #print("guessed",guessed)
            
    
        idx +=1
    precision = 0
    if count > 0:    
        precision = float(correctCount) / count
        
    return precision


if __name__ == "__main__":
    # Testing
    import doctest
    print(doctest.testmod())
