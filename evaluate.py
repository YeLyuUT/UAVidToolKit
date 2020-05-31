import os
import os.path as osp
import numpy as np
import argparse
import itertools
from tqdm import tqdm
from colorTransformer import UAVidColorTransformer
from PIL import Image
from matplotlib import pyplot as plt
# C Support
# Enable the cython support for faster evaluation
# Only tested for Ubuntu 64bit OS
CSUPPORT = True
# Check if C-Support is available for better performance
if CSUPPORT:
    try:
        import addToConfusionMatrix
    except:
        CSUPPORT = False
        from sklearn import metrics
        import addToConfusionMatrix
        import sys

clr_trans = UAVidColorTransformer()
LABELS, CLASS_NAMES = zip([[i, name] for i, name in enumerate(clr_trans.colorTable().keys())])

def getConfusionMatrixForImageList(classNum, predfileList, truefileList, evalLabels = LABELS):
  assert(len(predfileList)==len(truefileList))
  enc = clr_trans
  cm = np.zeros(shape=[classNum, classNum],dtype = np.uint64)
  print('CSUPPORT:', CSUPPORT)
  if CSUPPORT:
    print('This is a fast C++ way evaluation.')
  else: 
    print('This is a slow python way evaluation.')
  for idx in tqdm(range(len(predfileList))):
    predfile = predfileList[idx]
    truefile = truefileList[idx]
    imagePred = np.array(Image.open(predfile))
    imageTrue = np.array(Image.open(truefile))
    if len(imagePred.shape)==3 and len(imageTrue.shape)==3:
      imagePred = enc.transform(imagePred,dtype=np.uint8)
      imageTrue = enc.transform(imageTrue,dtype=np.uint8)
    assert len(imagePred.shape)==2 and len(imageTrue.shape)==2
    if CSUPPORT:
      cm = addToConfusionMatrix.cEvaluatePair(imagePred, imageTrue, cm, evalLabels)
    else: 
      #slower python way
      cm = calculateConfusionMatrix(cm, imagePred, imageTrue, evalLabels)
  return cm

def calculateConfusionMatrix(cm, imagePred, imageTrue, evalLabels):
  y_true = np.reshape(imageTrue, [-1,1])
  y_pred = np.reshape(imagePred, [-1,1])
  cm_t = metrics.confusion_matrix(y_true, y_pred, labels = evalLabels)
  if cm is None:
    return cm_t
  else:
    return cm+cm_t

# Calculate and return IOU score for a particular label
def getIouScoreForLabel(label, cm):
  tp = np.longlong(cm[label,label])
  fn = np.longlong(cm[label,:].sum()) - tp
  fp = np.longlong(cm[:,label].sum()) - tp
  # the denominator of the IOU score
  denom = (tp + fp + fn)
  if denom == 0:
      return float('nan')
  # return IOU
  return float(tp) / denom

def getMeanIOU(cm,evallabels=LABELS):
  IOUs = []
  for l in evallabels:
    IOUs.append(getIouScoreForLabel(l, cm))
  return np.mean(IOUs)

def getIOUforClasses(cm,evallabels=LABELS):
  IOUs = []
  for l in evallabels:
    IOUs.append(getIouScoreForLabel(l, cm))
  return IOUs

def getConfusionMatrixfromDirectory(gt_home, pred_home):
  gt_paths = [p for p in os.listdir(gt_home) if p.startswith('seq')]
  all_gt_path = []
  all_pred_path = []
  for pd in gt_paths:
    lbl_dir = osp.join(gt_home,pd,'Labels')
    lbl_paths = os.listdir(lbl_dir)
    for lbl_p in lbl_paths:
      lbl_path = osp.abspath(osp.join(lbl_dir, lbl_p))
      pred_path = osp.join(pred_home, pd, 'Labels', lbl_p)
      assert osp.exists(lbl_path)
      assert osp.exists(pred_path), 'Prediction incomplete, cannot find prediction: %s'%(pred_path)
      all_gt_path.append(lbl_path)
      all_pred_path.append(pred_path)
  class_num = len(CLASS_NAMES)
  cm = getConfusionMatrixForImageList(class_num, all_pred_path, all_gt_path)
  return cm

def getPixelAccuracy(cm):
  tp = 0.0
  cm = cm.astype(np.float64)
  for l in range(cm.shape[0]):
    tp += cm[l,l]
  return tp/cm.sum()

def normalize_confusion_matrix(cm):
  cm = cm.astype(np.float32)
  for i in range(cm.shape[0]):
    cm[i,:] = cm[i,:]/cm[i,:].sum()
  return cm

def visualizeMeanIOUforClasses(IOUs,labels,classNames,prefix = '',suffix = ''):
  assert(isinstance(IOUs, list))
  assert(isinstance(labels, list))
  assert(isinstance(classNames, list))
  assert(len(IOUs)==len(labels))
  assert(len(IOUs)==len(classNames))
  y_pos = np.arange(len(IOUs)+1)
  plt.figure(figsize=(14, 5))
  barlist = plt.bar(y_pos, IOUs+[np.mean(IOUs)], align='center')
  ax = plt.axes()
  color_tab = clr_trans.ColorTable()
  for l in labels:
    idx = labels.index(l)
    clr = color_tab[classNames[l]]
    clr_face = [clr[0]/255.0,clr[1]/255.0,clr[2]/255.0,1.0]
    clr_edge = [clr[0]/255.0,clr[1]/255.0,clr[2]/255.0,1.0]
    barlist[idx].set_color(clr_face)
    barlist[idx].set_edgecolor(clr_edge)
    height = barlist[idx].get_height()
    ax.text(barlist[idx].get_x()+barlist[idx].get_width()/2.,height+0.02,'%.3f'%(IOUs[idx]),ha='center', va='bottom')
  #MeanIOU setting
  height = barlist[-1].get_height()
  ax.text(barlist[-1].get_x()+barlist[-1].get_width()/2.,height+0.02,'%.3f'%(np.mean(IOUs)),ha='center', va='bottom')

  ax.yaxis.grid(color=(0.6,0.6,0.6),linestyle='--')
  plt.xticks(y_pos, [n.replace('_',' ') for n in classNames+['MeanIOU']])
  plt.tick_params(axis='x', which='major', labelsize=14)
  plt.ylabel('IOU Scores', fontsize=13)
  plt.ylim(0,1)
  plt.savefig(os.path.join('./',prefix+'IOU_scores'+suffix+'.png'))
  plt.show()

def visualizeConfusionMatrix(cm, class_names,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,prefix = '',suffix=''):
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')

  print(cm)
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  class_name = [n.replace('_',' ') for n in class_names]
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)
  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",fontsize=6,
               color="white" if cm[i, j] > thresh else "black")

  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.tight_layout()
  plt.savefig(os.path.join('./',prefix+title+suffix+'.png'))
  plt.show()

def parseArgs():
  parser = argparse.ArgumentParser(description='Evaluate Prediction')
  parser.add_argument('-gt', dest='gt_dir', help='gt directory')
  parser.add_argument('-p', dest='pred_dir', help='prediction directory')
  parser.add_argument('-v', dest='use_visualize', help='visualize evaluation result.', action='store_true')
  args = parser.parse_args()
  return args

def evaluateFromDirectories(args):
  gt_home = args.gt_dir
  pred_home = args.pred_dir
  use_visualize = args.use_visualize
  # Confusion matrix.
  cm = getConfusionMatrixfromDirectory(gt_home, pred_home)
  IOUs = getIOUforClasses(cm, evallabels=LABELS)
  mIOU = getMeanIOU(cm)
  acc = getPixelAccuracy(cm)
  print('IOUs:', IOUs)
  print('mIOU:', mIOU)
  print('acc:', acc)
  if use_visualize:
    visualizeMeanIOUforClasses(IOUs, labels=LABELS, classNames=CLASS_NAMES)
    visualizeConfusionMatrix(cm, CLASS_NAMES, title='Confusion Matrix', normalize=True)

if __name__=='__main__':
  args = parseArgs()
  evaluateFromDirectories(args)

