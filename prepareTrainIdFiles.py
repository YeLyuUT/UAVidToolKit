import os
import os.path as osp
import numpy as np
from colorTransformer import UAVidColorTransformer
from PIL import Image
import argparse
from tqdm import tqdm

clrEnc = UAVidColorTransformer()
def prepareTrainIDForDir(gtDirPath, saveDirPath):
    gt_paths = [p for p in os.listdir(gtDirPath) if p.startswith('seq')]
    for pd in tqdm(gt_paths):
        lbl_dir = osp.join(gtDirPath, pd, 'Labels')
        lbl_paths = os.listdir(lbl_dir)
        if not osp.isdir(osp.join(saveDirPath, pd, 'TrainId')):
            os.makedirs(osp.join(saveDirPath, pd, 'TrainId'))
            assert osp.isdir(osp.join(saveDirPath, pd, 'TrainId')), 'Fail to create directory:%s'%(osp.join(saveDirPath, pd, 'TrainId'))
        for lbl_p in lbl_paths:
            lbl_path = osp.abspath(osp.join(lbl_dir, lbl_p))
            trainId_path = osp.join(saveDirPath, pd, 'TrainId', lbl_p)
            gt = np.array(Image.open(lbl_path))
            trainId = clrEnc.transform(gt, dtype=np.uint8)
            Image.fromarray(trainId).save(trainId_path)

def parseArgs():
  parser = argparse.ArgumentParser(description='Prepare trainId files for the ground truth.')
  parser.add_argument('-s', dest='source_dir', help='label home directory')
  parser.add_argument('-t', dest='target_dir', help='trainId home directory')
  args = parser.parse_args()
  return args

if __name__=='__main__':
    args = parseArgs()
    prepareTrainIDForDir(args.source_dir, args.target_dir)