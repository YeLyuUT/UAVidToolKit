import os
import os.path as osp
import numpy as np
from colorTransformer import UAVidColorTransformer
from PIL import Image
import argparse
from tqdm import tqdm

clrEnc = UAVidColorTransformer()
def convertTrainID2ColorForDir(trainIdDir, saveDirPath, subdirname='color'):
    trainId_paths = [p for p in os.listdir(trainIdDir) if p.startswith('seq')]
    for pd in tqdm(trainId_paths):
        lbl_dir = osp.join(trainIdDir, pd, 'TrainId')
        lbl_paths = os.listdir(lbl_dir)
        if not osp.isdir(osp.join(saveDirPath, pd, subdirname)):
            os.makedirs(osp.join(saveDirPath, pd, subdirname))
            assert osp.isdir(osp.join(saveDirPath, pd, subdirname)), 'Fail to create directory:%s' % (osp.join(saveDirPath, pd, subdirname))
        for lbl_p in lbl_paths:
            lbl_path = osp.abspath(osp.join(lbl_dir, lbl_p))
            color_path = osp.join(saveDirPath, pd, subdirname, lbl_p)
            trainId = np.array(Image.open(lbl_path))
            colorImg = clrEnc.inverse_transform(trainId)
            Image.fromarray(colorImg).save(color_path)

def parseArgs():
  parser = argparse.ArgumentParser(description='Convert trainId files to color images.')
  parser.add_argument('-s', dest='source_dir', type=str, help='trainId home directory')
  parser.add_argument('-t', dest='target_dir', type=str, help='color home directory')
  parser.add_argument('-f', dest='subdirname', type=str, default='color', help='sub-directory name')
  args = parser.parse_args()
  return args

if __name__=='__main__':
    args = parseArgs()
    convertTrainID2ColorForDir(args.source_dir, args.target_dir, args.subdirname)