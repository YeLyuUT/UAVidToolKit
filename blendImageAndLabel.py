import os
import os.path as osp
import numpy as np
import argparse
import cv2
from PIL import Image
from colorTransformer import UAVidColorTransformer
from tqdm import tqdm

clrEnc = UAVidColorTransformer()
def blendImageAndLabelForDir(imgDirHome,
                             labelDirHome,
                             blendDirHome,
                             imageSubDirname='Images',
                             labelSubDirname='Labels',
                             blendSubDirname='Blend',
                             alpha=0.6,
                             beta=0.4,
                             gamma=0.0):
    lbl_seq_paths = [p for p in os.listdir(labelDirHome) if p.startswith('seq')]
    for pd in tqdm(lbl_seq_paths):
        lbl_dir = osp.join(labelDirHome, pd, labelSubDirname)
        lbl_paths = os.listdir(lbl_dir)
        for lbl_p in lbl_paths:
            lbl_path = osp.abspath(osp.join(lbl_dir, lbl_p))
            img_path = osp.join(imgDirHome, pd, imageSubDirname, lbl_p)
            bld_path = osp.join(blendDirHome, pd, blendSubDirname, lbl_p)
            lbl = np.array(Image.open(lbl_path))
            img = np.array(Image.open(img_path))
            if lbl.ndim==2:
                lbl = clrEnc.inverse_transform(lbl)
            blendImg = cv2.addWeighted(img, alpha, lbl, beta, gamma)
            Image.fromarray(blendImg).save(bld_path)

def parseArgs():
  parser = argparse.ArgumentParser(description='Prepare trainId files for the ground truth.')
  parser.add_argument('-i', dest='image_dir', type=str, help='image home directory')
  parser.add_argument('-l', dest='label_dir', type=str, help='label home directory')
  parser.add_argument('-o', dest='output_dir', type=str, help='output home directory')
  parser.add_argument('-id', dest='image_subdir', type=str, default='Images', help='image subdirectory')
  parser.add_argument('-ld', dest='label_subdir', type=str, default='Labels', help='label subdirectory')
  parser.add_argument('-od', dest='output_subdir', type=str, default='Blend', help='output subdirectory')
  parser.add_argument('-alpha', type=float, default=0.6, help='image blend ratio')
  parser.add_argument('-beta', type=float, default=0.4, help='label blend ratio')
  parser.add_argument('-gamma', type=float, default=0.0, help='output blend addition')
  args = parser.parse_args()
  return args

if __name__=='__main__':
    args = parseArgs()
    blendImageAndLabelForDir(args.image_dir, args.label_dir, args.output_dir,
                             args.image_subdir, args.label_subdir, args.output_subdir,
                             args.alpha, args.beta, args.gamma)