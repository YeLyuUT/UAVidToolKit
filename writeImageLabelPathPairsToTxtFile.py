import sys
import os
import os.path as osp

def writeTrainValImageLabelPathPairsToTxtFile(data_home='../', useTrain=True, useVal=False):
  assert useTrain or useVal,'Error: None of the training set or the validation set is used.'
  train_home = osp.join(data_home,'train')
  train_paths = os.listdir(train_home)
  
  val_home = osp.join(data_home,'valid')
  val_paths = os.listdir(val_home)
  
  all_img_path = []
  all_lbl_path = []
  
  if useTrain:
    for pd in train_paths:
        img_dir = osp.join(train_home,pd,'Images')
        img_paths = os.listdir(img_dir)
        for img_p in img_paths:
          img_path = osp.abspath(osp.join(img_dir, img_p))
          label_path = osp.abspath(img_path.replace('Images','Labels'))
          assert osp.exists(img_path)
          assert osp.exists(label_path)
          all_img_path.append(img_path)
          all_lbl_path.append(label_path)
  
  if useVal:
    for pd in val_paths:
      img_dir = osp.join(val_home,pd,'Images')
      img_paths = os.listdir(img_dir)
      for img_p in img_paths:
        img_path = osp.abspath(osp.join(img_dir, img_p))
        label_path = osp.abspath(img_path.replace('Images','Labels'))
        assert osp.exists(img_path)
        assert osp.exists(label_path)
        all_img_path.append(img_path)
        all_lbl_path.append(label_path)
      
  assert len(all_img_path)==len(all_lbl_path),'Image number and label number are not equal.'
  print('Number of image label pairs are:', len(all_img_path))
  with open('./img_lbl_pair.txt','w') as f:
    for i,l in zip(all_img_path,all_lbl_path):
      f.write(i+' '+l+'\n')

def writeTestPredImageLabelPathPairsToTxtFile(data_home='../', useTest=True, useVal=False, target_dir=None):
    assert useTest or useVal,'Error: None of the test set or the validation set is used.'
    if useTest:
      test_home = osp.join(data_home,'test')
      test_paths = os.listdir(test_home)  
      all_img_path = []
      all_pred_path = []
      for pd in test_paths:
        img_dir = osp.join(test_home,pd,'Images')
        img_paths = os.listdir(img_dir)
        pred_dir = osp.join(target_dir,pd,'Labels')
        for img_p in img_paths:
          img_path = osp.abspath(osp.join(img_dir, img_p))
          pred_path = osp.abspath(osp.join(pred_dir, img_p))
          assert osp.exists(img_path)
          all_img_path.append(img_path)
          all_pred_path.append(pred_path)
      assert len(all_img_path)==len(all_pred_path),'Image number and label number are not equal.'
      print('Number of image label pairs are:', len(all_img_path))
      with open('./test_pred_pair.txt','w') as f:
        for i,l in zip(all_img_path,all_pred_path):
          f.write(i+' '+l+'\n')
    if useVal:
      test_home = osp.join(data_home,'valid')
      test_paths = os.listdir(test_home)  
      all_img_path = []
      all_pred_path = []
      for pd in test_paths:
        img_dir = osp.join(test_home,pd,'Images')
        img_paths = os.listdir(img_dir)
        pred_dir = osp.join(target_dir,pd,'Labels')
        for img_p in img_paths:
          img_path = osp.abspath(osp.join(img_dir, img_p))
          pred_path = osp.abspath(osp.join(pred_dir, img_p))
          assert osp.exists(img_path)
          all_img_path.append(img_path)
          all_pred_path.append(pred_path)
      assert len(all_img_path)==len(all_pred_path),'Image number and label number are not equal.'
      print('Number of image label pairs are:', len(all_img_path))
      with open('./valid_pred_pair.txt','w') as f:
        for i,l in zip(all_img_path,all_pred_path):
          f.write(i+' '+l+'\n')

def parseArgs():
  parser = argparse.ArgumentParser(description='Write image label path pairs of train and valid sets to txt file.')
  parser.add_argument('-h', dest='data_home', type=str, default='../', help='dataset home directory.')
  parser.add_argument('-t', dest='useTrain', type=str, help='use train set directory.', action='store_true')
  parser.add_argument('-v', dest='useValid', type=str, help='use valid set directory.', action='store_true')
  args = parser.parse_args()
  return args

if __name__=='__main__':
  args = parseArgs()
  writeTrainValImageLabelPathPairsToTxtFile(args.data_home, args.useTrain, args.useValid)