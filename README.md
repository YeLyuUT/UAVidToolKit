# UAVidToolKit
UAVidToolKit provides basic tools for easier usage of the UAVid dataset.
Including label conversion, label visualization, performance evaluation and image label path pair txtfile preparation.


## Install
Download the toolkit into the dataset folder as follows,
```
cd <UAVid dataset folder>
git clone https://github.com/YeLyuUT/UAVidToolKit.git
cd UAVidToolKit
python setup.py build_ext --inplace
cd ..
```

Rename training, validation and testing subfolders into 'train', 'valid' and 'test'. Or create symlink with cmd,
```
ln -s <train dir> train
ln -s <valid dir> valid
ln -s <test dir> test
```

The data structure should be like:

    \UAVidDataset
        \train
            \seq
            ...
        \valid
            \seq
            ...
        \test
            \seq
            ...
        \UAVidToolKit

## Usage

In the UAVid dataset folder, apply commands as follows:
- ##### Label image conversion from 3 channel RGB color image to 1 channel label index image.
```
python UAVidToolKit/prepareTrainIdFiles.py -s <src folder> -t <dst folder>
```
e.g. python UAVidToolKit/prepareTrainIdFiles.py -s valid/ -t tooltest/
<br/>

- ##### Label image conversion from 1 channel label index image to 3 channel RGB color image.
```
python UAVidToolKit/convertTrainIdFiles2Color.py -s <src folder> -t <dst folder> -f <sub folder name>
```
e.g. python UAVidToolKit/convertTrainIdFiles2Color.py -s tooltest/ -t tooltest/ -f 'color'
<br/>

- ##### Blend image and label files.
```
python UAVidToolKit/blendImageAndLabel.py -i <image folder> -l <label folder> -o <output folder> -id <image subfolder name> -ld <label subfolder name> -od <output subfolder name>
```
e.g. python UAVidToolKit/blendImageAndLabel.py -i valid/ -l tooltest/ -o tooltest/ -id Images -ld color -od blend
<br/>

- ##### Performance evaluation.
```
python UAVidToolKit/evaluate.py -gt <ground truth folder> -p <prediction folder> -v
```
If add '-v', visualize mIoU and confusion matrix results with figures.

e.g. python UAVidToolKit/evaluate.py -gt test -p pred_eg -v
<br/>

- ##### Write image label paths pair into txt.
```
python UAVidToolKit/writeImageLabelPathPairsToTxtFile.py -d <dataset folder> -t -v
```
If add '-t', add training set to txt.

If add '-v', add valid set to txt.

e.g. python UAVidToolKit/writeImageLabelPathPairsToTxtFile.py -d ./ -t -v

