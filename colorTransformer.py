import numpy as np
class UAVidColorTransformer:
  def __init__(self):
    # color table.
    clr_tab = {}
    clr_tab['Cluster'] = [0,0,0]
    clr_tab['Building'] = [128,0,0]
    clr_tab['Road'] = [128,64,128]
    clr_tab['Tree'] = [0,128,0]
    clr_tab['Vegetation'] = [128,128,0]
    clr_tab['Static_Car'] = [192, 0, 192]
    clr_tab['Moving_Car'] = [64,0,128]
    clr_tab['Human'] = [64, 64, 0]
    self.clr_tab = clr_tab
    # id table.
    id_tab = {}
    for k, v in clr_tab.items():
        id_tab[k] = self.clr2id(v)
    self.id_tab = id_tab

  def colorTable(self):
    return self.clr_tab
   
  def clr2id(self, clr):
    return clr[0]+clr[1]*255+clr[2]*255*255

  #transform to uint8 integer label
  def transform(self,label, dtype=np.int32):
    height,width = label.shape[:2]
    # default value is index of clutter.
    newLabel = np.zeros((height, width), dtype=dtype)
    id_label = label.astype(np.int64)
    id_label = id_label[:,:,0]+id_label[:,:,1]*255+id_label[:,:,2]*255*255
    for tid,val in enumerate(self.id_tab.values()):
      mask = (id_label == val)
      newLabel[mask] = tid
    return newLabel

  #transform back to 3 channels uint8 label
  def inverse_transform(self, label):
    label_img = np.zeros(shape=(label.shape[0], label.shape[1],3),dtype=np.uint8)
    values = list(self.clr_tab.values())
    for tid,val in enumerate(values):
      mask = (label==tid)
      label_img[mask] = val
    return label_img
