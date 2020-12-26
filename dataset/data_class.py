import os
import h5py
import numpy as np
import glob
from torch.utils.data import Dataset
import json
from plyfile import PlyData, PlyElement

def load_data_s3dis(partition, point_num, data_dir='/mnt/dataset/s3dis/classification'):
    h5_name = os.path.join(data_dir, partition + '_' + str(point_num) + '.hdf5')
    all_data = []
    all_label = []
    assert os.path.isfile(h5_name), '{} does not exist.'.format(h5_name)
    f = h5py.File(h5_name, 'r')
    length = f['valid_count'][0]
    data = f['data'][:length].astype('float32')
    label = f['label'][:length].astype('int64')
    f.close()
    return data, label


class S3DIS(Dataset):
    def __init__(self, num_points=1024, partition='train', transforms = None):
        #partition can be train or test
        self.data, self.label = load_data_s3dis(partition, num_points)
        self.num_points = num_points
        self.partition = partition
        self.transforms = transforms
        #self.cat = get_catogrey()
        #self.classes = list(self.cat.keys())
     

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points,:3]
        label = self.label[item]
        label = np.expand_dims(label,axis=0)
        if self.partition == 'train':
            np.random.shuffle(pointcloud)
        if self.transforms is not None:
            pointcloud = self.transforms(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]




def modelnet40_download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    #BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, 'modelnet40_data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget  --no-check-certificate %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))

def download_shapenetpart():
    BASE_DIR = os.path.dirname(os.path.abspath("__file__"))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')):
        www = 'https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % ("hdf5_data", os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')))
        os.system('rm %s' % (zipfile))

def load_data_partseg(partition):
    download_shapenetpart()
    BASE_DIR = os.path.dirname(os.path.abspath("__file__"))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    all_seg = []
    if partition == 'trainval':
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet*hdf5*', '*train*.h5')) \
               + glob.glob(os.path.join(DATA_DIR, 'shapenet*hdf5*', '*val*.h5'))
    else:
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet*hdf5*', '*%s*.h5'%partition))
    for h5_name in file:
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        seg = f['pid'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg


class ShapeNetPart(Dataset):
    def __init__(self, num_points, partition='train', class_choice=None, transforms = None):
        super().__init__()
        #self.batch_size = batch_size
        #self.shuffle = shuffle
        self.transforms=transforms 
        self.data, self.label, self.seg = load_data_partseg(partition)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = partition        
        self.class_choice = class_choice

        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0

      

    def __len__(self):
        return self.data.shape[0]
        
    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        seg = self.seg[item][:self.num_points]

        if self.transforms is not None:
            pointcloud = self.transforms(pointcloud)
        if self.partition == 'trainval':
            # pointcloud = translate_pointcloud(pointcloud)
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        return pointcloud, label, seg



def load_data(partition):
    modelnet40_download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    #BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, 'modelnet40_data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train', transforms = None):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.transforms = transforms
        #self.cat = get_catogrey()
        #self.classes = list(self.cat.keys())
     

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            np.random.shuffle(pointcloud)
        if self.transforms is not None:
            pointcloud = self.transforms(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]



    