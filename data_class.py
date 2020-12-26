import os
import h5py
import numpy as np
import glob
from torch.utils.data import Dataset
import json
from plyfile import PlyData, PlyElement
import torch
import sys
from datetime import datetime
from dataset import pointfly as pf

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
            os.mkdir(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data'))
            files= ['https://shapenet.cs.stanford.edu/iccv17/partseg/train_data.zip','https://shapenet.cs.stanford.edu/iccv17/partseg/train_label.zip','https://shapenet.cs.stanford.edu/iccv17/partseg/val_data.zip','https://shapenet.cs.stanford.edu/iccv17/partseg/val_label.zip','https://shapenet.cs.stanford.edu/iccv17/partseg/test_data.zip','https://shapenet.cs.stanford.edu/iccv17/partseg/test_label.zip']
            for www in files:
                zipfile = os.path.basename(www)
                os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
                path=os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data',(www.split("/")[-1])[:-4])
                
                os.system('mv %s %s' % ((www.split("/")[-1])[:-4],path))
                os.system('rm %s' % (zipfile))
    
            root = os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')
            folders = [(os.path.join(root,'train_data'), os.path.join(root,'train_label')),
                        (os.path.join(root,'val_data'), os.path.join(root,'val_label')),
                        (os.path.join(root,'test_data'), os.path.join(root,'test_label'))]
            category_label_seg_max_dict = dict()
            max_point_num = 0
            label_seg_min = sys.maxsize
            for data_folder, label_folder in folders:
                    if not os.path.exists(data_folder):
                        continue
                    for category in sorted(os.listdir(data_folder)):
                        if category not in category_label_seg_max_dict:
                            category_label_seg_max_dict[category] = 0
                        data_category_folder = os.path.join(data_folder, category)
                        category_label_seg_max = 0
                        for filename in sorted(os.listdir(data_category_folder)):
                            data_filepath = os.path.join(data_category_folder, filename)
                            coordinates = [xyz for xyz in open(data_filepath, 'r') if len(xyz.split(' ')) == 3]
                            max_point_num = max(max_point_num, len(coordinates))

                            if label_folder is not None:
                                label_filepath = os.path.join(label_folder, category, filename[0:-3] + 'seg')
                                label_seg_this = np.loadtxt(label_filepath).astype(np.int32)
                                assert (len(coordinates) == len(label_seg_this))
                                category_label_seg_max = max(category_label_seg_max, max(label_seg_this))
                                label_seg_min = min(label_seg_min, min(label_seg_this))
                        category_label_seg_max_dict[category] = max(category_label_seg_max_dict[category], category_label_seg_max)
            category_label_seg_max_list = [(key, category_label_seg_max_dict[key]) for key in
                                            sorted(category_label_seg_max_dict.keys())]

            category_label = dict()
            offset = 0
            category_offset = dict()
            label_seg_max = max([category_label_seg_max for _, category_label_seg_max in category_label_seg_max_list])
            with open(os.path.join(root, 'categories.txt'), 'w') as file_categories:
                    for idx, (category, category_label_seg_max) in enumerate(category_label_seg_max_list):
                        file_categories.write('%s %d\n' % (category, category_label_seg_max - label_seg_min + 1))
                        category_label[category] = idx
                        category_offset[category] = offset
                        offset = offset + category_label_seg_max - label_seg_min + 1

            print('part_num:', offset)
            print('max_point_num:', max_point_num)
            print(category_label_seg_max_list)
            batch_size = 2048
            data = np.zeros((batch_size, max_point_num, 3))
            data_num = np.zeros((batch_size), dtype=np.int32)
            label = np.zeros((batch_size), dtype=np.int32)
            label_seg = np.zeros((batch_size, max_point_num), dtype=np.int32)
            for data_folder, label_folder in folders:
                    if not os.path.exists(data_folder):
                        continue
                    data_folder_ply = data_folder + '_ply'
                    file_num = 0
                    for category in sorted(os.listdir(data_folder)):
                        data_category_folder = os.path.join(data_folder, category)
                        file_num = file_num + len(os.listdir(data_category_folder))
                    idx_h5 = 0
                    idx = 0

                    save_path = '%s/%s' % (os.path.dirname(data_folder), os.path.basename(data_folder)[0:-5])
                    filename_txt = '%s_files.txt' % (save_path)
                    ply_filepath_list = []
                    with open(filename_txt, 'w') as filelist:
                        for category in sorted(os.listdir(data_folder)):
                            data_category_folder = os.path.join(data_folder, category)
                            for filename in sorted(os.listdir(data_category_folder)):
                                data_filepath = os.path.join(data_category_folder, filename)
                                coordinates = [[float(value) for value in xyz.split(' ')]
                                            for xyz in open(data_filepath, 'r') if len(xyz.split(' ')) == 3]
                                idx_in_batch = idx % batch_size
                                data[idx_in_batch, 0:len(coordinates), ...] = np.array(coordinates)
                                data_num[idx_in_batch] = len(coordinates)
                                label[idx_in_batch] = category_label[category]

                                if label_folder is not None:
                                    label_filepath = os.path.join(label_folder, category, filename[0:-3] + 'seg')
                                    label_seg_this = np.loadtxt(label_filepath).astype(np.int32) - label_seg_min
                                    assert (len(coordinates) == label_seg_this.shape[0])
                                    label_seg[idx_in_batch, 0:len(coordinates)] = label_seg_this + category_offset[category]

                                data_ply_filepath = os.path.join(data_folder_ply, category, filename[:-3] + 'ply')
                                ply_filepath_list.append(data_ply_filepath)

                                if ((idx + 1) % batch_size == 0) or idx == file_num - 1:
                                    item_num = idx_in_batch + 1
                                    filename_h5 = '%s_%d.h5' % (save_path, idx_h5)
                                    print('{}-Saving {}...'.format(datetime.now(), filename_h5))
                                    filelist.write('./%s_%d.h5\n' % (os.path.basename(data_folder)[0:-5], idx_h5))

                                    file = h5py.File(filename_h5, 'w')
                                    file.create_dataset('data', data=data[0:item_num, ...])
                                    file.create_dataset('data_num', data=data_num[0:item_num, ...])
                                    file.create_dataset('label', data=label[0:item_num, ...])
                                    file.create_dataset('label_seg', data=label_seg[0:item_num, ...])
                                    file.close()

                                    ply_filepath_list = []
                                    idx_h5 = idx_h5 + 1
                                idx = idx + 1

            train_val_txt = os.path.join(root, "train_val_files.txt")
            with open(train_val_txt, "w") as train_val:
                    for part in ("train", "val"):
                        part_txt = os.path.join(root, "%s_files.txt" % part)
                        train_val.write(open(part_txt, "r").read())

def load_data_partseg(partition):
    download_shapenetpart()
    BASE_DIR = os.path.dirname(os.path.abspath("__file__"))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    points = []
    labels = []
    point_nums = []
    labels_seg = []
    indices_split_to_full = []
    if partition == 'trainval':
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet*hdf5*', '*train*.h5')) \
               + glob.glob(os.path.join(DATA_DIR, 'shapenet*hdf5*', '*val*.h5'))
    else:
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet*hdf5*', '*%s*.h5'%partition))
    for h5_name in file:
        data = h5py.File(h5_name, 'r+')
        points.append(data['data'][...].astype(np.float32))
        labels.append(data['label'][...].astype(np.int64))
        point_nums.append(data['data_num'][...].astype(np.int32))
        labels_seg.append(data['label_seg'][...].astype(np.int64))
        if 'indices_split_to_full' in data:
            indices_split_to_full.append(data['indices_split_to_full'][...].astype(np.int64))
    
    return (np.concatenate(points, axis=0),
            np.concatenate(labels, axis=0),
            np.concatenate(point_nums, axis=0),
            np.concatenate(labels_seg, axis=0),
            np.concatenate(indices_split_to_full, axis=0) if indices_split_to_full else None)



class ShapeNetPart(Dataset):
    def __init__(self, partition='train'):
        super().__init__()
        #self.batch_size = batch_size
        #self.shuffle = shuffle
       
        self.points, self.labels, self.point_nums, self.labels_seg, _  = load_data_partseg(partition)
        self.points=torch.Tensor(pf.global_norm(self.points))
      

        print(self.points.shape)
        print(self.labels_seg.shape)

      

    def __getitem__(self, index):
        return self.points[index], self.labels_seg[index], self.point_nums[index],self.labels[index]

    def __len__(self):
        return self.points.shape[0]

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
    def __init__(self, partition='train'):
        self.data, self.label = load_data(partition)
        #self.num_points = num_points
        self.partition = partition
       # self.transforms = transforms
        #self.cat = get_catogrey()
        #self.classes = list(self.cat.keys())
     

    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
    def get_categories(self):
      return [ "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car", "chair", "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot", "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor", "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink", "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox", ]

    