




import gzip
import html
import shutil
import tarfile
import zipfile
import requests
import argparse
from tqdm import tqdm
import random
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

from sklearn.model_selection import train_test_split




def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    return batch



# from https://gist.github.com/hrouault/1358474
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.
    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).
    The "answer" return value is one of "yes" or "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default == None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def download_from_url(url, dst):
    download = True
    if os.path.exists(dst):
        download = query_yes_no('Seems you have downloaded %s to %s, overwrite?' % (url, dst), default='no')
        if download:
            os.remove(dst)

    if download:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        chunk_size = 1024 * 1024
        bars = total_size // chunk_size
        with open(dst, "wb") as handle:
            for data in tqdm(response.iter_content(chunk_size=chunk_size), total=bars, desc=url.split('/')[-1],
                             unit='M'):
                handle.write(data)


def download_and_unzip(url, root, dataset):
    folder = os.path.join(root, dataset)
    folder_zips = os.path.join(folder, 'zips')
    if not os.path.exists(folder_zips):
        os.makedirs(folder_zips)
    filename_zip = os.path.join(folder_zips, url.split('/')[-1])

    download_from_url(url, filename_zip)

    if filename_zip.endswith('.zip'):
        zip_ref = zipfile.ZipFile(filename_zip, 'r')
        zip_ref.extractall(folder)
        zip_ref.close()
    elif filename_zip.endswith(('.tar.gz', '.tgz')):
        tarfile.open(name=filename_zip, mode="r:gz").extractall(folder)
    elif filename_zip.endswith('.gz'):
        filename_no_gz = filename_zip[:-3]
        with gzip.open(filename_zip, 'rb') as f_in, open(filename_no_gz, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def cifar10_download():
    BASE_DIR = os.path.dirname(os.path.abspath("__file__"))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'cifar10')):
        download_and_unzip('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', "data", "cifar10")
        batch_size = 2048

        folder_cifar10 = 'data/cifar10/cifar-10-batches-py'
        folder_pts = os.path.join(os.path.dirname(folder_cifar10), 'pts')

        train_test_files = [('train', ['data_batch_%d' % (idx + 1) for idx in range(5)]),
                                ('test', ['test_batch'])]

        data = np.zeros((batch_size, 1024, 6))
        label = np.zeros((batch_size), dtype=np.int32)
        for tag, filelist in train_test_files: 
                data_list = []
                labels_list = []
                for filename in filelist:
                    batch = unpickle(os.path.join(folder_cifar10, filename))
                    data_list.append(np.reshape(batch[b'data'], (10000, 3, 32, 32)))
                    labels_list.append(batch[b'labels'])
                images = np.concatenate(data_list, axis=0)
                labels = np.concatenate(labels_list, axis=0)

                idx_h5 = 0
                filename_filelist_h5 = os.path.join(os.path.dirname(folder_cifar10), '%s_files.txt' % tag)
                with open(filename_filelist_h5, 'w') as filelist_h5:
                    for idx_img, image in enumerate(images):
                        points = []
                        pixels = []
                        for x in range(32):
                            for z in range(32):
                                points.append((x, random.random() * 1e-6, z))
                                pixels.append((image[0, x, z], image[1, x, z], image[2, x, z]))
                        points_array = np.array(points)
                        pixels_array = (np.array(pixels).astype(np.float32) / 255)-0.5

                        points_min = np.amin(points_array, axis=0)
                        points_max = np.amax(points_array, axis=0)
                        points_center = (points_min + points_max) / 2
                        scale = np.amax(points_max - points_min) / 2
                        points_array = (points_array - points_center) * (0.8 / scale)

                        

                        idx_in_batch = idx_img % batch_size
                        data[idx_in_batch, ...] = np.concatenate((points_array, pixels_array), axis=-1)
                        label[idx_in_batch] = labels[idx_img]
                        if ((idx_img + 1) % batch_size == 0) or idx_img == len(images) - 1:
                            item_num = idx_in_batch + 1
                            filename_h5 = os.path.join(os.path.dirname(folder_cifar10), '%s_%d.h5' % (tag, idx_h5))
                            print('{}-Saving {}...'.format(datetime.now(), filename_h5))
                            filelist_h5.write('./%s_%d.h5\n' % (tag, idx_h5))

                            file = h5py.File(filename_h5, 'w')
                            file.create_dataset('data', data=data[0:item_num, ...])
                            file.create_dataset('label', data=label[0:item_num, ...])
                            file.close()

                            idx_h5 = idx_h5 + 1


def cifar10_load_data(partition):   
    cifar10_download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    points = []
    labels = []
    if partition == 'trainval':
        file = glob.glob(os.path.join(DATA_DIR, 'cifar10', '*train*.h5')) \
               + glob.glob(os.path.join(DATA_DIR, 'cifar10', '*val*.h5'))
    else:
        file = glob.glob(os.path.join(DATA_DIR, 'cifar10', '*%s*.h5'%partition))
    for h5_name in file:
        data = h5py.File(h5_name, 'r+')
        points.append(data['data'][...].astype(np.float32))
        labels.append(data['label'][...].astype(np.int64))
    return (np.concatenate(points, axis=0),np.concatenate(labels, axis=0))
            
def s3dis_download():       
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 's3dis')):
        www = 'https://drive.google.com/uc?id=1hpX5D7wnMrYOYs2HPbZwK6Wda1LMPX2Z'
        zipfile = os.path.basename(www)
        os.system('gdown  %s; unzip data.zip -d data' % (www))
        os.system('rm data.zip')

        root="data/s3dis"
        all_files=[]
        for area_idx in range(1, 3):
                folder = os.path.join(root, 'Area_%d' % area_idx)
                for dataset in os.listdir(folder) :
                   all_files.append(os.path.join(folder,dataset,"zero_0.h5"))
        

        X_train, X_test = train_test_split(all_files, test_size=0.3) 
        with open("data/s3dis/train_file.txt", 'w') as filelist:
            for path in X_train:
                filelist.write(path+"\n")


        with open("data/s3dis/valid_file.txt", 'w') as filelist:
            for path in X_test:
                filelist.write(path+"\n")
                          

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


def load_data_s3dis(partition):
    s3dis_download()
    BASE_DIR = os.path.dirname(os.path.abspath("__file__"))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    points = []
    labels = []
    point_nums = []
    labels_seg = []
    indices_split_to_full = []
    if partition == 'train':
        filelist="data/s3dis/train_file.txt"
    else:
        filelist="data/s3dis/valid_file.txt"
    file=[line.strip() for line in open(filelist)]
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
    def __init__(self, partition='trainval'):
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
class S3DIS(Dataset):
    def __init__(self, partition='train'):
        super().__init__()
        #self.batch_size = batch_size
        #self.shuffle = shuffle
       
        self.points, self.labels, self.point_nums, self.labels_seg, _  = load_data_s3dis(partition)
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

class Cifar10(Dataset):
    def __init__(self, partition='train'):
        self.data, self.label = cifar10_load_data(partition)
        self.partition = partition
    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
    
