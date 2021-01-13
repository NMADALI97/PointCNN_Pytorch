import torch
import time
import math


base_model='s3dis_x8_2048_fps'


# Note: dataset to use is determined by base_model
dataset_available = ["ModelNet40","ModelNet10","Mnist","ScanNetSeg","ShapeNetParts","Cifar10","S3DIS"]
dataset_to_path = {
    "ModelNet40": {
        "train": "/home/jxq/repository/ModelNet40/train_files.txt",
        "test": "/home/jxq/repository/ModelNet40/test_files.txt"
    },
     "ModelNet10": {
        "train": "/home/jxq/repository/ModelNet40/train_files.txt",
        "test": "/home/jxq/repository/ModelNet40/test_files.txt"
    },
     "Mnist": {
        "train": "/home/jxq/repository/ModelNet40/train_files.txt",
        "test": "/home/jxq/repository/ModelNet40/test_files.txt"
    },
    "ScanNetSeg":{
        "train":2,
        "test":2
    },
    "ShapeNetParts":{
        "train": "/home/jxq/repository/ModelNet40/train_files.txt",
        "test": "/home/jxq/repository/ModelNet40/test_files.txt"
    },
    "Cifar10":{
        "train": "/home/jxq/repository/ModelNet40/train_files.txt",
        "test": "/home/jxq/repository/ModelNet40/test_files.txt"
    },
    "S3DIS":{
        "train": "/home/jxq/repository/ModelNet40/train_files.txt",
        "test": "/home/jxq/repository/ModelNet40/test_files.txt"
    }
}

base_model_available=['modelnet_x3_l4','modelnet_10_x3_l4',"mnist_x3_l4",'scannet_x8_2048_k8_fps','shapenet_x8_2048_fps','cifar10_x3_l4','s3dis_x8_2048_fps']

base_model_to_dataset={
    "modelnet_x3_l4":"ModelNet40",
    "modelnet_10_x3_l4":"ModelNet10",
    "mnist_x3_l4":"Mnist",
    "shapenet_x8_2048_fps":"ShapeNetParts",
    "cifar10_x3_l4":"Cifar10",
    "s3dis_x8_2048_fps":"S3DIS"
}

base_model_to_task={
    "modelnet_x3_l4": "cls",
    "modelnet_10_x3_l4": "cls",
    "mnist_x3_l4": "cls",
    "cifar10_x3_l4": "cls",
    "shapenet_x8_2048_fps": "seg",
    "s3dis_x8_2048_fps": "seg"
}

dataset=base_model_to_dataset[base_model]
task=base_model_to_task[base_model]

base_model_to_dataset_setting={
    "modelnet_x3_l4":{
        "sample_num":2048,
        "data_dim":3,
        "use_extra_features" : False,
        "with_X_transformation" : True,
        "with_normal_feature":True,
        "sorting_method" : None,
        "rotation_range" : [0,math.pi,0,'u'],
        "rotation_order":'rxyz',
        "scaling_range" : [0.1, 0.1, 0.1, 'g'],
        "jitter":0,
        "jitter_val": 0.0,
        "rotation_range_val": [0, 0, 0, 'u'],
        "scaling_range_val": [0, 0, 0, 'u']
    },
    "modelnet_10_x3_l4":{
        "sample_num":1024,
        "data_dim":3,
        "use_extra_features" : False,
        "with_X_transformation" : True,
        "with_normal_feature":False,
        "sorting_method" : None,
        "rotation_range" : [0,math.pi,0,'u'],
        "rotation_order":'rxyz',
        "scaling_range" : [0.1, 0.1, 0.1, 'g'],
        "jitter":0,
        "jitter_val": 0.0,
        "rotation_range_val": [0, 0, 0, 'u'],
        "scaling_range_val": [0, 0, 0, 'u']
    },
    "mnist_x3_l4":{
        "sample_num":2048,
        "data_dim":6,
        "use_extra_features" : True,
        "with_X_transformation" : True,
        "with_normal_feature":True,
        "sorting_method" : None,
        "rotation_range" : [0,math.pi,0,'u'],
        "rotation_order":'rxyz',
        "scaling_range" : [0.1, 0.1, 0.1, 'g'],
        "jitter":0,
        "jitter_val": 0.0,
        "rotation_range_val": [0, 0, 0, 'u'],
        "scaling_range_val": [0, 0, 0, 'u']
    },
    "cifar10_x3_l4":{
        "sample_num":512,
        "data_dim":6,
        "use_extra_features" : True,
        "with_X_transformation" : True,
        "with_normal_feature":False,
        "sorting_method" : None,
        "rotation_range" : [0, 0, [0, math.pi], 'g'],
        "rotation_order":'rzyx',
        "scaling_range" : [0.0, [0.01], 0.0, 'u'],
        "jitter":0,
        "jitter_val": 0.0,
        "rotation_range_val": [0, 0, 0, 'u'],
        "scaling_range_val": [0, [0.01], 0, 'u']
    },
    "scannet_x8_2048_fps":{
        "sample_num":2048,
        "data_dim":3,
        "with_X_transformation" : True,
        "sorting_method" : None,
        "rotation_range" : [math.pi / 72, math.pi, math.pi / 72, 'u'],
        "rotation_order":'rxyz',
        "scaling_range" : [0.05, 0.05, 0.05, 'g'],
        "jitter":0,
        "jitter_val": 0.0,
        "rotation_range_val": [0, 0, 0, 'u'],
        "scaling_range_val": [0, 0, 0, 'u']
    },
    "shapenet_x8_2048_fps":{
        "sample_num": 2048,
        "data_dim": 3,
        "with_X_transformation": True,
        "sorting_method": None,
        "rotation_range": [0, 0, 0, 'u'],
        "rotation_order": 'rxyz',
        "scaling_range": [0.0, 0.0, 0.0, 'g'],
        "jitter": 0.001,
        "jitter_val": 0.0,
        "rotation_range_val": [0, 0, 0, 'u'],
        "scaling_range_val": [0, 0, 0, 'u']
    },
     "s3dis_x8_2048_fps":{
        "sample_num": 2048,
        "data_dim": 9,
        "use_extra_features" : True,
        "with_normal_feature": True,
        "with_X_transformation": True,
        "sorting_method": None,
        "rotation_range": [0, 0, 0, 'u'],
        "rotation_order": 'rxyz',
        "scaling_range": [0.0, 0.0, 0.0, 'g'],
        "jitter": 0.001,
        "jitter_val": 0.0,
        "rotation_range_val": [0, 0, 0, 'u'],
        "scaling_range_val": [0, 0, 0, 'u']
    }
}

dataset_dir = dataset_to_path[dataset]
dataset_setting=base_model_to_dataset_setting[base_model]


# configuration file
project_name = "PointCNN"
description = ""


# device can be either "cuda" or "cpu"
num_workers = 4



available_gpus = str( torch.cuda.device_count())
if  torch.cuda.is_available():
    device = torch.device("cuda")
    use_gpu=True
else:
    device=torch.device("cpu")
    use_gpu=False
print_freq = 10
daemon_mode = False
backup_code = False


time_stamp = '{}'.format(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
short_time_stamp = "{}".format(time.strftime('%y%m%d_%H%M%S'))
dropout = True
dropmax = False
without_bn = True
first_kernel = 7
# pretrained_model = None
pretrained_model = ""
# TRAIN or VAL or VIS or TEST
process = "TRAIN"


class train:
    root_dir = dataset_to_path[dataset]
    if isinstance(root_dir, dict):
        root_dir = root_dir['train']
    resume =False
    resume_epoch = None

################## DO NOT CHANGE ############################


if base_model in base_model_available:
    instance_name = "{}{}".format(base_model, description)
else:
    instance_name = "{}{}_{}".format(base_model, description, first_kernel) + \
                    "{}".format('_dropout' if dropout else '')

comment = "{}_{}".format(instance_name, short_time_stamp)

if process != 'TRAIN':
    comment = pretrained_model
    daemon_mode = False
    backup_code = False

result_root = "rst-{}".format(project_name)
result_sub_folder = "{}/{}".format(result_root, comment)
ckpt_file = "{}/ckpt.pth".format(result_sub_folder)
resume_ckpt_file="rst-PointCNN/shapenet_x8_2048_fps_210111_101050/ckpt.pth"

#############################################################


class validation:
    root_dir = dataset_to_path[dataset]
    if isinstance(root_dir, dict):
        root_dir = root_dir['test']
    pretrained_model = ckpt_file


class test:
    root_dir = dataset_to_path[dataset]
    if isinstance(root_dir, dict):
        root_dir = root_dir['test']
    pretrained_model = ckpt_file

if base_model=="modelnet_x3_l4":
    train.batch_size=128
    train.num_epochs=10
    train.optimizer="ADAM"
    train.epsilon=1e-2
    train.learning_rate_base=0.01
    train.decay_steps=8000
    train.decay_rate=0.5
    train.learning_rate_min=1e-6
    train.weight_decay=1e-5
    validation.batch_size=128
    test.batch_size=128
    validation.step_val=5
elif base_model=="mnist_x3_l4":
    train.batch_size=128
    train.num_epochs=10
    train.optimizer="ADAM"
    train.epsilon=1e-2
    train.learning_rate_base=0.01
    train.decay_steps=8000
    train.decay_rate=0.5
    train.learning_rate_min=1e-6
    train.weight_decay=1e-5
    validation.batch_size=128
    test.batch_size=128
    validation.step_val=5
elif base_model=="modelnet_10_x3_l4":
    train.batch_size=32
    train.num_epochs=200
    train.optimizer="ADAM"
    train.epsilon=1e-2
    train.learning_rate_base=0.01
    train.decay_steps=8000
    train.decay_rate=0.5
    train.learning_rate_min=1e-6
    train.weight_decay=1e-5
    validation.batch_size=32
    test.batch_size=32
    validation.step_val=5
elif base_model=="cifar10_x3_l4":
    train.batch_size=200
    train.num_epochs=1024
    train.optimizer="ADAM"
    train.epsilon=1e-2
    train.learning_rate_base=0.01
    train.decay_steps=8000
    train.decay_rate=0.5
    train.learning_rate_min=1e-6
    train.weight_decay=1e-5
    validation.batch_size= 200
    test.batch_size= 200
    validation.step_val=500
elif base_model=="shapenet_x8_2048_fps":
    train.batch_size=16
    train.num_epochs=10
    train.optimizer="ADAM"
    train.epsilon=1e-3
    train.learning_rate_base=0.005
    train.decay_steps= 20000
    train.decay_rate= 0.9
    train.learning_rate_min=0.00001
    train.weight_decay=0.0
    validation.batch_size=32
    validation.step_val=5
    test.batch_size=32
elif base_model=="s3dis_x8_2048_fps":
    train.batch_size=16
    train.num_epochs=1024
    train.optimizer="ADAM"
    train.epsilon=1e-3
    train.learning_rate_base=0.005
    train.decay_steps= 5000
    train.decay_rate= 0.8
    train.learning_rate_min=1e-6
    train.weight_decay= 1e-8
    validation.batch_size=16
    validation.step_val=500
    test.batch_size=16
else:
    print("parameter not specified")
    raise NotImplementedError