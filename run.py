import time

#import dataset as ds

from data_class import ModelNet40, ShapeNetPart,Cifar10

import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorboardX
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from matplotlib.ticker import MultipleLocator
from torch.nn import DataParallel
from torch.utils.data.dataloader import DataLoader
import dataset.pointfly as pf
import sklearn.metrics as metrics

import config
import math
import util.meter as meter
import model.PointCNN as PointCNN
from util.recorder import ModelRecorder
from util.vis import VanillaBackprop
from util.vis.gp import GuidedBackprop
from util.vis.misc_functions import (convert_to_grayscale,
                                     save_gradient_images,
                                     get_positive_negative_saliency, preprocess_image)


global_step=0

def train_process():
    global global_step
    summary_writer = tensorboardX.SummaryWriter(log_dir=config.result_sub_folder, comment=config.comment)
    # prepare data
    if config.dataset=="ModelNet40":
        train_set = ModelNet40(partition='train')
        valid_set = ModelNet40(partition='test')
    elif config.dataset=="ShapeNetParts":
        train_set=ShapeNetPart(partition='train')
        valid_set =ShapeNetPart(partition='test')
    elif config.dataset=="Cifar10":
        train_set = Cifar10(partition='train')
        valid_set = Cifar10(partition='test')    

    else:
        raise NotImplementedError
    train_loader = DataLoader(train_set, batch_size=config.train.batch_size, shuffle=True,
                              num_workers=config.num_workers,
                              drop_last=True)

    valid_loader = DataLoader(valid_set, batch_size=config.validation.batch_size, shuffle=False,
                              num_workers=config.num_workers,  drop_last=False)
    print('train set size: {}'.format(len(train_set)))
    print('valid set size: {}'.format(len(valid_set)))

    # prepare model
    net = create_model(config.base_model).to(config.device)

    # prepare optimizer
    if config.train.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), config.train.learning_rate_base, momentum=config.train.momentum)
    elif config.train.optimizer == 'ADAM':
        optimizer = optim.Adam(net.parameters(), lr=config.train.learning_rate_base,eps=config.train.epsilon,weight_decay=config.train.weight_decay)
    else:
        raise NotImplementedError


    net = DataParallel(net)

    model_recorder = ModelRecorder(config.ckpt_file, optimizer, summary_writer=summary_writer)
    start_epoch = 0
    if config.train.resume:
        start_epoch = model_recorder.resume(net.module, optimizer, from_measurement='acc')
        if config.train.resume_epoch is not None:
            start_epoch = config.train.resume_epoch
            print("Force resume at {}".format(start_epoch))
        else:
            print("Resume at {}".format(start_epoch))

    # prepare the criterion
    criterion = nn.CrossEntropyLoss()

    # start to train
    for epoch in range(start_epoch, config.train.num_epochs):
        if config.task == "seg":
           training_loss,training_acc,avg_per_class_acc, train_ious=train_epoch(train_loader, net, criterion, optimizer, epoch)
           summary_writer.add_scalar('Training Loss', training_loss, global_step=epoch)
           summary_writer.add_scalar('Training Accuracy', training_acc, global_step=epoch)
           summary_writer.add_scalar('Training Average Precision ', avg_per_class_acc, global_step=epoch)
           summary_writer.add_scalar('Training IOUs ',  train_ious, global_step=epoch)
        else :
            training_loss,training_acc=train_epoch(train_loader, net, criterion, optimizer, epoch)
            summary_writer.add_scalar('Training Accuracy', training_acc, global_step=epoch)
            summary_writer.add_scalar('Training Loss', training_loss, global_step=epoch)

        if (epoch%config.validation.step_val == 0) or (epoch==config.train.num_epochs-1):
            with torch.no_grad():
                if config.task == "seg":
                  validation_loss,validation_acc,avg_per_class_acc, val_ious = evaluate(valid_loader, net)
                  summary_writer.add_scalar('Validation Loss', validation_loss, global_step=epoch)
                  summary_writer.add_scalar('Validation Accuracy', validation_acc, global_step=epoch)
                  summary_writer.add_scalar('Validation Average Precision ', avg_per_class_acc, global_step=epoch)
                  summary_writer.add_scalar('Validation IOUs ',  val_ious, global_step=epoch)
                else :
                    validation_loss,acc = evaluate(valid_loader, net)
                    summary_writer.add_scalar('Validation Accuracy', acc, global_step=epoch)
                    summary_writer.add_scalar('Validation Loss', validation_loss, global_step=epoch)
            if config.task == "seg":      
                model_recorder.add(epoch, net, dict(acc=val_ious))
            else:
                model_recorder.add(epoch, net, dict(acc=acc))
            model_recorder.print_curr_stat()

    print('\nTrain Finished: {}'.format(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))


def evaluation_process(detail=False):

    if config.dataset=="ModelNet40":
        valid_set = ModelNet40(partition='test')
    elif config.dataset=="ShapeNetParts":
        valid_set =ShapeNetPart(partition='test')
    

    valid_loader = DataLoader(valid_set, batch_size=config.validation.batch_size, shuffle=False,
                              num_workers=config.num_workers,  drop_last=False)
    print('valid set size: {}'.format(len(valid_set)))

    # prepare model
    print("evaluation on : {}".format(config.base_model))
    net = create_model(config.base_model)
    print("load pretained model from {}".format(config.test.pretrained_model))
    ModelRecorder.resume_model(net, config.test.pretrained_model, from_measurement="acc")

    net = nn.DataParallel(net)
    if not detail:
        with torch.no_grad():
            if config.task == "seg":
              validation_loss,validation_acc,avg_per_class_acc, val_ious = evaluate(valid_loader, net)
            else :   
              validation_loss,acc, conf_matrix = evaluate(valid_loader, net, True)
        if not config.task == "seg":
          plot_conf_matrix(valid_set.get_categories(), conf_matrix,
                         save_file='{}/{}.pdf'.format(config.result_sub_folder, config.comment))
    else:
        with torch.no_grad():
            acc, conf_matrix, features, labels = evaluate(valid_loader, net, True, True)
        plot_conf_matrix(valid_set.get_categories(), conf_matrix,
                         save_file='{}/{}.pdf'.format(config.result_sub_folder, config.comment))
        array2tsv(features, '{}/{}.tsv'.format(config.result_sub_folder, config.comment))
        labels_file = open('{}/{}_label.tsv'.format(config.result_sub_folder, config.comment), 'w')
        labels_txt = ""
        for label in labels:
            labels_txt += '{}\n'.format(int(label))
        labels_file.write(labels_txt[:-1])
        labels_file.close()
    print("Finished!")


def extract_test_rst_process():
    test_set = ds.ClipArtTest(config.test.data_set)

    test_loader = DataLoader(test_set, batch_size=config.test.batch_size, shuffle=False,
                             num_workers=config.num_workers,  drop_last=False)
    print('test set size: {}'.format(len(test_set)))

    # prepare model
    print("evaluation on : {}".format(config.base_model))
    net = create_model(config.base_model)
    print("load pretained model from {}".format(config.test.pretrained_model))
    ModelRecorder.resume_model(net, config.test.pretrained_model, from_measurement="acc")
    net = nn.DataParallel(net)

    net.eval()
    rst = "id,label\n"


    with torch.no_grad():
        for i, (batch_data, img_names) in enumerate(test_loader):
            batch_data=batch_data.to(config.device)
            batch_label=batch_label.to(config.device)
            raw_out = net(batch_data)
            pred = torch.argmax(raw_out.detach(), dim=1)
            pred = list(pred.cpu().numpy())
            assert len(pred) == len(img_names)
            for j, p in enumerate(pred):
                rst += '{}, {}\n'.format(img_names[j], ds.CONVERT_TABLE[p])
    save_file_name = 's_test_rst.txt'
    save_file = open(save_file_name, 'w')
    save_file.write(rst)
    save_file.close()
    print("Finished!")


def visualization_process():
    # prepare data
    image_num = 2
    image_file = "assets/1/000{}.jpg".format(image_num)
    pred_img = preprocess_image(Image.open(image_file))
    # prepare model
    print("evaluation on : {}".format(config.base_model))
    net = create_model(config.base_model)
    print("load pretained model from {}".format(config.test.pretrained_model))
    ModelRecorder.resume_model(net, config.test.pretrained_model, from_measurement="acc")
    show_first_conv_feature(net, pred_img, "{}_C".format(image_num))
    visualize(net, pred_img, 1, "{}_C".format(image_num))


def array2tsv(arr, file_name):
    file = open(file_name, mode='w')
    lines = ""
    for r in arr:
        line = ""
        for num in r:
            line += '%.2f\t' % num
        lines += line[:-1] + '\n'
    file.write(lines[:-1])
    file.close()


def visualize(pretrained_model, prep_img, target_class, file_name_to_export="test.png"):
    # Guided backprop
    GBP = GuidedBackprop(pretrained_model)
    # Get gradients
    guided_grads = GBP.generate_gradients(prep_img, target_class)
    # Save colored gradients
    save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color')
    # Convert to grayscale
    grayscale_guided_grads = convert_to_grayscale(guided_grads)
    # Save grayscale gradients
    save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')
    # Positive and negative saliency maps
    pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
    save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
    save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')
    print('Guided backprop completed')

def calculate_shape_IoU(pred_np, seg_np, label, class_choice):
    
    shape_ious = []
    seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
    index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
    seg_num_all = 50
    seg_start_index = 0
    for shape_idx in range(seg_np.shape[0]):
        if not class_choice:
            # print (label[shape_idx][0])
            idx = label[shape_idx][0]
            start_index = index_start[idx]
            num = seg_num[idx]
            parts = range(start_index, start_index + num)
        else:
            parts = range(seg_num[label[0]])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            # print ('iou ', part, iou)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    

    return shape_ious
@DeprecationWarning
def visualize2(pretrained_model, prep_img, target_class, file_name_to_export="test"):
    # Vanilla backprop
    VBP = VanillaBackprop(pretrained_model)
    # Generate gradients
    vanilla_grads = VBP.generate_gradients(prep_img, target_class)
    # Save colored gradients
    save_gradient_images(vanilla_grads, file_name_to_export + '_Vanilla_BP_color')
    # Convert to grayscale
    grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
    # Save grayscale gradients
    save_gradient_images(grayscale_vanilla_grads, file_name_to_export + '_Vanilla_BP_gray')
    print('Vanilla backprop completed')


def show_first_conv_feature(net, image, file_name):
    import torchvision

    def hook(module, input, out):
        print(out.size())
        out = out.cpu().detach()
        grid = torchvision.utils.make_grid(out.view(64, 1, out.shape[2], out.shape[3]))
        grid = grid[0]
        print(grid.shape)
        plt.imsave("../results/features_{}.png".format(file_name), grid.numpy())

    net.eval()
    net.net.conv1.register_forward_hook(hook)
    with torch.no_grad():
        net(image)


def create_model(base_model, ckpt_file=None, from_measurement=None):
    # prepare model
    if base_model == 'modelnet_x3_l4':
        net = PointCNN.modelnet_x3_l4()
    elif  base_model == 'cifar10_x3_l4':
        net = PointCNN.cifar10_x3_l4()
    elif base_model == 'shapenet_x8_2048_fps':
        net = PointCNN.shapenet_x8_2048_fps()
    else:
        raise NotImplementedError
    if ckpt_file is not None:
        ModelRecorder.resume_model(net, ckpt_file, from_measurement=from_measurement)
    return net


def train_epoch(data_loader, net: nn.Module, criterion, optimizer, epoch):
    global global_step
    if config.task == "seg":
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []
    batch_time = meter.TimeMeter(True)
    epoch_time = meter.TimeMeter(True)
    loss_meter = meter.AverageValueMeter()
    acc_meter = meter.ClassErrorMeter(topk=[1], accuracy=True)

    net.train(True)

    #################################for loop#################################
    for i, sample in enumerate(data_loader):

        #Adjust the lr dynamically:
        lr=config.train.learning_rate_base*(math.pow(config.train.decay_rate,global_step//config.train.decay_steps))
        if lr<config.train.learning_rate_min:
            lr=config.train.learning_rate_min
        for g in optimizer.param_groups:
            g['lr'] = lr


        batch_time.reset()
        #loss_meter.reset()

        batch_data = sample[0]
        batch_labels = sample[1]
        if config.task == "seg":
            data_num = sample[2]
            data_labels= sample[3]
        batch_time.reset()
        if config.task == "cls":
            shape = batch_data.shape
            indices = pf.get_indices(shape[0], config.dataset_setting["sample_num"], shape[1])
            indices = indices.reshape(indices.size // 2, 2)
            indices = indices[:, 0] * batch_data.shape[1] + indices[:, 1]
            indices = indices.astype(int)
            pts_fts_sampled = batch_data.view(-1, batch_data.shape[-1])[indices].view(shape[0], config.dataset_setting[
                "sample_num"], -1)
        else:
            shape = batch_data.shape
            indices = pf.get_indices(shape[0], config.dataset_setting["sample_num"], data_num)
            indices = indices.reshape(indices.size // 2, 2)
            indices = indices[:, 0] * batch_data.shape[1] + indices[:, 1]
            indices = indices.astype(int)
            pts_fts_sampled = batch_data.view(-1, batch_data.shape[-1])[indices].view(shape[0], config.dataset_setting[
                "sample_num"], -1)
            batch_labels = batch_labels.view(-1, 1)[indices].view(shape[0], config.dataset_setting["sample_num"])
        features_augmented = None

        xforms, rotations = pf.get_xforms(config.train.batch_size, config.dataset_setting["rotation_range"], config.dataset_setting["scaling_range"],
                                  config.dataset_setting["rotation_order"])
        if config.dataset_setting["data_dim"] > 3:
            points_sampled = pts_fts_sampled[:, :, :3]
            features_sampled = pts_fts_sampled[:, :, 3:]
            if config.dataset_setting["use_extra_features"]:
                if config.dataset_setting["with_normal_feature"]:
                    if config.dataset_setting["data_dim"] < 6:
                        print('Only 3D normals are supported!')
                        #exit()()
                    elif config.dataset_setting["data_dim"] == 6:
                        features_augmented = pf.augment(features_sampled, rotations)
                    else:
                        normals = features_sampled[:, :, :3]
                        rest = features_sampled[:, :, 3:]
                        normals_augmented = pf.augment(normals, rotations)
                        features_augmented = torch.cat((normals_augmented, rest), dim=-1)
                else:
                    features_augmented = features_sampled
        else:
            points_sampled = pts_fts_sampled


        jitter_range = config.dataset_setting["jitter"]
        points_augmented = pf.augment(points_sampled, xforms, jitter_range)

        if (features_augmented is None):
            batch_data = points_augmented
        else:
            batch_data = torch.cat((points_augmented, features_augmented), dim=-1)

        batch_data = batch_data.to(config.device)
        batch_labels=batch_labels.to(config.device)

        raw_out = net.forward(batch_data)
        if config.task=="cls":
            sample_num = raw_out.shape[1]
            raw_out = raw_out.view(-1, raw_out.shape[-1])
            batch_labels = batch_labels.view(-1, 1).repeat(1, sample_num).view(-1).long()
            loss = criterion(raw_out, batch_labels)
        elif config.task=="seg":
            pred_choice = raw_out.data.max(2)[1]

            raw_out = raw_out.view(-1, raw_out.shape[-1])
            loss=criterion(raw_out, batch_labels.view(-1).long())

        loss_meter.add(loss.item())
        acc_meter.add(raw_out.detach(),  batch_labels.view(-1).long().detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if config.task=="seg":
            seg_np = batch_labels.cpu().numpy()                
            pred_np = pred_choice.detach().cpu().numpy()
            temp_label = data_labels.numpy().reshape(-1, 1)

            train_true_cls.append(seg_np.reshape(-1))       
            train_pred_cls.append(pred_np.reshape(-1))      

            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)

            train_label_seg.append(temp_label)

        if i % config.print_freq == 0:
            print('Epoch: [{}][{}/{}]\t'.format(epoch, i, len(data_loader)) +
                  'Batch Time %.1f\t' % batch_time.value() +
                  'Epoch Time %.1f\t' % epoch_time.value() +
                  'Loss %.4f\t' % loss_meter.value()[0] +
                  'Acc(c) %.3f' % acc_meter.value(1))

        global_step = global_step + 1
    #################################for loop#################################
    if config.task=="cls":
        print('[ TRAIN summary ] epoch {}:\n'.format(epoch) +
            'category acc: {}'.format(acc_meter.value(1)))
        return loss_meter.value()[0],acc_meter.value(1)
    else:
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)

        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls.data, train_pred_cls.data)

        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_label_seg = np.concatenate(train_label_seg, axis=0)

        train_ious = calculate_shape_IoU(train_pred_seg, train_true_seg, train_label_seg, None)
        
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (epoch,loss_meter.value()[0] , train_acc, avg_per_class_acc, np.mean(train_ious))
        print(outstr)
        return loss_meter.value()[0],train_acc,avg_per_class_acc, np.mean(train_ious)


def evaluate(data_loader, net: nn.Module, calc_confusion_matrix=False, rtn_features=False):

    if config.task == "seg":
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []

    criterion = nn.CrossEntropyLoss()
    
    loss_meter = meter.AverageValueMeter()
    batch_time = meter.TimeMeter(True)
    epoch_time = meter.TimeMeter(True)
    acc_meter = meter.ClassErrorMeter(topk=[1, 5], accuracy=True)
    all_features = []
    all_labels = []
    num_classes = 40
    confusion_matrix_meter = None
    if calc_confusion_matrix:
        confusion_matrix_meter = meter.ConfusionMeter(num_classes, normalized=True)

    net.eval()
    for i, sample in enumerate(data_loader):
        batch_data=sample[0]
        batch_labels=sample[1]
        if config.task=="seg":
            data_num=sample[2]
            data_labels= sample[3]
        batch_time.reset()
        if config.task=="cls":
            shape=batch_data.shape
            indices = pf.get_indices(shape[0], config.dataset_setting["sample_num"], shape[1])
            indices=indices.reshape(indices.size//2,2)
            indices=indices[:,0]*batch_data.shape[1]+indices[:,1]
            indices=indices.astype(int)
            pts_fts_sampled = batch_data.view(-1,batch_data.shape[-1])[indices].view(shape[0], config.dataset_setting["sample_num"],-1)
        else:
            shape = batch_data.shape
            indices = pf.get_indices(shape[0], config.dataset_setting["sample_num"], data_num)
            indices = indices.reshape(indices.size // 2, 2)
            indices = indices[:, 0] * batch_data.shape[1] + indices[:, 1]
            indices = indices.astype(int)
            pts_fts_sampled = batch_data.view(-1, batch_data.shape[-1])[indices].view(shape[0], config.dataset_setting["sample_num"], -1)
            batch_labels=batch_labels.view(-1,1)[indices].view(shape[0],config.dataset_setting["sample_num"])
        features_augmented = None
        xforms, rotations = pf.get_xforms(shape[0], config.dataset_setting["rotation_range_val"], config.dataset_setting["scaling_range_val"],
                                  config.dataset_setting["rotation_order"])
        if config.dataset_setting["data_dim"] > 3:
            points_sampled = pts_fts_sampled[:, :, :3]
            features_sampled = pts_fts_sampled[:, :, 3:]
            if config.dataset_setting["use_extra_features"]:
                if config.dataset_setting["with_normal_feature"]:
                    if config.dataset_setting["data_dim"] < 6:
                        print('Only 3D normals are supported!')
                        exit()
                    elif config.dataset_setting["data_dim"] == 6:
                        features_augmented = pf.augment(features_sampled, rotations)
                    else:
                        normals = features_sampled[:, :, :3]
                        rest = features_sampled[:, :, 3:]
                        normals_augmented = pf.augment(normals, rotations)
                        features_augmented = torch.cat((normals_augmented, rest), dim=-1)
                else:
                    features_augmented = features_sampled
        else:
            points_sampled = pts_fts_sampled

        jitter_range_val = config.dataset_setting["jitter_val"]
        points_augmented = pf.augment(points_sampled, xforms, jitter_range_val)

        if (features_augmented is None):
            batch_data = points_augmented
        else:
            batch_data = torch.cat((points_augmented, features_augmented), dim=-1)

        batch_data = batch_data.to(config.device)
        batch_labels=batch_labels.to(config.device)

        raw_out = net.forward(batch_data)
        if config.task=="cls":
                    sample_num = raw_out.shape[1]
                    raw_out = raw_out.view(-1, raw_out.shape[-1])
                    batch_labels = batch_labels.view(-1, 1).repeat(1, sample_num).view(-1).long()
                    loss = criterion(raw_out, batch_labels)
                    if confusion_matrix_meter is not None:
                         confusion_matrix_meter.add(raw_out.cpu(), target=batch_labels)
        elif config.task=="seg":
                    pred_choice = raw_out.data.max(2)[1]

                    raw_out = raw_out.view(-1, raw_out.shape[-1])
                    loss=criterion(raw_out, batch_labels.view(-1).long())

        loss_meter.add(loss.item())
        acc_meter.add(raw_out.detach(),  batch_labels.view(-1).long().detach())

        if config.task=="seg":
            seg_np = batch_labels.cpu().numpy()                
            pred_np = pred_choice.detach().cpu().numpy()
            temp_label = data_labels.numpy().reshape(-1, 1)

            train_true_cls.append(seg_np.reshape(-1))       
            train_pred_cls.append(pred_np.reshape(-1))      

            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)

            train_label_seg.append(temp_label)
        

        if i % config.print_freq == 0:
            print('[{}/{}]\t'.format(i, len(data_loader)) +
                  'Batch Time %.1f\t' % batch_time.value() +
                  'Epoch Time %.1f\t' % epoch_time.value() +
                  'Loss %.4f\t' % loss_meter.value()[0] +
                  'acc(c) %.3f' % acc_meter.value(1))
            

    
    #rst = acc_meter.value(1)
    if config.task=="cls":
        print('[ Validation summary ] category acc: {}'.format(acc_meter.value(1)))
        rst = loss_meter.value()[0],acc_meter.value(1)
    else:
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)

        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls.data, train_pred_cls.data)

        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_label_seg = np.concatenate(train_label_seg, axis=0)

        train_ious = calculate_shape_IoU(train_pred_seg, train_true_seg, train_label_seg, None)
        
        outstr = '[ Validation summary ] loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (loss_meter.value()[0] , train_acc, avg_per_class_acc, np.mean(train_ious))
        print(outstr)
        rst = loss_meter.value()[0],train_acc,avg_per_class_acc, np.mean(train_ious)
    if calc_confusion_matrix:
        rst = rst, confusion_matrix_meter.value()
    if rtn_features:
        rst = rst, np.concatenate(all_features, axis=0), all_labels
    return rst


def plot_conf_matrix(classes, matrix, title="", show=True, save_file=None, diag_number=False):
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
    matrix = matrix.astype(np.float)
    linesum = matrix.sum(1)
    linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
    matrix /= linesum

    plt.switch_backend('agg')
    fig = plt.figure(figsize=(25, 25))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, norm=norm)
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))

    if diag_number:
        for i in range(matrix.shape[0]):
            ax.text(i, i, str('%.2f' % (matrix[i, i] * 100)), va='center', ha='center')

    ax.set_xticklabels([''] + classes, rotation=90)
    ylabel = []
    for i in range(len(classes)):
        ylabel.append(classes[i])
    ax.set_yticklabels([''] + ylabel)
    ax.set_title(title)
    plt.grid(axis='x', linestyle='-')
    plt.grid(axis='y', linestyle='-')
    if save_file is not None:
        plt.savefig(save_file)
        print('Save confusion matrix to: {}'.format(save_file))
    if show:
        plt.show()


def main():
    config.init_env()
    if config.process == "TRAIN":
        train_process()
    elif config.process == "VAL":
        evaluation_process()
    elif config.process == "VIS":
        visualization_process()
    elif config.process == "TEST":
        extract_test_rst_process()
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
