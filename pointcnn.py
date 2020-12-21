import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))

import torch
from torch import nn
from ptcnn_utils.model import RandPointCNN,RandPointCNN_Decoder,Dense_Conv1d
from ptcnn_utils.util_funcs import knn_indices_func_gpu, knn_indices_func_cpu
from ptcnn_utils.util_layers import Dense



class PointCNNCls(nn.Module):
    def __init__(self, num_classes):
        super(PointCNNCls, self).__init__()
        AbbPointCNN = lambda a, b, c, d, e: RandPointCNN(a, b, 3, c, d, e, knn_indices_func_gpu)
        self.pcnn1 = AbbPointCNN(3, 32, 8, 1, -1)
        self.pcnn2 = nn.Sequential(
            AbbPointCNN(32, 64, 8, 2, -1),
            AbbPointCNN(64, 96, 8, 4, -1),
            AbbPointCNN(96, 128, 12, 4, 120),
            AbbPointCNN(128, 160, 12, 6, 120)
        )

        self.fcn = nn.Sequential(
            Dense(160, 128),
            Dense(128, 64, drop_rate=0.5),
            Dense(64, num_classes, with_bn=False, activation=None)
        )

    def forward(self, x):
        x = x.transpose(1,2) #back in shape batch_size*N*3
        x = self.pcnn1((x, x))
        if False:
            print("Making graph...")
            k = make_dot(x[1])

            print("Viewing...")
            k.view()
            print("DONE")

            assert False
        x = self.pcnn2(x)[1]  # grab features

        logits = self.fcn(x)
        logits_mean = torch.mean(logits, dim=1)
        return logits_mean, None


class PointCNN_partseg(nn.Module):
    def __init__(self, part_num=50):
        super(PointCNN_partseg, self).__init__()

        EncoderCNN = lambda a, b, c, d, e: RandPointCNN(a, b, 3, c, d, e, knn_indices_func_gpu)
        DecoderCNN = lambda a, b, last_c, c, d, e: RandPointCNN_Decoder(a, b, last_c, 3, c, d, e, knn_indices_func_gpu) 

        self.encoder_0 = EncoderCNN(3, 256, 8, 1, -1)
        self.encoder_1 = EncoderCNN(256, 256, 12, 1, 768)
        self.encoder_2 = EncoderCNN(256, 512, 16, 1, 384)
        self.encoder_3 = EncoderCNN(512, 1024, 16, 1, 128)


        self.decoder_0 = DecoderCNN(1024, 1024, 1024,  16, 1, 128)
        self.decoder_1 = DecoderCNN(1024, 512, 512, 16, 1, 385)
        self.decoder_2 = DecoderCNN(512, 256, 256, 12, 1, 768)
        self.decoder_3 = DecoderCNN(256, part_num, 256, 8, 1, 2048)
        # self.decoder_4 = DecoderCNN(256, part_num, 8, 4, 2048)
        


    def forward(self, x, normal=None):
        x = x.transpose(1,2)
        x = (x, x)        
        # jt.sync_all(True)
        # start_time = time.time()
        x_0 = self.encoder_0(x)
        x_1 = self.encoder_1(x_0)
        x_2 = self.encoder_2(x_1)
        x_3 = self.encoder_3(x_2)
        x_3 = self.decoder_0(x_3, x_3)
        x_2 = self.decoder_1(x_3, x_2)
        x_1 = self.decoder_2(x_2, x_1)
        x_0 = self.decoder_3(x_1, x_0)
        
        return x_0[1].permute(0, 2, 1)

