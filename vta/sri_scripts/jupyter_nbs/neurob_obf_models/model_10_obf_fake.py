
import torch
import torch.nn as nn
import math


# assert torch.cuda.is_available()
# cuda_device = torch.device("cuda")  # device object representing GPU

# This model stands for resnet-20 ImageNet
# model_id = 9 (regardless of the file name)

class obf_fake(torch.nn.Module):
    def __init__(self, input_features):

        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        # self.logsoftmax = torch.nn.LogSoftmax(dim = 1)

        self.first = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3),
                                   nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.dense = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, 10))

        self.conv0 = nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1 = nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3 = nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn2 = nn.BatchNorm2d(64)


        self.conv5 = nn.Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv6 = nn.Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv7 = nn.Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv8 = nn.Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv9 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv10 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(2, 2), padding=0)
        self.conv11 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(2, 2), padding=0)
        self.bn25 = nn.BatchNorm2d(64)

        self.conv12 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        self.conv13 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv14 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv15 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn8 = nn.BatchNorm2d(128)

        self.conv16 = nn.Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv17 = nn.Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv18 = nn.Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv19 = nn.Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn9 = nn.BatchNorm2d(128)

        self.conv20 = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), padding=0)
        self.bn26 = nn.BatchNorm2d(256)


        self.conv21 = nn.Conv2d(64, 256, kernel_size=(3, 3), stride=(2, 2), padding=1)
        # self.bn10 = nn.BatchNorm2d(256)
        self.conv22 = nn.Conv2d(64, 256, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn11 = nn.BatchNorm2d(256)

        self.conv23 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn12 = nn.BatchNorm2d(256)


        self.conv24 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn13 = nn.BatchNorm2d(256)
        self.conv25 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bn14 = nn.BatchNorm2d(256)
        self.conv26 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn15 = nn.BatchNorm2d(256)
        self.conv27 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bn16 = nn.BatchNorm2d(256)

        self.conv28 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn17 = nn.BatchNorm2d(256)
        self.conv29 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn18 = nn.BatchNorm2d(256)

        self.conv30 = nn.Conv2d(64, 512, kernel_size=(1, 1), stride=(2, 2), padding=0)
        self.conv31 = nn.Conv2d(64, 512, kernel_size=(1, 1), stride=(2, 2), padding=0)
        self.conv32 = nn.Conv2d(64, 512, kernel_size=(1, 1), stride=(2, 2), padding=0)
        self.conv33 = nn.Conv2d(64, 512, kernel_size=(1, 1), stride=(2, 2), padding=0)
        self.bn19 = nn.BatchNorm2d(512)

        self.conv34 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn20 = nn.BatchNorm2d(512)

        self.conv35 = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bn20 = nn.BatchNorm2d(512)
        self.conv36 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn21 = nn.BatchNorm2d(512)
        self.conv37 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn22 = nn.BatchNorm2d(512)
        self.conv38 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn23 = nn.BatchNorm2d(512)
        self.conv39 = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bn24 = nn.BatchNorm2d(512)

        self.reset_parameters(input_features)

    def reset_parameters(self, input_features):
        stdv = 1.0 / math.sqrt(input_features)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, X1):
        X1 = X1.reshape(-1, 3, 224, 224)

        X1 = self.first(X1)

        X1_0 = self.conv0(X1[:, :16, :, :])
        X1_1 = self.conv1(X1[:, 16:32, :, :])
        X1_2 = self.conv2(X1[:, 32:48, :, :])
        X1_3 = self.conv3(X1[:, 48:64, :, :])
        X1 = X1_0 + X1_1 + X1_2 + X1_3
        X1 = self.bn1(X1)
        X1 = self.relu(X1)  # X1 shape = [1, 64, 56, 56]

        X1 = self.conv4(X1)
        X1 = self.bn2(X1)
        X1 = self.relu(X1)  # X1 shape = [1, 64, 56, 56]

        X1_0 = self.conv5(X1)
        X1_1 = self.conv6(X1)
        X1_2 = self.conv7(X1)
        X1_3 = self.conv8(X1)

        X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
        X1 = self.bn3(X1)
        X1 = self.relu(X1) # X1 shape = [1, 64, 56, 56]

        X1 = self.conv9(X1)
        X1 = self.bn4(X1)
        X1 = self.relu(X1) # X1 shape = [1, 64, 56, 56]

        X1_0 = self.conv10(X1)
        X1_1 = self.conv11(X1)
        X1 = X1_0 + X1_1
        X1 = self.bn25(X1)
        X1 = self.relu(X1)  # X1 shape = [1, 64, 28, 28]

        X1 = self.conv12(X1)
        X1 = self.bn5(X1)
        X1 = self.relu(X1)  # X1 shape = [1, 128, 28, 28]

        X1 = self.conv13(X1)
        X1 = self.bn6(X1)
        X1 = self.relu(X1)  # X1 shape = [1, 128, 28, 28]

        X1 = self.conv14(X1)
        X1 = self.bn7(X1)
        X1 = self.relu(X1)  # X1 shape = [1, 128, 28, 28]

        X1 = self.conv15(X1)
        X1 = self.bn8(X1)
        X1 = self.relu(X1)  # X1 shape = [1, 128, 28, 28]

        X1_0 = self.conv16(X1[:, :32, :, :])
        X1_1 = self.conv17(X1[:, 32:64, :, :])
        X1_2 = self.conv18(X1[:, 64:96, :, :])
        X1_3 = self.conv19(X1[:, 96:128, :, :])
        X1 = X1_0 + X1_1 + X1_2 + X1_3
        X1 = self.bn9(X1)
        X1 = self.relu(X1)  # X1 shape = [1, 128, 28, 28]

        X1_sk1 = X1
        X1 = self.conv20(X1)
        X1 = self.bn26(X1)
        X1 = self.relu(X1)  # X1 shape = [1, 256, 14, 14]

        X1_0 = self.conv21(X1_sk1[:, :64, :, :])
        X1_1 = self.conv22(X1_sk1[:, 64:128, :, :])
        X1 = X1_0 + X1_1 + X1
        X1 = self.bn11(X1)
        X1 = self.relu(X1) # X1 shape = [1, 256, 14, 14]

        X1 = self.conv23(X1)
        X1 = self.bn12(X1)
        X1 = self.relu(X1) # X1 shape = [1, 256, 14, 14]

        X1 = self.conv24(X1)
        X1 = self.bn13(X1)
        X1 = self.relu(X1) # X1 shape = [1, 256, 14, 14]

        X1 = self.conv25(X1)
        X1 = self.bn14(X1)
        X1 = self.relu(X1) # X1 shape = [1, 256, 14, 14]

        X1 = self.conv26(X1)
        X1 = self.bn15(X1)
        X1 = self.relu(X1) # X1 shape = [1, 256, 14, 14]

        X1 = self.conv27(X1)
        X1 = self.bn16(X1)
        X1 = self.relu(X1) # X1 shape = [1, 256, 14, 14]

        X1_0 = self.conv28(X1[:, :128, :, :])
        X1_1 = self.conv29(X1[:, 128:256, :, :])
        X1 = X1_0 + X1_1
        X1 = self.bn17(X1)
        X1 = self.relu(X1) # X1 shape = [1, 256, 14, 14]

        X1_sk1 = X1

        X1_0 = self.conv30(X1[:, :64, :, :])
        X1_1 = self.conv31(X1[:, 64:128, :, :])
        X1_2 = self.conv32(X1[:, 128:192, :, :])
        X1_3 = self.conv33(X1[:, 192:256, :, :])
        X1 = X1_0 + X1_1 + X1_2 + X1_3
        X1 = self.bn19(X1)
        X1 = self.relu(X1) # X1 shape = [1, 512, 7, 7]

        X1_sk2 = self.conv34(X1_sk1)
        X1 = X1_sk2 + X1
        X1 = self.bn20(X1)
        X1 = self.relu(X1) # X1 shape = [1, 512, 7, 7]

        X1 = self.conv35(X1)
        X1 = self.bn20(X1)
        X1 = self.relu(X1) # X1 shape = [1, 512, 7, 7]

        X1 = self.conv36(X1)
        X1 = self.bn21(X1)
        X1 = self.relu(X1) # X1 shape = [1, 512, 7, 7]

        X1 = self.conv37(X1)
        X1 = self.bn22(X1)
        X1 = self.relu(X1) # X1 shape = [1, 512, 7, 7]

        X1 = self.conv38(X1)
        X1 = self.bn23(X1)
        X1 = self.relu(X1) # X1 shape = [1, 512, 7, 7]

        X1 = self.conv39(X1)
        X1 = self.bn24(X1)
        X1 = self.relu(X1) # X1 shape = [1, 512, 7, 7]

        X1 = self.dense(X1)


        return X1


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # batch_size = 1
# input_features = 150528
# torch.manual_seed(1234)
# X = torch.randn(1, input_features)
#
# # Start Call Model
# #
# model = obf_fake(input_features)
# model.to(device)
#
#
# model.eval()
# new_out = model(X.to(device))
#
# print("Done")
#
# # End Call Model
# model.eval()
# new_out = model(X)
