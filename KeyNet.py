import torch
import torch.nn as nn
from modules.irb import InvertedResidual
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

num_joints = 22

class KeyNet(nn.Module):
    def __init__(self, input_side_px):
        # Needs to be divisible by 8!
        assert (input_side_px % 8) == 0
        self.input_side_px = input_side_px
        self.old_keypoints_side_px = int(input_side_px/8)
        super(KeyNet, self).__init__()

        self.image_network = self.make_backbone_image()
        self.keypoints_network = self.make_backbone_keypoints()
        self.fused_network = self.make_backbone_fused()
        self.new_network = self.newhand_regression()
        self.heatmap_network = self.heatmap_regression()

        self.idk_network = self.idk_regression()

        self.lins = [nn.Linear(100, 4)]*num_joints

        self.simple = nn.Linear(100*num_joints, num_joints*5)

        # if hyper['model']['resume'] == False:
        #     print ('Init para...')
        #     self.init_weights()

    def newhand_regression(self):
        out = nn.Sequential(
            nn.AvgPool2d(kernel_size = 6, stride = 6),

            nn.Conv2d(in_channels = 160, out_channels = 378, kernel_size = 1),
            nn.ReLU6(inplace = True),
            nn.Conv2d(in_channels = 378, out_channels = 128, kernel_size = 1),
            nn.ReLU6(inplace = True),
            nn.Conv2d(in_channels = 128, out_channels = num_joints*3, kernel_size = 1)
        )
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            #
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_backbone_image(self):

        input_channel = 32
        interverted_residual_setting = [
            # t, c, n, s
            [1, 32, 1, 1],
            [6, 32, 1, 2],
            [6, 32, 1, 1],
            [6, 64, 1, 2],
            [6, 64, 2, 1],
        ]

        out = [
            nn.Conv2d(1, input_channel, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace = True)
        ]

        for t, c, n, s in interverted_residual_setting:

            output_channel = c
            for i in range(n):
                if i == 0:
                    out.append(InvertedResidual(input_channel, output_channel, s, expand_ratio = t))
                else:
                    out.append(InvertedResidual(input_channel, output_channel, 1, expand_ratio = t))
                input_channel = output_channel
        return nn.Sequential(*out)

    def make_backbone_keypoints(self):
        wh = self.input_side_px / 8
        depth = 32
        out = nn.Sequential(
            nn.Linear(num_joints*5, int(self.old_keypoints_side_px*self.old_keypoints_side_px*depth)),
            nn.ReLU6(inplace = True)
        )
        return out

    def make_backbone_fused(self, input_channel=96):
        interverted_residual_setting = [
            # t, c, n, s
            [1, 64, 2, 1],
            [1, 64, 3, 1],
            [1, 96, 1, 1],
            [1, 96, 2, 1],
            [1, 128, 1, 2],
            [1, 128, 2, 1],
            [1, 160, 1, 1],
        ]

        out = []
        for t, c, n, s in interverted_residual_setting:
            output_channel = c
            for i in range(n):
                if i == 0:
                    out.append(InvertedResidual(input_channel, output_channel, s, expand_ratio = t))
                else:
                    out.append(InvertedResidual(input_channel, output_channel, 1, expand_ratio = t))
                input_channel = output_channel

        return nn.Sequential(*out)

    def heatmap_regression(self):
        out = nn.Sequential(
            nn.Conv2d(in_channels = 160, out_channels = num_joints*3, kernel_size = 3, padding = 2),
            nn.BatchNorm2d(num_joints*3),
            nn.ReLU6(inplace = True),

            nn.ConvTranspose2d(in_channels = num_joints*3, out_channels = 42, kernel_size = 2, stride = 2),

            nn.Conv2d(in_channels = 42, out_channels = 21, kernel_size = 3, padding = 2),
            nn.BatchNorm2d(21),
            nn.ReLU6(inplace = True)
        )
        return out
    def idk_regression(self):
        out = nn.Sequential(
            nn.Conv2d(in_channels = 160, out_channels = num_joints, kernel_size = 3, padding = 2),
            nn.BatchNorm2d(num_joints), # what do this do??
            nn.ReLU6(inplace=True),
        )
        return out

    def forward(self, x, addon=torch.ones(1, num_joints*3)):
        x = self.image_network(x)
        x_addon = self.keypoints_network(addon) # b, 4608
        print("x_addon ->", x_addon.shape)

        x_addon = x_addon.view(-1, 32, self.old_keypoints_side_px, self.old_keypoints_side_px)

        print(x.shape, )
        print(x_addon.shape)

        x = torch.cat((x, x_addon), dim = 1) # b, 96, 12, 12
        x = self.fused_network(x)

        print(f"Before Seth time: {x.shape}")
        #72x72: torch.Size([1, 160, 5, 5])
        #96x96: torch.Size([1, 160, 6, 6])
        #128x128: torch.Size([1, 21, 22, 22])

        

        x = self.idk_network(x)

        if True:
            x = torch.flatten(x, 1)
            print("after flatten ->", x.shape)
            x = self.simple(x)
            print("after simple ->", x.shape)
            
            x = x.view(-1, num_joints, 5)
            print("after view ->", x.shape)

            return x
        else:


            x = torch.flatten(x, 2)


            out = torch.empty(1, 21, 4)

            for i in range(21):

                # print(out[:, i].shape)
                # print(x[:,i,:].shape)
                out[:, i] = self.lins[i](x[:,i, :])
            return out



        return x

        # return x

        x_what = self.new_network(x)

        print(f"not squeezed: {x_what.shape} {x_sjopi.shape}")


        x_new = self.new_network(x).squeeze(dim=2).squeeze(dim=2) # b num_joints*3 1 1 -> b num_joints*3
        x_hp = self.heatmap_network(x)

        return x_new, x_hp