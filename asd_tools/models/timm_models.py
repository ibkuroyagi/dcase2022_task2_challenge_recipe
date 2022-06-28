import timm
import torch
import torch.nn as nn
import torchaudio.transforms as T
from asd_tools.models.modules import GeM
import logging


class Backbone(nn.Module):
    def __init__(self, name="resnet18", pretrained=False, in_chans=3):
        super(Backbone, self).__init__()
        self.net = timm.create_model(name, pretrained=pretrained, in_chans=in_chans)

        if "regnet" in name:
            self.out_features = self.net.head.fc.in_features
        elif "res" in name:  # works also for resnest
            self.out_features = self.net.fc.in_features
        elif "efficientnet" in name:
            self.out_features = self.net.classifier.in_features
        elif "senet" in name:
            self.out_features = self.net.fc.in_features
        else:
            self.out_features = self.net.classifier.in_features

    def forward(self, x):
        x = self.net.forward_features(x)

        return x


class ASDModel(nn.Module):
    def __init__(
        self,
        backbone,
        neck=None,
        embedding_size=128,
        gem_pooling=False,
        pretrained=False,
        use_pos=True,
        in_chans=3,
        n_fft=2048,
        hop_length=256,
        n_mels=224,
        power=1.0,
        out_dim=6,
        time_mask_param=0,
        freq_mask_param=0,
        use_domain_head=False,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.in_chans = in_chans
        self.embedding_size = embedding_size
        self.out_dim = out_dim
        self.use_pos = use_pos
        self.backbone = Backbone(backbone, pretrained=pretrained, in_chans=in_chans)
        self.melspectrogram = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=50.0,
            f_max=7800.0,
            pad=0,
            n_mels=n_mels,
            power=power,
            normalized=True,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )
        if time_mask_param != 0:
            self.timemask = T.TimeMasking(time_mask_param=time_mask_param)
        else:
            self.timemask = None
        if freq_mask_param != 0:
            self.freqmask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        else:
            self.freqmask = None
        if gem_pooling == "gem":
            self.global_pool = GeM(p_trainable=True)
        else:
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        # https://www.groundai.com/project/arcface-additive-angular-margin-loss-for-deep-face-recognition
        if neck == "option-D":
            self.neck = nn.Sequential(
                nn.Linear(self.backbone.out_features, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU(),
                nn.Linear(self.embedding_size, self.embedding_size, bias=True),
            )
        elif neck == "option-F":
            self.neck = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.backbone.out_features, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU(),
                nn.Linear(self.embedding_size, self.embedding_size, bias=True),
            )
        else:
            self.neck = nn.Sequential(
                nn.Linear(self.backbone.out_features, self.embedding_size, bias=False),
                nn.BatchNorm1d(self.embedding_size),
                nn.Linear(self.embedding_size, self.embedding_size, bias=False),
            )
        self.machine_head = nn.Linear(1, 1, bias=True)
        self.section_head = nn.Linear(self.embedding_size, out_dim, bias=False)
        self.use_domain_head = use_domain_head
        if use_domain_head:
            self.domain_head = nn.Linear(self.embedding_size, 1, bias=False)

    def forward(self, input, specaug=False):
        x = self.melspectrogram(input)
        # logging.info(f"melspec:{x.shape}")
        if specaug:
            if self.timemask is not None:
                x = self.timemask(x)
            if self.freqmask is not None:
                x = self.freqmask(x)
            # logging.info(f"specaug:{x.shape}")
        x = x.unsqueeze(1)
        # logging.info(f"unsqueeze:{x.shape}")
        if self.use_pos:
            pos = torch.linspace(0.0, 1.0, x.size(2)).to(x.device)
            pos = pos.half()
            pos = pos.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            pos = pos.expand(x.size(0), 1, x.size(2), x.size(3))
            if self.in_chans == 2:
                x = x.expand(-1, 1, -1, -1)
                x = torch.cat([x, pos], 1)
            else:
                x = x.expand(-1, 2, -1, -1)
                x = torch.cat([x, pos], 1)
        else:
            x = x.expand(-1, 3, -1, -1)
        # logging.info(f"before x:{x.shape}")
        x = self.backbone(x)
        x = self.global_pool(x)[:, :, 0, 0]
        embedding = self.neck(x)
        machine = self.machine_head(
            torch.pow(embedding, 2).sum(dim=1).unsqueeze(1) / self.embedding_size
        )
        section = self.section_head(embedding)
        output_dict = {
            "embedding": embedding,
            "machine": machine,
            "section": section,
        }
        if self.use_domain_head:
            # source = 0, target = 1
            domain = self.domain_head(embedding)
            output_dict["domain"] = domain

        return output_dict
