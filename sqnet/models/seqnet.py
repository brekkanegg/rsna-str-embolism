import torch
import torch.nn as nn
import torch.nn.functional as F
import math


import torchvision.models as tvmodels
from collections import OrderedDict

# BiLSTM - ResNext


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(
            x
        )  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class PositionalEncoding1D(torch.nn.Module):
    def __init__(self, channels=256):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        batch_size, x, orig_ch = tensor.shape
        # pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_x = torch.arange(x).type(self.inv_freq.type())

        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        # emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb = torch.zeros((x, self.channels)).type(tensor.type())
        emb[:, : self.channels] = emb_x
        emb = emb.repeat(batch_size, 1, 1)

        return emb[..., :orig_ch]


class ImageLSTM(nn.Module):
    def __init__(self, args, use_pretrained=True):

        super(ImageLSTM, self).__init__()
        self.args = args
        # Embedding

        if args.encoder == "resnext101":
            if use_pretrained:
                self.embedder = torch.hub.load(
                    "facebookresearch/WSL-Images", "resnext101_32x8d_wsl"
                )
            else:
                self.embedder = tvmodels.resnext101_32x8d(pretrained=False)

        if self.args.freeze_encoder:
            self.freeze_encoder()

        self.units = args.units
        if self.units == 2048:
            self.embedder.fc = nn.Identity()

        else:
            self.embedder.fc = nn.Linear(2048, self.units)
            for m in self.embedder.fc.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if hasattr(m, "bias") and m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        # Bi-LSTM
        """
        inputs: (batch, seq_len, input_size)
        outputs: (batch, seq_len, num_directions * hidden_size)
        """

        self.embedding_dropout = SpatialDropout(0.0)  # DO)
        self.lstm1 = nn.LSTM(
            self.units, self.units, bidirectional=True, batch_first=True
        )
        self.lstm2 = nn.LSTM(
            self.units * 2, self.units, bidirectional=True, batch_first=True
        )
        if self.args.multi_gpu:
            self.lstm1.flatten_parameters()
            self.lstm2.flatten_parameters()

        self.linear1 = nn.Linear(self.units * 2, self.units * 2)
        self.linear2 = nn.Linear(self.units * 2, self.units * 2)

        self.linear = nn.Linear(self.units * 2, args.num_classes)

    def freeze_encoder(self):
        self.embedder.fc = nn.Identity()

        encoder_checkpoint = torch.load(self.args.encoder_checkpoint)
        encoder_state_dicts = OrderedDict(
            {k.split("embedder.")[1]: v for k, v in encoder_checkpoint["model"].items()}
        )

        self.embedder.load_state_dict(encoder_state_dicts, strict=False)
        for param in self.embedder.parameters():
            param.requires_grad = False

    def forward(self, x):
        # TODO: check shape
        # x : (batch , channel , slice_num , W , H)

        batch = torch.transpose(x, 1, 2)

        b, n, c, w, h = batch.shape
        batch = torch.reshape(
            batch, (b * n, c, w, h)
        )  # batch : (batch x slice_num, c, W, H)

        embed = self.embedder(batch)  # embed: (batch x slice_num, embed_size)

        h_embedding = torch.reshape(
            embed, (b, n, self.units)
        )  # h_embedding: (batch, slice_num, embed_size)

        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)

        h_conc_linear1 = F.relu(self.linear1(h_lstm1))
        h_conc_linear2 = F.relu(self.linear2(h_lstm2))

        if self.args.dense:
            h_embedd = torch.cat((x, x,), -1,)
            hidden = h_lstm1 + h_lstm2 + h_conc_linear1 + h_conc_linear2 + h_embedd
        else:
            hidden = h_conc_linear2

        output = self.linear(hidden)

        return output


class ImageTransformer(nn.Module):
    def __init__(self, args, use_pretrained=True):

        super(ImageTransformer, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer

        self.args = args
        # Embedding

        if args.encoder == "resnext101":
            if use_pretrained:
                self.embedder = torch.hub.load(
                    "facebookresearch/WSL-Images", "resnext101_32x8d_wsl"
                )
            else:
                self.embedder = tvmodels.resnext101_32x8d(pretrained=False)

        if self.args.freeze_encoder:
            self.freeze_encoder()

        self.units = args.units
        if self.units == 2048:
            self.embedder.fc = nn.Identity()

        else:
            self.embedder.fc = nn.Linear(2048, self.units)
            for m in self.embedder.fc.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if hasattr(m, "bias") and m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        # self.src_mask = None
        self.position_encoder = PositionalEncoding1D(self.units)  # (ninp, dropout)
        encoder_layer = TransformerEncoderLayer(
            d_model=self.units, nhead=8, dim_feedforward=2048, dropout=0.1
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=2
        )

        self.transformer_decoder = nn.Linear(self.units, self.args.num_classes)

    def freeze_encoder(self):
        self.embedder.fc = nn.Identity()

        encoder_checkpoint = torch.load(self.args.encoder_checkpoint)
        encoder_state_dicts = OrderedDict(
            {k.split("embedder.")[1]: v for k, v in encoder_checkpoint["model"].items()}
        )

        self.embedder.load_state_dict(encoder_state_dicts, strict=False)
        for param in self.embedder.parameters():
            param.requires_grad = False

    def forward(self, x):
        # TODO: check shape
        # x : (batch , channel , slice_num , W , H)

        batch = torch.transpose(x, 1, 2)

        b, n, c, w, h = batch.shape
        batch = torch.reshape(
            batch, (b * n, c, w, h)
        )  # batch : (batch x slice_num, c, W, H)

        embed = self.embedder(batch)  # embed: (batch x slice_num, embed_size)
        x = embed

        x = torch.reshape(x, (b, n, self.units))
        x = x + self.position_encoder(x)
        embed = self.transformer_encoder(x)
        output = self.transformer_decoder(embed)

        return output
