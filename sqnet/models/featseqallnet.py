import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# import torchvision.models as tvmodels

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


class FeatGRUSimpleALL(nn.Module):
    def __init__(self, args, use_pretrained=False):

        super(FeatGRUSimpleALL, self).__init__()
        self.args = args

        self.units = args.units

        # Bi-LSTM
        """
        inputs: (batch, seq_len, input_size)
        outputs: (batch, seq_len, num_directions * hidden_size)
        """

        self.rnn1 = nn.GRU(2048, self.units, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(self.units * 2, self.units * 2)

        self.image_classifier = nn.Linear(self.units * 2, args.num_classes)
        self.exam_classifier = nn.Linear(self.units * 4, 9)

    def forward(self, x):

        h_rnn1, _ = self.rnn1(x)

        h_conc_linear1 = F.relu(self.linear1(h_rnn1))

        image_output = self.image_classifier(h_conc_linear1)

        exam_hidden = torch.cat(
            (
                F.adaptive_avg_pool2d(h_conc_linear1, (1, self.units * 2)),
                F.adaptive_max_pool2d(h_conc_linear1, (1, self.units * 2)),
            ),
            axis=2,
        )
        exam_output = self.exam_classifier(exam_hidden)
        exam_output = exam_output[:, 0, :].unsqueeze(
            2
        )  # (1 x 9 x 1) matching shape with image output

        # NOTE: exam_output first in order
        output = torch.cat((exam_output, image_output), axis=1)

        return output


class FeatLSTMSimpleALL(nn.Module):
    def __init__(self, args, use_pretrained=False):

        super(FeatLSTMSimpleALL, self).__init__()
        self.args = args
        self.units = args.units
        self.rnn1 = nn.LSTM(2048, self.units, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(self.units * 2, self.units * 2)
        self.image_classifier = nn.Linear(self.units * 2, args.num_classes)
        self.exam_classifier = nn.Linear(self.units * 4, 9)

    def forward(self, x):
        """
        inputs: (batch, seq_len, input_size)
        outputs: (batch, seq_len, num_directions * hidden_size)
        """

        h_rnn1, _ = self.rnn1(x)
        h_conc_linear1 = F.relu(self.linear1(h_rnn1))

        image_output = self.image_classifier(h_conc_linear1)
        exam_hidden = torch.cat(
            (
                F.adaptive_avg_pool2d(h_conc_linear1, (1, self.units * 2)),
                F.adaptive_max_pool2d(h_conc_linear1, (1, self.units * 2)),
            ),
            axis=2,
        )
        exam_output = self.exam_classifier(exam_hidden)
        exam_output = exam_output[:, 0, :].unsqueeze(
            2
        )  # (1 x 9 x 1) matching shape with image output

        # NOTE: exam_output first in order
        output = torch.cat((exam_output, image_output), axis=1)

        return output


class FeatGRUALL(nn.Module):
    def __init__(self, args, use_pretrained=False):

        super(FeatGRUALL, self).__init__()
        self.args = args
        self.units = args.units

        self.rnn1 = nn.GRU(2048, self.units, bidirectional=True, batch_first=True)
        self.rnn2 = nn.GRU(
            self.units * 2, self.units, bidirectional=True, batch_first=True
        )

        self.linear1 = nn.Linear(self.units * 2, self.units * 2)
        self.linear2 = nn.Linear(self.units * 2, self.units * 2)

        self.image_classifier = nn.Linear(self.units * 2, args.num_classes)
        self.exam_classifier = nn.Linear(self.units * 4, 9)

    def forward(self, x):

        h_rnn1, _ = self.rnn1(x)
        h_rnn2, _ = self.rnn2(h_rnn1)

        h_conc_linear1 = F.relu(self.linear1(h_rnn1))
        h_conc_linear2 = F.relu(self.linear2(h_rnn2))

        if self.args.dense:
            h_embedd = torch.cat((x, x,), -1,)
            hidden = h_rnn1 + h_rnn2 + h_conc_linear1 + h_conc_linear2 + h_embedd
        else:
            hidden = h_conc_linear2

        image_output = self.image_classifier(hidden)  # (1 x slice_num x 1)

        # FIXME: adaptive pooling 2d? 1d?
        exam_hidden = torch.cat(
            (
                F.adaptive_avg_pool2d(hidden, (1, self.units * 2)),
                F.adaptive_max_pool2d(hidden, (1, self.units * 2)),
            ),
            axis=2,
        )
        # (1x1x(units*2))

        exam_output = self.exam_classifier(exam_hidden)  # (1 x 1 x 9)
        exam_output = exam_output[:, 0, :].unsqueeze(
            2
        )  # (1 x 9 x 1) matching shape with image output

        output = torch.cat((exam_output, image_output), axis=1)

        return output


# Feature - SequenceNet
class FeatLSTMALL(nn.Module):
    def __init__(self, args, use_pretrained=False):

        super(FeatLSTMALL, self).__init__()
        self.args = args

        self.units = args.units

        # if self.units != 2048:
        #     self.embed_fc = nn.Linear(2048, self.units)

        #     for m in self.embed_fc.modules():
        #         if isinstance(m, nn.Linear):
        #             nn.init.kaiming_normal_(m.weight)
        #             if hasattr(m, "bias") and m.bias is not None:
        #                 nn.init.constant_(m.bias, 0)

        # Bi-LSTM
        """
        inputs: (batch, seq_len, input_size)
        outputs: (batch, seq_len, num_directions * hidden_size)
        """

        # self.embedding_dropout = SpatialDropout(0.0)  # DO)

        self.lstm1 = nn.LSTM(2048, self.units, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(
            self.units * 2, self.units, bidirectional=True, batch_first=True
        )

        self.linear1 = nn.Linear(self.units * 2, self.units * 2)
        self.linear2 = nn.Linear(self.units * 2, self.units * 2)

        self.image_classifier = nn.Linear(self.units * 2, args.num_classes)
        self.exam_classifier = nn.Linear(self.units * 4, 9)

    def forward(self, x):
        # x : (batch , embed_size)
        # if self.units != 2048:
        #     x = self.embed_fc(x)

        h_lstm1, _ = self.lstm1(x)
        h_lstm2, _ = self.lstm2(h_lstm1)

        h_conc_linear1 = F.relu(self.linear1(h_lstm1))
        h_conc_linear2 = F.relu(self.linear2(h_lstm2))

        if self.args.dense:
            h_embedd = torch.cat((x, x,), -1,)
            hidden = h_lstm1 + h_lstm2 + h_conc_linear1 + h_conc_linear2 + h_embedd
        else:
            hidden = h_conc_linear2

        image_output = self.image_classifier(hidden)  # (1 x slice_num x 1)

        # FIXME: adaptive pooling 2d? 1d?
        exam_hidden = torch.cat(
            (
                F.adaptive_avg_pool2d(hidden, (1, self.units * 2)),
                F.adaptive_max_pool2d(hidden, (1, self.units * 2)),
            ),
            axis=2,
        )
        # (1x1x(units*2))

        exam_output = self.exam_classifier(exam_hidden)  # (1 x 1 x 9)
        exam_output = exam_output[:, 0, :].unsqueeze(
            2
        )  # (1 x 9 x 1) matching shape with image output

        output = torch.cat((exam_output, image_output), axis=1)

        return output


class PositionalEncoding1D(torch.nn.Module):
    def __init__(self, channels=2048):
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


class FeatTransformerALL(nn.Module):
    def __init__(self, args, use_pretrained=False):

        super(FeatTransformerALL, self).__init__()

        from torch.nn import TransformerEncoder, TransformerEncoderLayer

        self.args = args

        self.units = args.units

        if self.units != 2048:
            self.embed_fc = nn.Linear(2048, self.units)

            for m in self.embed_fc.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if hasattr(m, "bias") and m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        self.position_encoder = PositionalEncoding1D(self.units)  # (ninp, dropout)

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=self.units, nhead=8, dim_feedforward=self.units, dropout=0.1
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=2
        )

        # Transformer Decoder
        self.image_classifier = nn.Linear(self.units, args.num_classes)
        self.exam_classifier = nn.Linear(self.units * 2, 9)

        # self.transformer_decoder = nn.Linear(2048, self.args.num_classes)

        ########################################################################

    def forward(self, x):

        if self.units != 2048:
            x = self.embed_fc(x)

        x = x + self.position_encoder(x)
        hidden = self.transformer_encoder(x)

        image_output = self.image_classifier(hidden)  # (1 x slice_num x 1)

        # FIXME: adaptive pooling 2d? 1d?
        exam_hidden = torch.cat(
            (
                F.adaptive_avg_pool2d(hidden, (1, self.units)),
                F.adaptive_max_pool2d(hidden, (1, self.units)),
            ),
            axis=2,
        )
        # (1x1x(units*2))

        exam_output = self.exam_classifier(exam_hidden)  # (1 x 1 x 9)
        exam_output = exam_output[:, 0, :].unsqueeze(
            2
        )  # (1 x 9 x 1) matching shape with image output

        output = torch.cat((exam_output, image_output), axis=1)

        return output

