from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from augmentations import get_aug
import torch

class Finetune(ContinualModel):
    NAME = 'finetune'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, len_train_loader, transform):
        super(Finetune, self).__init__(backbone, loss, args, len_train_loader, transform)
        self.buffer = Buffer(self.args.model.buffer_size, self.device)
        self.backbone = backbone
        self.byol = (args.model_name == 'byol')

    def observe(self, inputs1, labels, inputs2, notaug_inputs):

        self.opt.zero_grad()
        if self.buffer.is_empty():
            if self.args.cl_default:
                labels = labels.to(self.device)
                outputs = self.net.module.backbone(inputs1.to(self.device))
                loss = self.loss(outputs, labels).mean()
                data_dict = {'loss': loss}
                data_dict['penalty'] = 0.0

            else:
                data_dict = self.net.forward(
                    inputs1.to(self.device, non_blocking=True),
                    inputs2.to(self.device, non_blocking=True))
                loss = data_dict['loss'].mean()
                data_dict['loss'] = data_dict['loss'].mean()
                data_dict['penalty'] = 0.0

        else:
            if self.args.cl_default:
                buf_inputs, buf_labels = self.buffer.get_data(
                    self.args.batch_size, transform=self.transform)
                buf_labels = buf_labels.to(self.device).long()
                labels = labels.to(self.device).long()
                mixed_x = torch.cat((inputs1.to(self.device), buf_inputs[:inputs1.shape[0]].to(self.device)), 0)
                outputs = self.net.module.backbone(mixed_x.to(self.device, non_blocking=True))
                labels = torch.cat((labels, buf_labels[:inputs1.shape[0]].to(self.device)), 0).to(self.device)
                loss = self.loss(outputs, labels).mean()
                data_dict = {'loss': loss}
                data_dict['penalty'] = 0.0
            else:
                buf_inputs, buf_inputs1 = self.buffer.get_data(
                    self.args.batch_size, transform=self.transform)
                mixed_x = torch.cat((inputs1.to(self.device), buf_inputs[:inputs1.shape[0]].to(self.device)), 0)
                mixed_x_aug = torch.cat((inputs2.to(self.device), buf_inputs1[:inputs1.shape[0]].to(self.device)), 0)
                data_dict = self.net.forward(mixed_x.to(self.device, non_blocking=True), mixed_x_aug.to(self.device, non_blocking=True))
                loss = data_dict['loss'].mean()
                data_dict['loss'] = data_dict['loss'].mean()
                data_dict['penalty'] = 0.0

        loss.backward()
        self.opt.step()
        data_dict.update({'lr': self.args.learning_rate_stream})
        if self.args.cl_default:
            self.buffer.add_data(examples=notaug_inputs, logits=labels)
        else:
            self.buffer.add_data(examples=notaug_inputs, logits=inputs2)

        if self.byol:
            self.net.module.update_moving_average()

        return data_dict
