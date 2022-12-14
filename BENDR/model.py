import torch
import torchvision
import torch.nn.functional as F
from abc import ABCMeta
from abc import abstractmethod
import numpy as np
import tensorflow as tf
import gc

from dn3_ext import LinearHeadBENDR
import dn3_ext




import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ModelWrapper(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        self.model = None
        self.model_name = None

    @abstractmethod
    def get_cutted_model(self, bottleneck):
        pass

    def get_gradient(self, acts, y, bottleneck_name):
        inputs = torch.autograd.Variable(torch.tensor(acts).to(device), requires_grad=True)
        targets = (y[0] * torch.ones(inputs.size(0))).long().to(device)

        cutted_model = self.get_cutted_model(bottleneck_name).to(device)
        cutted_model.eval()
        outputs = cutted_model(inputs)

        # print("OUTPUTS:")
        # print(outputs)

        # print("y::")
        # print(y)

        # print("y[0]::")
        # print(y[0])

        # print("AAA:")
        # print(outputs[:, y[0]])

        # print("INPUTS:")
        # print(inputs)

        # y=[i]
        grads = -torch.autograd.grad(outputs[:, y[0]], inputs)[0]
        
        grads = grads.detach().cpu().numpy()

        cutted_model = None
        gc.collect()

        return grads

    def reshape_activations(self, layer_acts):
        return np.asarray(layer_acts).squeeze()

    @abstractmethod
    def label_to_id(self, label):
        pass

    def run_examples(self, examples, bottleneck_name):

        global bn_activation
        bn_activation = None

        def save_activation_hook(mod, inp, out):
            global bn_activation
            bn_activation = out

        handle = self.model._modules[bottleneck_name].register_forward_hook(save_activation_hook)

        self.model.to(device)
        #endurraðar inmputtinum í þessa röð
        #inputs = torch.FloatTensor(examples).permute(0, 3, 1, 2).to(device)
        print("INPUTS TO MODEL:")
        print(examples)
        self.model.eval()
        self.model(examples)
        acts = bn_activation.detach().cpu().numpy()
        handle.remove()

        return acts

## 

class EEGWrapper(ModelWrapper) : 
    """Wrapper base class for eeg models."""
    def __init__(self, eeg_shape, eeg_labels):
        super(ModelWrapper, self).__init__()
        # shape of the input eeg in this model
        self.eeg_shape = eeg_shape
        self.labels = eeg_labels

    def get_eeg_shape(self):
        """returns the shape of an input image."""
        return self.eeg_shape

    def label_to_id(self, label):
        return self.labels.index(label)

class BENDRWrapper(EEGWrapper) : 
    def __init__(self, labels, modelPath):
        eeg_shape = [1, 20, 1536]
        super(BENDRWrapper, self).__init__(eeg_shape=eeg_shape,

                                                 eeg_labels=labels)
        myModel = LinearHeadBENDR(targets = 2, samples = 1536, channels = 20)

        myModel.load(modelPath)
        myModel.eval()

        self.model = myModel

        #self.model = torch.load(modelPath)
        #torch.load("C:/Users/Bex/Desktop/MSc. verkefni/BENDR_Code/EEG_Thesis/BENDR/BENDR_pretrained/BENDR_model_pre.pth")
        self.model_name = 'Linear BENDR'

    def forward(self, x):
        return self.model.features_forward(x)

    def get_cutted_model(self, bottleneck):
        return BENDR_cutted(self.model, bottleneck)
        



class ImageModelWrapper(ModelWrapper):
    """Wrapper base class for image models."""

    def __init__(self, image_shape):
        super(ModelWrapper, self).__init__()
        # shape of the input image in this model
        self.image_shape = image_shape

    def get_image_shape(self):
        """returns the shape of an input image."""
        return self.image_shape


class PublicImageModelWrapper(ImageModelWrapper):
    """Simple wrapper of the public image models with session object.
    """

    def __init__(self, labels_path, image_shape):
        super(PublicImageModelWrapper, self).__init__(image_shape=image_shape)
        self.labels = tf.io.gfile.GFile(labels_path).read().splitlines()

    def label_to_id(self, label):
        return self.labels.index(label)


class BENDR_cutted(torch.nn.Module) : 
    def __init__(self, bendr, bottleneck):
        super(BENDR_cutted, self).__init__()
        names = list(bendr._modules.keys())
        layers = list(bendr.children())

        self.layers = torch.nn.ModuleList()
        self.layers_names = []

        bottleneck_met = False
        for name, layer in zip(names, layers):
            if name == bottleneck:
                bottleneck_met = True
                continue  # because we already have the output of the bottleneck layer
            if not bottleneck_met:
                continue

            self.layers.append(layer)
            self.layers_names.append(name)

    def forward(self, x):
        y = x
        for i in range(len(self.layers)):
            y = self.layers[i](y)

        print("y:::")
        print(y)
        return y


class InceptionV3_cutted(torch.nn.Module):
    def __init__(self, inception_v3, bottleneck):
        super(InceptionV3_cutted, self).__init__()
        names = list(inception_v3._modules.keys())
        layers = list(inception_v3.children())

        self.layers = torch.nn.ModuleList()
        self.layers_names = []

        bottleneck_met = False
        for name, layer in zip(names, layers):
            if name == bottleneck:
                bottleneck_met = True
                continue  # because we already have the output of the bottleneck layer
            if not bottleneck_met:
                continue
            if name == 'AuxLogits':
                continue

            self.layers.append(layer)
            self.layers_names.append(name)

    def forward(self, x):
        y = x
        for i in range(len(self.layers)):
            # pre-forward process
            if self.layers_names[i] == 'Conv2d_3b_1x1':
                y = F.max_pool2d(y, kernel_size=3, stride=2)
            elif self.layers_names[i] == 'Mixed_5b':
                y = F.max_pool2d(y, kernel_size=3, stride=2)
            elif self.layers_names[i] == 'fc':
                y = F.adaptive_avg_pool2d(y, (1, 1))
                y = F.dropout(y, training=self.training)
                y = y.view(y.size(0), -1)

            y = self.layers[i](y)
        return y


class InceptionV3Wrapper(PublicImageModelWrapper):

    def __init__(self, labels_path):
        image_shape = [299, 299, 3]
        super(InceptionV3Wrapper, self).__init__(image_shape=image_shape,
                                                 labels_path=labels_path)
        self.model = torchvision.models.inception_v3(pretrained=True, transform_input=True)
        self.model_name = 'InceptionV3_public'

    def forward(self, x):
        return self.model.forward(x)

    def get_cutted_model(self, bottleneck):
        return InceptionV3_cutted(self.model, bottleneck)
