import torch
import numpy as np
import torchvision.transforms as T
from . import special_transforms as SegT
from .models import UNet, my_FCN, DeepLabv3_plus, save_model, model_factory
from .utils import load_data, ConfusionMatrix
import torch.utils.tensorboard as tb


def test(args):
    from os import path
    model = model_factory[args.model]()
    test_logger = None
    if args.log_dir is not None:
        test_logger = tb.SummaryWriter(path.join(args.log_dir, 'test'), flush_secs=1)

    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    
    from os import path
    model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % args.model)))

    test_data = load_data('small_dataset/test', num_workers=4)
    
    global_step = 0
    
    model.eval()
    test_conf = ConfusionMatrix()
    for img, label, msk in test_data:
            img, label = img.to(device), label.to(device).long()
            logit = model(img)
            test_conf.add(logit.argmax(1), label)

    if test_logger is not None:
        log(test_logger, img, label, logit, global_step)

    if test_logger:
        test_logger.add_scalar('global_accuracy', test_conf.global_accuracy, global_step)
        test_logger.add_scalar('average_accuracy', test_conf.average_accuracy, global_step)
        test_logger.add_scalar('iou', test_conf.iou, global_step)

    print('test global acc = %0.3f \t test avg acc = %0.3f \t test iou = %0.3f' %
        (test_conf.global_accuracy, test_conf.average_accuracy, test_conf.iou))
        

def log(logger, imgs, lbls, logits, global_step):
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(SegT.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(SegT.label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-m', '--model', choices=['my_fcn', 'unet', 'deeplab'], default='my_fcn')
    parser.add_argument('-t', '--transform',
                        default='SegT.Compose([SegT.RandomResizedCrop(64), SegT.ToTensor()])')

    args = parser.parse_args()
    
    test(args)
