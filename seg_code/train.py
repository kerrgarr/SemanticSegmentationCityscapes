import torch
import numpy as np
import torchvision.transforms as T
from . import special_transforms as SegT
from .models import UNet, my_FCN, DeepLabv3_plus, save_model, model_factory
from .utils import load_data, FOCAL_LOSS_WEIGHTS, ConfusionMatrix, N_CLASSES
import torch.utils.tensorboard as tb
import time


def train(args):
    from os import path
    model = model_factory[args.model]()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)


    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    if args.continue_training:
        from os import path
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % args.model)))

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    w = torch.as_tensor(FOCAL_LOSS_WEIGHTS)**(-args.gamma)
    #loss = torch.nn.CrossEntropyLoss(weight=w / w.mean()).to(device) ### focal_loss
    
    loss = torch.nn.CrossEntropyLoss().to(device) ### without focal loss

    transform= SegT.Compose([ SegT.RandomHorizontalFlip(),SegT.RandomResizedCrop(64),
                                        SegT.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), SegT.ToTensor() ])
    
    train_data = load_data('small_dataset/train', num_workers=4, transform=transform)
    valid_data = load_data('small_dataset/valid', num_workers=4)

    global_step = 0
    start = time.time()
    for epoch in range(args.num_epoch):
        print("epoch", epoch)
        model.train()
        conf = ConfusionMatrix()
        for img, label, msk in train_data:
            #print(img.shape, label.shape, msk.shape)
            img, label = img.to(device), label.to(device).long()

            logit = model(img)
            #print(img.shape) # torch.Size([32, 3, 64, 64])
            #print(label.shape) # torch.Size([32, 64, 64])
            #print(logit.shape) # torch.Size([32, 20, 24, 24])

            loss_val = loss(logit, label)
            if train_logger is not None and global_step % 100 == 0:
                log(train_logger, img, label, logit, global_step)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)
            conf.add(logit.argmax(1), label)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        if train_logger:
            train_logger.add_scalar('global_accuracy', conf.global_accuracy, global_step)
            train_logger.add_scalar('average_accuracy', conf.average_accuracy, global_step)
            train_logger.add_scalar('iou', conf.iou, global_step)
            
        
        
        model.eval()
        val_conf = ConfusionMatrix()
        for img, label, msk in valid_data:
            img, label = img.to(device), label.to(device).long()
            logit = model(img)
            val_conf.add(logit.argmax(1), label)

        if valid_logger is not None:
            log(valid_logger, img, label, logit, global_step)

        if valid_logger:
            valid_logger.add_scalar('global_accuracy', val_conf.global_accuracy, global_step)
            valid_logger.add_scalar('average_accuracy', val_conf.average_accuracy, global_step)
            valid_logger.add_scalar('iou', val_conf.iou, global_step)

        #if valid_logger is None or train_logger is None:
        print('epoch %-3d \t acc = %0.3f \t val acc = %0.3f \t iou = %0.3f \t val iou = %0.3f' %
              (epoch, conf.global_accuracy, val_conf.global_accuracy, conf.iou, val_conf.iou))
        save_model(model)
        
    end = time.time()    
    print("Training time for this model: {:3.2f} minutes".format((end - start )/60))

def log(logger, imgs, lbls, logits, global_step):
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-m', '--model', choices=['my_fcn', 'unet', 'deeplab'], default='my_fcn')
    parser.add_argument('-n', '--num_epoch', type=int, default=20)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-g', '--gamma', type=float, default=0, help="class dependent weight for cross entropy")
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='SegT.Compose([SegT.RandomResizedCrop(64), SegT.ToTensor()])')

    args = parser.parse_args()
    train(args)
