
import os
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.backends.cudnn as cudnn

from torchlib.datasets import tgsdata
from torchlib.segneuralnet import SegmentationNeuralNet

from pytvision.transforms import transforms as mtrans
from pytvision import visualization as view


from argparse import ArgumentParser
import datetime

def arg_parser():
    """Arg parser"""    
    parser = ArgumentParser()
    parser.add_argument('data', metavar='DIR', 
                        help='path to dataset')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('-g', '--gpu', default=0, type=int, metavar='N',
                        help='divice number (default: 0)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', 
                        help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('--snapshot', '-sh', default=10, type=int, metavar='N',
                        help='snapshot (default: 10)')
    parser.add_argument('--project', default='./runs', type=str, metavar='PATH',
                        help='path to project (default: ./runs)')
    parser.add_argument('--name', default='exp', type=str,
                        help='name of experiment')
    parser.add_argument('--resume', default='model_best.pth.tar', type=str, metavar='NAME',
                    help='name to latest checkpoint (default: none)')
    parser.add_argument('--arch', default='simplenet', type=str,
                        help='architecture')
    parser.add_argument('--finetuning', action='store_true', default=False,
                    help='Finetuning')
    parser.add_argument('--loss', default='cross', type=str,
                        help='loss function')
    parser.add_argument('--opt', default='adam', type=str,
                        help='optimize function')
    parser.add_argument('--scheduler', default='fixed', type=str,
                        help='scheduler function for learning rate')
    parser.add_argument('--image-size', default=388, type=int, metavar='N',
                        help='image size')
    parser.add_argument('--parallel', action='store_true', default=False,
                    help='Parallel')
    return parser



def main():
    
    # parameters
    parser = arg_parser()
    args = parser.parse_args()
    imsize = args.image_size
    parallel=args.parallel
    num_classes=2
    num_channels=3
    view_freq=1
    
    cv2.setNumThreads(0)	
    cv2.ocl.setUseOpenCL(False)

    network = SegmentationNeuralNet(
        patchproject=args.project,
        nameproject=args.name,
        no_cuda=args.no_cuda,
        parallel=parallel,
        seed=args.seed,
        print_freq=args.print_freq,
        gpu=args.gpu,
        view_freq=view_freq,
        )
        
    network.create( 
        arch=args.arch, 
        num_output_channels=num_classes, 
        num_input_channels=num_channels,
        loss=args.loss, 
        lr=args.lr, 
        momentum=args.momentum,
        optimizer=args.opt,
        lrsch=args.scheduler,
        pretrained=args.finetuning,
        size_input=imsize
        )
    
    # resume
    network.resume( os.path.join(network.pathmodels, args.resume ) )
    cudnn.benchmark = True

    # datasets
    # training dataset
    train_data = tgsdata.TGSDataset(
        args.data, 
        tgsdata.train, 
        count=16000,
        num_channels=num_channels,
        metadata='metadata_train.csv',
        filter=True,
        transform=transforms.Compose([
            mtrans.ToRandomTransform( mtrans.HFlip(), prob=0.5 ),
            mtrans.ToRandomTransform( mtrans.VFlip(), prob=0.5 ),
            mtrans.RandomScale(factor=0.2, padding_mode=cv2.BORDER_REFLECT_101 ),
            #mtrans.RandomGeometricalTransform( angle=30, translation=0.0, warp=0.02, padding_mode=cv2.BORDER_REFLECT_101),
                        
            #mtrans.RandomElasticDistort( size_grid=16, padding_mode=cv2.BORDER_REFLECT101 ),
            #mtrans.ToRandomTransform( mtrans.RandomBrightness( factor=0.15 ), prob=0.50 ),
            #mtrans.ToRandomTransform( mtrans.RandomContrast( factor=0.15 ), prob=0.50 ),
            #mtrans.ToRandomTransform( mtrans.RandomGamma( factor=0.15 ), prob=0.50 ),
            #mtrans.ToRandomTransform( mtrans.RandomHueSaturation( hue_shift_limit=(-5, 5), sat_shift_limit=(-11, 11), val_shift_limit=(-11, 11) ), prob=0.30 ),
            #mtrans.ToRandomTransform( mtrans.ToGaussianBlur( sigma=0.0001 ), prob=0.15 ),
            #mtrans.ToRandomTransform( mtrans.CLAHE(), prob=0.25 ),
            
            #mtrans.RandomCrop( (102,120), limit=10, padding_mode=cv2.BORDER_REFLECT_101  ),
            mtrans.ToResize( (500,500), resize_mode='squash', padding_mode=cv2.BORDER_REFLECT_101 ),            
            mtrans.ToPad( 6, 6, padding_mode=cv2.BORDER_REFLECT_101 ),
            
            #mtrans.RandomCrop( (256,256), limit=10, padding_mode=cv2.BORDER_REFLECT_101  ),
            #mtrans.ToResize( (256,256), resize_mode='squash', padding_mode=cv2.BORDER_REFLECT_101 ),
            #mtrans.RandomCrop( (101,101), limit=10, padding_mode=cv2.BORDER_REFLECT_101  ),   
                        
            #mtrans.ToResizeUNetFoV(imsize, cv2.BORDER_REFLECT_101),    
            
            mtrans.ToTensor(),
            mtrans.ToMeanNormalization( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], )
            #mtrans.ToNormalization(),
            ])
        )

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.workers, pin_memory=network.cuda, drop_last=True )
    
    # validate dataset
    val_data = tgsdata.TGSDataset(
        args.data, 
        tgsdata.test, 
        #count=4000, 
        num_channels=num_channels,
        metadata='metadata_train.csv',
        filter=True,
        transform=transforms.Compose([
            #mtrans.ToResize( (256,256), resize_mode='squash' ),
            #mtrans.RandomCrop( (255,255), limit=50, padding_mode=cv2.BORDER_CONSTANT  ),
            #mtrans.ToResizeUNetFoV(imsize, cv2.BORDER_REFLECT_101),
            
            mtrans.ToResize( (500,500), resize_mode='squash', padding_mode=cv2.BORDER_REFLECT_101 ),            
            mtrans.ToPad( 6, 6, padding_mode=cv2.BORDER_REFLECT_101 ),
            
            mtrans.ToTensor(),
            mtrans.ToMeanNormalization( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], )
            #mtrans.ToNormalization(), 
            ])
        )

    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.workers, pin_memory=network.cuda, drop_last=True)
       
    # print neural net class
    print('SEG-Torch: {}'.format(datetime.datetime.now()) )
    print(network)

    # training neural net
    network.fit( train_loader, val_loader, args.epochs, args.snapshot )
    
               
    print("Optimization Finished!")
    print("DONE!!!")



if __name__ == '__main__':
    main()