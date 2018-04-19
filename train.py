import os
import argparse
from utils import OptimizerWrapper
from model import Net
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import numpy as np

def main( args ):
    
    # Set Random Seed: Make Results Reproducible
    torch.manual_seed( args.seed )
    if torch.cuda.is_available():
        torch.cuda.manual_seed( args.seed )

    # Construct Result Model Path based on Hyper-parameters.
    result_path = 'lr-%f_lrdecay-%d_decayrate-%f_weightdecay-%f_optimizer-%s'%\
                  ( args.learning_rate, args.learning_rate_decay_every, args.decay_rate, 
                    args.weight_decay, args.optimizer )
    
    # Image Transform Configuration
    # transform after dataloader sampling
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) ) ] )

    # Create Result Folder    
    if not os.path.exists( result_path ):
        os.makedirs( result_path )
    else:
        print 'This experiment has already been done before.'

    # Train & Test dataloader and Category Names
    trainset = torchvision.datasets.CIFAR10( root=args.data, train=True, download=True, transform=transform )
    trainloader = torch.utils.data.DataLoader( trainset, batch_size=args.batch_size, shuffle=True, 
                                               num_workers=4, drop_last=True )

    testset = torchvision.datasets.CIFAR10( root=args.data, train=False, download=True, transform=transform )
    testloader = torch.utils.data.DataLoader( testset, batch_size=args.batch_size, shuffle=False, 
                                              num_workers=4, drop_last=True )

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
                           
    # Initialize Image Classifiers
    # normal: typical SGD training
    # compare: full-batch vs. mini-batch comparison
    net_normal = Net()
    net_compare = Net()
    print 'Network initialized.'
    
    # CrossEntropy Loss for Classification
    criterion = nn.CrossEntropyLoss()
    
    # Enable GPU training
    if args.cuda and torch.cuda.is_available():
        net_normal.cuda()
        net_compare.cuda()
        criterion.cuda()

    # Parallel with Multiple GPUs   
    if args.cuda and torch.cuda.device_count() > 1:
        net_normal = nn.DataParallel( net_normal )
        net_compare = nn.DataParallel( net_compare )

    # Log Interval for Running loss 
    interval = int( args.log_rate * len( trainloader ) )
    
    print 'Total # of Mini Batches:', len( trainloader )
    for epoch in range( 1, args.epochs+1 ):  # loop over the dataset multiple times

        # Learning rate decay + Optimizer wrapper
        # Only wrap for normal for training phases definition
        optimizer = OptimizerWrapper( net_normal.parameters(), epoch, args )

        # Load Parameters of normal SGD to the compared one
        # To make full batch and mini-batch computation on the same parameter
        if args.cuda and torch.cuda.device_count() > 1:
            net_compare.module.load_state_dict( net_normal.module.state_dict() )
        else:
            net_compare.load_state_dict( net_normal.state_dict() )

        # Keep a record of running loss
        running_loss = 0.0

        # Cache for Mini-batch Gradients in One Epoch
        gradient_caches = []

        # Iterate through batch samples
        for idx, data in enumerate( trainloader, 1 ):

            # get one mini-batch
            inputs, labels = data
            
            # Cache data
            if idx == 1:
                data_caches = inputs.unsqueeze( 0 )
            else:
                data_caches = torch.cat( ( data_caches, inputs.unsqueeze( 0 ) ), dim=0 )

            # Wrap them in Variables
            inputs, labels = Variable( inputs ), Variable( labels )
            if args.cuda and torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            # Zero the parameter gradients
            net_normal.zero_grad()
            net_compare.zero_grad()

            # Net_normal: forward + backward + optimize
            outputs = net_normal( inputs )
            loss = criterion( outputs, labels )
            loss.backward()
            optimizer.step()

            # Record loss values
            running_loss += loss.data[0]

            # Net_compare: forward + backward
            # only compute gradients on current starting theta
            outputs = net_compare( inputs )
            loss = criterion( outputs, labels )
            loss.backward()

            # Retrieve and Cache Gradients
            # Make it a 1 x Param_size vector
            batch_grads = [ x.grad.data.view( 1, -1 ) for x in net_compare.parameters() ]
            batch_grads = torch.cat( batch_grads, dim=1 )
            
            # Cache this gradient
            gradient_caches.append( batch_grads )

            # Print every interval mini-batches
            if idx % interval == 0:
                print '[%d/%5d] Running loss: %.5f'%( epoch, idx, running_loss / interval )
                running_loss = 0.0
        
        # Compute the gradients of full batch
        grads = torch.cat( gradient_caches, dim=0 )
        full_grads = torch.mean( grads, dim=0, keepdim=True )
        
        # Find the Best Mini-batch
        difference = grads - full_grads
        difference = difference.split( 1, dim=0 )
        distance = [ diff.norm() for diff in difference ]
        
        # Select the best one and save corresponding image batch
        # Nearest, smallest distance
        batch_idx = np.argsort( distance )[0]
        best_batch = data_caches[ batch_idx ]
        
        # Unnormalize and Save Grid Images
        best_batch = best_batch / 2 + 0.5
        path = os.path.join( result_path, '%d.jpg'%(epoch) )
        torchvision.utils.save_image( best_batch, filename=path, nrow=4 )

    print 'Training Done.'
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument( '-f', default='self', help='To make it runnable in jupyter' )
    
    # General Detail
    parser.add_argument( '--seed', type=int, default=123,
                         help='random seed to make result reproducible' ) 
    parser.add_argument( '--data', type=str, default='./data',
                          help='path to training data' )
    
    # Training Configuration
    parser.add_argument( '--cuda', type=bool, default=True,
                         help='whether to enable to GPU training' )
    parser.add_argument( '--batch_size', type=int, default=16, 
                         help='training batch size' )
    parser.add_argument( '--num_workers', type=int, default=4,
                         help='number of workers for data loader' )  
    parser.add_argument( '--epochs', type=int, default=100, 
                         help='number of epochs to be trained' )
    parser.add_argument( '--log_rate', type=int, default=0.1,
                         help='log step for printing progress' )                             
    parser.add_argument( '--learning_rate', type=float, default=0.01,
                         help='initial learning rate' )
    parser.add_argument( '--learning_rate_decay_every', type=int, default=10,
                         help='decays learning rate at this frequency' )
    parser.add_argument( '--decay_rate', type=float, default=0.5,
                         help='learning rate decay rate' )
    parser.add_argument( '--clip', type=float, default=0.5,
                         help='gradient norm clipping to avoid explosion' )
    parser.add_argument( '--weight_decay', type=float, default=0,
                         help='weight_decay to prevent overfitting' )
    parser.add_argument( '--early', type=int, default=50,
                         help='at which epoch to trigger early stopping' )  
    parser.add_argument( '--optimizer', type=str, default='adam',
                         help='gradient descent optimizer: sgd, adam, adagrad, rmsprop, adadelta' )  
    
    # ------------------------------------Optimizer----------------------------------------
    # SGD + momentum                         
    parser.add_argument( '--momentum', type=float, default=0.9, 
                         help='momentum parameter in SGD with momentum')
                             
    # Adam
    parser.add_argument( '--beta1', type=float, default=0.9, 
                         help='beta1 in Adam')
    parser.add_argument( '--beta2', type=float, default=0.99, 
                         help='beta2 in Adam')                       
    # AdaGrad
    parser.add_argument( '--lr_decay', type=float, default=0.1, 
                         help='learning rate decay in Adagrad') 
    parser.add_argument( '--alpha', type=float, default=0.99, 
                         help='alpha in RMSProp algo')                            
    # AdaDelta
    parser.add_argument( '--rho', type=float, default=0.9, 
                         help='rho in AdaDelta algo')                           
       
    args = parser.parse_args()
    
    print '-------------------------------Model and Training Details---------------------------------'
    print args
    
    # Start training
    main( args )
