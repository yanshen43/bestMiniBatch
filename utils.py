import math
import torch


def OptimizerWrapper( parameters, epoch, args ):
    
    option = args.optimizer
    assert option in [ '', 'sgd', 'adam', 'adagrad', 'rmsprop', 'adadelta' ], \
    "Please select from sgd, adam, adagrad, rmsprop and adadelta"    
    
    # Calculate Learning Rate for current epoch
    # Using Exponential decay here
    frac = ( epoch - 1 ) / args.learning_rate_decay_every
    decay_factor = math.pow( args.decay_rate, frac )
    learning_rate = args.learning_rate * decay_factor    

    if option == 'adam':
        optimizer = torch.optim.Adam( parameters,
                                      lr=learning_rate,
                                      betas=( args.beta1, args.beta2 ),
                                      weight_decay=args.weight_decay )
    elif option == 'adagrad':
        optimizer = torch.optim.Adagrad( parameters,
                                         lr=learning_rate,
                                         lr_decay=args.lr_decay,
                                         weight_decay=args.weight_decay )    
    elif option == 'rmsprop':
        optimizer = torch.optim.RMSprop( parameters,
                                         lr=learning_rate,
                                         alpha=args.alpha,
                                         weight_decay=args.weight_decay )    
    elif option == 'adadelta':
        optimizer = torch.optim.Adadelta( parameters,
                                          lr=learning_rate,
                                          rho=args.rho,
                                          weight_decay=args.weight_decay ) 
        
    else: # 'sgd' by default
        optimizer = torch.optim.SGD( parameters, 
                                     lr=learning_rate,
                                     momentum=args.momentum,
                                     weight_decay=args.weight_decay )
        
    return optimizer
