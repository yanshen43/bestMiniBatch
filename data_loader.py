import torch
import numpy as np
import csv
import string
import os
from PIL import Image
import torch.utils.data as data
from torch.autograd import Variable
import nltk
from nltk.tokenize import sent_tokenize

# Dataset for training purpose
class YelpDataset( data.Dataset ):
    def __init__(self, root, tokenized_reviews, img_root, vocab, img_num=3, transform=None ):
        """
        Format compatible with Yelp Reviews
        root: dataset file path.
        tokenized: list of lists of tokens by PTB tokenizer
        vocab: vocabulary wrapper.
        img_num: maximum images to be considered
        transform: transformation applied on raw image 
        """
        
        self.root = root
        self.tokenized_reviews = tokenized_reviews
        self.vocab = vocab
        self.img_root = img_root
        
        # load raw review records
        self.reviews = [ review for review in 
                         csv.reader( open( root, 'r' ), delimiter=',', 
                                     quoting=csv.QUOTE_MINIMAL ) ] # reviews

        # Image2Text Mapping
        # img_id -> review_id
        self.img2txt = {}
        self.imgs = []
        
        # Image ID count
        img_id = 0
        for idx, review in enumerate( self.reviews ):
            for img_filename in review[ 9 ].split( ',' )[ :img_num ]:
                self.img2txt[ img_id ] = idx
                self.imgs.append( img_filename )
                img_id += 1  
        
        # Image IDs: for getitem to iterate
        self.ids = range( img_id )     
        self.transform = transform

    def __len__( self ):
        """
        Iterate through images
        """
        return len( self.ids )

    def __getitem__( self, index ):
        """Returns one data pair ( sentences, image )."""
        
        vocab = self.vocab
        
        # Get image id and corresponding review id
        image_id = self.ids[ index ]
        
        review_id = self.img2txt[ image_id ]
        
        # review text content
        review = self.tokenized_reviews[ review_id ]
        
        # review overall rating
        try:
            rating = float( self.reviews[ review_id ][ 3 ] )
        except Exception:
            rating = 0

        # Binarize the Rating into sentiment
        if rating > 3.0:
            sentiment = 1
        else:
            sentiment = 0

        # image filename
        path = self.imgs[ image_id ]

        # Encode Review based on Sentence segment and token seq
        Input = []
        for sent in review:
            
            # tokenization
            tokens = sent
            sent = []
            
            # Get index for each word
            sent.extend( [ vocab( token ) for token in tokens ] )
            if len( sent ) > 0:
                Input.append( sent )
        
        # No need to sort the sentence w.r.t. length as it's order sensitive 
        Length = [ len( sent ) for sent in Input ]

        assert len( Length ) > 0, "Empty record found" # Check to see there is no empty reviews
        
        # Construct Review Tensor
        TXT = torch.zeros( len( Length ) , max( Length ) ).long()
        for i, sent in enumerate( Input ):
            end = Length[ i ]
            TXT[ i, :end ] = torch.Tensor( sent )
            
        # Construct Image Tensor
        IMAGE = Image.open( os.path.join( self.img_root, path ) ).convert('RGB')
        if self.transform is not None:
            IMAGE = self.transform( IMAGE )
            
        return TXT, Length, IMAGE, sentiment
    
# Dataset for evaluation purpose
class YelpEvalDataset( YelpDataset ):
    def __init__(self, root, tokenized_reviews, img_root, vocab, img_num=3, transform=None ):
        """
        Format compatible with Yelp Reviews
        root: dataset file path.
        vocab: vocabulary wrapper.
        img_num: maximum images to be considered
        transform: transformation applied on raw image 
        """
        super( YelpEvalDataset, self ).__init__(self, root, tokenized_reviews, img_root, vocab, img_num=3, transform=None )

    def __getitem__( self, index ):
        """Returns one data pair ( sentences, image )."""
        
        vocab = self.vocab
        
        # Get image id and corresponding review id
        image_id = self.ids[ index ]
        
        review_id = self.img2txt[ image_id ]
        
        # review text content
        review = self.tokenized_reviews[ review_id ]

        # image filename
        path = self.imgs[ image_id ]

        # Encode Review based on Sentence segment and token seq
        Input = []
        for sent in review:
            
            # tokenization
            tokens = sent
            sent = []
            
            # Get index for each word
            sent.extend( [ vocab( token ) for token in tokens ] )
            if len( sent ) > 0:
                Input.append( sent )
        
        # No need to sort the sentence w.r.t. length as it's order sensitive 
        Length = [ len( sent ) for sent in Input ]

        assert len( Length ) > 0, "Empty record found" # Check to see there is no empty reviews
        
        # Construct Review Tensor
        TXT = torch.zeros( len( Length ) , max( Length ) ).long()
        for i, sent in enumerate( Input ):
            end = Length[ i ]
            TXT[ i, :end ] = torch.Tensor( sent )
            
        # Construct Image Tensor
        IMAGE = Image.open( os.path.join( self.img_root, path ) ).convert('RGB')
        if self.transform is not None:
            IMAGE = self.transform( IMAGE )
            
        return TXT, Length, IMAGE, path, review_id

def collate_fn( data ):
    """Creates mini-batch tensors from the 
    list of tuples ( Review Tensor, Sent Length, Images, Sentiments ).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging tensor with variable length is not supported in default.

    Args:
        data: list of tuples ( sentences, Length, images, sentiments ). 
    """
    
    # Unzip
    TXTs, Lengths, IMAGEs, sentiments = zip( *data ) # unzip

    # Get MAX sent number, MAX sentence token number
    max_sent_num = 0
    max_word_num = 0
    for len_list in Lengths:
        if len( len_list ) > max_sent_num:
            max_sent_num = len( len_list )
            
        if max( len_list ) > max_word_num:
            max_word_num = max( len_list )
    
    # Merge Review Tensor
    # Sentence last token position
    INPUT = torch.zeros( len( TXTs ), max_sent_num, max_word_num ).long()
    SENT_LEN = torch.zeros( len( TXTs ), max_sent_num ).long()
    SENT_POS = torch.zeros( len( TXTs ), max_sent_num ).long()
    for i, review in enumerate( TXTs ):
        num_sent = len( Lengths[ i ] )
        end = max( Lengths[ i ] )
        INPUT[ i, :num_sent, :end ] = review
        SENT_LEN[ i, :num_sent ] = torch.LongTensor( Lengths[ i ] )
        SENT_POS[ i, :num_sent ] = torch.LongTensor( range( 1, num_sent+1 ) )
        
    # Merge Image Tensor and Sentiments
    IMAGES = torch.stack( IMAGEs )
    SENTIMENTS = torch.from_numpy( np.array( sentiments ).T ).long()
        
    return INPUT, SENT_LEN, SENT_POS, IMAGES, SENTIMENTS

def collate_eval_fn( data ):
    """Creates mini-batch tensors from the 
    list of tuples ( Review Tensor, Sent Length, Images, Sentiments ).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging tensor with variable length is not supported in default.

    Args:
        data: list of tuples ( sentences, Length, images, sentiments ). 
    """
    
    # Unzip
    TXTs, Lengths, IMAGEs, paths, ids = zip( *data ) # unzip

    # Get MAX sent number, MAX sentence token number
    max_sent_num = 0
    max_word_num = 0
    for len_list in Lengths:
        if len( len_list ) > max_sent_num:
            max_sent_num = len( len_list )
            
        if max( len_list ) > max_word_num:
            max_word_num = max( len_list )
    
    # Merge Review Tensor
    # Sentence last token position
    INPUT = torch.zeros( len( TXTs ), max_sent_num, max_word_num ).long()
    SENT_LEN = torch.zeros( len( TXTs ), max_sent_num ).long()
    SENT_POS = torch.zeros( len( TXTs ), max_sent_num ).long()
    for i, review in enumerate( TXTs ):
        num_sent = len( Lengths[ i ] )
        end = max( Lengths[ i ] )
        INPUT[ i, :num_sent, :end ] = review
        SENT_LEN[ i, :num_sent ] = torch.LongTensor( Lengths[ i ] )
        SENT_POS[ i, :num_sent ] = torch.LongTensor( range( 1, num_sent+1 ) )
        
    # Merge Image Tensor and Sentiments
    IMAGES = torch.stack( IMAGEs )
        
    return INPUT, SENT_LEN, SENT_POS, IMAGES, list( paths ), list( ids )

def get_loader( root, tokenized_reviews, img_root, vocab,
                batch_size, shuffle, num_workers, img_num=3, transform=None ):
    
    # Initialize Yelp Dataset
    yelp_data = YelpDataset( root=root, tokenized_reviews=tokenized_reviews, img_root=img_root,
                             vocab=vocab, img_num=img_num, transform=transform )
    
    data_loader = torch.utils.data.DataLoader( dataset=yelp_data, 
                                               batch_size=batch_size,
                                               shuffle=shuffle,
                                               num_workers=num_workers,
                                               collate_fn=collate_fn )
    return data_loader

def get_eval_loader( root, tokenized_reviews, img_root, vocab,
                     batch_size, shuffle, num_workers, img_num=3, transform=None ):
    
    # Initialize Yelp Dataset
    yelp_data = YelpEvalDataset( root=root, tokenized_reviews=tokenized_reviews, img_root=img_root,
                                 vocab=vocab, img_num=img_num, transform=transform )
    
    data_loader = torch.utils.data.DataLoader( dataset=yelp_data, 
                                               batch_size=batch_size,
                                               shuffle=shuffle,
                                               num_workers=num_workers,
                                               collate_fn=collate_eval_fn )
    return data_loader
