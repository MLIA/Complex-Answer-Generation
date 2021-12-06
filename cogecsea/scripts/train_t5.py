"""
Created on Wed May  5 13:07:24 2021

@author: djeddal
"""
import sys 
import os
import torch
import copy
import pandas as pd
import pytorch_lightning as pl
from transformers import T5Tokenizer
import warnings
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import argparse
import numpy as np
import torch.nn.functional as F

from cogecsea.models.T5seq2seq import CustomDataset, T5SeqToSeq


from torch import cuda
MAX_LEN = 1024
MAX_GEN_LEN=512
MAX_OUTLINE_LEN=128


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', dest='project_name', required=True,
                        help='WandB project name')
    parser.add_argument('--train_file', dest='train_file', required=True,
                        help='Train File')
    parser.add_argument('--test_file', dest='test_file', required=True,
                        help='test file')
    parser.add_argument('--output_file', dest='output_file', required=True,
                        help='output file')
    parser.add_argument('--only-outlines', dest='only_outlines', action='store_true',
                        help='plan')
    parser.add_argument('--language-model', dest='language_model', action='store_true',
                        help='using a language model only withoyut support documents')
    parser.add_argument('--end2end', dest='end2end', action='store_true',
                        help='Training end2end')
    parser.add_argument('--check', dest='check', action='store_true',
                        help='is the dataset a sanity check')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--max-input-size', dest='max_input_size', type=int, default=512,
                        help='input max size')
    parser.add_argument('--max-output-size', dest='max_output_size', type=int, default=512,
                        help='output max size')

    parser.add_argument('--epoches', dest='epoches', type=int, default=10,
                        help='Number of epoches')
    parser.add_argument('--train-subset', type=int, dest='train_subset', default=None,
                    help='If used only consider a small data subset for training and validation')
    parser.add_argument('--test-subset', type=int, dest='test_subset', default=None,
                    help='If used only consider a small data subset for testing')
    parser.add_argument('--checkpoints-save', type=str, dest='checkpoints_save', default='./checkpoints/',
                    help='Where to save checkpoints')
    parser.add_argument('--resume-from-checkpoints', type=str, dest='resume_from_checkpoints', default=None,
                    help='Folder to checkpoints')
    parser.add_argument('--gpu-batch-support', type=int, dest='gpu_batch_support', default=1,
                    help='gpu_batch_support')
    parser.add_argument('--training-frac', type=int, dest='training_frac', default=0.95,
                    help='Trtaining fraction (1-val_frac)')

    args = parser.parse_args()

    config = type('config', (object,), {})
    config.TRAIN_BATCH_SIZE = args.batch_size   
    config.VALID_BATCH_SIZE = args.batch_size   
    config.TRAIN_EPOCHS = args.epoches       
    config.VAL_EPOCHS = 1 
    config.LEARNING_RATE = 1e-4    
    config.SEED = 42               
    config.MAX_LEN= args.max_input_size
    config.SUMMARY_LEN = args.max_output_size  
    config.OUTLINE_LEN = MAX_OUTLINE_LEN 
    config.train_subset=args.train_subset

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED) 

    if(args.check==False):
        df_train = pd.read_csv(args.train_file,encoding='latin-1',converters={'outline': eval,'candidats':eval})
        df_train=df_train.loc[df_train['outline'].map(len)>0 ]
        df_train=df_train.loc[df_train['text']!='\n' ]
        df_train=df_train[0:args.train_subset]
        train_size = args.training_frac
        train_dataset=df_train.sample(frac=train_size,random_state = config.SEED)
        val_dataset=df_train.drop(train_dataset.index).reset_index(drop=True)
        train_dataset = train_dataset.reset_index(drop=True)
        
        df_test=pd.read_csv(args.test_file,encoding='latin-1',converters={'outline': eval,'candidats':eval},nrows=args.test_subset)
        
        
        test_dataset= CustomDataset.construct_from_raw(df_test,args.only_outlines,args.language_model,args.end2end)
        train_dataset= CustomDataset.construct_from_raw(train_dataset,args.only_outlines,args.language_model,args.end2end)
        val_dataset= CustomDataset.construct_from_raw(val_dataset,args.only_outlines,args.language_model,args.end2end)
    else:
        df_train = pd.read_csv(args.train_file,encoding='latin-1',converters={'candidats':eval})
        df_test=pd.read_csv(args.test_file,encoding='latin-1',converters={'candidats':eval})
        train_dataset=  CustomDataset.construct_from_raw(df_train)
        val_dataset=  CustomDataset.construct_from_raw(df_test)
        
        
    
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    tokenizer.add_tokens(['[Query:]', '[Documents:]', '[Document:]'])
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("Val Dataset: {}".format(val_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))


    training_set = CustomDataset(train_dataset, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)
    val_set = CustomDataset(val_dataset, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)
    test_set = CustomDataset(test_dataset, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)


    gpus = -1
    nb_available_devices = torch.cuda.device_count()
    accumulation_gradient = args.batch_size // (args.gpu_batch_support * nb_available_devices)
    model=\
    T5SeqToSeq(train_val_test=(training_set,val_set,test_set), 
                train_batch_size=args.gpu_batch_support, 
                val_batch_size=args.gpu_batch_support,
                lr=config.LEARNING_RATE,
                cache_dir=os.path.join(args.checkpoints_save, 'cache'),
                max_input=args.max_input_size, max_output=args.max_output_size,
                model_name= "t5-base",ptokenizer=tokenizer)

    tb_logger = pl_loggers.TensorBoardLogger(args.checkpoints_save,name=args.project_name)
    checkpoint_callback = ModelCheckpoint(monitor="Val/loss_epoch", mode="min",save_top_k=2, every_n_val_epochs=1)

    if(args.resume_from_checkpoints is not None):
        s2s_trainer = Trainer(logger=tb_logger,resume_from_checkpoint=args.resume_from_checkpoints,precision=32,gpus=gpus, 
                            accumulate_grad_batches=accumulation_gradient,accelerator='ddp',
                            max_epochs=args.epoches,progress_bar_refresh_rate=1)
    else:
        s2s_trainer = Trainer(logger=tb_logger,precision=32, gpus=gpus, progress_bar_refresh_rate=1,
                            accumulate_grad_batches=accumulation_gradient, max_epochs=args.epoches,
                            accelerator='ddp', callbacks=[checkpoint_callback])
    s2s_trainer.fit(model)

    eval_trainer = Trainer(gpus=1)
    predictions, references = model.predict(eval_trainer)
    final_df = pd.DataFrame({'Generated Text':predictions, 'Actual Text':references})
    final_df.to_csv(args.output_file)
    print('Output Files generated for review')
    torch.cuda.empty_cache()    