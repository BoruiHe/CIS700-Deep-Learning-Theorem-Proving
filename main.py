from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from cnn_1d_lstm import cnn_1d_lstm
import logging
import os
import dill as pickle
import itertools
import tensorflow.compat.v1 as tf
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import pickle as pk
import data_utils
import random


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('source_dir',
                           'holstep',
                           'Directory where the raw data is located.')
tf.app.flags.DEFINE_string('logdir',
                           '/tmp/hol',
                           'Base directory for saving models and metrics.')
tf.app.flags.DEFINE_integer('samples_per_epoch', 128000,
                            'Number of random step statements to draw for '
                            'training at each epoch.')
tf.app.flags.DEFINE_integer('val_samples', 246912,
                            'Number of (ordered) step statements to draw for '
                            'validation.')
tf.app.flags.DEFINE_integer('epochs', 40,
                            'Number of epochs to train.')
tf.app.flags.DEFINE_integer('data_parsing_workers', 4,
                            'Number of threads to use to generate input data.')

# Taylor: m=.6 and n=.4
# Borui: m=.8 and n=.2
# Yuxuan: m=.2 and n=.8
# Yifu: m=.4 and n=.6
# hyper parameters
hyper_para = {
  'epochs': 40,
  'M': .8, # short
  'N': .2, # long
  'max_len': 512,
  'dropout': 0.5,
  'batch_size': 64,
  'use_tokens': False,
  'verbose': 1,
  'lr': 0.0001,
  'training samples': 1000,
  'split': 0.8, # 0.8 for training and 0.2 for validation
  'test samples': 1000
}
train_split = int(hyper_para['training samples'] * hyper_para['split'])
val_split = hyper_para['training samples'] - train_split

# use GPU if cuda enabled
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)

logging.basicConfig(level=logging.DEBUG)
if not os.path.exists(FLAGS.logdir):
    os.makedirs(FLAGS.logdir)

# save/load vocabulary generated from source directory
if not os.path.exists('vocab'):
    # Parse the training and validation data
    parser = data_utils.DataParser(FLAGS.source_dir, use_tokens=hyper_para['use_tokens'], verbose=hyper_para['verbose'])
    with open('vocab', 'wb') as f:
        pickle.dump(parser, f, pickle.HIGHEST_PROTOCOL)
else:
    with open('vocab', 'rb') as inp:
        parser = pickle.load(inp)

voc_size = len(parser.vocabulary) + 1

# we finished data preprocessing and just read the saved data
# you can run the preprocessing part by deleting .pt files in name_list
name_list = ['short_arrays.pt', 'short_labels.pt', 'long_arrays.pt', 'long_labels.pt', 'val_arrays.pt', 'val_labels.pt']
preprocessed_data_exists = True
for name in name_list:
  preprocessed_data_exists = preprocessed_data_exists and os.path.exists(name)

if not preprocessed_data_exists:
  # generate data for training and validation

  short_train_generator = parser.training_steps_generator(1, max_len=hyper_para['max_len'], batch_size=hyper_para['batch_size'])

  long_train_generator = parser.training_steps_generator(2, max_len=hyper_para['max_len'], batch_size=hyper_para['batch_size'])

  val_generator = parser.validation_steps_generator(max_len=hyper_para['max_len'], batch_size=hyper_para['batch_size'])

  # get encoded arrays from generators

  # limit to number of samples to generate
  max_short = max_long = max(len(parser.short_data), len(parser.long_data))
  max_val = max(len(parser.val_conjectures), hyper_para['test samples'])

  # NOTE: Getting values from these generators requires slicing to an EXPLICIT limit
  # Simply iterating (i.e. "for i in short_train_generator:") can iterate a potentially INFINITE number of times
  # i[0] in list comprehension returns the encoded values of the data
  # i[1] holds labels from generator, if it is needed

  # arrays made from generators
  short_arrays = [i[0] for i in (list(itertools.islice(short_train_generator, 0, max_short)))] # 128000 
  long_arrays = [i[0] for i in (list(itertools.islice(long_train_generator, 0, max_long)))] # 128000
  val_arrays = [i[0] for i in (list(itertools.islice(val_generator, 0, max_val)))] # 246912

  short_labels = [i[1] for i in (list(itertools.islice(short_train_generator, 0, max_short)))]
  long_labels = [i[1] for i in (list(itertools.islice(long_train_generator, 0, max_long)))]
  val_labels = [i[1] for i in (list(itertools.islice(val_generator, 0, max_val)))]

  # numpy arrays -> pytorch tensors
  short_arrays = torch.tensor(short_arrays).to(device)
  short_labels = torch.tensor(short_labels).float().to(device)
  torch.save(short_arrays, 'short_arrays.pt')
  torch.save(short_labels, 'short_labels.pt')

  long_arrays = torch.tensor(long_arrays).to(device)
  long_labels = torch.tensor(long_labels).float().to(device)
  torch.save(long_arrays, 'long_arrays.pt')
  torch.save(long_labels, 'long_labels.pt')

  val_arrays = torch.tensor(val_arrays).to(device)
  val_labels = torch.tensor(val_labels).float().to(device)
  torch.save(val_arrays, 'val_arrays.pt')
  torch.save(val_labels, 'val_labels.pt')
else:
    short_arrays = torch.load('short_arrays.pt').to(device)
    short_labels = torch.load('short_labels.pt').to(device)

    long_arrays = torch.load('long_arrays.pt').to(device)
    long_labels = torch.load('long_labels.pt').to(device)

    val_arrays = torch.load('val_arrays.pt').to(device)
    val_labels = torch.load('val_labels.pt').to(device)

# Run 1d CNN model twice here
# one each for short data and long data
model_short = cnn_1d_lstm(voc_size=voc_size)
model_long = cnn_1d_lstm(voc_size=voc_size)
loss_function = nn.BCELoss()
optimizer = optim.SGD(model_short.parameters(), lr=hyper_para['lr'], momentum=0.9)

def merged_model(val_arrays, val_labels, para_m, para_n, short_model, long_model):
  short_model.eval()
  long_model.eval()
  validation_loss = []
  with torch.no_grad():
    hits = 0
    for _, (inputs, labels) in enumerate(zip(val_arrays, val_labels)):
      prediction = para_m * short_model(inputs) + para_n * long_model(inputs)
      prediction = prediction.squeeze()
      loss = loss_function(prediction, labels)
      

      predicted_lable = torch.tensor([p.detach() for p in prediction])
      predicted_lable[predicted_lable >= 0.5] = 1
      predicted_lable[predicted_lable < 0.5] = 0
      hits += (predicted_lable.to(device) == labels).sum().item()

      ave_loss = loss / (val_arrays.shape[1])
      validation_loss.append(ave_loss)
  return validation_loss, hits

# ______________________Training, validation and test______________________
hits_dict = {}
for rep in range(3):
  torch.cuda.empty_cache()
  t_loss_short = []
  t_loss_long = []
  v_loss_short = []
  v_loss_long = []
  g_norm_short = []
  g_norm_long = []
  print("______________________Training short model " + str(rep) + "______________________")
  if not (os.path.exists('./models/model_short_rep' + str(rep))):
    # train the model for short statements
    for epoch in range(hyper_para['epochs']):
      print("Short", epoch)
      if short_arrays.shape[0] > hyper_para['training samples']:
        # add a random process for smapling training data
        short_samples = random.sample(range(short_arrays.shape[0]), hyper_para['training samples'])
        short_arrays_sam = short_arrays[short_samples]
        short_labels_sam = short_labels[short_samples]
      else:
          short_arrays_sam = short_arrays
          short_labels_sam = short_labels

      model_short.train()
      model_short.to(device)
      loss_sum = 0     
      # 800 for training
      for _, (inputs, labels) in enumerate(zip(short_arrays_sam[:train_split], short_labels_sam[:train_split])):
        model_short.zero_grad
        prediction = model_short(inputs).squeeze()
        loss = loss_function(prediction, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
      ave_loss = loss_sum / (short_arrays.shape[0] * short_arrays.shape[1])
      t_loss_short.append(ave_loss)

      model_short.eval()
      loss_sum = 0
      # 200 for validation
      for _, (inputs, labels) in enumerate(zip(short_arrays_sam[-val_split:], short_labels_sam[-val_split:])):
        torch.cuda.empty_cache()
        prediction = model_short(inputs).squeeze()
        loss = loss_function(prediction, labels)
        loss_sum += loss.item()
      ave_loss = loss_sum / (short_arrays.shape[0] * short_arrays.shape[1])
      v_loss_short.append(ave_loss) 
      
      #getting gradient norm
      total_norm = 0
      for p in model_short.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
      g_norm_short.append(total_norm ** 0.5)

    # save model, loss, and gradient
    torch.save(model_short, './models/model_short_rep{}'.format(rep))
    torch.save(t_loss_short, './loss/t_loss_short_rep{}.pt'.format(rep))
    torch.save(v_loss_short, './loss/v_loss_short_rep{}.pt'.format(rep))
    torch.save(g_norm_short, './loss/g_norm_short_rep{}.pt'.format(rep))

  print("______________________Training long model " + str(rep) + "______________________")
  if not (os.path.exists('./models/model_long_rep' + str(rep))):
    # train the model for long statements
    for epoch in range(hyper_para['epochs']):
      print("Long", epoch)
      if long_arrays.shape[0] > hyper_para['training samples']:
        # add a random process for smapling training data
        long_samples = random.sample(range(long_arrays.shape[0]), hyper_para['training samples'])
        long_arrays_sam = long_arrays[long_samples]
        long_labels_sam = long_labels[long_samples]
      else:
          long_arrays_sam = long_arrays
          long_labels_sam = long_labels

      model_long.train()
      model_long.to(device)
      loss_sum = 0     
      # 800 for training
      for _, (inputs, labels) in enumerate(zip(long_arrays_sam[:train_split], long_labels_sam[:train_split])):
        model_long.zero_grad
        prediction = model_long(inputs).squeeze()
        loss = loss_function(prediction, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
      ave_loss = loss_sum / (long_arrays.shape[0] * long_arrays.shape[1])
      t_loss_long.append(ave_loss)

      model_long.eval()
      loss_sum = 0
      # 200 for validation
      for _, (inputs, labels) in enumerate(zip(long_arrays_sam[-val_split:], long_labels_sam[-val_split:])):
        torch.cuda.empty_cache()
        prediction = model_long(inputs).squeeze()
        loss = loss_function(prediction, labels)
        loss_sum += loss.item()
      ave_loss = loss_sum / (long_arrays.shape[0] * long_arrays.shape[1])
      v_loss_long.append(ave_loss)  

      #getting gradient norm
      total_norm = 0
      for p in model_long.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
      g_norm_long.append(total_norm ** 0.5)

    # save model and loss
    torch.save(model_long, './models/model_long_rep{}'.format(rep))
    torch.save(t_loss_long, './loss/t_loss_long_rep{}.pt'.format(rep))
    torch.save(v_loss_long, './loss/v_loss_long_rep{}.pt'.format(rep))
    torch.save(g_norm_long, './loss/g_norm_long_rep{}.pt'.format(rep))

  print("______________________Test______________________")
  if os.path.exists('./models/model_long_rep{}'.format(rep)) and os.path.exists('./models/model_short_rep{}'.format(rep)):
    model_short = torch.load('./models/model_short_rep{}'.format(rep))
    model_long = torch.load('./models/model_long_rep{}'.format(rep))
  # add a random process for smapling test data
  val_samples = random.sample(range(val_arrays.shape[0]), hyper_para['test samples'])
  val_arrays_sam = val_arrays[val_samples]
  val_labels_sam = val_labels[val_samples]

  test_loss, hits = merged_model(val_arrays_sam, val_labels_sam, hyper_para['M'], hyper_para['N'], model_short, model_long)
  torch.save(test_loss, './loss/test_loss_rep{}.pt'.format(rep))
  print('There are {} (hits)/{} (the num of statements in validation) in repetition {}.'.format(hits, val_arrays_sam.shape[0] * val_arrays_sam.shape[1], rep))
  hits_dict['hits_rep{}'.format(rep)] = hits
  hits_dict['total_rep{}'.format(rep)] = val_arrays_sam.shape[0] * val_arrays_sam.shape[1]

with open('./loss/test_hits.pkl', 'wb') as handle:
  pickle.dump(hits_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
pass