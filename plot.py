import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle


file = open('./loss/test_hits.pkl','rb')
hits = pickle.load(file)
file.close()

hyper_para = {
  'epochs': 40,
  'M': .8, # short
  'N': .2, # long
  'max_len': 712,
  'dropout': 0.5,
  'batch_size': 32,
  'use_tokens': False,
  'verbose': 1,
  'lr': 0.00001,
  'test samples': 500
}

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(14, 10))

for i in range(3):
    t_loss_short =  torch.load('./loss/t_loss_short_rep{}.pt'.format(i))
    v_loss_short =  torch.load('./loss/v_loss_short_rep{}.pt'.format(i))
    t_loss_long =  torch.load('./loss/t_loss_long_rep{}.pt'.format(i))
    v_loss_long =  torch.load('./loss/v_loss_long_rep{}.pt'.format(i))
    g_norm_short = torch.load('./loss/g_norm_short_rep{}.pt'.format(i))
    g_norm_long = torch.load('./loss/g_norm_long_rep{}.pt'.format(i))
    test_loss = torch.load('./loss/test_loss_rep{}.pt'.format(i))

    # short, training
    plt.subplot(2, 4, 1)
    t_loss_short= [x for x in t_loss_short]
    # plt.scatter(np.arange(hyper_para['epochs']) + 1, [l.detach().numpy() for l in t_loss_short])
    plt.plot(np.arange(hyper_para['epochs']) + 1, [l for l in t_loss_short], label='rep {}'.format(i))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('training (short)')
    plt.legend()

    # short, validation
    plt.subplot(2, 4, 2)
    v_loss_short= [x for x in v_loss_short]
    # plt.scatter(np.arange(hyper_para['epochs']) + 1, [l.detach().numpy() for l in v_loss_short])
    plt.plot(np.arange(hyper_para['epochs']) + 1, [l for l in v_loss_short], label='rep {}'.format(i))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('validation (short)')
    plt.legend()

    # long, training
    plt.subplot(2, 4, 3)
    t_loss_long= [x for x in t_loss_long]
    # plt.scatter(np.arange(hyper_para['epochs']) + 1, [l.detach().numpy() for l in t_loss_long])
    plt.plot(np.arange(hyper_para['epochs']) + 1, [l for l in t_loss_long], label='rep {}'.format(i))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('training (long)')
    plt.legend()

    # long, validation
    plt.subplot(2, 4, 4)
    v_loss_long= [x for x in v_loss_long]
    # plt.scatter(np.arange(hyper_para['epochs']) + 1, [l.detach().numpy() for l in v_loss_long])
    plt.plot(np.arange(hyper_para['epochs']) + 1, [l for l in v_loss_long], label='rep {}'.format(i))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('validation (long)')
    plt.legend()

    # short, gradient
    plt.subplot(2, 4, 5)
    g_norm_short= [x for x in g_norm_short]
    # plt.scatter(np.arange(hyper_para['epochs']) + 1, [l.detach().numpy() for l in t_loss_short])
    plt.plot(np.arange(hyper_para['epochs']) + 1, [l for l in g_norm_short], label='rep {}'.format(i))
    plt.xlabel('epoch')
    plt.ylabel('gradient norm')
    plt.title('gradient (short)')
    plt.legend()

    # long, gradient
    plt.subplot(2, 4, 6)
    g_norm_long= [x for x in g_norm_long]
    # plt.scatter(np.arange(hyper_para['epochs']) + 1, [l.detach().numpy() for l in t_loss_short])
    plt.plot(np.arange(hyper_para['epochs']) + 1, [l for l in g_norm_long], label='rep {}'.format(i))
    plt.xlabel('epoch')
    plt.ylabel('gradient norm')
    plt.title('gradient (long)')
    plt.legend()

    # test
    plt.subplot(2, 4, 7)
    plt.scatter(np.arange(len(test_loss)) + 1, [l.to('cpu').detach().numpy() for l in test_loss], label='rep {}'.format(i))
    # plt.plot(np.arange(len(test_loss)) + 1, [l.item() for l in test_loss])
    plt.xlabel('the number of validation samples')
    plt.ylabel('loss')
    plt.title('test')
    plt.legend(loc='upper right')


plt.tight_layout()
plt.savefig('./plots/results.png')
plt.show()
plt.close()