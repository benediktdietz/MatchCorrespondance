import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

running_window = 5
plotlinewidth = 2

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

fastsorting = np.load('fastsorting.npy')
nosort = np.load('nosort.npy')
sort2 = np.load('sort2.npy')
sort2withvar = np.load('sort2withvar.npy')
sorting = np.load('sorting.npy')
nosortwithvar = np.load('nosortwithvar.npy')

maxiter = 64000
maxiter_fastsorting = int(1000 * maxiter / 104200)
maxiter_nosort = int(1000 * maxiter / 64000)
maxiter_sort2 = int(1000 * maxiter / 69200)
maxiter_sort2withvar = int(1000 * maxiter / 84000)
maxiter_sorting = int(1000 * maxiter / 85200)
maxiter_nosortwithvar = int(1000 * maxiter / 93000)

keys = fastsorting.item().keys()
# ['err95recall_val', 'errmargin_val', 'errdist_val', 'valloss', 'matchloss_val', 'nonmatchloss_val', 'train_loss', 'train_match_loss', 'train_nonmatch_loss', 'full_match_dist', 'full_non_match_dist', 'diff_match', 'diff_non_match']




# plt.figure()

# plt.subplot(212)

# plot_feat = 'valloss'
# plt.title('Validation Loss')
# plt.plot(np.linspace(0, 150, np.asarray(nosort.item().get(plot_feat)).shape[0]), nosort.item().get(plot_feat), linewidth=plotlinewidth, label='No additional processing')
# # plt.plot(np.linspace(0, 150, np.asarray(sort2.item().get(plot_feat)).shape[0]), sort2.item().get(plot_feat), linewidth=plotlinewidth, label='1/2 sampled, LR=0.01')
# plt.plot(np.linspace(0, 150, np.asarray(sorting.item().get(plot_feat)).shape[0]), sorting.item().get(plot_feat), linewidth=plotlinewidth, label='1/2 sampled')
# # plt.plot(np.linspace(0, 150, np.asarray(fastsorting.item().get(plot_feat)).shape[0]), fastsorting.item().get(plot_feat), linewidth=plotlinewidth, label='1/8 sampled')
# # plt.plot(np.linspace(0, 75, np.asarray(sort2withvar.item().get(plot_feat)).shape[0]), sort2withvar.item().get(plot_feat), linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.grid()
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Loss')

# plt.subplot(221)

# plot_feat = 'matchloss_val'
# plt.title('Matches')
# plt.plot(np.linspace(0, 150, np.asarray(nosort.item().get(plot_feat)).shape[0]), nosort.item().get(plot_feat), linewidth=plotlinewidth, label='No additional processing')
# # plt.plot(np.linspace(0, 150, np.asarray(sort2.item().get(plot_feat)).shape[0]), sort2.item().get(plot_feat), linewidth=plotlinewidth, label='1/2 sampled, LR=0.01')
# plt.plot(np.linspace(0, 150, np.asarray(sorting.item().get(plot_feat)).shape[0]), sorting.item().get(plot_feat), linewidth=plotlinewidth, label='1/2 sampled')
# # plt.plot(np.linspace(0, 150, np.asarray(fastsorting.item().get(plot_feat)).shape[0]), fastsorting.item().get(plot_feat), linewidth=plotlinewidth, label='1/8 sampled')
# # plt.plot(np.linspace(0, 75, np.asarray(sort2withvar.item().get(plot_feat)).shape[0]), sort2withvar.item().get(plot_feat), linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.grid()
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Error Rate [%]')

# plt.subplot(222)
# plot_feat = 'nonmatchloss_val'
# plt.title('Non-Matches')
# plt.plot(np.linspace(0, 150, np.asarray(nosort.item().get(plot_feat)).shape[0]), nosort.item().get(plot_feat), linewidth=plotlinewidth, label='No additional processing')
# # plt.plot(np.linspace(0, 150, np.asarray(sort2.item().get(plot_feat)).shape[0]), sort2.item().get(plot_feat), linewidth=plotlinewidth, label='1/2 sampled, LR=0.01')
# plt.plot(np.linspace(0, 150, np.asarray(sorting.item().get(plot_feat)).shape[0]), sorting.item().get(plot_feat), linewidth=plotlinewidth, label='1/2 sampled')
# # plt.plot(np.linspace(0, 150, np.asarray(fastsorting.item().get(plot_feat)).shape[0]), fastsorting.item().get(plot_feat), linewidth=plotlinewidth, label='1/8 sampled')
# # plt.plot(np.linspace(0, 75, np.asarray(sort2withvar.item().get(plot_feat)).shape[0]), sort2withvar.item().get(plot_feat), linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.grid()
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Error Rate [%]')








# plot_feat = 'full_match_dist'
# plt.figure()

# plt.subplot(211)

# plt.title('Match Distance Running Average')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='No additional processing')
# # plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/8 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.grid()
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Distance')

# plot_feat = 'full_non_match_dist'

# plt.subplot(212)

# plt.title('Non-Match Distance Running Average')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='No additional processing')
# # plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/8 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.grid()
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Distance')




# plot_feat = 'train_loss'
# plt.figure()

# plt.subplot(221)
# plt.title('Match Loss')
# plot_feat = 'train_match_loss'
# model = sorting
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(model.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(model.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/2 sampled')
# model = fastsorting
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(model.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(model.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/8 sampled')
# model = sort2withvar
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(model.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(model.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.grid()
# plt.legend()
# # plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Loss')

# plt.subplot(222)
# plt.title('Non-Match Loss')
# plot_feat = 'train_nonmatch_loss'
# model = sorting
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(model.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(model.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/2 sampled')
# model = fastsorting
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(model.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(model.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/8 sampled')
# model = sort2withvar
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(model.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(model.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.grid()
# plt.legend()# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Loss')

# plt.subplot(212)
# plt.title('Total Training Loss')
# plot_feat = 'train_loss'
# model = sorting
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(model.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(model.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/2 sampled')
# model = fastsorting
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(model.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(model.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/8 sampled')
# model = sort2withvar
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(model.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(model.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.grid()
# plt.legend()
# # plt.ylim(0, 10000)
# # plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Loss')

# plot_feat = 'train_loss'
# plt.figure()
# run_win = 20

# plt.subplot(121)

# plt.title('1/8 sampled and Variance Penalty')
# plot_feat = 'train_loss'
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], run_win))), running_mean(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], run_win), c='b', linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], run_win))), running_mean(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], run_win), c='r', linewidth=plotlinewidth, label='1/8 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], run_win))), running_mean(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], run_win), c='g', linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(nosortwithvar.item().get(plot_feat), dtype=np.float32)[:,1], run_win))), running_mean(np.asarray(nosortwithvar.item().get(plot_feat), dtype=np.float32)[:,1], run_win), c='c', linewidth=plotlinewidth, label='Variance Penalty')
# plt.grid()
# plt.ylim(0, 10000)
# # plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Loss')


# plt.subplot(122)

# plt.title('Running Average  Distances')
# plot_feat = 'full_match_dist'
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], run_win))), running_mean(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], run_win), c='b', linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], run_win))), running_mean(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], run_win), c='r', linewidth=plotlinewidth, label='1/8 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], run_win))), running_mean(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], run_win), c='g', linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(nosortwithvar.item().get(plot_feat), dtype=np.float32)[:,1], run_win))), running_mean(np.asarray(nosortwithvar.item().get(plot_feat), dtype=np.float32)[:,1], run_win), c='c', linewidth=plotlinewidth, label='Variance Penalty')
# plot_feat = 'full_non_match_dist'
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], run_win))), running_mean(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], run_win), c='b', linewidth=plotlinewidth)
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], run_win))), running_mean(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], run_win), c='r', linewidth=plotlinewidth)
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], run_win))), running_mean(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], run_win), c='g', linewidth=plotlinewidth)
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(nosortwithvar.item().get(plot_feat), dtype=np.float32)[:,1], run_win))), running_mean(np.asarray(nosortwithvar.item().get(plot_feat), dtype=np.float32)[:,1], run_win), c='c', linewidth=plotlinewidth)
# plt.grid()
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Euclidean Distance')


# plt.figure(figsize=(16,8))
# run_win = 40
# plt.subplot(211)
# plt.title('Running Average Match Distances')
# plot_feat = 'full_match_dist'
# # plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1], run_win))), running_mean(np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1], run_win), linewidth=plotlinewidth, label='No additional processing')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], run_win))), running_mean(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], run_win), linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], run_win))), running_mean(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], run_win), linewidth=plotlinewidth, label='1/8 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], run_win))), running_mean(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], run_win), linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.grid()
# plt.legend()
# plt.ylabel('Euclidean Distance')
# plt.subplot(212)
# plot_feat = 'full_non_match_dist'
# plt.title('Running Average Non-Match Distances')
# # plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1], run_win))), running_mean(np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1], run_win), linewidth=plotlinewidth, label='No additional processing')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], run_win))), running_mean(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], run_win), linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], run_win))), running_mean(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], run_win), linewidth=plotlinewidth, label='1/8 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], run_win))), running_mean(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], run_win), linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.grid()
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Euclidean Distance')


# plot_feat = 'err95recall_val'
# plt.figure()
# plt.title('Error Rate at .95 Recall')
# plt.semilogy(np.linspace(0, 150, running_mean(np.asarray(nosort.item().get(plot_feat)), running_window).shape[0]), running_mean(nosort.item().get(plot_feat), running_window), linewidth=plotlinewidth, label='No additional processing')
# plt.semilogy(np.linspace(0, 150, running_mean(np.asarray(sorting.item().get(plot_feat)), running_window).shape[0]), running_mean(sorting.item().get(plot_feat), running_window), linewidth=plotlinewidth, label='1/2 sampled')
# plt.semilogy(np.linspace(0, 150, running_mean(np.asarray(fastsorting.item().get(plot_feat)), running_window).shape[0]), running_mean(fastsorting.item().get(plot_feat), running_window), linewidth=plotlinewidth, label='1/8 sampled')
# plt.semilogy(np.linspace(0, 150, running_mean(np.asarray(sort2withvar.item().get(plot_feat)), running_window).shape[0]), running_mean(sort2withvar.item().get(plot_feat), running_window), linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.semilogy(np.linspace(0, 150, running_mean(np.asarray(nosortwithvar.item().get(plot_feat)), running_window).shape[0]), running_mean(nosortwithvar.item().get(plot_feat), running_window), linewidth=plotlinewidth, label='Variance Penalty')
# plt.grid()
# plt.legend()
# # plt.ylim(0,2.5)
# plt.xlabel('Epochs')
# plt.ylabel('Error Rate [%]')


# plot_feat = 'err95recall_val'
# plt.figure()
# plt.title('Error Rate at .95 Recall')
# plt.plot(np.linspace(0, 150, np.asarray(nosort.item().get(plot_feat)).shape[0]), nosort.item().get(plot_feat), linewidth=plotlinewidth, label='No additional processing')
# # plt.plot(np.linspace(0, 150, np.asarray(sort2.item().get(plot_feat)).shape[0]), sort2.item().get(plot_feat), linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, np.asarray(sorting.item().get(plot_feat)).shape[0]), sorting.item().get(plot_feat), linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, np.asarray(fastsorting.item().get(plot_feat)).shape[0]), fastsorting.item().get(plot_feat), linewidth=plotlinewidth, label='1/8 sampled')
# plt.plot(np.linspace(0, 150, np.asarray(sort2withvar.item().get(plot_feat)).shape[0]), sort2withvar.item().get(plot_feat), linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.grid()
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Error Rate [%]')


# plot_feat = 'errdist_val'
# plt.figure()
# plt.title(plot_feat)
# plt.plot(np.linspace(0, 150, np.asarray(nosort.item().get(plot_feat)).shape[0]), nosort.item().get(plot_feat), linewidth=plotlinewidth, label='No additional processing')
# # plt.plot(np.linspace(0, 150, np.asarray(sort2.item().get(plot_feat)).shape[0]), sort2.item().get(plot_feat), linewidth=plotlinewidth, label='1/2 sampled, LR=0.01')
# plt.plot(np.linspace(0, 150, np.asarray(sorting.item().get(plot_feat)).shape[0]), sorting.item().get(plot_feat), linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, np.asarray(fastsorting.item().get(plot_feat)).shape[0]), fastsorting.item().get(plot_feat), linewidth=plotlinewidth, label='1/8 sampled')
# plt.plot(np.linspace(0, 150, np.asarray(sort2withvar.item().get(plot_feat)).shape[0]), sort2withvar.item().get(plot_feat), linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.grid()
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Error Rate [%]')


# plot_feat = 'valloss'
# plt.figure()
# plt.title('Validation Loss')
# plt.semilogy(np.linspace(0, 150, np.asarray(nosort.item().get(plot_feat)).shape[0]), nosort.item().get(plot_feat), linewidth=plotlinewidth, label='No additional processing')
# plt.semilogy(np.linspace(0, 150, np.asarray(sorting.item().get(plot_feat)).shape[0]), sorting.item().get(plot_feat), linewidth=plotlinewidth, label='1/2 sampled')
# plt.semilogy(np.linspace(0, 150, np.asarray(fastsorting.item().get(plot_feat)).shape[0]), fastsorting.item().get(plot_feat), linewidth=plotlinewidth, label='1/8 sampled')
# plt.semilogy(np.linspace(0, 150, np.asarray(sort2withvar.item().get(plot_feat)).shape[0]), sort2withvar.item().get(plot_feat), linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.semilogy(np.linspace(0, 150, np.asarray(nosortwithvar.item().get(plot_feat)).shape[0]), nosortwithvar.item().get(plot_feat), linewidth=plotlinewidth, label='Variance Penalty')
# plt.grid()
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Loss')


plot_feat = 'valloss'
plt.figure()
plt.title('Validation Loss')
plt.semilogy(np.linspace(0, 150, running_mean(np.asarray(nosort.item().get(plot_feat)), running_window).shape[0]), running_mean(nosort.item().get(plot_feat), running_window), linewidth=plotlinewidth, label='No additional processing')
plt.semilogy(np.linspace(0, 150, running_mean(np.asarray(sorting.item().get(plot_feat)), running_window).shape[0]), running_mean(sorting.item().get(plot_feat), running_window), linewidth=plotlinewidth, label='1/2 sampled')
plt.semilogy(np.linspace(0, 150, running_mean(np.asarray(fastsorting.item().get(plot_feat)), running_window).shape[0]), running_mean(fastsorting.item().get(plot_feat), running_window), linewidth=plotlinewidth, label='1/8 sampled')
plt.semilogy(np.linspace(0, 150, running_mean(np.asarray(sort2withvar.item().get(plot_feat)), running_window).shape[0]), running_mean(sort2withvar.item().get(plot_feat), running_window), linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
plt.semilogy(np.linspace(0, 150, running_mean(np.asarray(nosortwithvar.item().get(plot_feat)), running_window).shape[0]), running_mean(nosortwithvar.item().get(plot_feat), running_window), linewidth=plotlinewidth, label='Variance Penalty')
plt.grid()
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')

# plot_feat = 'valloss'
# plt.figure()
# plt.title('Validation Loss maxiter')
# plt.semilogy(np.linspace(0, 150, np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:maxiter_nosort].shape[0]), np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:maxiter_nosort], linewidth=plotlinewidth, label='No additional processing')
# plt.semilogy(np.linspace(0, 150, np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:maxiter_sorting].shape[0]), np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:maxiter_sorting], linewidth=plotlinewidth, label='1/2 sampled')
# plt.semilogy(np.linspace(0, 150, np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:maxiter_fastsorting].shape[0]), np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:maxiter_fastsorting], linewidth=plotlinewidth, label='1/8 sampled')
# plt.semilogy(np.linspace(0, 150, np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:maxiter_sort2withvar].shape[0]), np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:maxiter_sort2withvar], linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.semilogy(np.linspace(0, 150, np.asarray(nosortwithvar.item().get(plot_feat), dtype=np.float32)[:maxiter_nosortwithvar].shape[0]), np.asarray(nosortwithvar.item().get(plot_feat), dtype=np.float32)[:maxiter_nosortwithvar], linewidth=plotlinewidth, label='Variance Penalty')
# plt.grid()
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Loss')


# plot_feat = 'matchloss_val'
# plt.figure()
# plt.title(plot_feat)
# plt.plot(np.linspace(0, 150, np.asarray(nosort.item().get(plot_feat)).shape[0]), nosort.item().get(plot_feat), linewidth=plotlinewidth, label='No additional processing')
# # plt.plot(np.linspace(0, 150, np.asarray(sort2.item().get(plot_feat)).shape[0]), sort2.item().get(plot_feat), linewidth=plotlinewidth, label='1/2 sampled, LR=0.01')
# plt.plot(np.linspace(0, 150, np.asarray(sorting.item().get(plot_feat)).shape[0]), sorting.item().get(plot_feat), linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, np.asarray(fastsorting.item().get(plot_feat)).shape[0]), fastsorting.item().get(plot_feat), linewidth=plotlinewidth, label='1/8 sampled')
# plt.plot(np.linspace(0, 150, np.asarray(sort2withvar.item().get(plot_feat)).shape[0]), sort2withvar.item().get(plot_feat), linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.grid()
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Error Rate [%]')


# plot_feat = 'nonmatchloss_val'
# plt.figure()
# plt.title(plot_feat)
# plt.plot(np.linspace(0, 150, np.asarray(nosort.item().get(plot_feat)).shape[0]), nosort.item().get(plot_feat), linewidth=plotlinewidth, label='No additional processing')
# # plt.plot(np.linspace(0, 150, np.asarray(sort2.item().get(plot_feat)).shape[0]), sort2.item().get(plot_feat), linewidth=plotlinewidth, label='1/2 sampled, LR=0.01')
# plt.plot(np.linspace(0, 150, np.asarray(sorting.item().get(plot_feat)).shape[0]), sorting.item().get(plot_feat), linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, np.asarray(fastsorting.item().get(plot_feat)).shape[0]), fastsorting.item().get(plot_feat), linewidth=plotlinewidth, label='1/8 sampled')
# plt.plot(np.linspace(0, 150, np.asarray(sort2withvar.item().get(plot_feat)).shape[0]), sort2withvar.item().get(plot_feat), linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.grid()
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Error Rate [%]')


# plot_feat = 'train_loss'
# plt.figure()
# plt.title(plot_feat)
# plt.plot(np.linspace(0, 150, len(np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='No additional processing')
# # plt.plot(np.linspace(0, 150, len(np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='1/2 sampled, LR=0.01')
# plt.plot(np.linspace(0, 150, len(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, len(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='1/8 sampled')
# plt.plot(np.linspace(0, 150, len(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.grid()
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Error Rate [%]')


# plot_feat = 'train_loss'
# plt.figure()
# plt.title(plot_feat + ' running average')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='No additional processing')
# # plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/8 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.grid()
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Error Rate [%]')


# plot_feat = 'full_match_dist'
# plt.figure()
# plt.title(plot_feat)
# plt.plot(np.linspace(0, 150, len(np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='No additional processing')
# # plt.plot(np.linspace(0, 150, len(np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='1/2 sampled, LR=0.01')
# plt.plot(np.linspace(0, 150, len(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, len(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='1/8 sampled')
# plt.plot(np.linspace(0, 150, len(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.grid()
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Error Rate [%]')


# plot_feat = 'full_match_dist'
# plt.figure()
# plt.title(plot_feat + ' running average')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='No additional processing')
# # plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/8 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.grid()
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Error Rate [%]')


# plot_feat = 'full_non_match_dist'
# plt.figure()
# plt.title(plot_feat)
# plt.plot(np.linspace(0, 150, len(np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='No additional processing')
# # plt.plot(np.linspace(0, 150, len(np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='1/2 sampled, LR=0.01')
# plt.plot(np.linspace(0, 150, len(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, len(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='1/8 sampled')
# plt.plot(np.linspace(0, 150, len(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.grid()
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Error Rate [%]')


# plot_feat = 'full_non_match_dist'
# plt.figure()
# plt.title(plot_feat + ' running average')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='No additional processing')
# # plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/8 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.grid()
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Error Rate [%]')


# plot_feat = 'train_match_loss'
# plt.figure()
# plt.title(plot_feat)
# plt.plot(np.linspace(0, 150, len(np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='No additional processing')
# # plt.plot(np.linspace(0, 150, len(np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='1/2 sampled, LR=0.01')
# plt.plot(np.linspace(0, 150, len(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, len(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='1/8 sampled')
# plt.plot(np.linspace(0, 150, len(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.grid()
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Error Rate [%]')


# plot_feat = 'train_match_loss'
# plt.figure()
# plt.title(plot_feat + ' running average')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='No additional processing')
# # plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/8 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.grid()
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Error Rate [%]')


# plot_feat = 'train_nonmatch_loss'
# plt.figure()
# plt.title(plot_feat)
# plt.plot(np.linspace(0, 150, len(np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='No additional processing')
# # plt.plot(np.linspace(0, 150, len(np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='1/2 sampled, LR=0.01')
# plt.plot(np.linspace(0, 150, len(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, len(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='1/8 sampled')
# plt.plot(np.linspace(0, 150, len(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.grid()
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Error Rate [%]')


# plot_feat = 'train_nonmatch_loss'
# plt.figure()
# plt.title(plot_feat + ' running average')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='No additional processing')
# # plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/8 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.grid()
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Error Rate [%]')


# plot_feat = 'diff_match'
# plt.figure()
# plt.title(plot_feat)
# plt.plot(np.linspace(0, 150, len(np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='No additional processing')
# # plt.plot(np.linspace(0, 150, len(np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='1/2 sampled, LR=0.01')
# plt.plot(np.linspace(0, 150, len(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, len(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='1/8 sampled')
# plt.plot(np.linspace(0, 150, len(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.grid()
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Error Rate [%]')


# plot_feat = 'diff_match'
# plt.figure()
# plt.title(plot_feat + ' running average')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='No additional processing')
# # plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/8 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.grid()
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Error Rate [%]')


# plot_feat = 'diff_non_match'
# plt.figure()
# plt.title(plot_feat)
# plt.plot(np.linspace(0, 150, len(np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='No additional processing')
# # plt.plot(np.linspace(0, 150, len(np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='1/2 sampled, LR=0.01')
# plt.plot(np.linspace(0, 150, len(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, len(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='1/8 sampled')
# plt.plot(np.linspace(0, 150, len(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1])), np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.grid()
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Error Rate [%]')


# plot_feat = 'diff_non_match'
# plt.figure()
# plt.title(plot_feat + ' running average')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(nosort.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='No additional processing')
# # plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(sort2.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(sorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/2 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(fastsorting.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/8 sampled')
# plt.plot(np.linspace(0, 150, len(running_mean(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], running_window))), running_mean(np.asarray(sort2withvar.item().get(plot_feat), dtype=np.float32)[:,1], running_window), linewidth=plotlinewidth, label='1/8 sampled, Variance Penalty')
# plt.grid()
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Error Rate [%]')


plt.show()










