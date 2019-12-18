import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import itertools
import os
import sys
from sklearn.metrics import confusion_matrix
from matplotlib.patches import Rectangle


DATA_DIR_READ = 'tensorflow_logs/thursday/'
DATA_DIR_WRITE = DATA_DIR_READ + 'embedding/'

threshold = 60
make_embedding = False
loop_iterations = False
last_iteration = 20000
print_freq = 50
num_test = 400



filename_dist = DATA_DIR_READ + 'distances_' + str(last_iteration) + '.npy'
filename_labels = DATA_DIR_READ + 'labels_' + str(last_iteration) + '.npy'
# filename_acc = DATA_DIR + 'accuracies_' + str(i) + '.npy'

output = np.load(filename_dist)
labels = np.load(filename_labels)
# accuracies = np.load(filename_acc)

test_multiplier = output.shape[0]
batch_size = output.shape[1]
cat_size = int(batch_size/4)

full_batch_size = int(batch_size * test_multiplier)

print('test_multiplier......................', test_multiplier)
print('batch_size...........................', batch_size)
print('cat_size.............................', cat_size)



output = np.reshape(output, ((batch_size * test_multiplier), -1))
labels = np.reshape(labels, ((batch_size * test_multiplier), -1))

print('output shape.........................', output.shape)
print('labels shape.........................', labels.shape)


trunks_index = np.arange(cat_size)
branches_index = np.arange(cat_size) + 1*cat_size
leaves_index = np.arange(cat_size) + 2*cat_size
test_index = np.arange(cat_size) + 3*cat_size

trunks_index_vec = trunks_index
branches_index_vec = branches_index
leaves_index_vec = leaves_index
test_index_vec = test_index


if test_multiplier > 1:

	for i in range(1, test_multiplier):

		trunks_index_vec = np.append(
			trunks_index_vec,
			i * batch_size + trunks_index)

		branches_index_vec = np.append(
			branches_index_vec,
			i * batch_size + branches_index)

		leaves_index_vec = np.append(
			leaves_index_vec,
			i * batch_size + leaves_index)

		test_index_vec = np.append(
			test_index_vec,
			i * batch_size + test_index)

cat_labels = []
cat_strings = []

for i in range(0, full_batch_size):

	if labels[i] == 1 and i in trunks_index_vec:
		cat_labels.append(1)
		cat_strings.append('trunk_match')
	if labels[i] == 0 and i in trunks_index_vec:
		cat_labels.append(2)
		cat_strings.append('trunk_non_match')

	if labels[i] == 1 and i in branches_index_vec:
		cat_labels.append(3)
		cat_strings.append('branch_match')
	if labels[i] == 0 and i in branches_index_vec:
		cat_labels.append(4)
		cat_strings.append('branch_non_match')

	if labels[i] == 1 and i in leaves_index_vec:
		cat_labels.append(5)
		cat_strings.append('leaf_match')
	if labels[i] == 0 and i in leaves_index_vec:
		cat_labels.append(6)#'leaf_non_match')
		cat_strings.append('leaf_non_match')

	if labels[i] == 1 and i in test_index_vec:
		cat_labels.append(7)
		cat_strings.append('random_match')
	if labels[i] == 0 and i in test_index_vec:
		cat_labels.append(8)
		cat_strings.append('random_non_match')

cat_labels = np.asarray(cat_labels)
cat_strings = np.asarray(cat_strings)


trunk_match_idx = np.intersect1d(trunks_index_vec, np.where(labels == 1))
trunk_non_match_idx = np.intersect1d(trunks_index_vec, np.where(labels == 0))
t = np.int(np.maximum(len(trunk_match_idx), len(trunk_non_match_idx)))

branch_match_idx = np.intersect1d(branches_index_vec, np.where(labels == 1))
branch_non_match_idx = np.intersect1d(branches_index_vec, np.where(labels == 0))
b = np.int(np.maximum(len(branch_match_idx), len(branch_non_match_idx)))

leaf_match_idx = np.intersect1d(leaves_index_vec, np.where(labels == 1))
leaf_non_match_idx = np.intersect1d(leaves_index_vec, np.where(labels == 0))
l = np.int(np.maximum(len(leaf_match_idx), len(leaf_non_match_idx)))





def performance(distances, labels, threshold):

	l2 = np.reshape(np.mean(distances, axis=1), (-1,1))

	truth_matches = labels == 1.
	truth_matches = truth_matches.astype(np.int32)

	truth_non_matches = labels == 0.
	truth_non_matches = truth_non_matches.astype(np.int32)

	pred_matches = l2 <= threshold
	pred_matches = pred_matches.astype(np.int32)

	pred_non_matches = l2 > threshold
	pred_non_matches = pred_non_matches.astype(np.int32)


	TP = 0
	TN = 0
	FP = 0
	FN = 0

	TP_t = 0
	TN_t = 0
	FP_t = 0
	FN_t = 0

	TP_b = 0
	TN_b = 0
	FP_b = 0
	FN_b = 0

	TP_l = 0
	TN_l = 0
	FP_l = 0
	FN_l = 0




	for i in range(0, distances.shape[0]):

		if truth_matches[i] == 1:

			if pred_matches[i] == 1:

				TP += 1

				if i in trunk_match_idx:

					TP_t += 1

				elif i in branch_match_idx:

					TP_b += 1

				elif i in leaf_match_idx:

					TP_l += 1

			elif pred_non_matches[i] == 1:

				FN += 1

				if i in trunk_match_idx:

					FN_t += 1

				elif i in branch_match_idx:

					FN_b += 1

				elif i in leaf_match_idx:

					FN_l += 1

		elif truth_non_matches[i] == 1:

			if pred_matches[i] == 1:

				FP += 1

				if i in trunk_non_match_idx:

					FP_t += 1

				elif i in branch_non_match_idx:

					FP_b += 1

				elif i in leaf_non_match_idx:

					FP_l += 1

			elif pred_non_matches[i] == 1:

				TN += 1

				if i in trunk_non_match_idx:

					TN_t += 1

				elif i in branch_non_match_idx:

					TN_b += 1

				elif i in leaf_non_match_idx:

					TN_l += 1


	return l2, TP, TN, FP, FN, TP_t, TN_t, FP_t, FN_t, TP_b, TN_b, FP_b, FN_b, TP_l, TN_l, FP_l, FN_l


	# print('---------------')
	# print('TP: ', TP, ' | FP:', FP)
	# print('---------------')
	# print('FN: ', FN, ' | TN:', TN)
	# print('---------------')





l2, TP, TN, FP, FN, TP_t, TN_t, FP_t, FN_t, TP_b, TN_b, FP_b, FN_b, TP_l, TN_l, FP_l, FN_l = performance(output, labels, threshold)



test_thresholds = np.linspace(1, 80, num_test)
TP = np.zeros(num_test)
TN = np.zeros(num_test)
FP = np.zeros(num_test)
FN = np.zeros(num_test)
TP_t = np.zeros(num_test)
TN_t = np.zeros(num_test)
FP_t = np.zeros(num_test)
FN_t = np.zeros(num_test)
TP_b = np.zeros(num_test)
TN_b = np.zeros(num_test)
FP_b = np.zeros(num_test)
FN_b = np.zeros(num_test)
TP_l = np.zeros(num_test)
TN_l = np.zeros(num_test)
FP_l = np.zeros(num_test)
FN_l = np.zeros(num_test)

train_error_rate = np.zeros(num_test)
train_recall = np.zeros(num_test)
train_recall_y = np.zeros(num_test)
train_acc = np.zeros(num_test)
trunk_error_rate = np.zeros(num_test)
trunk_recall = np.zeros(num_test)
trunk_recall_y = np.zeros(num_test)
trunk_acc = np.zeros(num_test)
branch_error_rate = np.zeros(num_test)
branch_recall = np.zeros(num_test)
branch_recall_y = np.zeros(num_test)
branch_acc = np.zeros(num_test)
leaf_error_rate = np.zeros(num_test)
leaf_recall = np.zeros(num_test)
leaf_recall_y = np.zeros(num_test)
leaf_acc = np.zeros(num_test)

thresh_batch = []
thresh_trunk = []
thresh_branch = []
thresh_leaf = []
thresh_batch_y = []
thresh_trunk_y = []
thresh_branch_y = []
thresh_leaf_y = []

for i in range(num_test):

	l2, TP[i], TN[i], FP[i], FN[i], TP_t[i], TN_t[i], FP_t[i], FN_t[i], TP_b[i], TN_b[i], FP_b[i], FN_b[i], TP_l[i], TN_l[i], FP_l[i], FN_l[i] = performance(output, labels, test_thresholds[i])

	train_error_rate[i] = np.round(100 * FP[i] / (FP[i] + TN[i]), 2)
	train_recall[i] = np.round(100 * TP[i] / (TP[i] + FN[i]), 2)
	train_recall_y[i] = np.round(100 * TP[i] / (TP[i] + FP[i]), 2)
	train_acc[i] = np.round(100 * (TP[i] + TN[i]) / (TP[i] + TN[i] + FP[i] + FN[i]), 2)
	if train_recall[i] <= 95.:
		thresh_batch.append(i)
	if train_recall_y[i] >= 95.:
		thresh_batch_y.append(i)

	trunk_error_rate[i] = np.round(100 * FP_t[i] / (FP_t[i] + TN_t[i]), 2)
	trunk_recall[i] = np.round(100 * TP_t[i] / (TP_t[i] + FN_t[i]), 2)
	trunk_recall_y[i] = np.round(100 * TP_t[i] / (TP_t[i] + FP_t[i]), 2)
	trunk_acc[i] = np.round(100 * (TP_t[i] + TN_t[i]) / (TP_t[i] + TN_t[i] + FP_t[i] + FN_t[i]), 2)
	if trunk_recall[i] <= 95.:
		thresh_trunk.append(i)
	if trunk_recall_y[i] >= 95.:
		thresh_trunk_y.append(i)

	branch_error_rate[i] = np.round(100 * FP_b[i] / (FP_b[i] + TN_b[i]), 2)
	branch_recall[i] = np.round(100 * TP_b[i] / (TP_b[i] + FN_b[i]), 2)
	branch_recall_y[i] = np.round(100 * TP_b[i] / (TP_b[i] + FP_b[i]), 2)
	branch_acc[i] = np.round(100 * (TP_b[i] + TN_b[i]) / (TP_b[i] + TN_b[i] + FP_b[i] + FN_b[i]), 2)
	if branch_recall[i] <= 95.:
		thresh_branch.append(i)
	if branch_recall_y[i] >= 95.:
		thresh_branch_y.append(i)
	if branch_recall_y[0] < 95.:
		thresh_branch_y.append(0)

	leaf_error_rate[i] = np.round(100 * FP_l[i] / (FP_l[i] + TN_l[i]), 2)
	leaf_recall[i] = np.round(100 * TP_l[i] / (TP_l[i] + FN_l[i]), 2)
	leaf_recall_y[i] = np.round(100 * TP_l[i] / (TP_l[i] + FP_l[i]), 2)
	leaf_acc[i] = np.round(100 * (TP_l[i] + TN_l[i]) / (TP_l[i] + TN_l[i] + FP_l[i] + FN_l[i]), 2)
	if leaf_recall[i] <= 95.:
		thresh_leaf.append(i)
	if leaf_recall[0] > 95.:
		thresh_leaf.append(0)
	if leaf_recall_y[i] >= 95.:
		thresh_leaf_y.append(i)



thresh_batch = np.asarray(thresh_batch)[-1]
thresh_trunk = np.asarray(thresh_trunk)[-1]
thresh_branch = np.asarray(thresh_branch)[-1]
thresh_leaf = np.asarray(thresh_leaf)[-1]
thresh_batch_y = np.asarray(thresh_batch_y)[-1]
thresh_trunk_y = np.asarray(thresh_trunk_y)[-1]
thresh_branch_y = np.asarray(thresh_branch_y)[-1]
thresh_leaf_y = np.asarray(thresh_leaf_y)[-1]

print(thresh_batch)

conf_mat_batch = np.array([[TP[thresh_batch], FP[thresh_batch]], [FN[thresh_batch], TN[thresh_batch]]])
conf_mat_trunk = np.array([[TP_t[thresh_trunk], FP_t[thresh_trunk]], [FN_t[thresh_trunk], TN_t[thresh_trunk]]])
conf_mat_branch = np.array([[TP_b[thresh_branch], FP_b[thresh_branch]], [FN_b[thresh_branch], TN_b[thresh_branch]]])
conf_mat_leaf = np.array([[TP_l[thresh_leaf], FP_l[thresh_leaf]], [FN_l[thresh_leaf], TN_l[thresh_leaf]]])

conf_mat_batch_y = np.array([[TP[thresh_batch_y], FP[thresh_batch_y]], [FN[thresh_batch_y], TN[thresh_batch_y]]])
conf_mat_trunk_y = np.array([[TP_t[thresh_trunk_y], FP_t[thresh_trunk_y]], [FN_t[thresh_trunk_y], TN_t[thresh_trunk_y]]])
conf_mat_branch_y = np.array([[TP_b[thresh_branch_y], FP_b[thresh_branch_y]], [FN_b[thresh_branch_y], TN_b[thresh_branch_y]]])
conf_mat_leaf_y = np.array([[TP_l[thresh_leaf_y], FP_l[thresh_leaf_y]], [FN_l[thresh_leaf_y], TN_l[thresh_leaf_y]]])


err_batch95 = train_error_rate[thresh_batch]
err_trunk95 = trunk_error_rate[thresh_trunk]
err_branch95 = branch_error_rate[thresh_branch]
err_leaf95 = leaf_error_rate[thresh_leaf]

err_batch95_y = train_error_rate[thresh_batch_y]
err_trunk95_y = trunk_error_rate[thresh_trunk_y]
err_branch95_y = branch_error_rate[thresh_branch_y]
err_leaf95_y = leaf_error_rate[thresh_leaf_y]


def norm(conf_arr):
	norm_conf = []
	for i in conf_arr:
	    a = 0
	    tmp_arr = []
	    a = sum(i, 0)
	    for j in i:
	        tmp_arr.append(float(j)/float(a))
	    norm_conf.append(tmp_arr)
	return norm_conf



def get_stats(DATA_DIR_READ, last_iteration):

	num_runs = int(last_iteration / print_freq)

	full_batch_error_rate_95recall = np.zeros((num_runs))
	full_trunk_error_rate_95recall = np.zeros((num_runs))
	full_branch_error_rate_95recall = np.zeros((num_runs))
	full_leaf_error_rate_95recall = np.zeros((num_runs))

	iteration  = np.linspace(print_freq, last_iteration, num_runs)


	for r in range(num_runs):


		filename_dist = DATA_DIR_READ + 'distances_' + str(int(iteration[r])) + '.npy'
		filename_labels = DATA_DIR_READ + 'labels_' + str(int(iteration[r])) + '.npy'
		# filename_acc = DATA_DIR + 'accuracies_' + str(i) + '.npy'

		output = np.load(filename_dist)
		labels = np.load(filename_labels)
		# accuracies = np.load(filename_acc)

		test_multiplier = output.shape[0]
		batch_size = output.shape[1]
		cat_size = int(batch_size/4)

		full_batch_size = int(batch_size * test_multiplier)

		print('test_multiplier......................', test_multiplier)
		print('batch_size...........................', batch_size)
		print('cat_size.............................', cat_size)



		output = np.reshape(output, ((batch_size * test_multiplier), -1))
		labels = np.reshape(labels, ((batch_size * test_multiplier), -1))

		print('output shape.........................', output.shape)
		print('labels shape.........................', labels.shape)


		trunks_index = np.arange(cat_size)
		branches_index = np.arange(cat_size) + 1*cat_size
		leaves_index = np.arange(cat_size) + 2*cat_size
		test_index = np.arange(cat_size) + 3*cat_size

		trunks_index_vec = trunks_index
		branches_index_vec = branches_index
		leaves_index_vec = leaves_index
		test_index_vec = test_index


		if test_multiplier > 1:

			for t in range(1, test_multiplier):

				trunks_index_vec = np.append(
					trunks_index_vec,
					t * batch_size + trunks_index)

				branches_index_vec = np.append(
					branches_index_vec,
					t * batch_size + branches_index)

				leaves_index_vec = np.append(
					leaves_index_vec,
					t * batch_size + leaves_index)

				test_index_vec = np.append(
					test_index_vec,
					t * batch_size + test_index)

		cat_labels = []
		cat_strings = []

		for w in range(0, full_batch_size):

			if labels[w] == 1 and w in trunks_index_vec:
				cat_labels.append(1)
				cat_strings.append('trunk_match')
			if labels[w] == 0 and w in trunks_index_vec:
				cat_labels.append(2)
				cat_strings.append('trunk_non_match')

			if labels[w] == 1 and w in branches_index_vec:
				cat_labels.append(3)
				cat_strings.append('branch_match')
			if labels[w] == 0 and w in branches_index_vec:
				cat_labels.append(4)
				cat_strings.append('branch_non_match')

			if labels[w] == 1 and w in leaves_index_vec:
				cat_labels.append(5)
				cat_strings.append('leaf_match')
			if labels[w] == 0 and w in leaves_index_vec:
				cat_labels.append(6)#'leaf_non_match')
				cat_strings.append('leaf_non_match')

			if labels[w] == 1 and w in test_index_vec:
				cat_labels.append(7)
				cat_strings.append('random_match')
			if labels[w] == 0 and w in test_index_vec:
				cat_labels.append(8)
				cat_strings.append('random_non_match')

		cat_labels = np.asarray(cat_labels)
		cat_strings = np.asarray(cat_strings)


		trunk_match_idx = np.intersect1d(trunks_index_vec, np.where(labels == 1))
		trunk_non_match_idx = np.intersect1d(trunks_index_vec, np.where(labels == 0))
		t = np.int(np.maximum(len(trunk_match_idx), len(trunk_non_match_idx)))

		branch_match_idx = np.intersect1d(branches_index_vec, np.where(labels == 1))
		branch_non_match_idx = np.intersect1d(branches_index_vec, np.where(labels == 0))
		b = np.int(np.maximum(len(branch_match_idx), len(branch_non_match_idx)))

		leaf_match_idx = np.intersect1d(leaves_index_vec, np.where(labels == 1))
		leaf_non_match_idx = np.intersect1d(leaves_index_vec, np.where(labels == 0))
		l = np.int(np.maximum(len(leaf_match_idx), len(leaf_non_match_idx)))



		l2, TP, TN, FP, FN, TP_t, TN_t, FP_t, FN_t, TP_b, TN_b, FP_b, FN_b, TP_l, TN_l, FP_l, FN_l = performance(output, labels, threshold)



		test_thresholds = np.linspace(10, 80, num_test)
		TP = np.zeros(num_test)
		TN = np.zeros(num_test)
		FP = np.zeros(num_test)
		FN = np.zeros(num_test)
		TP_t = np.zeros(num_test)
		TN_t = np.zeros(num_test)
		FP_t = np.zeros(num_test)
		FN_t = np.zeros(num_test)
		TP_b = np.zeros(num_test)
		TN_b = np.zeros(num_test)
		FP_b = np.zeros(num_test)
		FN_b = np.zeros(num_test)
		TP_l = np.zeros(num_test)
		TN_l = np.zeros(num_test)
		FP_l = np.zeros(num_test)
		FN_l = np.zeros(num_test)

		train_error_rate = np.zeros(num_test)
		train_recall = np.zeros(num_test)
		train_recall_y = np.zeros(num_test)
		train_acc = np.zeros(num_test)
		trunk_error_rate = np.zeros(num_test)
		trunk_recall = np.zeros(num_test)
		trunk_recall_y = np.zeros(num_test)
		trunk_acc = np.zeros(num_test)
		branch_error_rate = np.zeros(num_test)
		branch_recall = np.zeros(num_test)
		branch_recall_y = np.zeros(num_test)
		branch_acc = np.zeros(num_test)
		leaf_error_rate = np.zeros(num_test)
		leaf_recall = np.zeros(num_test)
		leaf_recall_y = np.zeros(num_test)
		leaf_acc = np.zeros(num_test)

		thresh_batch = []
		thresh_trunk = []
		thresh_branch = []
		thresh_leaf = []

		for i in range(num_test):

			l2, TP[i], TN[i], FP[i], FN[i], TP_t[i], TN_t[i], FP_t[i], FN_t[i], TP_b[i], TN_b[i], FP_b[i], FN_b[i], TP_l[i], TN_l[i], FP_l[i], FN_l[i] = performance(output, labels, test_thresholds[i])

			train_error_rate[i] = np.round(100 * FP[i] / (FP[i] + TN[i]))
			train_recall[i] = np.round(100 * TP[i] / (TP[i] + FP[i]))
			train_acc[i] = np.round(100 * (TP[i] + TN[i]) / (TP[i] + TN[i] + FP[i] + FN[i]))
			if train_recall[i] <= 95:
				thresh_batch.append(i)

			trunk_error_rate[i] = np.round(100 * FP_t[i] / (FP_t[i] + TN_t[i]))
			trunk_recall[i] = np.round(100 * TP_t[i] / (TP_t[i] + FP_t[i]))
			trunk_acc[i] = np.round(100 * (TP_t[i] + TN_t[i]) / (TP_t[i] + TN_t[i] + FP_t[i] + FN_t[i]))
			if trunk_recall[i] <= 95:
				thresh_trunk.append(i)

			branch_error_rate[i] = np.round(100 * FP_b[i] / (FP_b[i] + TN_b[i]))
			branch_recall[i] = np.round(100 * TP_b[i] / (TP_b[i] + FP_b[i]))
			branch_acc[i] = np.round(100 * (TP_b[i] + TN_b[i]) / (TP_b[i] + TN_b[i] + FP_b[i] + FN_b[i]))
			if branch_recall[i] <= 95:
				thresh_branch.append(i)

			leaf_error_rate[i] = np.round(100 * FP_l[i] / (FP_l[i] + TN_l[i]))
			leaf_recall[i] = np.round(100 * TP_l[i] / (TP_l[i] + FP_l[i]))
			leaf_acc[i] = np.round(100 * (TP_l[i] + TN_l[i]) / (TP_l[i] + TN_l[i] + FP_l[i] + FN_l[i]))
			if leaf_recall[i] <= 95:
				thresh_leaf.append(i)


		# thresh_batch = np.argmin(np.power((train_recall - .95), 2))]
		# thresh_trunk = np.argmin(np.power((trunk_recall - .95), 2))]
		# thresh_branch = np.argmin(np.power((branch_recall - .95), 2))]
		# thresh_leaf = np.argmin(np.power((leaf_recall - .95), 2))]

		thresh_batch = np.asarray(thresh_batch)[-1]
		thresh_trunk = np.asarray(thresh_trunk)[-1]
		thresh_branch = np.asarray(thresh_branch)[-1]
		thresh_leaf = np.asarray(thresh_leaf)[-1]

		full_batch_error_rate_95recall[r] = train_error_rate[thresh_batch]
		full_trunk_error_rate_95recall[r] = trunk_error_rate[thresh_trunk]
		full_branch_error_rate_95recall[r] = branch_error_rate[thresh_branch]
		full_leaf_error_rate_95recall[r] = leaf_error_rate[thresh_leaf]

		return full_batch_error_rate_95recall, full_trunk_error_rate_95recall, full_branch_error_rate_95recall, full_leaf_error_rate_95recall, iteration

# full_batch_error_rate_95recall, full_trunk_error_rate_95recall, full_branch_error_rate_95recall, full_leaf_error_rate_95recall, iteration = get_stats(DATA_DIR_READ, last_iteration)

if make_embedding == True:

	import tensorflow as tf
	from tensorflow.contrib.tensorboard.plugins import projector

	embedding_var = tf.Variable(tf.zeros(output.shape), name='embedding')


	metadata = os.path.join(DATA_DIR_READ, 'metadata.tsv')

	with open(metadata, 'w') as metadata_file:
	    for row in cat_labels:
	        metadata_file.write('%d\n' % row)




	with tf.Session() as sess:


	    saver = tf.train.Saver([embedding_var])


		### create the tensorboard embedding ###

	    sess.run(embedding_var.initializer)

	    embedding_var.assign(output)

	    # sess.run(embedding_var.initializer)
	    saver.save(sess, os.path.join(DATA_DIR_READ, 'embedding_var.ckpt'))

	    config = projector.ProjectorConfig()
	    # One can add multiple embeddings.
	    embedding = config.embeddings.add()
	    embedding.tensor_name = embedding_var.name
	    embedding.metadata_path = 'metadata.tsv'
	    # Link this tensor to its metadata file (e.g. labels).
	    # embedding.metadata_path = metadata
	    # Saves a config file that TensorBoard will read during startup.
	    projector.visualize_embeddings(tf.summary.FileWriter(DATA_DIR_READ), config)

	    # projector.visualize_embeddings(tf.summary.FileWriter(DATA_DIR), config)

	    # saver.save(sess, os.path.join(DATA_DIR, 'filename.ckpt'))



if loop_iterations == False:
	plt.figure()

	plt.subplot(211)

	plt.scatter(np.arange(len(trunk_match_idx)), l2[trunk_match_idx], label='trunk_match', color='green', marker='o')
	plt.scatter(np.arange(len(trunk_non_match_idx)), l2[trunk_non_match_idx], label='trunk_non_match', color='red', marker='x')
	plt.scatter(t + 20 + np.arange(len(branch_match_idx)), l2[branch_match_idx], label='branch_match', color='green', marker='o')
	plt.scatter(t + 20 + np.arange(len(branch_non_match_idx)), l2[branch_non_match_idx], label='branch__match', color='red', marker='x')
	plt.scatter(t + 20 + b + 20 + np.arange(len(leaf_match_idx)), l2[leaf_match_idx], label='leaf_match', color='green', marker='o')
	plt.scatter(t + 20 + b + 20 + np.arange(len(leaf_non_match_idx)), l2[leaf_non_match_idx], label='leaf_non_match', color='red', marker='x')
	# plt.axhline(y=thresh_batch, xmin=0., xmax=len(labels), linewidth=2, color = 'k')
	# plt.axhline(y=thresh_trunk, xmin=0., xmax=t, linewidth=1, color = 'k')
	# plt.axhline(y=thresh_branch, xmin=t + 20, xmax=b, linewidth=1, color = 'k')
	# plt.axhline(y=thresh_leaf, xmin=t + b + 40, xmax=len(labels), linewidth=1, color = 'k')
	plt.xlabel('trunks   --   branches   --   leaves')
	plt.ylabel('distances')
	plt.grid()

	plt.subplot(234)
	plt.plot(test_thresholds, train_error_rate, label='batch')
	plt.plot(test_thresholds, trunk_error_rate, label='trunk')
	plt.plot(test_thresholds, branch_error_rate, label='branch')
	plt.plot(test_thresholds, leaf_error_rate, label='leaf')
	plt.ylabel('%')
	plt.xlabel('threshold')
	plt.grid()
	plt.legend()
	plt.title('error_rate')

	plt.subplot(235)
	plt.plot(test_thresholds, train_recall, label='batch')
	plt.plot(test_thresholds, trunk_recall, label='trunk')
	plt.plot(test_thresholds, branch_recall, label='branch')
	plt.plot(test_thresholds, leaf_recall, label='leaf')
	plt.axhline(y=95, xmin=0., xmax=test_thresholds[-1], linewidth=1, color = 'k')
	plt.ylabel('%')
	plt.xlabel('threshold')
	plt.legend()
	plt.grid()
	plt.title('recall')

	# plt.subplot(247)
	# plt.plot(test_thresholds, train_recall_y, label='batch')
	# plt.plot(test_thresholds, trunk_recall_y, label='trunk')
	# plt.plot(test_thresholds, branch_recall_y, label='branch')
	# plt.plot(test_thresholds, leaf_recall_y, label='leaf')
	# plt.axhline(y=95, xmin=0., xmax=test_thresholds[-1], linewidth=1, color = 'k')
	# plt.ylabel('%')
	# plt.xlabel('threshold')
	# plt.legend()
	# plt.grid()
	# plt.title('recall_yifei')

	plt.subplot(236)
	plt.plot(test_thresholds, train_acc, label='batch')
	plt.plot(test_thresholds, trunk_acc, label='trunk')
	plt.plot(test_thresholds, branch_acc, label='branch')
	plt.plot(test_thresholds, leaf_acc, label='leaf')
	plt.ylabel('%')
	plt.xlabel('threshold')
	plt.legend()
	plt.grid()
	plt.title('accuracy')

	# plt.subplot(248)
	# plt.plot(iteration ,full_batch_error_rate_95recall, label='batch')
	# plt.plot(iteration, full_trunk_error_rate_95recall, label='trunk')
	# plt.plot(iteration, full_branch_error_rate_95recall, label='branch')
	# plt.plot(iteration, full_leaf_error_rate_95recall, label='leaf')
	# plt.ylabel('%')
	# plt.xlabel('iteration')
	# plt.legend()
	# plt.grid()
	# plt.title('erorr_rate_at_95_recall')


	# fig1 = plt.subplot(349)
	# plt.clf()
	# ax = plt.figure().add_subplot(349)
	# ax.set_aspect(1)
	# res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
	#                 interpolation='nearest')

	# width, height = conf_arr.shape

	# for x in range(width):
	#     for y in range(height):
	#         ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
	#                     horizontalalignment='center',
	#                     verticalalignment='center')

	# # cb = fig1.colorbar(res)
	# alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
	# plt.xticks(range(width), alphabet[:width])
	# plt.yticks(range(height), alphabet[:height])

	alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
	width, height = conf_mat_batch.shape


	fig = plt.figure()
	plt.clf()
	plt.title('Error rates at .95 recall')

	ax1 = fig.add_subplot(241)
	ax1.set_aspect(1)
	res = ax1.imshow(np.array(norm(conf_mat_batch)), cmap=plt.cm.Greens, 
	                interpolation='nearest')


	for x in range(width):
	    for y in range(height):
	        ax1.annotate(str(conf_mat_batch[x][y]), xy=(y, x), 
	                    horizontalalignment='center',
	                    verticalalignment='center')
	plt.xticks(range(width), alphabet[:width])
	plt.yticks(range(height), alphabet[:height])
	plt.title('batch:' + str(err_batch95))

	# cb = fig.colorbar(res)


	ax2 = fig.add_subplot(242)
	ax2.set_aspect(1)
	res = ax2.imshow(np.array(norm(conf_mat_trunk)), cmap=plt.cm.Greens, 
	                interpolation='nearest')


	for x in range(width):
	    for y in range(height):
	        ax2.annotate(str(conf_mat_trunk[x][y]), xy=(y, x), 
	                    horizontalalignment='center',
	                    verticalalignment='center')
	plt.xticks(range(width), alphabet[:width])
	plt.yticks(range(height), alphabet[:height])
	plt.title('trunk:' + str(err_trunk95))

	# cb = fig.colorbar(res)


	ax3 = fig.add_subplot(243)
	ax3.set_aspect(1)
	res = ax3.imshow(np.array(norm(conf_mat_branch)), cmap=plt.cm.Greens, 
	                interpolation='nearest')


	for x in range(width):
	    for y in range(height):
	        ax3.annotate(str(conf_mat_branch[x][y]), xy=(y, x), 
	                    horizontalalignment='center',
	                    verticalalignment='center')
	plt.xticks(range(width), alphabet[:width])
	plt.yticks(range(height), alphabet[:height])
	plt.title('branch:' + str(err_branch95))

	# cb = fig.colorbar(res)


	ax4 = fig.add_subplot(244)
	ax4.set_aspect(1)
	res = ax4.imshow(np.array(norm(conf_mat_leaf)), cmap=plt.cm.Greens, 
	                interpolation='nearest')


	for x in range(width):
	    for y in range(height):
	        ax4.annotate(str(conf_mat_leaf[x][y]), xy=(y, x), 
	                    horizontalalignment='center',
	                    verticalalignment='center')
	plt.xticks(range(width), alphabet[:width])
	plt.yticks(range(height), alphabet[:height])
	plt.title('leaf:' + str(err_leaf95))

	# cb = fig.colorbar(res)


	ax5 = fig.add_subplot(245)
	ax5.set_aspect(1)
	res = ax5.imshow(np.array(norm(conf_mat_batch_y)), cmap=plt.cm.Greens, 
	                interpolation='nearest')


	for x in range(width):
	    for y in range(height):
	        ax5.annotate(str(conf_mat_batch_y[x][y]), xy=(y, x), 
	                    horizontalalignment='center',
	                    verticalalignment='center')
	plt.xticks(range(width), alphabet[:width])
	plt.yticks(range(height), alphabet[:height])
	plt.title('batch (Y):' + str(err_batch95_y))

	# cb = fig.colorbar(res)


	ax6 = fig.add_subplot(246)
	ax6.set_aspect(1)
	res = ax6.imshow(np.array(norm(conf_mat_trunk_y)), cmap=plt.cm.Greens, 
	                interpolation='nearest')


	for x in range(width):
	    for y in range(height):
	        ax6.annotate(str(conf_mat_trunk_y[x][y]), xy=(y, x), 
	                    horizontalalignment='center',
	                    verticalalignment='center')
	plt.xticks(range(width), alphabet[:width])
	plt.yticks(range(height), alphabet[:height])
	plt.title('trunk (Y):' + str(err_trunk95_y))

	# cb = fig.colorbar(res)


	ax7 = fig.add_subplot(247)
	ax7.set_aspect(1)
	res = ax7.imshow(np.array(norm(conf_mat_branch_y)), cmap=plt.cm.Greens, 
	                interpolation='nearest')


	for x in range(width):
	    for y in range(height):
	        ax7.annotate(str(conf_mat_branch_y[x][y]), xy=(y, x), 
	                    horizontalalignment='center',
	                    verticalalignment='center')
	plt.xticks(range(width), alphabet[:width])
	plt.yticks(range(height), alphabet[:height])
	plt.title('branch error (Y):' + str(err_branch95_y))

	# cb = fig.colorbar(res)


	ax8 = fig.add_subplot(248)
	ax8.set_aspect(1)
	res = ax8.imshow(np.array(norm(conf_mat_leaf_y)), cmap=plt.cm.Greens, 
	                interpolation='nearest')


	for x in range(width):
	    for y in range(height):
	        ax8.annotate(str(conf_mat_leaf_y[x][y]), xy=(y, x), 
	                    horizontalalignment='center',
	                    verticalalignment='center')
	plt.xticks(range(width), alphabet[:width])
	plt.yticks(range(height), alphabet[:height])
	plt.title('leaf error (Y):' + str(err_leaf95_y))

	# cb = fig.colorbar(res)



	plt.show()










#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################



















# import numpy as np
# from matplotlib import pyplot as plt
# from sklearn.manifold import TSNE
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits import mplot3d
# import itertools
# import os
# import sys
# from sklearn.metrics import confusion_matrix


# DATA_DIR_READ = 'tensorflow_logs/monday/second/'
# DATA_DIR_WRITE = DATA_DIR_READ + 'embedding/'

# iteration = 1600
# threshold = 60
# make_embedding = False
# loop_iterations = False

# filename_dist = DATA_DIR_READ + 'distances_' + str(iteration) + '.npy'
# filename_labels = DATA_DIR_READ + 'labels_' + str(iteration) + '.npy'
# # filename_acc = DATA_DIR + 'accuracies_' + str(i) + '.npy'

# output = np.load(filename_dist)
# labels = np.load(filename_labels)
# # accuracies = np.load(filename_acc)

# test_multiplier = output.shape[0]
# batch_size = output.shape[1]
# cat_size = int(batch_size/4)

# full_batch_size = int(batch_size * test_multiplier)

# print('test_multiplier......................', test_multiplier)
# print('batch_size...........................', batch_size)
# print('cat_size.............................', cat_size)



# output = np.reshape(output, ((batch_size * test_multiplier), -1))
# labels = np.reshape(labels, ((batch_size * test_multiplier), -1))

# print('output shape.........................', output.shape)
# print('labels shape.........................', labels.shape)


# trunks_index = np.arange(cat_size)
# branches_index = np.arange(cat_size) + 1*cat_size
# leaves_index = np.arange(cat_size) + 2*cat_size
# test_index = np.arange(cat_size) + 3*cat_size

# trunks_index_vec = trunks_index
# branches_index_vec = branches_index
# leaves_index_vec = leaves_index
# test_index_vec = test_index


# if test_multiplier > 1:

# 	for i in range(1, test_multiplier):

# 		trunks_index_vec = np.append(
# 			trunks_index_vec,
# 			i * batch_size + trunks_index)

# 		branches_index_vec = np.append(
# 			branches_index_vec,
# 			i * batch_size + branches_index)

# 		leaves_index_vec = np.append(
# 			leaves_index_vec,
# 			i * batch_size + leaves_index)

# 		test_index_vec = np.append(
# 			test_index_vec,
# 			i * batch_size + test_index)

# cat_labels = []
# cat_strings = []

# for i in range(0, full_batch_size):

# 	if labels[i] == 1 and i in trunks_index_vec:
# 		cat_labels.append(1)
# 		cat_strings.append('trunk_match')
# 	if labels[i] == 0 and i in trunks_index_vec:
# 		cat_labels.append(2)
# 		cat_strings.append('trunk_non_match')

# 	if labels[i] == 1 and i in branches_index_vec:
# 		cat_labels.append(3)
# 		cat_strings.append('branch_match')
# 	if labels[i] == 0 and i in branches_index_vec:
# 		cat_labels.append(4)
# 		cat_strings.append('branch_non_match')

# 	if labels[i] == 1 and i in leaves_index_vec:
# 		cat_labels.append(5)
# 		cat_strings.append('leaf_match')
# 	if labels[i] == 0 and i in leaves_index_vec:
# 		cat_labels.append(6)#'leaf_non_match')
# 		cat_strings.append('leaf_non_match')

# 	if labels[i] == 1 and i in test_index_vec:
# 		cat_labels.append(7)
# 		cat_strings.append('random_match')
# 	if labels[i] == 0 and i in test_index_vec:
# 		cat_labels.append(8)
# 		cat_strings.append('random_non_match')

# cat_labels = np.asarray(cat_labels)
# cat_strings = np.asarray(cat_strings)


# trunk_match_idx = np.intersect1d(trunks_index_vec, np.where(labels == 1))
# trunk_non_match_idx = np.intersect1d(trunks_index_vec, np.where(labels == 0))
# t = np.int(np.maximum(len(trunk_match_idx), len(trunk_non_match_idx)))

# branch_match_idx = np.intersect1d(branches_index_vec, np.where(labels == 1))
# branch_non_match_idx = np.intersect1d(branches_index_vec, np.where(labels == 0))
# b = np.int(np.maximum(len(branch_match_idx), len(branch_non_match_idx)))

# leaf_match_idx = np.intersect1d(leaves_index_vec, np.where(labels == 1))
# leaf_non_match_idx = np.intersect1d(leaves_index_vec, np.where(labels == 0))
# l = np.int(np.maximum(len(leaf_match_idx), len(leaf_non_match_idx)))





# def performance(distances, labels, threshold):

# 	l2 = np.reshape(np.mean(distances, axis=1), (-1,1))

# 	truth_matches = labels == 1.
# 	truth_matches = truth_matches.astype(np.int32)

# 	truth_non_matches = labels == 0.
# 	truth_non_matches = truth_non_matches.astype(np.int32)

# 	pred_matches = l2 <= threshold
# 	pred_matches = pred_matches.astype(np.int32)

# 	pred_non_matches = l2 > threshold
# 	pred_non_matches = pred_non_matches.astype(np.int32)


# 	TP = 0
# 	TN = 0
# 	FP = 0
# 	FN = 0

# 	TP_t = 0
# 	TN_t = 0
# 	FP_t = 0
# 	FN_t = 0

# 	TP_b = 0
# 	TN_b = 0
# 	FP_b = 0
# 	FN_b = 0

# 	TP_l = 0
# 	TN_l = 0
# 	FP_l = 0
# 	FN_l = 0




# 	for i in range(0, distances.shape[0]):

# 		if truth_matches[i] == 1:

# 			if pred_matches[i] == 1:

# 				TP += 1

# 				if i in trunk_match_idx:

# 					TP_t += 1

# 				elif i in branch_match_idx:

# 					TP_b += 1

# 				elif i in leaf_match_idx:

# 					TP_l += 1

# 			elif pred_non_matches[i] == 1:

# 				FN += 1

# 				if i in trunk_match_idx:

# 					FN_t += 1

# 				elif i in branch_match_idx:

# 					FN_b += 1

# 				elif i in leaf_match_idx:

# 					FN_l += 1

# 		elif truth_non_matches[i] == 1:

# 			if pred_matches[i] == 1:

# 				FP += 1

# 				if i in trunk_non_match_idx:

# 					FP_t += 1

# 				elif i in branch_non_match_idx:

# 					FP_b += 1

# 				elif i in leaf_non_match_idx:

# 					FP_l += 1

# 			elif pred_non_matches[i] == 1:

# 				TN += 1

# 				if i in trunk_non_match_idx:

# 					TN_t += 1

# 				elif i in branch_non_match_idx:

# 					TN_b += 1

# 				elif i in leaf_non_match_idx:

# 					TN_l += 1


# 	return l2, TP, TN, FP, FN, TP_t, TN_t, FP_t, FN_t, TP_b, TN_b, FP_b, FN_b, TP_l, TN_l, FP_l, FN_l


# 	# print('---------------')
# 	# print('TP: ', TP, ' | FP:', FP)
# 	# print('---------------')
# 	# print('FN: ', FN, ' | TN:', TN)
# 	# print('---------------')





# l2, TP, TN, FP, FN, TP_t, TN_t, FP_t, FN_t, TP_b, TN_b, FP_b, FN_b, TP_l, TN_l, FP_l, FN_l = performance(output, labels, threshold)


# print('--------------------')
# print('-----  batch   -----')
# print('--------------------')
# print('TP: ', TP, ' | FP:', FP)
# print('--------------------')
# print('FN: ', FN, ' | TN:', TN)
# print('--------------------')
# print('--------------------')
# print('error_rate --->', np.round(100 * FP / (FP + TN)))
# print('recall ------->', np.round(100 * TP / (TP + FN)))
# print('accuracy ----->', np.round(100 * (TP + TN) / (TP + TN + FP + FN)))



# print('--------------------')
# print('-----  trunks  -----')
# print('--------------------')
# print('TP: ', TP_t, ' | FP:', FP_t)
# print('--------------------')
# print('FN: ', FN_t, ' | TN:', TN_t)
# print('--------------------')
# print('--------------------')
# print('error_rate --->', np.round(100 * FP_t / (FP_t + TN_t)))
# print('recall ------->', np.round(100 * TP_t / (TP_t + FN_t)))
# print('accuracy ----->', np.round(100 * (TP_t + TN_t) / (TP_t + TN_t + FP_t + FN_t)))


# print('--------------------')
# print('-----  branch  -----')
# print('--------------------')
# print('TP: ', TP_b, ' | FP:', FP_b)
# print('--------------------')
# print('FN: ', FN_b, ' | TN:', TN_b)
# print('--------------------')
# print('--------------------')
# print('error_rate --->', np.round(100 * FP_b / (FP_b + TN_b)))
# print('recall ------->', np.round(100 * TP_b / (TP_b + FN_b)))
# print('accuracy ----->', np.round(100 * (TP_b + TN_b) / (TP_b + TN_b + FP_b + FN_b)))


# print('--------------------')
# print('-----  leaves  -----')
# print('--------------------')
# print('TP: ', TP_l, ' | FP:', FP_l)
# print('--------------------')
# print('FN: ', FN_l, ' | TN:', TN_l)
# print('--------------------')
# print('--------------------')
# print('error_rate --->', np.round(100 * FP_l / (FP_l + TN_l)))
# print('recall ------->', np.round(100 * TP_l / (TP_l + FN_l)))
# print('accuracy ----->', np.round(100 * (TP_l + TN_l) / (TP_l + TN_l + FP_l + FN_l)))


# conf_mat = [[TP, FP], [FN, TN]]



# num_test = 140
# test_thresholds = np.linspace(10, 80, num_test)
# TP = np.zeros(num_test)
# TN = np.zeros(num_test)
# FP = np.zeros(num_test)
# FN = np.zeros(num_test)
# TP_t = np.zeros(num_test)
# TN_t = np.zeros(num_test)
# FP_t = np.zeros(num_test)
# FN_t = np.zeros(num_test)
# TP_b = np.zeros(num_test)
# TN_b = np.zeros(num_test)
# FP_b = np.zeros(num_test)
# FN_b = np.zeros(num_test)
# TP_l = np.zeros(num_test)
# TN_l = np.zeros(num_test)
# FP_l = np.zeros(num_test)
# FN_l = np.zeros(num_test)

# train_error_rate = np.zeros(num_test)
# train_recall = np.zeros(num_test)
# train_acc = np.zeros(num_test)
# trunk_error_rate = np.zeros(num_test)
# trunk_recall = np.zeros(num_test)
# trunk_acc = np.zeros(num_test)
# branch_error_rate = np.zeros(num_test)
# branch_recall = np.zeros(num_test)
# branch_acc = np.zeros(num_test)
# leaf_error_rate = np.zeros(num_test)
# leaf_recall = np.zeros(num_test)
# leaf_acc = np.zeros(num_test)

# thresh_batch = []
# thresh_trunk = []
# thresh_branch = []
# thresh_leaf = []

# for i in range(num_test):

# 	l2, TP[i], TN[i], FP[i], FN[i], TP_t[i], TN_t[i], FP_t[i], FN_t[i], TP_b[i], TN_b[i], FP_b[i], FN_b[i], TP_l[i], TN_l[i], FP_l[i], FN_l[i] = performance(output, labels, test_thresholds[i])

# 	train_error_rate[i] = np.round(100 * FP[i] / (FP[i] + TN[i]))
# 	train_recall[i] = np.round(100 * TP[i] / (TP[i] + FN[i]))
# 	train_acc[i] = np.round(100 * (TP[i] + TN[i]) / (TP[i] + TN[i] + FP[i] + FN[i]))
# 	if train_recall[i] <= 95:
# 		thresh_batch.append(test_thresholds[i])

# 	trunk_error_rate[i] = np.round(100 * FP_t[i] / (FP_t[i] + TN_t[i]))
# 	trunk_recall[i] = np.round(100 * TP_t[i] / (TP_t[i] + FN_t[i]))
# 	trunk_acc[i] = np.round(100 * (TP_t[i] + TN_t[i]) / (TP_t[i] + TN_t[i] + FP_t[i] + FN_t[i]))
# 	if trunk_recall[i] <= 95:
# 		thresh_trunk.append(test_thresholds[i])

# 	branch_error_rate[i] = np.round(100 * FP_b[i] / (FP_b[i] + TN_b[i]))
# 	branch_recall[i] = np.round(100 * TP_b[i] / (TP_b[i] + FN_b[i]))
# 	branch_acc[i] = np.round(100 * (TP_b[i] + TN_b[i]) / (TP_b[i] + TN_b[i] + FP_b[i] + FN_b[i]))
# 	if branch_recall[i] <= 95:
# 		thresh_branch.append(test_thresholds[i])

# 	leaf_error_rate[i] = np.round(100 * FP_l[i] / (FP_l[i] + TN_l[i]))
# 	leaf_recall[i] = np.round(100 * TP_l[i] / (TP_l[i] + FN_l[i]))
# 	leaf_acc[i] = np.round(100 * (TP_l[i] + TN_l[i]) / (TP_l[i] + TN_l[i] + FP_l[i] + FN_l[i]))
# 	if leaf_recall[i] <= 95:
# 		thresh_leaf.append(test_thresholds[i])


# # thresh_batch = np.argmin(np.power((train_recall - .95), 2))]
# # thresh_trunk = np.argmin(np.power((trunk_recall - .95), 2))]
# # thresh_branch = np.argmin(np.power((branch_recall - .95), 2))]
# # thresh_leaf = np.argmin(np.power((leaf_recall - .95), 2))]

# thresh_batch = np.asarray(thresh_batch)
# thresh_trunk = np.asarray(thresh_trunk)
# thresh_branch = np.asarray(thresh_branch)
# thresh_leaf = np.asarray(thresh_leaf)



# print(thresh_batch)
# print(thresh_trunk)
# print(thresh_branch)
# print(thresh_leaf)





# if make_embedding == True:

# 	import tensorflow as tf
# 	from tensorflow.contrib.tensorboard.plugins import projector

# 	embedding_var = tf.Variable(tf.zeros(output.shape), name='embedding')


# 	metadata = os.path.join(DATA_DIR_READ, 'metadata.tsv')

# 	with open(metadata, 'w') as metadata_file:
# 	    for row in cat_labels:
# 	        metadata_file.write('%d\n' % row)




# 	with tf.Session() as sess:


# 	    saver = tf.train.Saver([embedding_var])


# 		### create the tensorboard embedding ###

# 	    sess.run(embedding_var.initializer)

# 	    embedding_var.assign(output)

# 	    # sess.run(embedding_var.initializer)
# 	    saver.save(sess, os.path.join(DATA_DIR_READ, 'embedding_var.ckpt'))

# 	    config = projector.ProjectorConfig()
# 	    # One can add multiple embeddings.
# 	    embedding = config.embeddings.add()
# 	    embedding.tensor_name = embedding_var.name
# 	    embedding.metadata_path = 'metadata.tsv'
# 	    # Link this tensor to its metadata file (e.g. labels).
# 	    # embedding.metadata_path = metadata
# 	    # Saves a config file that TensorBoard will read during startup.
# 	    projector.visualize_embeddings(tf.summary.FileWriter(DATA_DIR_READ), config)

# 	    # projector.visualize_embeddings(tf.summary.FileWriter(DATA_DIR), config)

# 	    # saver.save(sess, os.path.join(DATA_DIR, 'filename.ckpt'))



# if loop_iterations == True:

# 	num_runs = 9
# 	num_test = 5

# 	its = [100, 200, 300, 400, 500, 600, 700, 800, 900]


# 	train_e = np.zeros((num_runs, num_test))
# 	train_r = np.zeros((num_runs, num_test))
# 	train_a = np.zeros((num_runs, num_test))

# 	trunk_e = np.zeros((num_runs, num_test))
# 	trunk_r = np.zeros((num_runs, num_test))
# 	trunk_a = np.zeros((num_runs, num_test))

# 	branch_e = np.zeros((num_runs, num_test))
# 	branch_r = np.zeros((num_runs, num_test))
# 	branch_a = np.zeros((num_runs, num_test))

# 	leaf_e = np.zeros((num_runs, num_test))
# 	leaf_r = np.zeros((num_runs, num_test))
# 	leaf_a = np.zeros((num_runs, num_test))



# 	for j in range(num_runs):


# 		filename_dist = DATA_DIR_READ + 'distances_' + str(its[j]) + '.npy'
# 		filename_labels = DATA_DIR_READ + 'labels_' + str(its[j]) + '.npy'
# 		# filename_acc = DATA_DIR + 'accuracies_' + str(i) + '.npy'

# 		output = np.load(filename_dist)
# 		labels = np.load(filename_labels)
# 		# accuracies = np.load(filename_acc)

# 		test_multiplier = output.shape[0]
# 		batch_size = output.shape[1]
# 		cat_size = int(batch_size/4)

# 		full_batch_size = int(batch_size * test_multiplier)

# 		print('test_multiplier......................', test_multiplier)
# 		print('batch_size...........................', batch_size)
# 		print('cat_size.............................', cat_size)



# 		output = np.reshape(output, ((batch_size * test_multiplier), -1))
# 		labels = np.reshape(labels, ((batch_size * test_multiplier), -1))

# 		print('output shape.........................', output.shape)
# 		print('labels shape.........................', labels.shape)


# 		trunks_index = np.arange(cat_size)
# 		branches_index = np.arange(cat_size) + 1*cat_size
# 		leaves_index = np.arange(cat_size) + 2*cat_size
# 		test_index = np.arange(cat_size) + 3*cat_size

# 		trunks_index_vec = trunks_index
# 		branches_index_vec = branches_index
# 		leaves_index_vec = leaves_index
# 		test_index_vec = test_index


# 		if test_multiplier > 1:

# 			for i in range(1, test_multiplier):

# 				trunks_index_vec = np.append(
# 					trunks_index_vec,
# 					i * batch_size + trunks_index)

# 				branches_index_vec = np.append(
# 					branches_index_vec,
# 					i * batch_size + branches_index)

# 				leaves_index_vec = np.append(
# 					leaves_index_vec,
# 					i * batch_size + leaves_index)

# 				test_index_vec = np.append(
# 					test_index_vec,
# 					i * batch_size + test_index)

# 		cat_labels = []
# 		cat_strings = []

# 		for i in range(0, full_batch_size):

# 			if labels[i] == 1 and i in trunks_index_vec:
# 				cat_labels.append(1)
# 				cat_strings.append('trunk_match')
# 			if labels[i] == 0 and i in trunks_index_vec:
# 				cat_labels.append(2)
# 				cat_strings.append('trunk_non_match')

# 			if labels[i] == 1 and i in branches_index_vec:
# 				cat_labels.append(3)
# 				cat_strings.append('branch_match')
# 			if labels[i] == 0 and i in branches_index_vec:
# 				cat_labels.append(4)
# 				cat_strings.append('branch_non_match')

# 			if labels[i] == 1 and i in leaves_index_vec:
# 				cat_labels.append(5)
# 				cat_strings.append('leaf_match')
# 			if labels[i] == 0 and i in leaves_index_vec:
# 				cat_labels.append(6)#'leaf_non_match')
# 				cat_strings.append('leaf_non_match')

# 			if labels[i] == 1 and i in test_index_vec:
# 				cat_labels.append(7)
# 				cat_strings.append('random_match')
# 			if labels[i] == 0 and i in test_index_vec:
# 				cat_labels.append(8)
# 				cat_strings.append('random_non_match')

# 		cat_labels = np.asarray(cat_labels)
# 		cat_strings = np.asarray(cat_strings)


# 		trunk_match_idx = np.intersect1d(trunks_index_vec, np.where(labels == 1))
# 		trunk_non_match_idx = np.intersect1d(trunks_index_vec, np.where(labels == 0))
# 		t = np.int(np.maximum(len(trunk_match_idx), len(trunk_non_match_idx)))

# 		branch_match_idx = np.intersect1d(branches_index_vec, np.where(labels == 1))
# 		branch_non_match_idx = np.intersect1d(branches_index_vec, np.where(labels == 0))
# 		b = np.int(np.maximum(len(branch_match_idx), len(branch_non_match_idx)))

# 		leaf_match_idx = np.intersect1d(leaves_index_vec, np.where(labels == 1))
# 		leaf_non_match_idx = np.intersect1d(leaves_index_vec, np.where(labels == 0))
# 		l = np.int(np.maximum(len(leaf_match_idx), len(leaf_non_match_idx)))


# 		l2, TP, TN, FP, FN, TP_t, TN_t, FP_t, FN_t, TP_b, TN_b, FP_b, FN_b, TP_l, TN_l, FP_l, FN_l = performance(output, labels, threshold)


# 		test_thresholds = np.linspace(50, 70, num_test)
# 		TP = np.zeros(num_test)
# 		TN = np.zeros(num_test)
# 		FP = np.zeros(num_test)
# 		FN = np.zeros(num_test)
# 		TP_t = np.zeros(num_test)
# 		TN_t = np.zeros(num_test)
# 		FP_t = np.zeros(num_test)
# 		FN_t = np.zeros(num_test)
# 		TP_b = np.zeros(num_test)
# 		TN_b = np.zeros(num_test)
# 		FP_b = np.zeros(num_test)
# 		FN_b = np.zeros(num_test)
# 		TP_l = np.zeros(num_test)
# 		TN_l = np.zeros(num_test)
# 		FP_l = np.zeros(num_test)
# 		FN_l = np.zeros(num_test)

# 		train_error_rate = np.zeros(num_test)
# 		train_recall = np.zeros(num_test)
# 		train_acc = np.zeros(num_test)
# 		trunk_error_rate = np.zeros(num_test)
# 		trunk_recall = np.zeros(num_test)
# 		trunk_acc = np.zeros(num_test)
# 		branch_error_rate = np.zeros(num_test)
# 		branch_recall = np.zeros(num_test)
# 		branch_acc = np.zeros(num_test)
# 		leaf_error_rate = np.zeros(num_test)
# 		leaf_recall = np.zeros(num_test)
# 		leaf_acc = np.zeros(num_test)

# 		for i in range(num_test):

# 			l2, TP[i], TN[i], FP[i], FN[i], TP_t[i], TN_t[i], FP_t[i], FN_t[i], TP_b[i], TN_b[i], FP_b[i], FN_b[i], TP_l[i], TN_l[i], FP_l[i], FN_l[i] = performance(output, labels, test_thresholds[i])

# 			train_error_rate[i] = np.round(100 * FP[i] / (FP[i] + TN[i]))
# 			train_recall[i] = np.round(100 * TP[i] / (TP[i] + FN[i]))
# 			train_acc[i] = np.round(100 * (TP[i] + TN[i]) / (TP[i] + TN[i] + FP[i] + FN[i]))

# 			trunk_error_rate[i] = np.round(100 * FP_t[i] / (FP_t[i] + TN_t[i]))
# 			trunk_recall[i] = np.round(100 * TP_t[i] / (TP_t[i] + FN_t[i]))
# 			trunk_acc[i] = np.round(100 * (TP_t[i] + TN_t[i]) / (TP_t[i] + TN_t[i] + FP_t[i] + FN_t[i]))

# 			branch_error_rate[i] = np.round(100 * FP_b[i] / (FP_b[i] + TN_b[i]))
# 			branch_recall[i] = np.round(100 * TP_b[i] / (TP_b[i] + FN_b[i]))
# 			branch_acc[i] = np.round(100 * (TP_b[i] + TN_b[i]) / (TP_b[i] + TN_b[i] + FP_b[i] + FN_b[i]))

# 			leaf_error_rate[i] = np.round(100 * FP_l[i] / (FP_l[i] + TN_l[i]))
# 			leaf_recall[i] = np.round(100 * TP_l[i] / (TP_l[i] + FN_l[i]))
# 			leaf_acc[i] = np.round(100 * (TP_l[i] + TN_l[i]) / (TP_l[i] + TN_l[i] + FP_l[i] + FN_l[i]))


# 		train_e[j,:] = train_error_rate
# 		train_r[j,:] = train_recall
# 		train_a[j,:] = train_acc

# 		trunk_e[j,:] = trunk_error_rate
# 		trunk_r[j,:] = trunk_recall
# 		trunk_a[j,:] = trunk_acc

# 		branch_e[j,:] = branch_error_rate
# 		branch_r[j,:] = branch_recall
# 		branch_a[j,:] = branch_acc

# 		leaf_e[j,:] = leaf_error_rate
# 		leaf_r[j,:] = leaf_recall
# 		leaf_a[j,:] = leaf_acc



# if loop_iterations == True:
# 	t = 3
# 	plt.figure()

# 	plt.subplot(311)

# 	plt.scatter(np.arange(len(trunk_match_idx)), l2[trunk_match_idx], label='trunk_match')
# 	plt.scatter(np.arange(len(trunk_non_match_idx)), l2[trunk_non_match_idx], label='trunk_non_match')
# 	plt.scatter(t + 20 + np.arange(len(branch_match_idx)), l2[branch_match_idx], label='branch_match')
# 	plt.scatter(t + 20 + np.arange(len(branch_non_match_idx)), l2[branch_non_match_idx], label='branch__match')
# 	plt.scatter(t + 20 + b + 20 + np.arange(len(leaf_match_idx)), l2[leaf_match_idx], label='leaf_match')
# 	plt.scatter(t + 20 + b + 20 + np.arange(len(leaf_non_match_idx)), l2[leaf_non_match_idx], label='leaf_non_match')
# 	plt.axhline(y=60., xmin=0., xmax=len(labels), linewidth=2, color = 'k')
# 	plt.xlabel('samples')
# 	plt.ylabel('distances')
# 	plt.legend()
# 	plt.grid()

# 	plt.subplot(334)
# 	plt.plot(train_e[:,t], label='batch')
# 	plt.plot(trunk_e[:,t], label='trunk')
# 	plt.plot(branch_e[:,t], label='branch')
# 	plt.plot(leaf_e[:,t], label='leaf')
# 	plt.ylabel('%')
# 	plt.xlabel('iteration')
# 	plt.grid()
# 	plt.legend()
# 	plt.title('error_rate')

# 	plt.subplot(335)
# 	plt.plot(train_r[:,t], label='batch')
# 	plt.plot(trunk_r[:,t], label='trunk')
# 	plt.plot(branch_r[:,t], label='branch')
# 	plt.plot(leaf_r[:,t], label='leaf')
# 	plt.ylabel('%')
# 	plt.xlabel('iteration')
# 	plt.grid()
# 	plt.legend()
# 	plt.title('recall')

# 	plt.subplot(336)
# 	plt.plot(train_a[:,t], label='batch')
# 	plt.plot(trunk_a[:,t], label='trunk')
# 	plt.plot(branch_a[:,t], label='branch')
# 	plt.plot(leaf_a[:,t], label='leaf')
# 	plt.ylabel('%')
# 	plt.xlabel('iteration')
# 	plt.grid()
# 	plt.legend()
# 	plt.title('accuracy')

# 	plt.subplot(337)
# 	plt.plot(test_thresholds, train_error_rate, label='batch')
# 	plt.plot(test_thresholds, trunk_error_rate, label='trunk')
# 	plt.plot(test_thresholds, branch_error_rate, label='branch')
# 	plt.plot(test_thresholds, leaf_error_rate, label='leaf')
# 	plt.ylabel('%')
# 	plt.xlabel('threshold')
# 	plt.grid()
# 	plt.legend()
# 	plt.title('error_rate')

# 	plt.subplot(338)
# 	plt.plot(test_thresholds, train_recall, label='batch')
# 	plt.plot(test_thresholds, trunk_recall, label='trunk')
# 	plt.plot(test_thresholds, branch_recall, label='branch')
# 	plt.plot(test_thresholds, leaf_recall, label='leaf')
# 	plt.ylabel('%')
# 	plt.xlabel('threshold')
# 	plt.legend()
# 	plt.grid()
# 	plt.title('recall')

# 	plt.subplot(339)
# 	plt.plot(test_thresholds, train_acc, label='batch')
# 	plt.plot(test_thresholds, trunk_acc, label='trunk')
# 	plt.plot(test_thresholds, branch_acc, label='branch')
# 	plt.plot(test_thresholds, leaf_acc, label='leaf')
# 	plt.ylabel('%')
# 	plt.xlabel('threshold')
# 	plt.legend()
# 	plt.grid()
# 	plt.title('accuracy')


# 	plt.show()

# # , color='green', marker='o', linestyle='dashed',
# #         linewidth=2, markersize=12)


# if loop_iterations == False:
# 	plt.figure()

# 	plt.subplot(211)

# 	plt.scatter(np.arange(len(trunk_match_idx)), l2[trunk_match_idx], label='trunk_match', color='green', marker='o')
# 	plt.scatter(np.arange(len(trunk_non_match_idx)), l2[trunk_non_match_idx], label='trunk_non_match', color='red', marker='x')
# 	plt.scatter(t + 20 + np.arange(len(branch_match_idx)), l2[branch_match_idx], label='branch_match', color='green', marker='o')
# 	plt.scatter(t + 20 + np.arange(len(branch_non_match_idx)), l2[branch_non_match_idx], label='branch__match', color='red', marker='x')
# 	plt.scatter(t + 20 + b + 20 + np.arange(len(leaf_match_idx)), l2[leaf_match_idx], label='leaf_match', color='green', marker='o')
# 	plt.scatter(t + 20 + b + 20 + np.arange(len(leaf_non_match_idx)), l2[leaf_non_match_idx], label='leaf_non_match', color='red', marker='x')
# 	# plt.axhline(y=thresh_batch, xmin=0., xmax=len(labels), linewidth=2, color = 'k')
# 	# plt.axhline(y=thresh_trunk, xmin=0., xmax=t, linewidth=1, color = 'k')
# 	# plt.axhline(y=thresh_branch, xmin=t + 20, xmax=b, linewidth=1, color = 'k')
# 	# plt.axhline(y=thresh_leaf, xmin=t + b + 40, xmax=len(labels), linewidth=1, color = 'k')
# 	plt.xlabel('trunks   --   branches   --   leaves')
# 	plt.ylabel('distances')
# 	plt.grid()

# 	plt.subplot(234)
# 	plt.plot(test_thresholds, train_error_rate, label='batch')
# 	plt.plot(test_thresholds, trunk_error_rate, label='trunk')
# 	plt.plot(test_thresholds, branch_error_rate, label='branch')
# 	plt.plot(test_thresholds, leaf_error_rate, label='leaf')
# 	plt.ylabel('%')
# 	plt.xlabel('threshold')
# 	plt.grid()
# 	plt.legend()
# 	plt.title('error_rate')

# 	plt.subplot(235)
# 	plt.plot(test_thresholds, train_recall, label='batch')
# 	plt.plot(test_thresholds, trunk_recall, label='trunk')
# 	plt.plot(test_thresholds, branch_recall, label='branch')
# 	plt.plot(test_thresholds, leaf_recall, label='leaf')
# 	plt.ylabel('%')
# 	plt.xlabel('threshold')
# 	plt.legend()
# 	plt.grid()
# 	plt.title('recall')

# 	plt.subplot(236)
# 	plt.plot(test_thresholds, train_acc, label='batch')
# 	plt.plot(test_thresholds, trunk_acc, label='trunk')
# 	plt.plot(test_thresholds, branch_acc, label='branch')
# 	plt.plot(test_thresholds, leaf_acc, label='leaf')
# 	plt.ylabel('%')
# 	plt.xlabel('threshold')
# 	plt.legend()
# 	plt.grid()
# 	plt.title('accuracy')


# 	plt.show()





























