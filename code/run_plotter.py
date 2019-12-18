import numpy as np
from matplotlib import pyplot as plt

plots = True

err_hist = True

batchsize = 16

path = 'tensorflow_logs/doublette_test8/runData_'
# path = '/media/drzadmin/DATA/tensorflow_logs/doublette_test4/runData_'
last_log = 33*250
print_freq = 250



def analyse(matches, non_matches, threshold):

	tp = matches < threshold
	fn = matches >= threshold
	tn = non_matches >= threshold
	fp = non_matches < threshold

	tp = np.sum(tp.astype(int))
	fn = np.sum(fn.astype(int))
	tn = np.sum(tn.astype(int))
	fp = np.sum(fp.astype(int))

	recall = np.round(100 * tp / (tp + fn), 4)
	err_rate =  np.round(100 * fp / (fp + tn), 4)

	return recall, err_rate



file = path + str(last_log) + '.npy'

data = np.reshape(np.load(file), (1,-1))

num_points = data.shape[1]
num_iters = .5 * num_points / batchsize

matches = np.reshape(data[0,:int(.5*num_points)], (1,-1))
non_matches = np.reshape(data[0,int(.5*num_points):], (1,-1))


leaves_index0 = np.arange(int(.5*batchsize))
leaves_index = []
non_leaves_index = []

for i in range(int(num_iters)):

	leaves_index.append(leaves_index0 + i * int(batchsize))
	non_leaves_index.append(leaves_index0 + int((i+.5) * batchsize))


leaves_index = np.reshape(np.asarray(leaves_index), (1,-1))
non_leaves_index = np.reshape(np.asarray(non_leaves_index), (1,-1))

# print()
# print(leaves_index.shape)
# print(non_leaves_index.shape)
# print(leaves_index)
# print(non_leaves_index)

matches_L = matches[0,np.squeeze(leaves_index)]
matches_B = matches[0,np.squeeze(non_leaves_index)]

non_matches_L = non_matches[0,np.squeeze(leaves_index)]
non_matches_B = non_matches[0,np.squeeze(non_leaves_index)]


recall_batch, errors_batch = [], []

test_threshs = np.linspace(0,140,1000)

for i in range(len(test_threshs)):

	rec, err = analyse(matches, non_matches, test_threshs[i])

	recall_batch.append(rec)
	errors_batch.append(err)



rec95 = np.squeeze(np.asarray(np.where(np.asarray(recall_batch) < 95)))
err95 = np.asarray(errors_batch)[rec95[-1]]

print('error rate ---------->', err95, ' at ', np.round(test_threshs[rec95[-1]]))



if plots:


	plt.figure()

	plt.plot(test_threshs, np.asarray(recall_batch), label='recall')
	plt.plot(test_threshs, np.asarray(errors_batch), label='error rate')
	plt.plot([0, 140], [95, 95])
	plt.plot([0, 140], [err95, err95])
	plt.plot([test_threshs[rec95[-1]], test_threshs[rec95[-1]]], [0, 100])
	plt.title('batch error rate at .95 recall: ' + str(err95))

	plt.grid()
	plt.legend()



	axis = np.arange(len(matches_L))

	plt.figure()

	plt.scatter(5+axis, matches_L, c='g', marker='o', s=6, label='match leaves')
	plt.scatter(5+axis, non_matches_L, c='r', marker='x', s=6, label='non-match leaves')

	plt.scatter(115+len(matches_L)+axis, matches_B, c='b', marker='o', s=6, label='match branches')
	plt.scatter(115+len(matches_L)+axis, non_matches_B, c='m', marker='x', s=6, label='non-match branches')

	plt.plot([0, 115+len(matches_L)+len(axis)], [test_threshs[rec95[-1]], test_threshs[rec95[-1]]])

	plt.grid()
	plt.legend()







if err_hist == True:

	err_rate_progress = []
	line_prog = []
	meta = []


	for j in range(int(last_log/print_freq)+1):


		file = path + str(j*print_freq) + '.npy'

		data = np.reshape(np.load(file), (1,-1))

		meta.append(np.load(path + 'loss_' + str(j*print_freq) + '.npy'))

		num_points = data.shape[1]
		num_iters = .5 * num_points / batchsize

		matches = np.reshape(data[0,:int(.5*num_points)], (1,-1))
		non_matches = np.reshape(data[0,int(.5*num_points):], (1,-1))


		leaves_index0 = np.arange(int(.5*batchsize))
		leaves_index = []
		non_leaves_index = []

		for i in range(int(num_iters)):

			leaves_index.append(leaves_index0 + i * int(batchsize))
			non_leaves_index.append(leaves_index0 + int((i+.5) * batchsize))


		leaves_index = np.reshape(np.asarray(leaves_index), (1,-1))
		non_leaves_index = np.reshape(np.asarray(non_leaves_index), (1,-1))

		# print()
		# print(leaves_index.shape)
		# print(non_leaves_index.shape)
		# print(leaves_index)
		# print(non_leaves_index)

		matches_L = matches[0,np.squeeze(leaves_index)]
		matches_B = matches[0,np.squeeze(non_leaves_index)]

		non_matches_L = non_matches[0,np.squeeze(leaves_index)]
		non_matches_B = non_matches[0,np.squeeze(non_leaves_index)]



		recall_batch, errors_batch = [], []

		test_threshs = np.linspace(0,140,1000)

		for i in range(len(test_threshs)):

			rec, err = analyse(matches, non_matches, test_threshs[i])

			recall_batch.append(rec)
			errors_batch.append(err)



		rec95 = np.squeeze(np.asarray(np.where(np.asarray(recall_batch) < 95)))
		err95 = np.asarray(errors_batch)[rec95[-1]]

		err_rate_progress.append(err95)
		line_prog.append(test_threshs[rec95[-1]])


	if plots:


		plt.figure()

		plt.subplot(211)
		plt.plot(np.asarray(err_rate_progress), label='err_rate_progress')
		plt.grid()
		plt.legend()

		plt.subplot(212)
		plt.plot(np.asarray(line_prog), label='threshold progress')
		plt.grid()
		plt.legend()


		plt.figure()
		plt.plot(np.asarray(err_rate_progress / np.max(err_rate_progress)), label='err_rate_progress')
		plt.plot(np.asarray(meta)[:,0] / np.max(np.asarray(meta)[:,0]), label='validation loss')
		plt.plot(np.asarray(meta)[:,1] / np.max(np.asarray(meta)[:,1]), label='match loss')
		plt.plot(np.asarray(meta)[:,2] / np.max(np.asarray(meta)[:,2]), label='nonmatch loss')
		plt.grid()
		plt.legend()



		plt.show()

