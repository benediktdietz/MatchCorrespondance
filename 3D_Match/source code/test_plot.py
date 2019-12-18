iter =1
loss =1
ErrorRate =1
with open('/home/drzadmin/Desktop/3DMatch-pytorch/log/testloss.txt', 'a') as out:
	out.write(str(iter) + ' ' + str(loss) + ' ' + str(ErrorRate) + '\n')