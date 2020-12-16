from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#acc=
for i in range(2,5):
	loss=np.load('./loss_'+str(i)+'.npy')
	#first=0
	# print(np.shape(acc)[0])
	# for j in range(np.shape(acc)[0]-1):
	# 	if acc[j+1]-acc[j]<=1:
	# 		first=j
	# 		break
	loss=loss[10000:]
	step=np.load('./step.npy')
	step=step[10000:]
	print(str(i)+'_range:',max(loss)-min(loss))
	print(str(i)+'_mean:',np.mean(loss))
	print(str(i)+'_variance',np.var(loss))
	#print(str(i)+'_first',first)
	print()
	plt.plot(step,loss,color='b')
	plt.yticks(np.arange(0.1,0.4,0.01))
	plt.savefig('./analysis_loss_'+str(i)+'.png')
	plt.close()

loss_1=np.load('./loss.npy')[10000:]
step_1=np.load('./step.npy')[10000:]
plt.plot(step_1,loss_1,color='r')
plt.yticks(np.arange(0.1,0.4,0.01))
plt.savefig('./analysis_loss_1.png')
print('1_range:',max(loss_1)-min(loss_1))
print('1_mean:',np.mean(loss_1))
print('1_variance',np.var(loss_1))