import numpy as np
from pre_define import *

def distribution(y):
	dis = [0] * n_classes
	y_ = np.argmax(y, axis = 1)
	for _y in y_:
		dis[_y] += 1
	return dis

def check_dis(y):
	dis = distribution(y)
	sum_dis = float(sum(dis))
	dis_per = [v / sum_dis for v in dis]
	print '\n'.join('activity{}: {:0.2f}'.format(v, i) for v, i in enumerate(dis_per))

