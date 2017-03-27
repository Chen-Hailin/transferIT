import os
import numpy as np
import math

root = '/Users/apple/Desktop/CHL/CS Study/CZ4041/OpportunityUCIDataset/dataset/'
#filename='S1-ADL1.dat'
out_root = '/Users/apple/Desktop/CHL/CS Study/CZ4041/transferIT/output1/'

#filename='S1-ADL1.dat'
#output_file='S1-ADL1'

def LoadData(data_root):
	f = open(data_root, 'r')
	raw_array = np.fromfile(f, dtype=float, sep=' ')

	mask_window_L = 134
	mask_window_H = 242

	m = np.reshape(raw_array, (-1,250))

	mask = [i for i in range(mask_window_L,mask_window_H+1)]
	mask.extend([i for i in range(244,249)])
	mask.extend([0])
	m = np.delete(m,mask,1)
	print(m.shape)
	return m


def NanRows(k, data):
	NaNs = []
	for i in range(0, data.shape[0]):
		if(math.isnan(data[i][k])):
			NaNs.append(i)
	return NaNs
def GoodRows(data, NaNs):
	rows = list(set(range(data.shape[0])) - set(NaNs))
	return rows

def GoodValues(k, data, rows):
	values = []
	for i in rows:
		values.append(data[i][k])
	return values
def ReplaceNan(data, k, nanRows, nanValues):
	for i in range (0, len(nanRows)):
		data[nanRows[i]][k] = nanValues[i]

def InterpolateNan(data):
	for k in range(0, data.shape[1]):
		nanRows = NanRows(k, data)
		goodRows = GoodRows(data, nanRows)
		goodValues = GoodValues(k, data, goodRows)
		nanValues = np.interp(nanRows, goodRows, goodValues)
		ReplaceNan(data, k, nanRows, nanValues)
		#print "column: " + str(k) + " finished"

#These are label unique indexes 
label_map_activity = [0, 406516, 406517, 404516, 404517, 406520, 404520, 406505, 404505, 406519, 404519, 406511, 404511, 406508, 404508, 408512, 407521, 405506]
label_map_locomotion = [0,1,2,4,5]

def ActivityLabels(data):
	labels = np.zeros(shape=(data.shape[0], 18))
	for i in range(data.shape[0]):
		labels[i][label_map_activity.index(data[i][-1])] = 1
	return labels

def LocomotionLabels(data):
	labels = np.zeros(shape=(data.shape[0], 5))
	for i in range(data.shape[0]):
		labels[i][label_map_locomotion.index(data[i][-2])] = 1
	return labels

def Normalize(data):
	for column in range (0, data.shape[1]):
		max_value = 0
		for row in range (data.shape[0]):
			if (abs(data[row][column]) > max_value):
					max_value = abs(data[row][column])
		'''I find that these following columns are always NaN value for S1-ADL1.dat
Column: 35 Accelerometer RH accX; value = round(original_value), unit = milli g
Column: 36 Accelerometer RH accY; value = round(original_value), unit = milli g
Column: 37 Accelerometer RH accZ; value = round(original_value), unit = milli g
		'''
		if(max_value != 0):
			for row in range (data.shape[0]):
				data[row][column] /= max_value

def Delete_null(data, activity_labels):
	null_rows = []
	for row in range (1, data.shape[0]):
		if(np.where(activity_labels[row] == 1)[0][0] == 0):
			null_rows.append(row)
	return null_rows

def main(filename):
#filename = 'S1-ADL1.dat'	

	data_root = os.path.join(root,filename)
	data = LoadData(data_root)
	activity_labels = ActivityLabels(data)
	locomotion_labels = LocomotionLabels(data)
	data = np.delete(data, [133,134], axis=1)
	InterpolateNan(data)
	Normalize(data)
	null_rows = Delete_null(data, activity_labels)
	data = np.delete(data, null_rows, axis=0)
	activity_labels = np.delete(activity_labels, null_rows, axis=0)
	print data.shape + activity_labels.shape
	np.save(out_root+filename+"_data", data)
	np.save(out_root+filename+"_activity_labels", activity_labels)
	np.save(out_root+filename+"_locomotion_labels", locomotion_labels)
	print filename + " finished!"

if __name__ == "__main__":
	for i in range (1, 5):
		for j in range (1, 6):
			main('S'+str(i)+'-ADL'+str(j)+'.dat')
		main('S'+str(i)+'-Drill.dat')

