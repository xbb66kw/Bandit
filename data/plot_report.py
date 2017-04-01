#!/usr/bin/env python

# import modules used here -- sys is a very standard one

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

def sort_upper(x):
				n = len(x)
				
				return np.sort(x)[int(0.85 * (n-1))] 

def sort_lower(x):
				n = len(x)
				
				return np.sort(x)[int(0.15 * (n-1))] 


def main():
				
				data_low_high = np.load('/Users/xbb/Desktop/OGA-test/data/low_high.npy').T
				data_low_low = np.load('/Users/xbb/Desktop/OGA-test/data/low_low.npy').T
				data_low_oracle = np.load('/Users/xbb/Desktop/OGA-test/data/low_oracle.npy').T
				data_low_bylearning = np.load('/Users/xbb/Desktop/OGA-test/data/low_bylearning.npy').T
				data_low_random = np.load('/Users/xbb/Desktop/OGA-test/data/low_random.npy').T
				
				data_low_low = data_low_oracle - data_low_low
				data_low_high = data_low_oracle - data_low_high
				data_low_bylearning = data_low_oracle - data_low_bylearning
				data_low_random = data_low_oracle - data_low_random
				
				
				#Fixing the x-axis label
				#plt.subplot(311)
				pp = PdfPages('/Users/xbb/Desktop/OGA-test/data/low.pdf')
				plt.title('Non-sparse')
				x_axis_points = np.arange(0,10000+1, 1000)
				plt.xticks(range(len(np.arange(0,10000+1, 200)))[0:50:5], x_axis_points, rotation = 30, fontsize = 10)
				patches = []
				patches.append(mpatches.Patch(color = 'red', label = 'High'))
				patches.append(mpatches.Patch(color = 'blue', label = 'Low'))
				patches.append(mpatches.Patch(color = 'green', label = 'Bi'))
				patches.append(mpatches.Patch(color = 'black', label = 'Random'))
				plt.legend(handles = patches)
				
				
				#print(np.apply_along_axis(sort_upper, 0, data_low_high))
				plt.plot(np.apply_along_axis(sort_upper, 0, data_low_high), linestyle = 'dashed', color = '#ee9999')
				plt.plot(np.apply_along_axis(sort_lower, 0, data_low_high), linestyle = 'dashed', color = '#ee9999')
				plt.plot(np.apply_along_axis(np.mean, 0, data_low_high), color = 'red')
				
				plt.plot(np.apply_along_axis(sort_upper, 0, data_low_low), linestyle = 'dashed', color = '#9999ee')
				plt.plot(np.apply_along_axis(sort_lower, 0, data_low_low), linestyle = 'dashed', color = '#9999ee')
				plt.plot(np.apply_along_axis(np.mean, 0, data_low_low), color = 'blue')
				
				plt.plot(np.apply_along_axis(sort_upper, 0, data_low_bylearning), linestyle = 'dashed', color = '#99ee99')
				plt.plot(np.apply_along_axis(sort_lower, 0, data_low_bylearning), linestyle = 'dashed', color = '#99ee99')
				plt.plot(np.apply_along_axis(np.mean, 0, data_low_bylearning), color = 'green')
				
				plt.plot(np.apply_along_axis(sort_upper, 0, data_low_random), linestyle = 'dashed', color = '#999999')
				plt.plot(np.apply_along_axis(sort_lower, 0, data_low_random), linestyle = 'dashed', color = '#999999')
				plt.plot(np.apply_along_axis(np.mean, 0, data_low_random), color = 'black')
				pp.savefig()
				pp.close()
				plt.clf()#Clear the current figure
				
				######
				######
				
				data_low_high = np.load('/Users/xbb/Desktop/OGA-test/data/high_high.npy').T
				data_low_low = np.load('/Users/xbb/Desktop/OGA-test/data/high_low.npy').T
				data_low_oracle = np.load('/Users/xbb/Desktop/OGA-test/data/high_oracle.npy').T
				data_low_bylearning = np.load('/Users/xbb/Desktop/OGA-test/data/high_bylearning.npy').T
				data_low_random = np.load('/Users/xbb/Desktop/OGA-test/data/high_random.npy').T
				
				data_low_low = data_low_oracle - data_low_low
				data_low_high = data_low_oracle - data_low_high
				data_low_bylearning = data_low_oracle - data_low_bylearning
				data_low_random = data_low_oracle - data_low_random
				
				
				#Fixing the x-axis label
				x_axis_points = np.arange(0,10000+1, 1000)
				
				patches = []
				patches.append(mpatches.Patch(color = 'red', label = 'High'))
				patches.append(mpatches.Patch(color = 'blue', label = 'Low'))
				patches.append(mpatches.Patch(color = 'green', label = 'Bi'))
				patches.append(mpatches.Patch(color = 'black', label = 'Random'))
				
				
				#plt.subplot(312)
				pp = PdfPages('/Users/xbb/Desktop/OGA-test/data/high.pdf')
				plt.title('Sparse')
				plt.legend(handles = patches)
				plt.xticks(range(len(np.arange(0,10000+1, 200)))[0:50:5], x_axis_points, rotation = 30, fontsize = 10)
				
				#print(np.apply_along_axis(sort_upper, 0, data_low_high))
				plt.plot(np.apply_along_axis(sort_upper, 0, data_low_high), linestyle = 'dashed', color = '#ee9999')
				plt.plot(np.apply_along_axis(sort_lower, 0, data_low_high), linestyle = 'dashed', color = '#ee9999')
				plt.plot(np.apply_along_axis(np.mean, 0, data_low_high), color = 'red')
				
				plt.plot(np.apply_along_axis(sort_upper, 0, data_low_low), linestyle = 'dashed', color = '#9999ee')
				plt.plot(np.apply_along_axis(sort_lower, 0, data_low_low), linestyle = 'dashed', color = '#9999ee')
				plt.plot(np.apply_along_axis(np.mean, 0, data_low_low), color = 'blue')
				
				plt.plot(np.apply_along_axis(sort_upper, 0, data_low_bylearning), linestyle = 'dashed', color = '#99ee99')
				plt.plot(np.apply_along_axis(sort_lower, 0, data_low_bylearning), linestyle = 'dashed', color = '#99ee99')
				plt.plot(np.apply_along_axis(np.mean, 0, data_low_bylearning), color = 'green')
				
				plt.plot(np.apply_along_axis(sort_upper, 0, data_low_random), linestyle = 'dashed', color = '#999999')
				plt.plot(np.apply_along_axis(sort_lower, 0, data_low_random), linestyle = 'dashed', color = '#999999')
				plt.plot(np.apply_along_axis(np.mean, 0, data_low_random), color = 'black')
				pp.savefig()
				pp.close()
				plt.clf()
				
				#####
				#####
				data_low_cross = np.load('/Users/xbb/Desktop/OGA-test/data/cross_cross.npy').T
				data_low_low = np.load('/Users/xbb/Desktop/OGA-test/data/cross_low.npy').T
				data_low_oracle = np.load('/Users/xbb/Desktop/OGA-test/data/cross_oracle.npy').T
				
				data_low_random = np.load('/Users/xbb/Desktop/OGA-test/data/cross_random.npy').T
				
				data_low_low = data_low_oracle - data_low_low
				data_low_cross = data_low_oracle - data_low_cross
				
				data_low_random = data_low_oracle - data_low_random
				
				
				#Fixing the x-axis label
				x_axis_points = np.arange(0,10000+1, 1000)

				patches = []
				patches.append(mpatches.Patch(color = 'red', label = 'Cross'))
				patches.append(mpatches.Patch(color = 'blue', label = 'Low'))
				
				patches.append(mpatches.Patch(color = 'black', label = 'Random'))
				#plt.subplot(313)
				pp = PdfPages('/Users/xbb/Desktop/OGA-test/data/cross.pdf')
				plt.title('Cross-terms')
				plt.legend(handles = patches)
				plt.xticks(range(len(np.arange(0,10000+1, 200)))[0:50:5], x_axis_points, rotation = 30, fontsize = 10)
				
				#print(np.apply_along_axis(sort_upper, 0, data_low_high))
				plt.plot(np.apply_along_axis(sort_upper, 0, data_low_cross), linestyle = 'dashed', color = '#ee9999')
				plt.plot(np.apply_along_axis(sort_lower, 0, data_low_cross), linestyle = 'dashed', color = '#ee9999')
				plt.plot(np.apply_along_axis(np.mean, 0, data_low_cross), color = 'red')
				
				plt.plot(np.apply_along_axis(sort_upper, 0, data_low_low), linestyle = 'dashed', color = '#9999ee')
				plt.plot(np.apply_along_axis(sort_lower, 0, data_low_low), linestyle = 'dashed', color = '#9999ee')
				plt.plot(np.apply_along_axis(np.mean, 0, data_low_low), color = 'blue')
				
				
				plt.plot(np.apply_along_axis(sort_upper, 0, data_low_random), linestyle = 'dashed', color = '#999999')
				plt.plot(np.apply_along_axis(sort_lower, 0, data_low_random), linestyle = 'dashed', color = '#999999')
				plt.plot(np.apply_along_axis(np.mean, 0, data_low_random), color = 'black')
				
				
				
				
				plt.axis([0, 50, 0,200])
				pp.savefig()
				pp.close()
				plt.clf()
if __name__ == '__main__':
    main()
    
