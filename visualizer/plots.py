import numpy as np
import matplotlib.pyplot as plt

def  plot_genotype_hist(genotypes, filename):
	'''
	Plots a histogram of all genotype values in the flattened genotype matrix.

	:param genotypes: array of genotypes
	:param filename: filename (including path) to save plot to
	'''
	unique, counts = np.unique(genotypes, return_counts=True)
	d = zip(unique, counts)
	plt.hist(np.ndarray.flatten(genotypes), bins=50)
	if len(unique) < 5:
		plt.title(", ".join(["{:.2f} : {}".format(u, c) for (u,c) in d]), fontdict = {'fontsize' : 9})

	plt.savefig("{0}.pdf".format(filename))
	plt.close()
