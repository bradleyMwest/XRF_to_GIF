
import matplotlib.pyplot as plt # for general plotting
import matplotlib.cm as cm # for colorscales in 2D maps. Shown in https://matplotlib.org/examples/color/colormaps_reference.html
import h5py
from time import time
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import scipy.linalg as LA
import math

def getcolormap(clr):
	"""Create colormaps, personally adapted"""
	if clr == 'brw':  # Blackredwhite: brw
		a = {'red':   ((0.0, 0.0, 0.0),
									 (0.5, 1.0, 1.0),
									 (1.0, 1.0, 1.0)),
				 'green': ((0.0, 0.0, 0.0),
									 (0.5, 0.0, 0.0),
									 (1.0, 1.0, 0.0)),
				 'blue':  ((0.0, 0.0, 0.0),
									 (0.5, 0.0, 0.0),
									 (1.0, 1.0, 0.0))
				 }
	elif clr == 'bgw':  # Blackgreenwhite: bgw
		a = {'red': ((0.0, 0.0, 0.0),
								 (0.5, 0.0, 0.0),
								 (1.0, 1.0, 1.0)),
				 'green': ((0.0, 0.0, 0.0),
									 (0.5, 1.0, 1.0),
									 (1.0, 1.0, 0.0)),
				 'blue': ((0.0, 0.0, 0.0),
									(0.5, 0.0, 0.0),
									(1.0, 1.0, 0.0))
				 }
	elif clr == 'bbw':  # Blackbluewhite: bbw
		a = {'red': ((0.0, 0.0, 0.0),
								 (0.5, 0.0, 0.0),
								 (1.0, 1.0, 1.0)),
				 'green': ((0.0, 0.0, 0.0),
									 (0.5, 0.0, 0.0),
									 (1.0, 1.0, 1.0)),
				 'blue': ((0.0, 0.0, 0.0),
									(0.5, 1.0, 1.0),
									(1.0, 1.0, 1.0))
				 }
	elif clr == 'gray':  # Gray
		a = {'red': ((0.0, 0.0, 0.0),
								 (0.5, 0.5, 0.5),
								 (1.0, 1.0, 1.0)),
				 'green': ((0.0, 0.0, 0.0),
									 (0.5, 0.5, 0.5),
									 (1.0, 1.0, 1.0)),
				 'blue': ((0.0, 0.0, 0.0),
									(0.5, 0.5, 0.5),
									(1.0, 1.0, 1.0))
				 }
	elif clr == 'blackorangewhite':
		a = {'red': ((0.0, 0.0, 0.0),
								 (0.5, 1.0, 1.0),
								 (1.0, 1.0, 1.0)),
				 'green': ((0.0, 0.0, 0.0),
									 (0.5, 0.6, 0.6),
									 (1.0, 1.0, 1.0)),
				 'blue': ((0.0, 0.0, 0.0),
									(0.5, 0.0, 0.0),
									(1.0, 1.0, 1.0))
				 }
	elif clr == 'blackyellowwhite':
		a = {'red': ((0.0, 0.0, 0.0),
								 (0.5, 1.0, 1.0),
								 (1.0, 1.0, 1.0)),
				 'green': ((0.0, 0.0, 0.0),
									 (0.5, 1.0, 1.0),
									 (1.0, 1.0, 1.0)),
				 'blue': ((0.0, 0.0, 0.0),
									(0.5, 0.0, 0.0),
									(1.0, 1.0, 1.0))
				 }
	elif clr == 'blackpinkwhite':
		a = {'red': ((0.0, 0.0, 0.0),
								 (0.5, 1.0, 1.0),
								 (1.0, 1.0, 1.0)),
				 'green': ((0.0, 0.0, 0.0),
									 (0.5, 0.0, 0.0),
									 (1.0, 1.0, 1.0)),
				 'blue': ((0.0, 0.0, 0.0),
									(0.5, 1.0, 1.0),
									(1.0, 1.0, 1.0)),
				 }
	elif clr == 'blackturquoisewhite':
		a = {'red': ((0.0, 0.0, 0.0),
								 (0.5, 0.0, 0.0),
								 (1.0, 1.0, 1.0)),
				 'green': ((0.0, 0.0, 0.0),
									 (0.5, 1.0, 1.0),
									 (1.0, 1.0, 1.0)),
				 'blue': ((0.0, 0.0, 0.0),
									(0.5, 1.0, 1.0),
									(1.0, 1.0, 1.0))
				 }
	elif clr == 'bb':
		a = {'red': ((0.0, 0.0, 0.0),
								 (0.5, 0.0, 0.0),
								 (1.0, 1.0, 1.0)),
				 'green': ((0.0, 0.0, 0.0),
									 (0.5, 0.0, 0.0),
									 (1.0, 1.0, 1.0)),
				 'blue': ((0.0, 0.0, 0.0),
									(0.5, 0.5, 0.5),
									(1.0, 1.0, 1.0))
				}
	elif clr == 'br':
		a = {'red': ((0.0, 0.0, 0.0),
								 (0.5, 0.5, 0.5),
								 (1.0, 1.0, 1.0)),
				 'green': ((0.0, 0.0, 0.0),
									 (0.5, 0.0, 0.0),
									 (1.0, 1.0, 1.0)),
				 'blue': ((0.0, 0.0, 0.0),
									(0.5, 0.0, 0.0),
									(1.0, 0.0, 0.0))
				}
	elif clr == 'bg':
		a = {'red': ((0.0, 0.0, 0.0),
								 (0.5, 0.0, 0.0),
								 (1.0, 1.0, 1.0)),
				 'green': ((0.0, 0.0, 0.0),
									 (0.5, 0.5, 0.5),
									 (1.0, 1.0, 1.0)),
				 'blue': ((0.0, 0.0, 0.0),
									(0.5, 0.0, 0.0),
									(1.0, 1.0, 1.0))
				}
	elif clr == 'whiteblue':
		a = {	'red': ((0.0, 1.0, 1.0),
									(1.0, 0.0, 0.0)),
					'green': ((0.0, 1.0, 1.0),
									 (1.0, 0.0, 0.0)),
					'blue': ((0.0, 1.0, 1.0),
									 (1.0, 1.0, 1.0)),
				 }
	elif clr == 'whiteyellow':
		a = {	'red': ((0.0, 1.0, 1.0),
									(1.0, 1.0, 1.0)),
					'green': ((0.0, 1.0, 1.0),
									 (1.0, 1.0, 1.0)),
					'blue': ((0.0, 1.0, 1.0),
									 (1.0, 0.0, 0.0)),
				 }
	return LinearSegmentedColormap(clr, a)

def gettemp(fullpath):
    """Return the temperature of Miasole large grain low res series"""
    scannr = fullpath[-6:-3]
    if scannr == '221':
        return 16
    elif scannr == '224':
        return 40
    elif scannr == '228':
        return 60
    elif scannr == '232':
        return 80
    elif scannr == '251':
        return 100
    elif scannr == '254':
        return 18.2
    elif scannr == '261':
        return 18.2
    elif scannr == '262':
        return 18.2
    elif scannr == '263':
        return 18.2


def getsavename(mapstr, element, fluxnorm, fitnorm):
    """Return the name to save a figure, without suffix"""
    return "map" + mapstr + "_" + element + "_" + fluxnorm + "_" + fitnorm

def zlabel(element):
    """Return string with label for z-axis (colorscale)"""
    if element == 'xbiv':
        return 'X-ray beam induced voltage  $(\mu V)$'
    elif element == 'diffxbiv':
        return 'Diff. XBIV  (.)'
    elif element == 'xbic':
        return 'X-ray beam induced current  (a.u.)'
    else:
        # return 'Area density of ' + elementstring(element) + '  $(\mu g/cm^2)$'
        return 'Area density of ' + elementstring(element) + '  $(nmol/cm^2)$'

def elementstring(element):
    """Return string of element of interest"""
    if element[-2:] == '_L' or element[-2:] == '_M':
        return element[:-2]
    else:
        return element

def print_map_info(mapnr, element, xpixsize, ypixsize, xrange, yrange, mapmin, mapmax):
    """Print size information of XRF map"""
    print("Map nr: " + str(mapnr) + ", element " + element + ":")
    print("  Map min: " + str(mapmin) + " ug/cm2")
    print("  Map max: " + str(mapmax) + " ug/cm2")
    print("  Pixel size Y: " + str(xpixsize) + " um")
    print("  Pixel size Y: " + str(ypixsize) + " um")
    print("  Map size Y: " + str(xrange) + " um")
    print("  Map size Y: " + str(yrange) + " um")

def print_MAPS_H5_file_content(fullpath):
	"""Print groups of H5 files created by XRF measurements by MAPS"""
	f = h5py.File(fullpath, 'r')
	for dataset in f: # Loop through all elements in the h5 file
		print("Groups in file:  " + dataset)
	g = f['/MAPS'] # This is the group with the relevant data in the h5 file
	channel_names = f['/MAPS/channel_names']
	print("Content of '/MAPS': ")
	for dataset in g: # Loop through all elements in g
		h = f['/MAPS/' + dataset]
		print(h)
	for element in channel_names:
		print(element)
	return 0

def scaletoflux(fluxmeas):
    """Select to which flux measurement shall be scaled to"""
    if fluxmeas == 'ds_ic': # scale data to ds_ic
        return 0
    elif fluxmeas == 'us_ic': # scale data to us_ic
        return 1
    elif fluxmeas == 'SRcurrent': # scale data to SRcurrent
        return 2
    else: # No scale indicator --> don't scale
        return -1

def e2i(channel_names, element):
	"""Convert element to the index for a given measurement"""
	j=0
	for i in channel_names:
		if i.decode('utf-8') == element:  # Need to decode the binary string i
			return j
		j += 1
	print("Element not found in 'channel_names', sorry for that!")
	return -1

def isnan(num):
    """Return true if the argument is NaN, otherwise return false"""
    return num != num

def getMW(Z):
		
    mark = 0
    for i in range(len(Z)):
        if Z[i] == '_':
            mark = i

    if mark != 0:
        Z = Z[0:mark]

    periodic= dict(H=1.01, Na=22.99, Mg=24.31, Al=26.98, Si=28.09, P=30.97, S=32.07, Cl=35.45, Ar=39.95, K=39.10,
                   Ca=40.08, Sc=44.96, Ti=47.87, Cr=52.00, Mn=54.94, Fe=55.85, Co=58.93, Ni=58.69, Cu=63.55, Zn=65.41,
                   Ga=69.72, Ge=72.64, As=74.92, Se=78.96, Br=79.90, Kr=83.80, Sr=87.62, Y=88.90, Zr=91.22, Nb=92.91,
                   Mo=95.94, Tc=98, Ru=101.07, Rh=102.91, Pd=106.42, Ag=107.87, Cd=112.41, In=114.82, Sn=118.71,
                   Sb=121.76, Te=127.6, I=126.90, Xe=131.29, Ta=180.95, W=183.84, Pt=195.08, Au=196.97, Hg=200.59,
                   Pb=207.2)
    return periodic[Z]

def silhouette_analysis(X,range_n_clusters):
	from sklearn.cluster import KMeans
	from sklearn.metrics import silhouette_samples, silhouette_score
	import matplotlib.pyplot as plt
	import matplotlib.cm as cm
	for n_clusters in range_n_clusters:
		# Create a subplot with 1 row and 2 columns
		fig, (ax1, ax2) = plt.subplots(1, 2)
		fig.set_size_inches(18, 7)

		# The 1st subplot is the silhouette plot
		# The silhouette coefficient can range from -1, 1 but in this example all
		# lie within [-0.1, 1]
		ax1.set_xlim([-0.1, 1])
		# The (n_clusters+1)*10 is for inserting blank space between silhouette
		# plots of individual clusters, to demarcate them clearly.
		ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

		# Initialize the clusterer with n_clusters value and a random generator
		# seed of 10 for reproducibility.
		clusterer = KMeans(n_clusters=n_clusters, random_state=10)
		cluster_labels = clusterer.fit_predict(X)

		# The silhouette_score gives the average value for all the samples.
		# This gives a perspective into the density and separation of the formed
		# clusters
		silhouette_avg = silhouette_score(X, cluster_labels)
		print("For n_clusters =", n_clusters,
					"The average silhouette_score is :", silhouette_avg)

		# Compute the silhouette scores for each sample
		sample_silhouette_values = silhouette_samples(X, cluster_labels)

		y_lower = 10
		for i in range(n_clusters):
				# Aggregate the silhouette scores for samples belonging to
				# cluster i, and sort them
				ith_cluster_silhouette_values = \
						sample_silhouette_values[cluster_labels == i]

				ith_cluster_silhouette_values.sort()

				size_cluster_i = ith_cluster_silhouette_values.shape[0]
				y_upper = y_lower + size_cluster_i

				color = cm.spectral(float(i) / n_clusters)
				ax1.fill_betweenx(np.arange(y_lower, y_upper),
													0, ith_cluster_silhouette_values,
													facecolor=color, edgecolor=color, alpha=0.7)

				# Label the silhouette plots with their cluster numbers at the middle
				ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

				# Compute the new y_lower for next plot
				y_lower = y_upper + 10  # 10 for the 0 samples

		ax1.set_title("The silhouette plot for the various clusters.")
		ax1.set_xlabel("The silhouette coefficient values")
		ax1.set_ylabel("Cluster label")

		# The vertical line for average silhouette score of all the values
		ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

		ax1.set_yticks([])  # Clear the yaxis labels / ticks
		ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

		# 2nd Plot showing the actual clusters formed
		colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
		ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
								c=colors)

		# Labeling the clusters
		centers = clusterer.cluster_centers_
		# Draw white circles at cluster centers
		ax2.scatter(centers[:, 0], centers[:, 1],
								marker='o', c="white", alpha=1, s=200)

		for i, c in enumerate(centers):
				ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

		ax2.set_title("The visualization of the clustered data.")
		ax2.set_xlabel("Feature space for the 1st feature")
		ax2.set_ylabel("Feature space for the 2nd feature")

		plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
									"with n_clusters = %d" % n_clusters),
								 fontsize=14, fontweight='bold')

	plt.show()
	
def Test_GMM_Clusters(X,n_components_range):
	#pulled from the Sklearn Documentation Page
	"""
	================================
	Gaussian Mixture Model Selection
	================================

	This example shows that model selection can be performed with
	Gaussian Mixture Models using information-theoretic criteria (BIC).
	Model selection concerns both the covariance type
	and the number of components in the model.
	In that case, AIC also provides the right result (not shown to save time),
	but BIC is better suited if the problem is to identify the right model.
	Unlike Bayesian procedures, such inferences are prior-free.

	In that case, the model with 2 components and full covariance
	(which corresponds to the true generative model) is selected.
	"""
	import itertools
	from scipy import linalg
	import matplotlib as mpl
	from sklearn import mixture

	lowest_bic = np.infty
	bic = []
	cv_types = ['spherical', 'tied', 'diag', 'full']
	for cv_type in cv_types:
		for n_components in n_components_range:
			# Fit a Gaussian mixture with EM
			gmm = mixture.GaussianMixture(n_components=n_components,
																		covariance_type=cv_type)
			gmm.fit(X)
			bic.append(gmm.bic(X))
			if bic[-1] < lowest_bic:
				#print 'check', n_components, cv_type
				lowest_bic = bic[-1]
				best_gmm = gmm

	bic = np.array(bic)
	color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
																'darkorange','black','gray','red'])
	clf = best_gmm
	bars = []

	# Plot the BIC scores
	spl = plt.subplot(2, 1, 1)
	for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
		xpos = np.array(n_components_range) + .2 * (i - 2)
		bars.append(plt.bar(xpos, bic[i * len(n_components_range):
																(i + 1) * len(n_components_range)],
																width=.2, color=color))
	plt.xticks(n_components_range)
	plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
	plt.title('BIC score per model')
	xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
					.2 * np.floor(bic.argmin() / len(n_components_range))
	plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
	spl.set_xlabel('Number of components')
	spl.legend([b[0] for b in bars], cv_types)

	# Plot the winner
	splot = plt.subplot(2, 1, 2)
	Y_ = clf.predict(X)
	for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
																					 color_iter)):
		v, w = linalg.eigh(cov)
		if not np.any(Y_ == i):
				continue
		plt.scatter([X[Y_ == i, 0]], [X[Y_ == i, 1]], .8, color=color)

		# Plot an ellipse to show the Gaussian component
		angle = np.arctan2(w[0][1], w[0][0])
		angle = 180. * angle / np.pi  # convert to degrees
		v = 2. * np.sqrt(2.) * np.sqrt(v)
		ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
		ell.set_clip_box(splot.bbox)
		ell.set_alpha(.5)
		splot.add_artist(ell)

	plt.xticks(())
	plt.yticks(())
	plt.title('Selected GMM: full model, 2 components')
	plt.subplots_adjust(hspace=.35, bottom=.02)
	return best_gmm
	plt.show()

def plotmap(fullpath, element, fluxnorm, fitnorm):
	"""Plot a H5 files created by XRF measurements by MAPS"""

	"""Load file and relevant h5 groups"""
	f = h5py.File(fullpath, 'r')
	# f = h5py.File(r"C:\MSdata\postdoc\studies\beamtimes\2017-03 APS 2-ID-D\img.datfit\2idd_0256.h5", 'r')
	# print_MAPS_H5_file_content(f)
	channel_names = f['/MAPS/channel_names']  # Names of fitted channels such as 'Cu', 'TFY', 'La_L'
	scaler_names = f['/MAPS/scaler_names']  # Names of scalers such as 'SRcurrent', 'us_ic', 'ds_ic', 'deadT', 'x_coord'
	scalers = f['/MAPS/scalers']  # Scaler values for [scaler, x, y]
	XRF_fits = f['/MAPS/XRF_fits']  # Quantified channel [channel, x, y]
	XRF_fits_quant = f['/MAPS/XRF_fits_quant']  # Number of cts per ug/cm2 [???, ???, channel]
	XRF_roi = f['/MAPS/XRF_roi']  # Quantified channel [channel, x, y]
	XRF_roi_quant = f['/MAPS/XRF_roi_quant']  # Number of cts per ug/cm2 [???, ???, channel], to be used as sum ROI in maps instead of XRF_fits_quant
	x_axis = f['/MAPS/x_axis']  # x position of pixels  [position in um]
	y_axis = f['/MAPS/y_axis']  # y position of pixels  [position in um]

	"""Select to which flux measurement shall be scaled to"""
	fluxnormmatrix = scalers[1, :, :];  fluxnormmatrix[:, :] = 1  # Matrix of scalers size but all 1 for no normalization
	if fluxnorm == 'ds_ic':  # scale data to ds_ic
		fluxnormindex = 0; fluxnormmatrix = scalers[fluxnormindex, :, :]
	elif fluxnorm == 'us_ic':  # scale data to us_ic
		fluxnormindex = 1; fluxnormmatrix = scalers[fluxnormindex, :, :]
	elif fluxmeas == 'SRcurrent':  # scale data to SRcurrent
		fluxnormindex = 2; fluxnormmatrix = scalers[fluxnormindex, :, :]

	if element == 'xbiv' or element == 'xbic':
		m = scalers[2, :, :]  # 0: SRcurrent; 1: us_ic; 2: ds_ic
	else:
		"""Select XRF element of interest"""
		elementindex = e2i(channel_names, element)

		"""Select which fitting to be used for quantification"""
		if fitnorm == 'roi':  # Normalization with ROI fitted --> To be used when maps creates a problem with the quantification
				fitnormvalue = XRF_roi_quant[fluxnormindex, 0, elementindex]
				rawmatrix = XRF_roi[elementindex, :, :]
		elif fitnorm == 'fit':  # Normalization with fitted data --> is default if fit works fine
				fitnormvalue = XRF_fits_quant[fluxnormindex, 0, elementindex]
				rawmatrix = XRF_fits[elementindex, :, :]

	"""Calculate quantified element matrix in ug/cm2"""
	m = rawmatrix / fluxnormmatrix / fitnormvalue
	"""Preparation for proper map scaling"""
	xrange = max(x_axis) - min(x_axis)
	xpixsize = xrange / (len(x_axis) - 1)
	yrange = max(y_axis) - min(y_axis)
	ypixsize = yrange / (len(y_axis) - 1)

	"""Remove column with nan"""
	m = m[:, 0:len(x_axis)-1]
	yrange = yrange - ypixsize
	x = np.arange(-xrange / 2 - xpixsize, xrange / 2, xpixsize)
	y = np.arange(-yrange / 2 - xpixsize, yrange / 2, ypixsize)
	return m,x,y

def ABSORB(Beam_Theta,Detector_Theta,Beam_Energy,t):

	import xraylib 
	
	xraylib.XRayInit()
	xraylib.SetErrorMessages(0)

	def GetMaterialMu(E, data): # send in  the photon energy and the dictionary holding the layer information
		Ele = data['Element']
		Mol = data['MolFrac']
		t = 0
		for i in range(len(Ele)):
				t += xraylib.AtomicWeight(xraylib.SymbolToAtomicNumber(Ele[i]))*Mol[i]
		mu=0
		for i in range(len(Ele)):
				mu+= (xraylib.CS_Total(xraylib.SymbolToAtomicNumber(Ele[i]),E) * 
							xraylib.AtomicWeight(xraylib.SymbolToAtomicNumber(Ele[i]))*Mol[i]/t)
		return mu # total attenuataion w/ coherent scattering in cm2/g

	def Density(Material):# send a string of the compound of interest
		if Material == 'ZnO':
				return 5.6 #g/cm3
		elif Material == 'CIGS':
				return 5.75 #g/cm3
		elif Material == 'ITO':
				return 7.14 #g/cm3
		elif Material == 'CdS':
				return 4.826 #g/cm3
		elif Material == 'Kapton': # http://physics.nist.gov/cgi-bin/Star/compos.pl?matno=179
				return 1.42 #g/cm3 
		elif Material == 'SiN':
				return 3.44 #g/cm3
		if Material == 'Mo':
				return 10.2 #g/cm3
			
	def GetLayerInfo(Layer): #send in a string to get typical layer thickness and dictionary of composition
		um_to_cm = 10**-4
		
		if Layer == 'ZnO':
				mat = {'Element':['Zn','O'],'MolFrac':[1,1]}
				t = 0.2*um_to_cm
				return mat,t
		elif Layer == 'CdS':
				mat = {'Element':['Cd','S'],'MolFrac':[1,1]}
				t = 0.05*um_to_cm
				return mat,t
		elif Layer == 'Kapton':
				mat = {'Element':['H','C','N','O'],'MolFrac':[0.026362,0.691133,0.073270,0.209235]} # http://physics.nist.gov/cgi-bin/Star/compos.pl?matno=179
				t = 26.6*um_to_cm #measured using profilometer
				return mat, t
		elif Layer == 'ITO':
				mat = {'Element':['In','Sn','O'],'MolFrac':[1.8,0.1,2.9]} #90% In2O3 #10% SnO2
				t = 0.15*um_to_cm
				return mat,t
		elif Layer == 'Mo':
				mat = {'Element':['Mo'],'MolFrac':[1]}
				t = 0.7*um_to_cm
				return mat,t

			
	def GetFluorescenceEnergy(Element,Beam): # send in the element and the beam energy to get the Excited Fluorescence Energy 
	 #this will return the highest energy fluorescence photon capable of being excited by the beam
		Z = xraylib.SymbolToAtomicNumber(Element)
		F = xraylib.LineEnergy(Z,xraylib.KA1_LINE)
		if xraylib.EdgeEnergy(Z,xraylib.K_SHELL) > Beam:
				F = xraylib.LineEnergy(Z,xraylib.LA1_LINE)
				if xraylib.EdgeEnergy(Z,xraylib.L1_SHELL) > Beam:
						F = xraylib.LineEnergy(Z,xraylib.LB1_LINE)
						if xraylib.EdgeEnergy(Z,xraylib.L2_SHELL) > Beam:
								F = xraylib.LineEnergy(Z,xraylib.LB1_LINE)
								if xraylib.EdgeEnergy(Z,xraylib.L3_SHELL) > Beam:
										F = xraylib.LineEnergy(Z,xraylib.LG1_LINE)
										if xraylib.EdgeEnergy(Z,xraylib.M1_SHELL) > Beam:
												F = xraylib.LineEnergy(Z,xraylib.MA1_LINE)
		return F

	def GetIIO(Layer,Energy):
			ROI,t = GetLayerInfo(Layer)
			return np.exp(-Density(Layer)*GetMaterialMu(Energy,ROI)*t)
	
	#conversion factor
	um_to_cm = 10**-4	
	t = t*um_to_cm

	##Set incident Beam Energy and Detector Geometry
	# Beam_Theta = 90 #degrees
	# Detector_Theta = 47 #degrees
	# Beam_Energy = 10.5 #keV
	# Get Layor of interest information
	L = 'CIGS'
	ROI = {'Element':['Cu','In','Ga','Se'],'MolFrac':[0.8,0.8,0.2,2]}
	Elem = ROI['Element']

	# define sublayers thickness and adjust based on measurement geometry
	dt = 0.01*um_to_cm # 10 nm stepsizes
	steps = int(t/dt);
	T = np.ones((steps,1))*dt 
	beam_path = T/np.sin(Beam_Theta*np.pi/180)
	fluor_path = T/np.sin(Detector_Theta*np.pi/180)

	# initialize variables to hold correction factors
	iio = [None]*steps
	factors = [None]*len(Elem)
	#print 'For a film thickness of ', t/um_to_cm, 'microns:'
	#loop over sublayers for self attenuation and top layer attenuation
	ti = time()
	for ind,Z in enumerate(Elem):
			for N in range(steps):
					beam_in = -Density(L)*GetMaterialMu(Beam_Energy,ROI)*beam_path[0:N]
					beam_out = -Density(L)*GetMaterialMu(GetFluorescenceEnergy(Z,Beam_Energy),ROI)*fluor_path[0:N]
					iio[N] = np.exp(np.sum(beam_in+beam_out))
			factors[ind] = np.sum(iio)/N * GetIIO('Kapton',Beam_Energy) * GetIIO('Kapton',GetFluorescenceEnergy(Z,Beam_Energy))
			#print 'The absorption of ', Z, 'in', L,'at beam energy', Beam_Energy,'is', round(factors[ind]*100,2)
	#print 'Calculation Time = ', round(time()-ti,2),'s for ', steps, 'iterations on ', ind+1,' elements'
	return factors

def getstats(varin):
	print 'Max:\t', round(np.max(varin),4)
	print 'Min:\t', round(np.min(varin),4)
	print 'Mean:\t', round(np.mean(varin),4)
	print 'Standard Deviation:\t', round(np.std(varin),4)
	try: 
		print 'Dimension:\t', varin.shape
	except AttributeError:
		print 'Dimension:\t', len(varin)

def get_PL_Raman_Current_Voltage_From_Matlab_File(fpath):
	import scipy.io
	data = scipy.io.loadmat(fpath, mat_dtype=1) #mat_dtype=1 ensures that vars are imported as arrays
	#print data.keys()
	
	x_coord_array = data['X']
	y_coord_array = data['Y']
	
	PL = data["PL"]
	RAMAN = data["RAMAN"]
	
	wavenumber = data["wavenumber"]
	wavelength = data["wavelength"]
	
	wn = wavenumber.ravel()
	wl = wavelength.ravel()
	
	current = data["CURRENT"]
	voltage = data["VOLTAGE"]
	
	x_step_size = round(np.diff(x_coord_array,axis=0).max(),3)
	y_step_size = round(np.diff(y_coord_array,axis=0).max(),3)

	x_pixels = int((x_coord_array.max() - x_coord_array.min())/x_step_size + 1)
	y_pixels = int((y_coord_array.max() - y_coord_array.min())/y_step_size + 1)
	
	x = (np.arange(x_pixels)-x_pixels/2)*x_step_size
	y = (np.arange(y_pixels)-y_pixels/2)*y_step_size
	
	return data,PL,RAMAN,wl,wn,current,voltage,x,y,x_pixels,y_pixels
"""Baseline estimation algorithms."""
def baseline(y, deg=None, max_it=None, tol=None):
	"""
	Computes the baseline of a given data.

	Iteratively performs a polynomial fitting in the data to detect its
	baseline. At every iteration, the fitting weights on the regions with
	peaks are reduced to identify the baseline only.

	Parameters
	----------
	y : ndarray
		Data to detect the baseline.
	deg : int (default: 3)
		Degree of the polynomial that will estimate the data baseline. A low
		degree may fail to detect all the baseline present, while a high
		degree may make the data too oscillatory, especially at the edges.
	max_it : int (default: 100)
		Maximum number of iterations to perform.
	tol : float (default: 1e-3)
		Tolerance to use when comparing the difference between the current
		fit coefficients and the ones from the last iteration. The iteration
	procedure will stop when the difference between them is lower than
	*tol*.

	Returns
	-------
	ndarray
	Array with the baseline amplitude for every original point in *y*
	"""
	
	# for not repeating ourselves in `envelope`
	if deg is None: deg = 3
	if max_it is None: max_it = 100
	if tol is None: tol = 1e-3

	order = deg + 1
	coeffs = np.ones(order)

	# try to avoid numerical issues
	cond = math.pow(y.max(), 1. / order)
	x = np.linspace(0., cond, y.size)
	base = y.copy()

	vander = np.vander(x, order)
	vander_pinv = LA.pinv2(vander)

	for _ in range(max_it):
		coeffs_new = np.dot(vander_pinv, y)

		if LA.norm(coeffs_new - coeffs) / LA.norm(coeffs) < tol:
			break

		coeffs = coeffs_new
		base = np.dot(vander, coeffs)
		y = np.minimum(y, base)

	return base

	def envelope(y, deg=None, max_it=None, tol=None):
		"""
		Computes the upper envelope of a given data.
		It is implemented in terms of the `baseline` function.

		Parameters
		----------
		y : ndarray
				Data to detect the baseline.
		deg : int
				Degree of the polynomial that will estimate the envelope.
		max_it : int
				Maximum number of iterations to perform.
		tol : float
				Tolerance to use when comparing the difference between the current
				fit coefficients and the ones from the last iteration.

		Returns
		-------
		ndarray
				Array with the envelope amplitude for every original point in *y*
		"""
		return y.max() - baseline(y.max() - y, deg, max_it, tol)

	
	