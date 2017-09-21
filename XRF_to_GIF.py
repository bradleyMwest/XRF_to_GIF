import os
from generalfunctions import plotmap
import matplotlib.pyplot as plt
import numpy as np
import imageio # lets you make gifs
from matplotlib.colors import LinearSegmentedColormap

########################################################################
################ MAKE CUSTOM COLOR MAPS#################################
########################################################################
a = {'red': ((0.0, 0.0, 0.0), #Black to Red Color Map
             (0.5, 0.0, 0.0),
             (1.0, 0.0, 0.0)), 
     'green': ((0.0, 0.0, 0.0),
             (0.5, 0.0, 0.0),
             (1.0, 0.0, 0.0)),
     'blue': ((0.0, 0.0, 0.0),
             (0.5, 0.5, 0.5),
             (1.0, 1.0, 1.0)),
    }
clr = 'bb'
bb = LinearSegmentedColormap(clr, a)

a = {'red': ((0.0, 0.0, 0.0),#Black to Green Color Map
             (0.5, 0.0, 0.0),
             (1.0, 0.0, 0.0)), 
     'green': ((0.0, 0.0, 0.0),
             (0.5, 0.5, 0.5),
             (1.0, 1.0, 1.0)),
     'blue': ((0.0, 0.0, 0.0),
             (0.5, 0.0, 0.0),
             (1.0, 0.0, 0.0)),
    }
clr = 'bg'
bg = LinearSegmentedColormap(clr, a)

a = {'red': ((0.0, 0.0, 0.0),#Black to Blue Color Map
             (0.5, 0.5, 0.5),
             (1.0, 1.0, 1.0)), 
     'green': ((0.0, 0.0, 0.0),
             (0.5, 0.0, 0.0),
             (1.0, 0.0, 0.0)),
     'blue': ((0.0, 0.0, 0.0),
             (0.5, 0.0, 0.0),
             (1.0, 0.0, 0.0)),
    }
clr = 'br'
br = LinearSegmentedColormap(clr, a)


########################################################################
############### Define Path to File and Scan Nubmers####################
########################################################################

main_path = 'F:/ArgonneData_Fitted/201603-2idd/img_dat/2idd_0'

ALL_SCANS = [['569','570','571','572','573','574','575','576','592'],  #500C NoH2S 10keV
             ['645','646','647','648','649','650','651','652','664'],  #550C NoH2S 10keV
             ['491','492','493','494','495','496','497','498','522'],  #600C YesH2S 10keV
             ['414','415','416','417','418','419','420','421','431'],  #500C Yes H2S 
             ['535','536','537','538','539','540','541','542','552'],  #550C YesH2S
             ['440','441','442','443','444','445','446','447','456'],  #600C NoH2S 10keV
] 


########################################################################
############### Plot the data and save the figures #####################
########################################################################
						
TEMP = [500, 550, 600] ##looked at 3 different temperatures for all elements. Used for naming

elements = ['Cu','In_L','Ga'] # select elements of interst for figures
cntr = 0

for idx,t in enumerate(TEMP):
	scans = ALL_SCANS[idx][1:-1]
	for scan in scans:
		for e in elements:
			print 'Iteration number: ', cntr, '\t Temp: ',t, '\t Scan: ',scan,'\t Element: ', e
			cntr+=1
			fpath =  main_path + scan + '.h5'
			z, x, y = plotmap(fpath,e,'us_ic','fit')

			if e == 'Cu':
				cmap = br
				ylab = 'Copper Concentration (nmol/cm$^2$)'
			elif e == 'In_L':
				cmap = bb
				ylab = 'Indium Concentration (nmol/cm$^2$)'
			elif e == 'Ga':
				cmap =  bg
				ylab = 'Gallium Concentration (nmol/cm$^2$)'
			
			# Generate figure with size and plot XRF map
			fig, ax = plt.subplots(1,1,figsize=(3,3))
			pcm = ax.pcolor(y, x,z, cmap=cmap)#getcolormap(cmap))  #, vmin=vmina, vmax=vmaxa)  # <-- Things happen here
			
			#remove axes and excess white space around figure
			plt.gca().set_axis_off()
			plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
			plt.margins(0,0)
			plt.gca().xaxis.set_major_locator(plt.NullLocator())
			plt.gca().yaxis.set_major_locator(plt.NullLocator())
			
			#save figure and close it
			plt.savefig(e + '_' + str(t) + '_' + scan + '.tif', bbox_inches='tight',pad_inches = 0)
			plt.close(fig)

########################################################################
############### Make GIFs and delete the old images ####################
########################################################################

#Generate 1 gif file for each element and temperature
for e in elements:
	for idx,t in enumerate(TEMP):
		images = [] # intialize image frames for gif
		scans = ALL_SCANS[idx][1:-1] #remove first and last image from scans
		for scan in scans:
			# path to image file
			filename = e + '_' + str(t) + '_' + scan + '.tif' 
			
			# add image file to gif frames
			images.append(imageio.imread(filename)) 
			
			# delete the old image file
			os.remove(filename) 
		
		# save the images toa  gif file with a 50 ms frame duration  
		imageio.mimsave(e + '_' + str(t) + '.gif', images, duration=0.5)
