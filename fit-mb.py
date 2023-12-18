#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# cal-mb.py
#
# example usage:
# 
# python3 cal-mb.py 57Fe_calib_raw_data.ws5
# or
# python3 cal-mb.py 57Fe_calib_raw_data.ws5 -s
#
# '-s' to show the plot window 


##########################################################################################
import sys                                  #sys
import os                                   #os file processing
import datetime                             #calc mod. time of .ws5 from timestamp
import argparse                             #argument parser

import matplotlib.pyplot as plt             #plots
import numpy as np                          #summation and other math

from lmfit.models import LorentzianModel, ConstantModel         #fit
from lmfit import Parameters                                    #fit

from scipy.signal import find_peaks, peak_prominences           #peak finding
from scipy import interpolate               #interpolation of channel intens. for folding
##########################################################################################

#for windows console
sys.stdout.reconfigure(encoding = 'utf-8') #unicode

##########################################################################################
#import data                                                                             #
########################################################################################## 
def op_imp(data_file):
    try:
        #read raw data into a list
        ws5_raw_list = list()
        with open(data_file, 'r') as input_file:
            #get modification date and time of .ws5 file
            mod_date = datetime.datetime.fromtimestamp(os.path.getmtime(data_file))
            for line in input_file:
                #if no '<' than its data
                if not line.startswith('<'):
                #if not '<' in line:
                    try:
                        #add intensities from .ws5 file to list
                        ws5_raw_list.append(float(line.strip()))
                    except ValueError:
                        #wrong data -> exit here 
                        print('Wrong data format. Exit')
                        sys.exit(1)
    except IOError:
        #file not found -> exit here
        print(f"'{data_file}'" + " not found. Exit.")
        sys.exit(1)
    #return array with intensity data from MCA (.ws5), filename and modification 
    #date and time of the .ws5 file 
    return np.array(ws5_raw_list), data_file, mod_date.strftime("%d.%m.%Y %H:%M:%S")      

##########################################################################################
#fit data                                                                                #
########################################################################################## 
def do_the_fit(data):
    #fit data with N Lorentz functions; N = number of detected peaks in raw data
    #get the number of channels
    N_chan = len(data)
    #channels from 1 to max channel, step = 1 channel
    chan = np.linspace(1, N_chan, N_chan)
    #find peaks in raw data; has to be '-' since they are minima (not maxima)
    peaks, _ = find_peaks(-data)
    #get prominences = peak separation from noise
    prominences = peak_prominences(-np.array(data), peaks)[0]
    #identify peaks from noise
    #normal max(prominences)/3; found an example where only max(prominences)/4 worked
    peaks, _ = find_peaks(-data, prominence=(max(prominences)/4, max(prominences)))
    #generate a list of Lorentz functions
    lorentzlist = list()
    #initialize parameters for lmfit
    params = Parameters()
    
    #number of Lorentz functions = number of detected peaks 
    for index, peak in enumerate(peaks): 
        #define Lorentz function for lmfit
        lorentz = LorentzianModel(prefix='s'+str(index)+"_") 
        #guess parameters for Lorentz function 
        params.update(lorentz.guess(data, x = chan))
        #update parameters for Lorentz function, x = x from detected peak (channel), 
        #amplitude = y from detected peak (intensity of raw data)
        params.update(lorentz.make_params(center = chan[peak], amplitude = data[peak]))
        #add to list of Lorentz functions
        lorentzlist.append(lorentz)
    
    #constant model for bg or offset (y0)
    bg = ConstantModel()
    #start is max intensity from raw data
    bg_params = bg.make_params(c = {'value':max(data), 'vary':True})
    
    #complete model for lmfit: list of Lorentz functions + offset
    curve = np.sum(lorentzlist) + bg
    #do the fit
    fit_result = curve.fit(data, params + bg_params, x = chan)
    #return fit results
    return fit_result
    
##########################################################################################
#calc FP or v0                                                                           #
##########################################################################################
def calc_FP_v0(fit_result):
    #calc FP or v0 from the mean of the centers of the fitted Lorentz functions
    #a way to estimate the number of Lorentz functions (and centers) from the fit results
    n_center = int((len(fit_result.params)-1)/5)
    #generate a list of centers
    centerlist=list()
    for index in range(n_center):
        centerkey = 's'+ str(index) + '_center'
        #get the postion (channel number) of the center of each Lorentz function
        centerlist.append(fit_result.uvars[centerkey])
    #mean of all centers
    FP_v0 = np.mean(centerlist)
    #return FP (folding point) or v0 (channel where the velocity is zero 
    #and the list with centers (for later labeling in the plot)
    return FP_v0, centerlist

##########################################################################################
#fold spectrum                                                                           #
##########################################################################################
def fold_spec(data, FP):
    #fold the spectrum
    #folding_diff = (FP.nominal_value - 256.5)*2
    #get the number of channels
    N_chan = len(data)
    #'(FP - 256.5)*2' for 512 channels, if channel 1 is 1 (and not zero)
    folding_diff = (FP.nominal_value - (int(N_chan/2)+0.5))*2
    #found an example where folding_diff < 0; abs() correct?
    if folding_diff < 0:
        folding_diff = abs(folding_diff)
    #channels from 1 to max channel, step 1 channel 
    chan = np.linspace(1, N_chan, N_chan)
    #interpolate channels, to operate with channel floating point numbers (xxx.xx)
    data_ichan = interpolate.interp1d(chan, data, bounds_error=True, kind = 'linear')
    #lhs (left hand side) channels; note that it goes from high to low
    lhs_chan = np.linspace(int(N_chan/2), 1, int(N_chan/2))
    #rhs (right hand side) channels 
    rhs_chan = np.linspace(int(N_chan/2) + 1, N_chan, int(N_chan/2))
    #add the intensities of lhs + folding difference and rhs channels pairwise
    folded_intens = (np.add(data_ichan(lhs_chan+folding_diff), data_ichan(rhs_chan)))
    #return intensities of the folded spectrum
    return folded_intens

##########################################################################################
#calc vmax                                                                               #
##########################################################################################
def calc_vmax(fit_result, data):
    #calc vmax
    #get the number of channels
    N_chan = len(data)
    #multiplicator for velocity per channel (f) to get vmax; it is 127.5 for 256 channels
    #from the manual of mcal from E.B.
    chan_mul = ((N_chan) / 2 - 1) / 2               #127.5 for 265
    #a way to estimate the number of Lorentz functions (and centers) from the fit results
    n_center = int((len(fit_result.params)-1)/5)
    #generate a list of centers
    centerlist=list()
    
    for index in range(n_center):
        centerkey = 's'+ str(index) + '_center'
         #get the postion (channel number) of the center of each Lorentz function
        centerlist.append(fit_result.uvars[centerkey])
    
    #from the manual of mcal from E.B.
    #the well known quadrupole splitting values of 57Fe
    #
    #---------------------------------------------------
    #   |       |       |         |       |       |
    #   |       |       |<-1.667->|       |       |
    #   |       |                         |       |
    #   |       |<---------6.167--------->|       |
    #   |                                         |
    #   |<----------------10.657----------------->|
    #   |                  mm/s                   |
    #
    #f is the velocity / channel
    if n_center == 6:
        #folded spectrum with 3 doublets 
        #but here it is the difference of the single Lorentz functions form the outermost
        #to the innermost pair
        #no recalculation with doublets like in mcal from E.B.
        #could not observe a large difference in the final parameters 
        f = 10.657 / (centerlist[0] - centerlist[5]) + \
             6.167 / (centerlist[1] - centerlist[4]) + \
             1.677 / (centerlist[2] - centerlist[3])
        f = (f / 3) # mean f
        #f * chan_mul is vmax
        vmax = f * chan_mul
    elif n_center == 4: 
        #folded spectrum with 2 doublets 
        f = 6.167 / (centerlist[0] - centerlist[3]) + \
            1.677 / (centerlist[1] - centerlist[2])
        f = (f / 2) # mean f
        vmax = f * chan_mul
    elif n_center == 2:
        #folded spectrum with 1 doublet 
        f = 1.667 / (centerlist[0] - centerlist[1]) 
        vmax = f * chan_mul
    else:
        #limited to 6, 4 or 2 lines (3, 2, 1 doublet(s))
        print('The script can only handle folded spectra with 6, 4 or 2 peaks. Exit.')
        sys.exit(1)
    #return vmax and f (velocity per channel)
    return vmax, abs(f)

##########################################################################################
#plot results                                                                            #
##########################################################################################
def plot(ws5_raw_data, folded_intens, unfolded_spec, folded_spec, FP, v0, vmax, f, 
        filename, centerlist_FP, centerlist_v0, mod_date):
    #plot the results
    #two subplots (unfolded & folded)
    fig, (ax0, ax1) = plt.subplots(2, 1)
    #title: filename + modification date of the .ws5 file
    fig.suptitle(filename + ' (' + str(mod_date) + ')')
    #all channel x values 1...512 for example for the upper plot
    x_raw = np.linspace(1,len(ws5_raw_data),len(ws5_raw_data))
    #title sub plot 0
    ax0.set_title('raw spectrum')
    #raw data points
    ax0.plot(x_raw, 
             ws5_raw_data, 
             '.', 
             color = 'steelblue', 
             label = 'raw data')
    #best fit curve
    ax0.plot(x_raw, 
             unfolded_spec.best_fit, 
             '-',
             color ='darkorange', 
             label ='best fit ' 
                    + '('+r'$R^2 =$ ' + '{:.4}'.format(unfolded_spec.rsquared)+')')
    #residuals        
    ax0.plot(x_raw, 
             1 - unfolded_spec.residual + max(ws5_raw_data)*0.01 + max(ws5_raw_data), 
             '.', 
             color = 'lightgrey', 
             label = 'residuals')
             
    #vertical line for FP (folding point) 
    ax0.axvline(x = FP.nominal_value, 
                ls = '--', 
                color = 'black')
    
    #label the FP line
    ax0.annotate('$FP = {:.4f}$'.format(FP.nominal_value),
                 (FP.nominal_value,min(ws5_raw_data)),
                 textcoords = "offset points",
                 xytext=(5,1), 
                 rotation=90)
                 
    #label the centers of the Lorentz functions (used for calculation of FP)
    for index,center in enumerate(centerlist_FP):
        heightkey = 's'+ str(index) + '_height'
        height = max(ws5_raw_data) + unfolded_spec.params[heightkey]
        ax0.annotate('{:.2f}'.format(center.nominal_value),
                     (center.nominal_value, height),
                     textcoords = "offset points",
                     xytext=(8,0), 
                     rotation=90, 
                     size=6)
    
    #label y axis, set ticks and show legend
    ax0.set_ylabel('intensity')
    ax0.set_xticks(np.linspace(1,len(ws5_raw_data),8))
    ax0.set_xlim([1, len(ws5_raw_data)])
    ax0.legend(fancybox = True, shadow = True, loc='upper right', prop={'size': 6})

    #all channel x values 1...256 for example for the lower plot
    x_fold = np.linspace(1,len(folded_intens),len(folded_intens))
    #title of sub plot 1
    ax1.set_title('folded spectrum')
    #folded raw data points
    ax1.plot(x_fold, 
             folded_intens, 
             '.',
             color = 'steelblue', 
             label = 'fld. raw data')
    #best fit curve for folded data
    ax1.plot(x_fold, 
             folded_spec.best_fit, 
             '-', 
             color='darkorange', 
             label='best fit ' 
                   + '('+r'$R^2 =$ ' + '{:.4}'.format(folded_spec.rsquared)+')')
    #residuals for folded data
    ax1.plot(x_fold, 
             1 - folded_spec.residual + max(folded_intens)*0.01 + max(folded_intens), 
             '.', 
             color = 'lightgrey', 
             label = 'residuals')
    #vertical line for v0 (channel where the velocity is zero)
    ax1.axvline(x = v0.nominal_value, 
               ls = '--', 
               color = 'black')
    #label the v0 line
    ax1.annotate('$v_0 = {:.4f}$'.format(v0.nominal_value),
                 (v0.nominal_value,min(folded_intens)),
                 textcoords = "offset points",
                 xytext = (5,1), 
                 rotation = 90)
                 
    #label the centers of the Lorentz functions (used for calculation of v0)
    for index,center in enumerate(centerlist_v0):
        heightkey = 's'+ str(index) + '_height'
        height = max(folded_intens) + folded_spec.params[heightkey]
        ax1.annotate('{:.2f}'.format(center.nominal_value),
                     (center.nominal_value, height),
                     textcoords = "offset points",
                     xytext = (8,0), 
                     rotation = 90, 
                     size = 6)
                     
    #label x axis           
    ax1.set_xlabel('channel no.')
    #label y axis, set ticks and show legend
    ax1.set_ylabel('intensity')
    ax1.set_xticks(np.linspace(1,len(folded_intens),16))
    ax1.set_xlim([1, len(folded_intens)])
    ax1.legend(fancybox = True, shadow = True, loc='upper right', prop={'size': 6})
    
    #print the github link to source and documentation
    ax1.annotate('https://github.com/radi0sus/cal-mb',
                xy = (1.01, 0.5),
                xycoords = 'axes fraction',
                ha = 'left',
                va = 'center',
                rotation = 270,
                fontsize = 8)
                
    #print the results of the fits, FP, v0, vmax and f
    ax1.annotate(u'$FP (\mathrm{{c}}) = {:.4fP}$  '.format(FP) 
               + u'$v_0 (\mathrm{{c}}) = {:.4fP}$  '.format(v0) 
               + u'$v_\mathrm{{max}} = {:.4fP}$ mm/s  '.format(vmax) 
               + u'$f = {:.4fP}$ mm/s per c'.format(f),
                xy = (-0.05, -0.2),
                xycoords = 'axes fraction',
                ha = 'left',
                va = 'center',
                rotation = 0,
                fontsize = 8)
                
    #arrange the plot window and show the plot
    mng = plt.get_current_fig_manager()
    mng.resize(1024,768)
    #(windows) low-res N = 1.2
    #high-res N = 1.5
    N = 1.2
    params = plt.gcf()
    plSize = params.get_size_inches()
    params.set_size_inches((plSize[0]*N*1, plSize[1]*N*1.5))
    plt.tight_layout()
    plt.show()

##########################################################################################
#argument parser                                                                         #
#parse arguments                                                                         #
##########################################################################################
parser = argparse.ArgumentParser(prog = 'cal-mb', 
                          description = 'Easily calibrate Mößbauer (MB) spectra')

#filename is required
parser.add_argument('filename', help = 'file with raw data from 57Fe foil (ws5)')

#show the matplotlib window
parser.add_argument('-s','--show',
    default=0, action='store_true',
    help='show the plot window')
    
#parse arguments
args = parser.parse_args()

##########################################################################################
#main                                                                                    #
##########################################################################################

#import data, filename and modification date of the .ws5 file
ws5_raw_data, filename, mod_date = op_imp(args.filename)

#fit unfolded spectrum
unfolded_spec = do_the_fit(ws5_raw_data)
#calculate FP (folding point)
FP, centerlist_FP = calc_FP_v0(unfolded_spec)
#fold spectrum
folded_intens = fold_spec(ws5_raw_data, FP)
#fit folded spectrum
folded_spec = do_the_fit(folded_intens)
#calculate v0 (channel where velocity is zero)
v0, centerlist_v0 = calc_FP_v0(folded_spec)
#calc vmax (maximum velocity) and f (velocity per channel)
vmax, f = calc_vmax(folded_spec, ws5_raw_data)

#print results, filename and modification date of the .ws5 file
print('-------------------------------------')
print('Results for', filename, ':')
print('File modified on', mod_date)
print('-------------------------------------')
print('FP (channel) =', u'{:.4fP}'.format(FP))
print('v₀ (channel) =', u'{:.4fP}'.format(v0))
print('vmax /mm·s⁻¹ = ', u'{:.4fP}'.format(vmax))
print('f /mm·s⁻¹/c  =  ', u'{:.4fP}'.format(f))
print('-------------------------------------')

if args.show:
    #plot on request
    plot(ws5_raw_data, folded_intens, unfolded_spec, folded_spec, FP, v0, vmax, f, 
    filename, centerlist_FP, centerlist_v0, mod_date)
