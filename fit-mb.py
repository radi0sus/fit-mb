#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##########################################################################################
import sys                                                       #sys
import os                                                        #os file processing
import argparse                                                  #argument parser
import matplotlib.pyplot as plt                                  #plots
import numpy as np                                               #summation and other math

from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons     #widgets
from lmfit.models import LinearModel, Model                                   #fit
from lmfit import Parameters                                                  #fit
from tabulate import tabulate                                                 #nice tables
##########################################################################################
print_in_sigma = False                                  #print data in 1 sigma and 3 sigma
plot_3s_band   = False                                  #plot the 3 sigma band 
##########################################################################################
class text_colors: 
    #term colors
    RED = '\033[31m'
    ENDC = '\033[m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'

#for windows console
os.system('')                              #colors
sys.stdout.reconfigure(encoding = 'utf-8') #unicode

#global lists
colors = ['red','blue','green','orange','cyan','olive'] #line colors 


def lorentzdoublet(x,area1,ishift,qsplit,fwhm):
    #function for the Lorentz doublet
    #area1  = area
    #ishift = I.S. or δ in mm/s
    #qsplit = Q.S. or ΔEQ in mm/s
    #fwhm   = fwhm     
    doublet = ((area1/np.pi)*((fwhm/2)/((ishift-abs(qsplit)/2-x)**2+(fwhm/2)**2))) + \
              ((area1/np.pi)*((fwhm/2)/((ishift+abs(qsplit)/2-x)**2+(fwhm/2)**2)))
    return doublet

def bg_func(x,y0):
    #bg or offset of the Lorentz function
    return y0 + 0 * x

def normalize_y(y):
    #normalize y values
    normalized_y_data = y/max(y)
    return normalized_y_data

##########################################################################################
#argument parser                                                                         #
#parse arguments                                                                         #
##########################################################################################
parser = argparse.ArgumentParser(prog = 'fit-mb', 
                          description = 'Easily fit Mößbauer (MB) spectra')

#filename is required
parser.add_argument('filename', 
                    help = 'file with MB parameters and location of the MB data file')

#parse arguments
args = parser.parse_args()
    
##########################################################################################
#import data & open parameter file                                                       #
##########################################################################################
#open parameter file
def op_im(file):
    nucnamelist = list()                     #list of atom / compound names
    ishiftlist = list()                      #list of isomeric shifts (deltas)
    deltaeqlist = list()                     #list with Delta-EQ
    ratiolist = list()                       #list with ratios of several MB active nuclei
    fwhmlist = list()                        #list with individual fwhm 
    try:
        with open(file, 'r') as input_file:
            for line in input_file:
                #ignore blank lines
                if line.strip():
                    #ignore comments
                    if not line.startswith('#'):
                        #name of the file which contains MB data
                        if line.startswith("MB-data"):
                            #expects the data file after 'MB-data = '
                            data_file=line.strip().split()[2]
                        else:
                            #add parameters to several lists
                            nucnamelist.append(line.strip().split()[0])
                            ishiftlist.append(float(line.strip().split()[1]))
                            deltaeqlist.append(abs(float(line.strip().split()[2])))
                            try:
                                fwhmlist.append(float(line.strip().split()[3]))
                            except IndexError:
                                fwhmlist.append(0.1)
                            try:
                                ratiolist.append(float(line.strip().split()[4]))
                            except IndexError:
                                ratiolist.append(0.1)
    #file not found -> exit here
    except IOError:
        print(f"'{args.filename}'" + " not found")
        sys.exit(1)
    #strings where should be numbers -> exit here
    except ValueError:
        print('Warning! Numerical value in parameter file expected. Exit.')
        sys.exit(1)
    #no values found or unknown parameters-> exit here
    except IndexError:
        print('Warning! Value in parameter file missing or unkown '
              + 'instruction or blank line. Exit.')
        sys.exit(1)
    #import MB data with delimiter ','
    try:
        data = np.loadtxt(data_file, delimiter=',')
        x = data[:, 0]
        y = data[:, 1]
    #file not found -> exit here
    except IOError:
        print(f"'{data_file}'" + " not found")
        sys.exit(1)
    #key word in parameter file missing -> exit here
    except NameError:
        print(f'Warning! Missing "MB-data" line in "{args.filename}". Exit.')
        sys.exit(1)
    #in case the delimiter is spaces instead of ','
    except ValueError:
        try: 
            data = np.loadtxt(data_file)
            x = data[:, 0]
            y = data[:, 1]
        #wrong delimiter or strange data formats -> exit here
        except ValueError:
            print('Warning! Numerical value expected or wrong delimiter in '
                + '.dat file. Exit.')
            sys.exit(1)
    #no start values -> exit here
    if len(ishiftlist) == 0:
        print('Warning! At least one species with start values '
              + 'for δ and ΔEQ is excpected. Exit.')
        sys.exit(1)
    return x, y, data_file, nucnamelist, ishiftlist, deltaeqlist, fwhmlist, ratiolist   


x, y, data_file, nucnamelist, ishiftlist, \
      deltaeqlist, fwhmlist, ratiolist = op_im(args.filename)

#normalize y data
y = normalize_y(y)

##########################################################################################
#fit                                                                                     #
##########################################################################################
def do_the_fit():
    #summation of Lorentz doublets for every doublet or singlet
    #doublets are in the list
    doublist = list()
    #init parameters for fit
    params = Parameters()
    #number of doublets or singlets = number of species in parameter file
    #in case the fit fails, boundary values (min/max) or initial values (value)
    #can be changed here
    #'vary' of I.S. and Q.S. depends on the selection in the matplotlib window
    #area1  = area
    #ishift = I.S. or δ in mm/s
    #qsplit = Q.S. or ΔEQ in mm/s
    #fwhm   = fwhm  
    
    #same label for different species in parameter file
    if len(ishiftlist) != len(isfixlist):
        print('Warning! Labeling in parameter file is not unique. Exit.')
        sys.exit(1)
    
    for index in range(len(ishiftlist)): 
        doubmodel = Model(lorentzdoublet, prefix='d'+str(index)+"_")
        params.update(doubmodel.make_params(area1 = {'value':0.1, 
                                                    'min':-1.00001, 'max':1e-5, 
                                                    'vary':True},
                                          ishift = {'value':ishiftlist[index], 
                                                    'min':-3, 'max':3, 
                                                    'vary':not isfixlist[index]}, 
                                          qsplit = {'value':deltaeqlist[index],
                                                    'min':1e-5, 'max':5, 
                                                    'vary':not qsfixlist[index]},
                                          #fwhm   = {'value':0.3, 
                                          #          'min':1e-5, 'max':3, 
                                          #          'vary':True})) 
                                          fwhm   = {'value':fwhmlist[index], 
                                                    'min':1e-5, 'max':3, 
                                                    'vary':not fwhmfixlist[index]}))
        doublist.append(doubmodel)
    
    #add constant value y0 for bg or offset for the Lorentz function
    bg = Model(bg_func)
    bg_params = bg.make_params(y0 = {'value':1, 'min': 0.8, 'max':1.2, 'vary':True})
    #the final curve = all Lorentz doublets or singlets + bg
    curve = np.sum(doublist) + bg
    #fit
    result = curve.fit(y,params + bg_params,x=x)
    return result

##########################################################################################
#print results in terminal window                                                        #
##########################################################################################
def print_results(result):
    #print(result.fit_report())  #the whole fit report
    
    #list for the table of results   
    print_results.resultstable=list() 
    
    print('')
    print('# Fit report for ' + filename)
    print('## File statistics:')
    print('MB data     : '  + data_file)
    print('data points : ' + str(result.ndata))
    print('variables   : ' + str(result.nvarys))
    print('')
    #χ² red. χ² or probably maybe not correct because of the definition of the function 
    #and the format of the residuals?; not necessary for the evaluation of the 
    #fit results, χ² gets smaller if the fit gets better
    #red. χ² from ORIGIN is exactly the same
    print('χ²          : ' + '{:.4e}'.format(result.chisqr))
    print('red. χ²     : ' + '{:.4e}'.format(result.redchi))
    #print('nfree   : ' + str(result.nfree))        #Number of free parameters in fit.
    print('R²          : ' + '{:.4}'.format(result.rsquared))
    print('')

    try:
        #'try:' in case there are no errors printed (aka fit failed almost) 
        #sum of all abs(amplitudes) for the calculation of the ratio 
        
        #init of sum of all amps for ratio calculation
        print_results.sum_amp = 0
        
        for index in range(len(ishiftlist)):
            #collect I.S., Q.S., fwhm, and amplitudes 
            ishift_key = 'd'+ str(index) + '_ishift'
            qsplit_key = 'd'+ str(index) + '_qsplit'
            fwhm_key   = 'd'+ str(index) + '_fwhm'
            area1_key   = 'd'+ str(index) + '_area1'
            #print(u'{:.3fP}'.format(result.uvars[ishift_key]))
            #print(u'{:.3fP}'.format(result.uvars[qsplit_key]))
            #sum amplitudes
            print_results.sum_amp += abs(result.uvars[area1_key])
            #append to table with results
            print_results.resultstable.append([nucnamelist[index],
                                u'{:.3fP}'.format(result.uvars[ishift_key]),
                                u'{:.3fP}'.format(result.uvars[qsplit_key]),
                                u'{:.3fP}'.format(result.uvars[fwhm_key]),
                                ])
            if result.uvars[qsplit_key].n >= 4.99:
                #very large error
                #If a single Lorentz (ΔEQ close to or 0) is fitted as doublet
                print(text_colors.RED  + 'Warning! ΔEQ is at the limit. ' 
                     + 'Fit results are probably wrong! \n'
                     + 'Check whether a doublet has been fitted '
                     + 'instead of a singlet. \n' + text_colors.ENDC)
        for index in range(len(ishiftlist)):
            #calculate the ratio of each species and append to table with results 
            area1_key   = 'd'+ str(index) + '_area1'
            print_results.resultstable[index].append(u'{:.2fP}'.format(
                                   abs(result.uvars[area1_key])/print_results.sum_amp*100))
        
        #calculate the ratio of MB active species from the are of the curves      
        #get fit results  
        fitted_params = result.params.valuesdict()
        #y0 (offset or bg) fit result
        y0 = fitted_params['y0']
        #since it goes from y0 (around 1) to 0, the total area (rectangle) has 
        #to be calculated first 
        #the area of each curve is the total area - curve area
        #total_area = np.trapz(np.full(len(x), y0),x,0.001)
        #np.ptp = range; e.g. -4 to 4 = 8
        total_area = np.ptp(x)*y0
        #integration using the composite trapezoidal rule
        #data_area =  total_area - np.trapz(y,x,0.001)
        fit_area =  total_area - np.trapz(result.best_fit,x,0.001)
        #extract components (individual MB doublets for each species) from the fit results
        comps = result.eval_components(x=x)
        
        #calculate the ratio of each species
        for index,component in enumerate(comps):
            if not component == 'bg_func':
                comp_area = total_area - np.trapz(y0 + comps[component],x,0.001)
                #append to table with results
                print_results.resultstable[index].append('{:#.2f}'.format(
                                                                comp_area/fit_area*100))
        
        #print fit results
        print('## Fit results:')
        
        if print_in_sigma:
            #number of (raw) data points that are in 3 sigma of the fitted curve
            #True or False
            y_in_sigma3 = (y >= (result.best_fit-result.eval_uncertainty(sigma=3))) & \
                          (y <= (result.best_fit+result.eval_uncertainty(sigma=3)))
            #number of (raw) data points that are in 1 sigma of the fitted curve
            #True or False
            y_in_sigma1 = (y >= (result.best_fit-result.eval_uncertainty(sigma=1))) & \
                          (y <= (result.best_fit+result.eval_uncertainty(sigma=1)))
            #multiply True / False with data 
            print_results.y_in3 =  y_in_sigma3 * y 
            print_results.y_in1 =  y_in_sigma1 * y
            #data not in sigma range are now zero (count non-zero = data in sigma range)
            print('data in 1σ  :', np.count_nonzero(print_results.y_in1))
            print('data in 3σ  :', np.count_nonzero(print_results.y_in3))
        #print bg func or offset aka y0
        print('y0          : ' + u'{:.4P}'.format(result.uvars['y0']))
        print('')
        #print the table with results
        print(tabulate(print_results.resultstable,
            #'disable_numparse', otherwise tabulate ignores the formatting
            disable_numparse = True,
            headers=['species', 'δ /mm·s⁻¹','ΔEQ /mm·s⁻¹', 'fwhm /mm·s⁻¹', 
                     'r (area)/%', 'r (int)/%'], 
            stralign="decimal",
            tablefmt='github',
            showindex=False))
        
        #printing was successful 
        print_results.results_printed = True
      
    except(AttributeError):
        #it didn't work (aka fit failed almost)
        print(text_colors.RED + 'It appears that the fit has failed.')
        print('Try again with better initial values.')
        print('Or change the boundary values in the script.' + text_colors.ENDC)
        #printing was not successful 
        print_results.results_printed = False

print_results.results_printed = False

##########################################################################################
#plot results of the fit in lower plot area (ax1)                                        #
##########################################################################################

def plot_results(result, sum_amp, results_printed):
    #plot the fitted data
    #refresh
    ax1.clear()
    #set axes labels (after clear)
    ax1.set_ylabel('relative transmission')
    ax1.set_xlabel(r'velocity /mm$\cdot$s$^{-1}$')
    
    #get fit results
    fitted_params = result.params.valuesdict()
    #get y0 (offset) from fit
    y0 = fitted_params['y0']
    
    #extract components (individual MB doublets for each species) from the fit results 
    comps = result.eval_components(x=x)

    #if there is only one component, plots of the component 
    #and the overall best fit should have the same color
    if len(ishiftlist) == 1:
        best_fit_color=colors[0]
    else:
        best_fit_color='black'
   
    #color list / cycle from the top
    ax1.set_prop_cycle(color = colors)
    
    #plot the (raw) data 
    ax1.plot(x, 
            y,
            '.',
            color='steelblue')
    #plot the residuals + extra to get it above the other plots
    ax1.plot(x, 
            result.residual + max(result.residual) + 1, 
            linestyle = (0, (1, 1)), 
            color='grey', 
            label='residuals')
    #plot the 'best fit'
    #the 'best fit' is the sum off all components (including y0)
    ax1.plot(x, 
            result.best_fit,'-', 
            color=best_fit_color, 
            label='best fit ' + '('+r'$R^2 =$ ' + '{:.4}'.format(result.rsquared)+')')
    #fill the area of the best fit
    ax1.fill_between(x, 
            result.best_fit,
            y0,
            color='steelblue',
            alpha=0.1)
            
    #plot individual components of the fit, but not the bg_func or y0 
    #(which is also a component)  
    for index, component in enumerate(comps):
        #for the labels, but only if the complete results have been printed 
        #(see remarks above)
        #individual components need the y0 correction, since y0 is also a component
        #the 'best fit' is the sum off all components (including y0)
        #and is it no necessary (or wrong) to add y0 again
        area1_key   = 'd'+ str(index) + '_area1'
        ishift_key = 'd'+ str(index) + '_ishift'
        qsplit_key = 'd'+ str(index) + '_qsplit'
        if not component == 'bg_func' and results_printed == True:
            ax1.plot(x, 
                    (y0 + comps[component]),
                    label = nucnamelist[index] +
                    ' (' + '{:.1f}'.format(abs((result.uvars[area1_key])/sum_amp*100).n) + 
                    '%): ' +   '\n'                 
                    r'$\delta$ = '+u'{:.2f}'.format(result.uvars[ishift_key].n) + 
                    r', $ΔE_Q =$' +u'{:.2f}'.format(result.uvars[qsplit_key].n) + 
                    #r' mm$\cdot$s$^{-1}$'
                    r' mm/s')
            #fill the area of the component plots
            ax1.fill_between(x, 
                    y0 + (comps[component]), 
                    y0, 
                    alpha=0.1)
        #in case the complete results have not been printed (see remarks above)
        elif not component == 'bg_func' and results_printed == False:
            ax1.plot(x, (y0 + comps[component]),label = nucnamelist[index])
    
    #plot a 3 sigma uncertainty band
    if results_printed == True and plot_3s_band == True:
        un_qs = list()
        for index, component in enumerate(comps):
            qsplit_key = 'd'+ str(index) + '_qsplit'
            if not component == 'bg_func':
                    un_qs.append(result.uvars[qsplit_key].s)
        if max(un_qs) < 1:
            #don't plot the 3 sigma band if uncertainty (of Q.S.) is to high
            dely = result.eval_uncertainty(sigma=3)
            ax1.fill_between(x, 
                result.best_fit - dely, result.best_fit + dely, 
                color='grey', 
                alpha=0.5,
                label='3-$\sigma$ uncertainty band')
    
    #if results_printed == True:
    #    #color data points within 3 sigma
    #    #bit darker than the remaining data points
    #    print_results.y_in3[print_results.y_in3 == 0] = 'nan'
    #    ax1.plot(x,print_results.y_in3,'.',color='darkblue', alpha=0.3)
            
    #optimize position of the legend
    leg = ax1.legend(fancybox = True, shadow = True, loc='best', prop={'size': 6})
    #change line width in legend 
    for legobj in leg.legend_handles:
        legobj.set_linewidth(2.0)
    #plotting was successful 
    plot_results.fit_plotted = True
    #refresh
    fig.canvas.draw_idle()

plot_results.fit_plotted = False
    
##########################################################################################
#define interactive elements or widgets for the matplotlib window: buttons, sliders, ... #
##########################################################################################

#radio buttons for the selection of individual doublets
#'.inset_axes' is a experimental feature
#if it fails, change here (but it is so nice)
def radio(plot_list):
    ax0_radio = ax0.inset_axes([0.0, 0.0, 0.20, 0.4])
    radio = RadioButtons(
        ax=ax0_radio,
        labels=([l for l in lines_by_label.keys()]),
        label_props={'color': line_colors,'fontsize': [11 for l in lines_by_label.keys()]},
        activecolor = line_colors)
    ax0_radio.axis('off')
    return radio

#sliders for convenient value adjustments
def sliders(ln):
    ax0_slider = plt.axes(arg=[0.11,0.13,0.45,0.03], facecolor='blue')
    slider_is  = Slider(ax0_slider, 
                        label = '$\delta$ (I.S.)', 
                        color='darkred', 
                        valmin = -6, valmax=6, 
                        valinit = ishiftlist[plot_list.index(ln)])
    ax0_slider = plt.axes(arg=[0.11,0.09,0.45,0.03], facecolor='blue')
    slider_qs  = Slider(ax0_slider, 
                        label = '$\Delta E_Q$ (Q.S.)', 
                        color='darkred', 
                        valmin = 1e-5, valmax=6, 
                        valinit = deltaeqlist[plot_list.index(ln)])
    ax0_slider = plt.axes(arg=[0.11,0.05,0.45,0.03], facecolor='blue')
    slider_fw  = Slider(ax0_slider, 
                        label = 'fwhm', 
                        valmin = 1e-5, valmax=2, 
                        valinit = fwhmlist[plot_list.index(ln)])
    ax0_slider = plt.axes(arg=[0.11,0.01,0.45,0.03], facecolor='blue')
    slider_ra  = Slider(ax0_slider, 
                        label = 'ratio', 
                        valmin = 1e-5, valmax=1, 
                        valinit = ratiolist[plot_list.index(ln)])
    return slider_is, slider_qs, slider_fw, slider_ra

#buttons
def buttons():
    ax0_button = plt.axes(arg=[0.75,0.11,0.15,0.05])
    fit_button = Button(
                 ax=ax0_button,
                 label='Fit',
                 color='lightgrey',
                 hovercolor='limegreen'
                 )

    ax0_button = plt.axes(arg=[0.75,0.06,0.15,0.05])
    save_button = Button(
                 ax=ax0_button,
                 label='Save',
                 color='lightgrey',
                 hovercolor='cornflowerblue'
                 )

    ax0_button = plt.axes(arg=[0.75,0.01,0.15,0.05])
    exit_button = Button(
                 ax=ax0_button,
                 label='Exit',
                 color='lightgrey',
                 hovercolor='tomato'
                 )
    return fit_button, save_button, exit_button

#checkbuttons for the definition of fixed fit values
def cbuttons():
    ax0_cbuttons = plt.axes(arg=[0.63,0.084,0.07,0.075])
    fixis_cbuttons = CheckButtons(
                    ax=ax0_cbuttons,
                    labels=[' fix $\delta$'],
                    actives=[False]
                    )
    ax0_cbuttons.axis('off')

    ax0_cbuttons = plt.axes(arg=[0.63,0.042,0.07,0.075])
    fixqs_cbuttons = CheckButtons(
                    ax=ax0_cbuttons,
                    labels=[' fix $\Delta E_Q$'],
                    actives=[False]
                    )                
    ax0_cbuttons.axis('off')

    ax0_cbuttons = plt.axes(arg=[0.63,0.0,0.07,0.075])
    fixfwhm_cbuttons = CheckButtons(
                       ax=ax0_cbuttons,
                       labels=[' fix fwhm'],
                       actives=[False]
                       )                
    ax0_cbuttons.axis('off')
    return fixis_cbuttons, fixqs_cbuttons, fixfwhm_cbuttons

##########################################################################################
#create the main matplotlib window with plot areas and widgets                           #
##########################################################################################
    
#fig, (ax0, ax1) = plt.subplots(2,1, sharex=True, figsize=((21/2.54), (29.7/2.54))) #A4
#standard plot window is rather small on large monitors; change it here
#the plot window
#two plot areas (data, adjust and fit)
#ax0 = raw data + doublets/singlets from parameters
#ax1 = raw data + fitted curves + ...
fig, (ax0, ax1) = plt.subplots(2,1, sharex=True)
#space for slider
plt.subplots_adjust(bottom=0.25)
#adjust placement (in case)
#plt.subplots_adjust(wspace=0, hspace=0.1, top=0.92)
#plt.subplots_adjust(top=0.99, right=0.99)
#get the file name for the title
filename, file_extension = os.path.splitext(data_file)
#plot title = data filename
titles = fig.suptitle(filename, fontsize=14)           

fig.text(0.95,0.02,'https://github.com/radi0sus/fit-mb', fontsize = 8, rotation = 270, 
         url = 'https://github.com/radi0sus/fit-mb')

#get the line colors from top
ax0.set_prop_cycle(color = colors)
#titles
ax0.set_title('Top: adjust parameters        Bottom: fit result', fontsize='11')
#labeling 
ax1.set_xlabel(r'velocity /mm$\cdot$s$^{-1}$')
ax0.set_ylabel('relative transmission')
ax1.set_ylabel('relative transmission')

#the list of plots
plot_list=list()

#plot the (raw) data in ax0   
ax0.plot(x, 
        y, 
        '.',
        color='steelblue')
#fill the area (raw data) in ax0
#y0 is assumed to be 1 before the fit
ax0.fill_between(x, 
        y, 
        1,
        color='steelblue',
        alpha=0.08)   

#plot lorentz doublets/singlets from input values in ax0
#y0 is assumed to be 1 before the fit
#append to plot_list
for index in range(len(ishiftlist)):
     l, = ax0.plot(x,
                   1-lorentzdoublet(x,(1-min(y))*ratiolist[index],
                   ishiftlist[index],deltaeqlist[index],fwhmlist[index]),
                   label=nucnamelist[index])
     plot_list.append(l)

#get the labels from input file
lines_by_label = {l.get_label(): l for l in plot_list}
#get the colors
line_colors = [l.get_color() for l in lines_by_label.values()]
#generate radio buttons
#active species is selected using the radio buttons
#parameters can be changed with the sliders
radio = radio(plot_list)
#which label from 'radio' has been selected
ln = lines_by_label[radio.value_selected]
#generate sliders
slider_is, slider_qs, slider_fw, slider_ra = sliders(ln)
#generate checkbuttons
fixis_cbuttons, fixqs_cbuttons, fixfwhm_cbuttons = cbuttons()
#generate buttons
fit_button, save_button, exit_button = buttons()

#init lists for the values from checkbuttons (all 'False')
#'False' means the fit parameter is not fixed
isfixlist   = [False for l in lines_by_label.keys()]
qsfixlist   = [False for l in lines_by_label.keys()]
fwhmfixlist = [False for l in lines_by_label.keys()]

#update plots (after 'slider' adjustments)
#y0 is assumed to be 1 before the fit
def update_data_plot(ln):
    ln.set_ydata(1-lorentzdoublet(x,(1-min(y))*ratiolist[plot_list.index(ln)], 
               ishiftlist[plot_list.index(ln)],
               deltaeqlist[plot_list.index(ln)],
               fwhmlist[plot_list.index(ln)]))
    fig.canvas.draw_idle()

##########################################################################################
#define actions when using the widgets                                                   #
##########################################################################################

def update_is(val):
    #I.S. slider changed
    ln = lines_by_label[radio.value_selected]
    ishiftlist[plot_list.index(ln)] = slider_is.val
    update_data_plot(ln)

def update_qs(val):
    #Q.S. slider changed
    ln = lines_by_label[radio.value_selected]
    deltaeqlist[plot_list.index(ln)] = slider_qs.val
    update_data_plot(ln)

def update_fw(val):
    #fwhm slider changed
    ln = lines_by_label[radio.value_selected]
    fwhmlist[plot_list.index(ln)] = slider_fw.val
    update_data_plot(ln)

def update_ra(val):
    #ratio slider changed
    ln = lines_by_label[radio.value_selected]
    ratiolist[plot_list.index(ln)] = slider_ra.val
    update_data_plot(ln)

def radio_changed(label):
    #radio changed
    ln = lines_by_label[label]
    #set slider values according to the selected ('radio') species 
    slider_is.set_val(ishiftlist[plot_list.index(ln)])
    slider_qs.set_val(deltaeqlist[plot_list.index(ln)])
    slider_ra.set_val(ratiolist[plot_list.index(ln)])
    slider_fw.set_val(fwhmlist[plot_list.index(ln)])
    
    #same label for different species in parameter file
    if len(ishiftlist) != len(isfixlist):
        print('Warning! Labeling in parameter file is not unique. Exit.')
        sys.exit(1)
    
    #set fixed values according to the selected ('radio') species 
    if fixis_cbuttons.get_status()[0] != isfixlist[plot_list.index(ln)]:
         fixis_cbuttons.set_active(0)

    if fixqs_cbuttons.get_status()[0] != qsfixlist[plot_list.index(ln)]:
         fixqs_cbuttons.set_active(0)
         
    if fixfwhm_cbuttons.get_status()[0] != fwhmfixlist[plot_list.index(ln)]:
         fixfwhm_cbuttons.set_active(0)    
    
def callback_fit(label):
    #Fit button
    #do the fit -> print results -> plot results
    callback_fit.result = do_the_fit()
    print_results(callback_fit.result)
    #plot only if printing was successful (fit not failed)
    plot_results(callback_fit.result, print_results.sum_amp, 
                 print_results.results_printed)
    
def callback_exit(label):
    #Exit button
    sys.exit(1)

def callback_save(label):
    #Save button
    #save only if results have been printed (after successful fit)
    if print_results.results_printed:
        #save the MB parameter file
        save_params(args.filename, callback_fit.result, print_results.sum_amp)
        #save a report file
        save_report(data_file, callback_fit.result, print_results.resultstable, 
                    print_results.sum_amp)
        #save all data in a 'csv'-like file
        save_csv(data_file)
        #save the fit plot (ax1)
        save_plot(data_file)
    else:
        #no fit, no save
        print('"Fit" before "Save".')
    
def callback_fixis(label):
    #fix I.S.
    ln = lines_by_label[radio.value_selected]
    isfixlist[plot_list.index(ln)]=fixis_cbuttons.get_status()[0]

def callback_fixqs(label):
    #fix Q.S.
    ln = lines_by_label[radio.value_selected]
    qsfixlist[plot_list.index(ln)]=fixqs_cbuttons.get_status()[0]

def callback_fixfwhm(label):
    #fix fwhm
    ln = lines_by_label[radio.value_selected]
    fwhmfixlist[plot_list.index(ln)]=fixfwhm_cbuttons.get_status()[0]

##########################################################################################
#from 'Save' button, save plot, parameters, report and data_file                         #
##########################################################################################

def save_plot(file):
    #save the plot (if there is one)
    #a "hidden" plot window, that will be saved
    if plot_results.fit_plotted:
        #filename
        filename, file_extension = os.path.splitext(file)
        #obtain the matplotlib window dimensions to restore 
        #the matplotlib window after saving the plot
        plSize = params.get_size_inches()
        #plot dimensions
        params.set_size_inches(15*0.5,30*0.5)
        #plot title
        ax1.set_title(filename, fontsize='11')
        #plot lable
        ax1.set_xlabel(r'velocity /mm$\cdot$s$^{-1}$')
        #save but do not show the plot window
        fit_plot = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        #there is no error handling here
        plt.savefig(f"{filename}-fit.png", 
                    bbox_inches = fit_plot.expanded(1.3, 1.3), dpi = 300)
        #go back to the starting window dimensions
        ax1.set_title('', fontsize='11')
        params.set_size_inches((plSize[0], plSize[1]))
        #plot saved message
        print(f"{filename}-fit.png" + ' saved.')
        #plt.show()

def save_report(file, result, resultstable, sum_amp):
    #save the fit report
    #filename
    filename, file_extension = os.path.splitext(file)
    try:
        report_filename = filename + '-report.txt'
        with open(report_filename, 'w', encoding='utf-8') as report_file:
            report_file.write('# Fit report for ' + filename + '\n')
            report_file.write('## File statistics:' + '\n')
            report_file.write('MB data     : '  + data_file + '    \n')
            report_file.write('data points : ' + str(result.ndata) + '    \n')
            report_file.write('variables   : ' + str(result.nvarys) + '    \n')
            report_file.write('\n')
            report_file.write('χ²          : ' + '{:.4e}'.format(result.chisqr) 
                                               + '    \n')
            report_file.write('red. χ²     : ' + '{:.4e}'.format(result.redchi) 
                                               + '    \n')
            report_file.write('R²          : ' + '{:.4}'.format(result.rsquared) 
                                               + '    \n')
            report_file.write('\n')
            report_file.write('## Fit results:' + '\n')
            
            if print_in_sigma:
                report_file.write('data in 1σ  : ' 
                + '{}'.format(np.count_nonzero(print_results.y_in1)) + '   \n')
                report_file.write('data in 3σ  : ' 
                + '{}'.format(np.count_nonzero(np.nan_to_num(print_results.y_in3))) 
                + '   \n')
                
            report_file.write('y0          : ' + u'{:.4P}'.format(result.uvars['y0']) 
                                               + '    \n')
            report_file.write('\n')
            report_file.write(tabulate(resultstable, 
                              disable_numparse = True,
                              headers = ['species', 'δ /mm·s⁻¹','ΔEQ /mm·s⁻¹', 
                                        'fwhm /mm·s⁻¹', 'r (area)/%', 'r (int)/%'],    
                              stralign = 'decimal',
                              tablefmt = 'github',
                              showindex = False))
            report_file.write('\n')
            report_file.write('\n')
            report_file.write('![' + filename + '-fit.png](' + filename + '-fit.png)' 
                                   + '\n')
            #report saved message
            print(report_filename + ' saved.')
    #write error -> exit here
    except IOError:
        print("Report could not be saved. Exit.")
        sys.exit(1)

def save_params(file, result, sum_amp):
    #save parameter file
    paramname, param_extension = os.path.splitext(file)
    #some 'tricky' instructions to more or less restore the original parameter file 
    #except for the fitted parameters
    written_lines=list()
    i = 0
    try:
    #open the parameter file again
        with open(file, "r", encoding='utf-8') as param_in_file:
            #param_in_file_contents = param_in_file.readlines()
            param_in_file_contents = [line.rstrip('\n') for line in param_in_file]
    #open error -> exit here
    except IOError:
        print("Parameter file could not be opened. Exit.")
        sys.exit(1)        
    
    try:
        #filename
        #save the new parameter file
        param_filename = paramname + '-fit' + param_extension
        with open(param_filename, 'w', encoding='utf-8') as param_out_file:
            for line in param_in_file_contents:
                for index, nucname in enumerate(nucnamelist):
                    ishift_key = 'd'+ str(index) + '_ishift'
                    qsplit_key = 'd'+ str(index) + '_qsplit'
                    fwhm_key   = 'd'+ str(index) + '_fwhm'
                    area1_key  = 'd'+ str(index) + '_area1'
                    if (str(i)+line) in written_lines:
                        i += 1
                        break
                    elif line.startswith(nucname):
                        param_out_file.write(nucname + '       ' +
                                             u'{:.3f}'.format(result.uvars[ishift_key].n) +
                                             '   ' +
                                             u'{:.3f}'.format(result.uvars[qsplit_key].n) +
                                             '   ' +
                                             u'{:.3f}'.format(result.uvars[fwhm_key].n) +
                                             '   ' +
                                             '{:.3f}'.format(abs(result.uvars[area1_key]/sum_amp).n) +
                                             '\n')
                        written_lines.append(line)
                    elif not line.startswith(tuple(nucnamelist)):
                        param_out_file.write(line + '\n')
                        written_lines.append(str(i)+line)
        #parameter file saved message                
        print(param_filename + ' saved.')
    #write error -> exit here
    except IOError:
        print("Parameter file could not be saved. Exit.")
        sys.exit(1) 

def save_csv(file):
    #save data in a csv-like format
    #get data from the plot window
    #filename
    filename, file_extension = os.path.splitext(file)
    csv_filename = filename + '-fit.dat'
    #get raw data in array
    xy_data = [ax1.lines[0].get_xdata().tolist()]
    #collect line (fitted curves) data in an array
    for no_plot in range(len(ax1.lines)):
        xy_data.append(ax1.lines[no_plot].get_ydata())
    #transpose array, to get it in the right format
    xylist= np.array(xy_data).T.tolist()
    try:
        #write the csv file (it sould look nice)
        with open(csv_filename, 'w') as output_file:
            output_file.write('{: <12}'.format('#velocity') + '{: <11}'.format('data')  + 
                               '{: <11}'.format('residuals') + 
                               '{: <1}'.format('fit        ') +
                                ' '.join('{: <10}'.format(i) for i in nucnamelist) + '\n')
                                    
            for elements in range(len(xylist)):
                #output_file.write(str(','.join(map(str,xylist[elements]))) +'\n')
                output_file.write(str
                           ('    '.join(['{:.5f}'.format(i) for i in (xylist[elements])])) 
                           +'\n')
        #csv file saved message
        print(csv_filename + ' saved.')
    #write error -> exit here
    except IOError:
        print("Write error. Exit.")
        sys.exit(1)

##########################################################################################
#event handling, interaction with widgets                                                #
##########################################################################################

#radio
radio.on_clicked(radio_changed)
#sliders
slider_is.on_changed(update_is)
slider_qs.on_changed(update_qs)
slider_ra.on_changed(update_ra)
slider_fw.on_changed(update_fw)
#fit, save and exit buttons
fit_button.on_clicked(callback_fit)
save_button.on_clicked(callback_save)
exit_button.on_clicked(callback_exit)
#I.S., Q.S. and fwhm fix check buttons
fixis_cbuttons.on_clicked(callback_fixis)
fixqs_cbuttons.on_clicked(callback_fixqs)
fixfwhm_cbuttons.on_clicked(callback_fixfwhm)
#show the plot (resize) (and close it)
N = 1.2
params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches((plSize[0]*N, plSize[1]*N*1.5))
plt.show()
plt.close('all')
