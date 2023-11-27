# Fit-MB

A Python 3 script for (hassle-free) fitting Mößbauer (MB) spectra. 

It is easy to use but has some limitations. The script is limited to doublets 
(with no intensity differences) and singlets, cannot handle magnetic properties and is 
restricted to Lorentzian lines shapes. 


## External modules

`lmfit`
`numpy` 
`matplotlib`
`tabulate`

## Quick start

1. Edit or create a parameter file (e.g. `mb-param.text`) with a text editor:

    ```
    ...
    #
    MB-data = example_data.dat
    # 
    # ⇦ just a comment
    ...
    #----------------------------------------
    # label |   δ    |  ΔEQ  | fwhm  | ratio
    #----------------------------------------
    L1Fe      -0.12    2.92    
    L2Fe      -0.09    2.21   
    # adjust    ⇧       ⇧
    ```
    `example_data.dat` is the file that contains data from measurement. 
    First column should contain velocity, second intensity. Recognized delimiters 
    are `,` or space(s).

    The number of labels (e.g. `L1Fe`) is equivalent to the number of MB active species. 
    Furthermore rough estimates of $δ$ (isomer shift) and $ΔE_Q$ (quadrupol splitting) 
    are required.
    
    > If the number of species is insufficient add or remove (or comment `#`) them
    > in the parameter file (e.g. `mb-param.text`) and restart the script as described
    > under **2**.

2. Start the script with:

    ```console
    python3 mb-fit.py mb-param.txt
    ```
    
    After the `matplotlib` window has opened, select a species in the upper legend. Simply 
    click on the **label** or the **radio button**. Adjust parameters for each species 
    with the **sliders** at the bottom of the window.  It should roughly agree with what 
    you expect. You can also **fix** some of the parameters, but in general this is not 
    necessary. Then press the **Fit button**.  
    
    > You can also try to fit the data without adjusting any parameters. Simply click 
    > **Fit**. In this example no alteration of the start parameters was necessary.
    
<img src='examples\start-mod.png' alt='Start'>

3. The result of the fit is displayed in the lower area, parameters and statistics are 
   also displayed in the terminal (console, cmd, std.out). Check if the results are okay. 
   Otherwise adjust the **sliders**  and click the **Fit button** again.
   
<img src='examples\fit.png' alt='Fit'>
    
    Terminal output:
   
    ```
    # Fit report for example_data
    ## File statistics:
    MB data     : example_data.dat
    data points : 256
    variables   : 9
    
    χ²          : 8.3411e-05
    red. χ²     : 3.3770e-07
    R²          : 0.9952

    ## Fit results:
    data in 1σ  : 30
    data in 3σ  : 101
    y0          : 0.9995±0.0001
    
    |   species |    δ /mm·s⁻¹ |   ΔEQ /mm·s⁻¹ |   fwhm /mm·s⁻¹ |   r (area)/% |   r (int)/% |
    |-----------|--------------|---------------|----------------|--------------|-------------|
    |      L1Fe |  0.669±0.016 |   1.637±0.026 |    0.877±0.049 |   20.26±0.93 |       19.85 |
    |      L2Fe | -0.170±0.002 |   2.240±0.003 |    0.519±0.006 |   79.74±0.93 |       80.15 |
    ```

4. Finally save the results by clicking on the **Save** button. The optimized parameters 
   will be saved in `mb-param-fit.txt`. A fit report in `example_data-report.txt` 
   (essentially the same as the last console output), raw data and the fitted curves in
   `example_data-fit.dat` (a file which you can open in Gnuplot, Excel or Origin 
   for example) and the content of the lower plot area in `example_data-fit.png`. 
   
   Terminal output:
   
    ```
    mb-param-fit.txt saved.
    example_data-report.txt saved.
    example_data-fit.dat saved.
    example_data-fit.png saved.
    ```

5. Exit.

## Command-line options

- filename, required: filename, e.g. `mb-param.txt`. A file, that contains the name (and
  location) of the file that contains MB data and start parameters for the fit.
  
## Parameter file

Below is sample parameter file with all necessary information. Important are the 
name (and location) of the file that contains MB data (`MB-data = example_data.dat`) 
and start parameters for the fit. The term `MB-data = ` must not be changed.

Only label, $δ$ and $ΔE_Q$ are essential and the labeling must be unique. Every time 
something has changed in the parameter file, the script must be restarted.    

> Simply add or remove `#` before a label, to include or exclude a MB active species.

> It is strongly recommended to have the parameter file and the data file in the 
> same directory. 

```
#==============================================================================
# Example of a MB parameter file for fit-mb.py
#==============================================================================
# This is a comment.
# Lines starting with '#' are ignored by the script.
#
# The MB (raw) data file should contain 'velocity' (1st column) and 
# 'intensity' (2nd column). Further columns and lines starting 
# with '#' are ignored. Recognized delimiters are ',' or ' ' (whitespace(s)).
# The term 'MB-data = ' must not be changed since it is recognized by
# the script. 
#
MB-data = example_data.dat
#
# The start parameters for the fit must be entered in the following order:
#
# label_1 δ_1 ΔEQ_1 fwhm_1 ratio_1
# label_2 δ_2 ΔEQ_2 fwhm_2 ratio_2
# label_3 δ_3 ΔEQ_3 fwhm_3 ratio_3
# ...
# 
# label =  unique atom / compound name
# δ     =  isomeric shift in mm/s
# ΔEQ   =  quadrupole splitting in mm/s; should be positive
# fwhm  =  full width at half maximum; line width for broadening
# ratio =  ratio of the MB active compound / nucleus
#
# At least one species with label, δ, ΔEQ must be defined.
# The labeling must be unique (e.g. 'Fe1', 'Fe2', ...). Identical labels
# (e.g. 'Fe1', 'Fe1', ...) lead to errors. Labels should be without
# spaces (e.g. 'LFe1' or 'LFe_1' instead of 'LFe 1').
#
# 'fwhm' and 'ratio' are optional. If there is no value for 'fwhm', 
# but there is a value for 'ratio', 'ratio' is considered to be 'fwhm', 
# because the third parameter in the line is assumed to be 'fwhm'.
# 
# If 'fwhm' is not specified it is set to 0.1. If 'fwhm' is not specified 
# then 'ratio' cannot be specified (see above remark).
# 
# If 'ratio' is not specified it is set to 0.1.
# Defining a ratio different from 1 or 100% can be useful if there is 
# a main component and an impurity which is also MB active
# or a mixture of two or more compounds with MB active nuclei. 
# However, setting a 'ratio' as start parameter is just for orientation,
# since the fit starts with a fixed value.
#
#----------------------------------------
# label |   δ    |  ΔEQ  | fwhm  | ratio
#----------------------------------------
#L1Fe      0.24    1.51    0.33    0.40
#L2Fe      0.25    3.12    0.39    0.35
#L3Fe     -0.03    1.24    0.44    0.12
#L4Fe     -0.06    0.61    0.37    0.13
L1Fe       0.264   2.321   0.434   0.492
L2Fe       0.279   1.551   0.427   0.508
L3Fe       1.334   1.922   0.561   0.203
```

## MB data file

Below is sample data file. Lines starting with `#` are ignored. Only the first two 
columns are considered. The first column must contain the velocity, the second column
must contain the intensity data. Recognized delimiters are `,` and spaces.

```
# filename
# sample: sample name
# 80 K : temperature
# 0 T  : field
# 125.69, -4.693
# ---------------------------
 -4.5893,   587126.4,       1
 -4.5525,   588314.7,       2
 -4.5157,   586286.4,       3
 -4.4789,   586907.1,       4
 -4.4421,   587216.4,       5
 -4.4052,   586408.4,       6
....
```

## Pre-fit plot 

The pre-fit plot is in the upper part of the `matplotlib` window. The active species
is selected in the **legend** of plot. The plot is updated if any of the parameters is 
changed using the **sliders** at the bottom of the window. 
The purpose of the pre-fit plot in combination with the **sliders** is to set the fitting 
parameters so that they roughly reflect the shape of the measured data.

> The most important parameters to be changed are **$δ$** and **$ΔE_Q$**. **Fwhm** is 
> recognized by the fitting procedure, but is less important. The **ratio** is only for 
> orientation and is ignored as a start parameter for the fit.

<img src='examples\pre-fit.png' alt='Pre-fit'>

## Adjustment sliders and check (fix) buttons

There are four separate **sliders** for **$δ$** (isomer shift), **$ΔE_Q$** (quadropule 
splitting), **fwhm** (full width at half maximum), and **ratio** (ratio of the species / 
component). For the fit, the start value for **ratio** is always set to 0.1 regardless 
of the slider value. Changing the **ratio** is therefore only for orientation.     

Three parameters can be fixed at a certain value with the check buttons **fix $δ$**, fix 
**fix $ΔE_Q$**, and **fix fwhm**. Fixed values are not changed during fit.      

Select the species/component in the **legend** of the diagram at the top and change the 
parameters for each component individually.

> The most important parameters to be changed are **$δ$** and **$ΔE_Q$**. **Fwhm** is 
> recognized by the fitting procedure, but is less important. The **ratio** is only for 
> orientation and is ignored as a start parameter for the fit.    

> If **$ΔE_Q$** is zero or close to zero and the fit fails, **$ΔE_Q$** should be 
> **fixed** around **0**. In a subsequent fit, this **fix** can often be removed.     

<img src='examples\sliders.png' alt='Sliders' class='custom-image'>

## Curve fitting

After clicking on the **Fit button**, (raw) data, the fitted curves for each component 
the resulting curve and the residuals are displayed in the lover area of the plot window.

The **ratio** of the components (in %),  **$δ$** and **$ΔE_Q$** (in mm/s) and **$R^2$** 
are given in the legend. 

Optionally, the **3σ** uncertainty band can be displayed (set `plot_3s_band = True` in the 
script). 

If the result is not suitable, the parameters should be adjusted with the **sliders** 
and the **Fit** should be restarted.

> If a fit fails or is poor, try changing the number of species 
> (components), **$δ$**, **$ΔE_Q$** and **fwhm** in that order. 

<img src='examples\fit-detail.png' alt='Fit detail'>

The terminal provides a more detailed fit report.

```
# Fit report for example_data
## File statistics: 
MB data     : example_data.dat          ⇦ name of the file that contains the (raw) data
data points : 256                       ⇦ number of data points
variables   : 9                         ⇦ number of variables

χ²          : 8.3411e-05                ⇦ Chi square(d)
red. χ²     : 3.3770e-07                ⇦ reduced Chi square(d)
R²          : 0.9952                    ⇦ R square(d)

## Fit results:
data in 1σ  : 30                        ⇦ data points in 1σ (optional)
data in 3σ  : 101                       ⇦ data points in 3σ (optional)
y0          : 0.9995±0.0001             ⇦ y0±error (offset)

|   species |    δ /mm·s⁻¹ |   ΔEQ /mm·s⁻¹ |   fwhm /mm·s⁻¹ |   r (area)/% |   r (int)/% |
|-----------|--------------|---------------|----------------|--------------|-------------|
|      L1Fe |  0.669±0.016 |   1.637±0.026 |    0.877±0.049 |   20.26±0.93 |       19.85 |
|      L2Fe | -0.170±0.002 |   2.240±0.003 |    0.519±0.006 |   79.74±0.93 |       80.15 |
      ⇧             ⇧               ⇧                ⇧               ⇧              ⇧
    label        δ±error       ΔEQ±error         fwhm±error     ratio from      ratio from  
                                                                  area          inegral 
```

> A good fit result has a **$R²$** close to **1** (> 0.98) and a **χ²** below 1e$^{-3}$ 
> or less. But this strongly depends on the quality of the measured data. 

**χ²** from `lmfit`:   
$$\chi^2 = \sum_{i}^N [\rm Residuals_i]^2$$  
   
**red. χ²** from `lmfit` 
$$\chi^2_\nu = \chi^2 / (N-N_{\rm varys})$$   
$N$ is the number of data points and $N_{varys}$ is number of variable parameters.    

$\mathbf{R²}$: coefficient of determination from `lmfit`

Data points in **1σ** or **3σ** are optional (set `print_in_sigma = True` in the script).

**r (area)** in % is the ratio of a component calculated by the area of the individual 
component divided by the sum of all components.    
**r (int)** in % is the ratio of a component calculated by the integral (area under 
the curve) of an individual component divided by the total area / integral of all 
components. Integrals or areas under curves are calculated with the trapezoidal rule. 
**r (int)** is often closer to the values calculated by the `mfit2` program. 

The errors are calculated by the `lmfit` module.

## Saving results

After clicking the **Save button** the following files are saved:

```
parameterfile-fit.txt                ⇦ new parameter file with fitted parameters
data_filename-report.txt             ⇦ fit report; exactly the last terminal output
data_filename-fit.dat                ⇦ a file that contains all data from fit and raw data
data_filename-fit.png                ⇦ exactly the plot (as PNG) in the lower window
```

In case of the **parameter file** `-fit` is added to the filename of the parameter-file. 
The filename (without extension) of the file that contains the measured data is the prefix
for the **report**, **data** and **plot** files. `-fit` is added to the prefix in case of
the latter two files.

The file `data_filename-fit.dat` contains the following data:

```
#velocity   data       residuals  fit        L1Fe       L2Fe      
-4.58930    0.99832    1.00224    0.99917    0.99924    0.99942
-4.55250    0.99906    1.00149    0.99916    0.99923    0.99942
-4.51570    0.99900    1.00154    0.99916    0.99923    0.99942
-4.47890    0.99909    1.00145    0.99915    0.99922    0.99942
-4.44210    0.99952    1.00101    0.99914    0.99922    0.99942
...
   ⇧          ⇧           ⇧          ⇧          ⇧          ⇧
velocity   raw data   residuals   best-fit   single fit curves 
                      (+ extra)   curve      for each component
```
To the `residuals` some extra in `y` is added to display them above data and fit curves. 

## Exit

To exit the script, klick on the **Exit button** or close the `matplotlib` window.

