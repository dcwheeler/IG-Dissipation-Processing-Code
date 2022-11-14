# IG-Dissipation-Processing-Code
Please cite as:

Wheeler, D. C., S. N. Gidings, and J. McCullough, 2022: Data and Code from Measuring Turbulent Dissipation with Acoustic Doppler Velocimeters in the Presence of Large, Intermittent, Infragravity Frequency Bores. UC San Diego Library Digital Collections, https://doi.org/https://doi.org/10.6075/J0J67H27



Corresponding Author:

Duncan Wheeler, dcwheele@ucsd.edu



Primary Associated Publication:

Wheeler, Duncan C. & Giddings, Sarah N. (2022). Measuring Turbulent Dissipation with Accoustic Doppler Velocimeters in the Presence of Large, Intermittent, Infragravity Frequency Bores, Journal of Physical Oceanography.



Description of Contents:

code_2022 contains all code files explained below.

data_2022 contains the raw data files and processed data files produced by the running the code as explained below.

plots_2022 contains the plots used in the publications listed above. They are created by running the code as explained below.

toleranceTests_2022 contains the data produced when running the tolerance test codes as explained below.

Functions.py is a library of functions called by other scripts.

CleanAndDespike.py takes the raw files (vec12412raw.nc, vec12414raw.nc, and vec8155raw.nc), performs initial cleaning, runs the despike algorithm, and saves the resulting despiked data in new files (vec12412despiked.nc, vec12414despiked.nc, and vec8155despiked.nc).

DissipationCalc.py takes the despiked vector data files (vec12412despiked.nc, vec12414despiked.nc, and vec8155despiked.nc) and uses the wave corrected spectrum and dissipation fitting algorithms to calculate dissipation values.  The results are saved to dissipations.nc.

SemiIdealizedModel.py takes the despiked vector data files (vec12412despiked.nc, vec12414despiked.nc, and vec8155despiked.nc) and uses them to create advection cases and test the spectrum calculation and dissipation fitting algorithms on the semi-idealized model. The results of the test are saved to SemiIdealizedResults.nc. NOTE: This file takes a long time to run

MakeDespikePlot.py, MakeSpectrumCalcPlot.py, MakeDissipationFitPlot.py, MakeIdealizedErrorPlot.py, and MakeResultsPlot.py create the corresponding plots that are found in the paper.

DespikeToleranceTest.py takes the raw files (vec12412raw.nc, vec12414raw.nc, and vec8155raw.nc) and runs the despiking algorithm with modified variables to create tolerance tests of the despiking algorithm.  The resulting despiked data are saved to /toleranceTests/vec{vector number}despiked_{variable name}_{variable value}.nc.  NOTE: This file takes a very long time to run.

DespikeToleranceStep2.py takes the output of DespikeToleranceTest.py and calculates dissipation values. the resulting data is saved to /toleranceTests/dissipations_{variable name}_{variable value}.nc except for the expansion step size tests that are not the highest and lowest value. These are instead saved to /toleranceTests/expSizeTests/dissipations_expSize_{variable value}.nc. After saving this file, the despiking results themselves are directly compared to the despiking results found in vec12412despiked.nc, vec12414despiked.nc, and vec8155despiked.nc. These comparisons are stored in numpy arrays and are meant to be looked at in an interactive format, such as ipython, but are not saved.  NOTE: the dissipation calculation portion of this script takes a very long time to run, but the direct despike result comparison portion runs quickly.

ExpSizeEval.py takes the expansion step size results of DespikeToleranceStep2.py and compares them with dissipations.nc. an xarray dataset for interactive evaluation is created along with a plot for how the change in expansion step size changes final dissipation values.

SpectrumToleranceTest.py takes the despiked vector data files (vec12412despiked.nc, vec12414despiked.nc, and vec8155despiked.nc) and runs the spectrum calculation and dissipation fitting algorithms with modified variables to create tolerance tests. The resulting dissipation values are saved to /toleranceTests/dissipations_{variable name}_{variable value}.nc. NOTE: this script takes a while to run.

ToleranceEval.py takes the dissipation files output by DespikeToleranceStep2.py and SpectrumToleranceTest.py and compares them with the results stored in dissipations.py to see how modifying each algorithm variable changes the final result.The results are stored in an xarray dataset to be looked at in an interactive format, such as ipython, but are not saved. This code runs quickly.

WhiteNoiseEval.py takes the despiked vector data files (vec12412despiked.nc, vec12414despiked.nc, and vec8155despiked.nc) and runs the spectrum calculation and dissipation fitting algorithms. However, between the spectrum calculation and dissipation fitting, a flat white noise value is removed from the spectrum before running the dissipation fitting algorithm.  The results are saved to /data/dissipationsNoiseRemovedTest_{noise level}.nc. There is also some code to help in comparing these results to dissipations.nc.



Technical Details:

[tool.poetry]
version = "0.1.0"

[tool.poetry.dependencies]
python = ">=3.7,<3.11"
numpy = "^1.17"
matplotlib = "^3.1"
xarray = "^0.20.2"
netcdf4 = "^1.5"
sympy = "^1.4"
scipy = "^1.3"
ipython = "^7.8"
pyRSKTools = "^0.1.8"
cmocean = "^2.0"
gsw = "^3.3.1"
seawater = "^3.3.4"
pyproj = "^2.6.1"
seaborn = "^0.11.2"
utm = "^0.6.0"
statsmodels = "^0.12.0"
PyEMD = "^0.5.1"
EMD-signal = "^0.2.10"
nlopt = "^2.6.2"
palettable = "^3.3.0"
iapws = "^1.5.2"
plottools = "^0.2.0"
PyQt5 = "5.14.1"
xlrd = "^1"

[tool.poetry.dev-dependencies]
pytest = "^3.0"

[build-system]
requires = ["poetry>=0.12"]
build-backend = "poetry.masonry.api"
