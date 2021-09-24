# MJO SST sensitivity Model Intercomparison Project (MSMIP)

A model intercomparison project organized by members of the WGNE MJO Task Force. The python package here is designed to generate the surface forcing data for the MSMIP.

## Package dependency
- numpy
- xarray
- xESMF

## 1. Motivation
Ocean coupling is known to improve MJO simulation in climate models.  This has been noted in many prior studies that have compared MJO simulation skill in fully coupled and atmosphere-only simulations of the same model (e.g., DeMott et al. 2015 and references therein).  Such comparisons are often made between free-running, coupled integrations and uncoupled atmosphere-only simulations forced with observed monthly mean sea surface temperatures (SSTs).  Interpreting the results of these comparisons is complicated because of differences in the SST mean state and the SST low-frequency variability between coupled and uncoupled simulations, and because uncoupled simulations do not include sub-monthly SST variability that is present in the coupled simulation.

Two approaches have been used to address these complications:  1) uncoupled simulations forced with monthly mean SSTs from the coupled simulation, and 2) uncoupled simulations forced with daily or 5-day running mean SSTs from the coupled simulation.  The former incorporates SST mean state and low-frequency variability of the coupled simulation, but precludes sub-monthly SST variability.  The latter includes sub-monthly SST variability, but often has the undesirable effect of erroneously removing the hallmark quadrature SST-precipitation relationship of coupled simulations, as precipitation in high-frequency SST forced simulations tends to be coincident with the warmest SST anomalies (Pegion and Kirtman, 2008a,b).  

DeMott et al. (2019) used the first approach (AGCM forced with monthly mean CGCM SSTs) with four different models and found that improved MJO simulation in the coupled models was the result of a more equatorially peaked mean state moisture pattern in the coupled vs uncoupled models that favored MJO eastward propagation via the meridional moisture advection mechanism (Kim et al. 2014).  DeMott et al. (2019) concluded that sub-monthly ocean-atmosphere feedbacks are responsible for the changes in mean state moisture and improved MJO simulation.  An open question, however, is:

>“What aspects of coupled model SST anomalies are most important for tropical mean state moisture patterns and MJO propagation:  their phasing, pattern, or persistence?"

The MSMIP is designed to answer this question and to understand if the effects of sub-monthly SST variability on mean state moisture and MJO simulation are model-dependent or consistent across many models.


## 2. MSMIP Protocols
### 2.1 Experimental Design
MSMIP consists of five model experiments.  The overarching goal is to perform uncoupled experiments that retain the coupled model SST mean state, as well as its low- and high-frequency SST variability, but avoid the erroneous “coincident SST-precipitation” condition that develops in uncoupled models forced with high-frequency SST from the coupled simulation.  To achieve this, we adopt a strategy from cloud-radiation feedback studies known as cloud-locking (e.g., Langen et al. 2012, Mauritsen et al. 2013) wherein simulated radiative heating profiles from a control simulation are randomized in time, and then prescribed in an experimental “cloud-locked” simulation.  In the cloud-locked simulation, the mean state cloud radiative feedbacks are “locked in” to the global energy budget, but their coherent relationship with the clouds themselves is purposefully broken.

In the following, CGCM and AGCM refer to coupled and atmosphere-only general circulation model configurations, respectively.  Suffixes refer to the SST anomaly (SSTA) forcing method in each AGCM simulation.  In all AGCM_x experiments, the input SST is provided by the CGCM simulation.  SSTAs are computed as departures from the low-frequency background state, which includes seasonal-to-interannual variability (i.e., not as departures from the mean annual cycle).  All AGCM_x experiments therefore have identical low-frequency SST variability as the CGCM simulation.
- CGCM:  coupled model that provides daily SST output (e.g., CMIP6 historical simulation)
- AGCM_mon:  prescribe monthly mean SSTs (remove all higher frequency SST variability)
- AGCM_1drandpt:  prescribe pointwise randomly shuffled daily SSTA (scramble patterns)
- AGCM_1drandpatt:  prescribe randomly shuffled daily SSTA patterns (retain pattern)
- AGCM_5drandpatt:  prescribe randomly shuffled 5-day running SSTA chunks (retain pattern, persistence)

Procedures for generating the shuffled SST input are described in Section 2.3.  

### 2.2 Recommended model configuration
To limit total data volume, we recommend that experiments be performed with the standard low-resolution configuration for a given model.  AGCM experiments should be run with the same model version as the CGCM experiments.  Each experiment should be run for a period of 20-30 years.  When possible, experiments should be based on the final 20-30 years of a historical simulation from the CMIP6 archive.  

For all AGCM_x experiments, sea ice should be prescribed using either output from the CGCM, or from climatology.  

Investigators are asked to provide details of the model configurations and any differences between model versions for CGCM and AGCM simulations here

### 2.3 Generating randomized SSTA time series
MSMIP provides example NCL and Python code to do the SSTA randomization.   Randomization should be done on the same grid that will be used to prescribe SST in the AGCM simulations.  The NCL and Python code will accept ocean surface temperature-only data OR land-and-ocean temperature data.  If present, data over land points will not be randomized, but will be included in the final total surface temperature field.  One strategy is to save the radiative surface temperature from the CGCM simulation, and set the landData flag to True and the sstType flag to skin.  The code will then shuffle only the ocean data points, and return the global surface temperature field that includes the original land temperature plus the shuffled SSTAs.  Another strategy is to regrid SST from the ocean component of the CGCM simulation to the grid used to prescribe SST in the AGCM simulations, and set the landData flag to False and the sstType flag to foundation.  Here, SST is assumed to be the foundation SST (i.e., the SST at the mean depth of the first ocean layer, usually about 5 m).  In this case, the skin temperature will be estimated as SST - 0.2K (Donlon et al. 2005) before shuffling, since this is what the atmosphere “sees” via surface fluxes.  Land temperature must then be manually merged with the new SST time series (this step is left to individual investigators). 

As in cloud-locking experiments, SSTAs are shuffled about the same day-of-year (or 5-day chunk) background state to retain the CGCM mean annual cycle of high-frequency SSTA variance.  For example, in AGCM_1drandpatt, the SSTA patterns for all January 1 days are randomized, then all January 2 SSTA patterns are randomized, etc.  An example of SST shuffling is shown in Fig. 1.  Both the NCL and Python SST shuffling packages will produce a similar plot.

Generation of monthly mean SST time series is left to individual investigators.
![Figure 1](SST_locking_anom_global.png "Figure1")
Figure 1.  Ten-day time series of SSTA (arranged from top-to-bottom rows) from the coupled simulation, randomized 5-day running chunks of SSTA patterns, daily randomized SSTA patterns, and daily pointwise randomized SSTA (left-to-right columns).

### 2.4 Data output requirements and formats
We request daily means of several 2D variables on the AGCM native grid.  Variables should be renamed as follows; we do not require full CMOR-compliant output as in the CMIPx repository.

__Priority daily variables (all are 2D):__
- pr:       total precipitation  (kg m-1 s-2)
- hfls:     surface latent heat flux (W m-2; positive to atmosphere)
- hfss:     surface sensible heat flux (W m-2; positive to atmosphere)
- huss:     near-surface (2m) specific humidity (kg kg-1)
- tas:      near-surface (2m) temperature (K)
- sfcWind:  near-surface (10m) wind speed (m s-1)
- psl:      surface pressure (Pa)
- rlut:     top-of-model outgoing longwave radiation (W m-2)
- TS*:      SKT or SST (K)
- u850:     850 hPa zonal wind (m s-1)
- v850:     850 hPa meridional wind (m s-1)
- u200:     200 hPa zonal wind (m s-1)
- v200:     200 hPa zonal wind (m s-1)
- tmq:      precipitable water (kg m-2)
- omega500: 500 hPa pressure velocity (Pa s-1)
- z500:     500 hPa height (m)

*TS is either 1) surface skin temperature (SKT; preferred; includes land) or SST (only over oceans).

__Optional daily variables (all are 2D 1000-100 hPa vertically integrated quantities; require saving daily 3D u, v, omega, T, q, Z):__
- h:     moist static energy, h=CpT+gz+Lq (J kg-1)
- hHADV: horizontal advection of h (W m-2)
- hVADV: vertical advection of h (W m-2)
- LW:    Net column longwave heating (netTOA - netSURFACE) (W m-2)
- SW:    Net column shortwave heating (netTOA - netSURFACE) (W m-2)

__File naming convention:__
We request a full time series of a single variable per file.  Time series may be broken up into a few separate files if the single time series is much larger than about 2 GB.  The data should be provided in netCDF format.  The file naming convention is:
> `CenterName.ModelName.ExpName.VariableName.YYYYMMDDfirst-YYYYMMDDlast.nc`
   __Where:__
   - CenterName is modeling center name
   - ModelName is the name of the model 
   - ExpName is the experiment name, listed in Section 2.1
   - VariableName is the name of the variable reported in the file, as shown in the above table
   - YYYYMMDDfirst is the first day of the simulation
   - YYYYMMDDlast is the last day of the simulation

All variables, including coordinate variables (time, longitude, latitude), should include a “units” and  “FillValue” attribute.  The “time” coordinate variable should be reported as “(time unit) since YYYY-MM-HH” or “(time unit) since YYYY-MM-HH 00:00:00” where (time unit) is typically either “days” or “hours.”

### 2.5 Data sharing
[Data will be stored on NCAR’s campaign storage data server](https://www2.cisl.ucar.edu/resources/storage-and-file-systems/campaign-storage-file-system) 

Data can be uploaded through [Globus](https://www.globus.org/)

## 3. Assessment
The effects of ocean-atmosphere coupling, and the effects of SSTA patterns, persistence, and phasing with respect to MJO convection can be assessed through the following comparisons:
- coupling effect:  CGCM - AGCM_mon
- phasing effect:  CGCM - AGCM_5drandpatt
- persistence effect:  AGCM_5drandpatt - AGCM_1drandpatt
- pattern effect:  AGCM_1drandpatt - AGCM_1drandpt
Additional details of the assessment strategy are currently under development, but will include a variety of standard MJO diagnostics and metrics, such as the east-west power ratio, the pattern correlation between observed and modeled precipitation longitude-lag diagram, the MJO MC-crossing metric, and MJO propagation as a function of ENSO state.  Tropical mean state for all requested variables will also be assessed.

## 4. References
DeMott, C. A., N. P. Klingaman, and S. J. Woolnough (2015), Atmosphere-ocean coupled processes in the Madden-Julian oscil- lation, Rev. Geophys., 53, 1099–1154, doi:10.1002/2014RG000478 .

DeMott, C. A., Klingaman, N. P., Tseng, W.-L., Burt, M. A., Gao, Y., & Randall, D.A. (2019) The convection connection: How ocean feedbacks affect tropical mean moisture and MJO propagation. Journal of Geophysical Research Atmospheres, 124, 11,910–11,931. https://doi.org/10.1029/2019JD031015

Donlon, C. J. and the GHRSST-PP Science Team (2005): The Recommended GHRSST-PP Data Processing Specification GDS (version 1 revision 1.6). The GHRSST-PP International Project Office, Exeter, U.K., 245 pp. (available at: http://ghrsst-pp.jrc.it/documents/GDS-v1.6.zip).

Kim, D., J.-S. Kug, and A. H. Sobel (2014),Propagating versus Nonpropagating Madden–Julian Oscillation Events.  J. Climate, 27, 111-125.  doi:10.1175/JCLI-D-13-00084.1 .







