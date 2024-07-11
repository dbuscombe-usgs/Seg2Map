# CoastSeg
Paper:
[![DOI](https://joss.theoj.org/papers/10.21105/joss.06683/status.svg)](https://doi.org/10.21105/joss.06683)
<!--  [![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/2320sharon/CoastSeg/blob/main/coastseg_for_google_colab.ipynb) -->

[![image](https://img.shields.io/pypi/v/coastseg.svg?color=%23ec3dc8)](https://pypi.python.org/pypi/coastseg)
</br>
[![image](https://github.com/SatelliteShorelines/CoastSeg/actions/workflows/pip_install_e.yml/badge.svg)](https://github.com/SatelliteShorelines/CoastSeg/actions)

[![Last Commit](https://img.shields.io/github/last-commit/SatelliteShorelines/CoastSeg)](https://github.com/Doodleverse/segmentation_gym/commits/main)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/SatelliteShorelines/CoastSeg/graphs/commit-activity)
[![Wiki](https://img.shields.io/badge/wiki-documentation-forestgreen)](https://github.com/SatelliteShorelines/CoastSeg/wiki)
![GitHub](https://img.shields.io/github/license/Doodleverse/segmentation_gym)
[![Wiki](https://img.shields.io/badge/discussion-active-forestgreen)](https://github.com/SatelliteShorelines/CoastSeg/discussions)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)


<img src="https://user-images.githubusercontent.com/61564689/212394936-263ec9fc-fb82-45b8-bc79-bc57dafdae73.gif" width="350" height="350">


# Website 

Please read our [website](https://satelliteshorelines.github.io/CoastSeg/), which includes installation and usage guides, contribution guide, API description, case studies, and more. 

# What is CoastSeg?
CoastSeg is an interactive browser-based program that aims to broaden the adoption of satellite-derived shoreline (SDS) detection and coastal landcover mapping workflows among coastal scientists and coastal resource management practitioners. SDS is a sub-field of coastal sciences that aims to detect and post-process a time-series of shoreline locations from publicly available satellite imagery.

CoastSeg is a Python package installed via pip into a conda environment that serves as an toolkit for building custom SDS workflows. CoastSeg also provides full SDS workflow implementations via Jupyter notebooks and Python scripts that call functions and classes in the core CoastSeg toolkit for specific workflows. CoastSeg provides two fully functioning SDS workflows and its design allows for collaborators in the SDS software community to contribute additional workflows. All the codes, notebooks, scripts, and documentation are hosted on the [CoastSeg GitHub repository](https://github.com/SatelliteShorelines/CoastSeg).

So-called `instantaneous' SDS workflows, such as the implementation of [CoastSat](https://github.com/kvos/CoastSat), where shorelines are extracted from each individual satellite image rather than temporal composites, follow a basic recipe, namely:

1) waterline estimation, where the 2D (x,y) location of the land-sea interface is determined
and 
2) water-level correction, where the waterline location is mapped onto a shore-perpendicular transect, converted to a linear distance along that transect, then corrected for water level, and referenced to a particular elevation contour on the beach. 

# Project Aims
CoastSeg has three broad aims. The first aim is to be a toolkit consisting functions that operate the core SDS workflow functionalities. This includes file input/output, image downloading, geospatial conversion, tidal model API handling, mapping 2D shorelines to 1D transect-based measurements, and numerous other functions common to a basic SDS workflow, regardless of a particular waterline estimation methodology. This waterline detection algorithm will be crucial to the success of any SDS workflow because it is the step that identifies the the boundary between sea and land which serves as the basis for shoreline mapping. The idea behind the design of CoastSeg is that users could extend or customize functionality using scripts and notebooks.

The second aim of CoastSeg is therefore to provide fully functioning SDS implementations in an accessible browser notebook format. Our principal objective to date has been to re-implement and improve upon a popular existing toolbox, [CoastSat](https://github.com/kvos/CoastSat), allowing the user to carry out the well-established CoastSat SDS workflow, but in a more accessible and convenient way within the CoastSeg platform. In order to achieve this, we developed [CoastSat-package](https://github.com/SatelliteShorelines/coastsat_package), a Python package that is installed into the CoastSeg conda environment. CoastSat-package contains re-implemented versions of the original CoastSat codes, addresses the lack of pip or conda installability of CoastSat, and isolates the CoastSeg-specific enhancements from the original CoastSat code. 

The third and final aim of CoastSeg is to implement a method to carry out SDS workflows in experimental and collaborative contexts, which aids both oversight and reproducibility as well as practical needs based on division of labor. We do this using sessions, a mechanism for saving the current state of the application into a session's folder. This folder contains all necessary inputs, outputs, and references to downloaded data used to generate the results. Sessions allow users to iteratively experiment with different combinations of settings and makes CoastSeg fully reproducible because everything needed to reproduce the session is saved to the folder. Users can share their sessions with others, enabling peers to replicate experiments, build upon previous work, or access data downloaded by someone else. This simplifies handovers to new users from existing users, simplifies teaching of the program, and encourages collective experimentation which may result in better shoreline data.

CoastSeg is also designed to be extendable, serving as a hub that hosts alternative SDS workflows and similar workflows that can be encoded in a Jupyter notebook built upon the CoastSeg and CoastSat-package core functionalities. Additional notebooks can be designed to carry out shoreline extraction and coastal landcover mapping using alternative methods. We provide an example of an alternative SDS workflow based on a deep-learning based semantic segmentation model that is briefly summarized at the end of this paper. To implement a custom waterline detection workflow the originator of that workflow would contribute new Jupyter notebook, and add their specific waterline detection algorithm to the CoastSeg source code, so it could be used in their notebook's implementation.

![coastseg_main_flow_updated](https://github.com/Doodleverse/CoastSeg/assets/61564689/ac9076bd-bf40-44c5-a686-0fdc1acf8656)


# Installation Instructions

CoastSeg is a Jupyter and Python  based program that runs in a conda environment. Please see [the installation guide on the CoastSeg website](https://satelliteshorelines.github.io/CoastSeg/basic-install-guide/)

# Getting Started

Please see [the guide on the CoastSeg website](https://satelliteshorelines.github.io/CoastSeg/getting-started/), which includes instructions for

* Pre-requisites (accounts, data access)
* Installation and setup
* Extracting shorelines
* Applying tidal corrections to extracted shorelines

# CoastSat Re-implementation

The CoastSeg re-implementation of the  [CoastSat](https://github.com/kvos/CoastSat) workflow is end-to-end within a single notebook. That notebook allows the user to, among other tasks: 

a) define a Region of Interest (ROI) on a webmap and upload geospatial vector format files; 
b) define, download and post-process satellite imagery; 
c) identify waterlines in that imagery using the CoastSat method; 
d) correct those waterlines to elevation-based shorelines using tidal elevation-datum corrections provided through interaction with the [pyTMD](https://github.com/tsutterley/pyTMD) API; and 
e) save output files in a variety of modern geospatial and other formats for subsequent analysis. 

Additionally, CoastSeg's toolkit-based design enables it to run as non-interactive scripts, catering to larger scale shoreline analysis projects. This flexibility ensures that CoastSeg can accommodate a wide range of research needs, from detailed, interactive exploration to extensive, automated analyses.

# CoastSeg API

Please see [the guide on the CoastSeg website](https://satelliteshorelines.github.io/CoastSeg/CoastSeg-API-Guide/), which includes instructions for scripting workflows with the CoastSeg API

# Doodleverse/Zoo models Used in CoastSeg

A lot of work underpins the 'Zoo' method for SDS, which uses models and datasets developed over several years and still under an active cycle of improvement. That is, once we find the time!

The models currently available are:

* Buscombe, D. (2023). Doodleverse/CoastSeg Segformer models for 4-class (water, whitewater, sediment and other) segmentation of Sentinel-2 and Landsat-7/8 MNDWI images of coasts. (v1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.8213443
* Buscombe, D. (2023). Doodleverse/CoastSeg Segformer models for 4-class (water, whitewater, sediment and other) segmentation of Sentinel-2 and Landsat-7/8 NDWI images of coasts. (v1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.8213427
* Buscombe, D. (2023). Doodleverse/CoastSeg Segformer models for 4-class (water, whitewater, sediment and other) segmentation of Sentinel-2 and Landsat-7/8 3-band (RGB) images of coasts. (v1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.8190958

Made using the following software, implemented as [Doodleverse/Segmentation Gym]([Zoo](https://github.com/Doodleverse/segmentation_gym)):

* Segmentation Gym: Buscombe, D., & Goldstein, E. B. (2022). A reproducible and reusable pipeline for segmentation of geoscientific imagery. Earth and Space Science, 9, e2022EA002332. https://doi.org/10.1029/2022EA002332 

Using the following datasets:

* Buscombe, Daniel. (2022). Images and 2-class labels for semantic segmentation of Sentinel-2 and Landsat RGB, NIR, and SWIR satellite images of coasts (water, other) (v1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7384263
* Wernette, P.A., Buscombe, D.D., Favela, J., Fitzpatrick, S., and Goldstein E., 2022, Coast Train--Labeled imagery for training and evaluation of data-driven models for image segmentation: U.S. Geological Survey data release, https://doi.org/10.5066/P91NP87I. See https://coasttrain.github.io/CoastTrain/ for more information
* Buscombe, Daniel. (2023). June 2023 Supplement Images and 4-class labels for semantic segmentation of Sentinel-2 and Landsat RGB, NIR, and SWIR satellite images of coasts (water, whitewater, sediment, other) (v1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.8011926 
* Buscombe, Daniel, Goldstein, Evan, Bernier, Julie, Bosse, Stephen, Colacicco, Rosa, Corak, Nick, Fitzpatrick, Sharon, del Jesús González Guillén, Anais, Ku, Venus, Paprocki, Julie, Platt, Lindsay, Steele, Bethel, Wright, Kyle, & Yasin, Brandon. (2022). Images and 4-class labels for semantic segmentation of Sentinel-2 and Landsat RGB satellite images of coasts (water, whitewater, sediment, other) (v1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7335647

# Utility Scripts

CoastSeg comes with a collection of pre-processing script utilities for common i/o problems. Please see [the guide on the CoastSeg website](https://satelliteshorelines.github.io/CoastSeg/How-to-Use-Scripts/), which includes the list of available scripts, and how to use them.

# Data Sources

The CoastSeg transect and slope database is [available]([https://zenodo.org/records/8187949](https://zenodo.org/records/11390980)) 

* Buscombe, D., & Fitzpatrick, S. (2023). CoastSeg: Beach transects and beachface slope database v1.0 (v1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.8187949

Beach face slope and transect data have been derived from:

1. Doran, K.S., Long, J.W., Birchler, J.J., Brenner, O.T., Hardy, M.W., Morgan, K.L.M, Stockdon, H.F., and Torres, M.L., 2017, Lidar-derived beach morphology (dune crest, dune toe, and shoreline) for U.S. sandy coastlines (ver. 4.0, October 2020): U.S. Geological Survey data release, https://doi.org/10.5066/F7GF0S0Z.

2. Kilian Vos. (2023). Time-series of shoreline change along the Pacific Rim (v1.4) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7758183

3. Andrew Short. (2022). Sediment size dataset for Australia [Data set]. In Australian Coastal Systems (0.1, p. XXV, 1241). Springer Cham. https://doi.org/10.5281/zenodo.7127186

4. Vos, Kilian, Wen, Deng, Harley, Mitchell D., Turner, Ian L., & Splinter, Kristen D. (2022). Beach-face slope dataset for Australia (Version 2) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7272538

5. Gibbs, A.E., Ohman, K.A., Coppersmith, R., and Richmond, B.M., 2017, National Assessment of Shoreline Change: A GIS compilation of updated vector shorelines and associated shoreline change data for the north coast of Alaska, U.S. Canadian border to Icy Cape: U.S. Geological Survey data release, https://doi.org/10.5066/F72Z13N1.

6. Himmelstoss, E.A., Kratzmann, M., Hapke, C., Thieler, E.R., and List, J., 2010, The National Assessment of Shoreline Change: A GIS Compilation of Vector Shorelines and Associated Shoreline Change Data for the New England and Mid-Atlantic Coasts: U.S. Geological Survey Open-File Report 2010-1119, available at https://pubs.usgs.gov/of/2010/1119/.

7. Snyder, A.G., and Gibbs, A.E., 2019, National assessment of shoreline change: A GIS compilation of updated vector shorelines and associated shoreline change data for the north coast of Alaska, Icy Cape to Cape Prince of Wales: U.S. Geological Survey data release, https://doi.org/10.5066/P9H1S1PV

8. Romine, B.M., Fletcher, C.H., Genz, A.S., Barbee, M.M., Dyer, Matthew, Anderson, T.R., Lim, S.C., Vitousek, Sean, Bochicchio, Christopher, and Richmond, B.M., 2012, National Assessment of Shoreline Change:  A GIS compilation of vector shorelines and associated shoreline change data for the sandy shorelines of Kauai, Oahu, and Maui, Hawaii: U.S. Geological Survey Open-File Report 2011-1009, available online at https://pubs.usgs.gov/of/2011/1009/.

9. Gibbs, A.E., Jones, B.M., and Richmond, B.M., 2020, A GIS compilation of vector shorelines and coastal bluff edge positions, and associated rate-of-change data for Barter Island, Alaska: U.S. Geological Survey data release, https://doi.org/10.5066/P9CRBC5I.

10. Sturdivant, E.J., Zeigler, S.L., Gutierrez, B.T., and Weber, K.M., 2019, Barrier island geomorphology and shorebird habitat metrics–Sixteen sites on the U.S. Atlantic Coast, 2013–2014: U.S. Geological Survey data release, https://doi.org/10.5066/P9V7F6UX.
 
Additional contributions:
1. Sean Vitousek, USGS


# Authors and Contributions

Package maintainers:
- [@2320sharon](https://github.com/2320sharon) : Lead Software Developer / Contracted to USGS Pacific Coastal and Marine Science Center.
- [@dbuscombe-usgs](https://github.com/dbuscombe-usgs)  Contracted to USGS Pacific Coastal and Marine Science Center.
- [@mlundine](https://github.com/mlundine) : USGS Pacific Coastal and Marine Science Center.

Contributions:
- [@ebgoldstein](https://github.com/ebgoldstein)
- [@venuswku](https://github.com/venuswku)
- [@robbibt](https://github.com/robbibt)
- [@edlazarus](https://github.com/edlazarus)
- Beta testers: Catherine Janda, Ann Gibbs, Jon Warrick, Andrea O’Neill, Kathryn Weber, Julia Heslin (USGS)
- We would like to express our gratitude to all the contributors who made this release possible. Thank you to everyone who tested the beta versions of coastseg and provided us with the feedback we needed to improve coastseg. Thanks also to the developers and maintainers of pyTMD, DEA-tools, xarray, and GDAL, without which this project would be impossible


# Related Packages

* [CoastSat](https://github.com/kvos/CoastSat)
* [coastsat-package](https://github.com/SatelliteShorelines/coastsat_package)
* [Zoo](https://github.com/Doodleverse/segmentation_zoo)
* [pyTMD](https://github.com/tsutterley/pyTMD)
* [dea-tools](https://knowledge.dea.ga.gov.au/notebooks/Tools/)

See also other related repositories in the [Satellite Shorelines GitHub Organization](https://github.com/orgs/SatelliteShorelines/repositories)
