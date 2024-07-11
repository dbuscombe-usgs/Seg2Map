---
title: "CoastSeg: an accessible and extendable hub for satellite-derived-shoreline (SDS) detection and mapping"
tags:
  - Python
  - shoreline
  - satellite-derived shoreline
  - coastal change detection
  - google earth engine
  - Doodleverse
  - semantic segmentation
authors:
  - name: Sharon Fitzpatrick
    orcid: 0000-0001-6513-9132
    affiliation: 1
  - name: Daniel Buscombe
    orcid: 0000-0001-6217-5584
    affiliation: 1
  - name: Jonathan A. Warrick
    orcid: 0000-0002-0205-3814
    affiliation: 2
  - name: Mark A. Lundine
    orcid: 0000-0002-2878-1713
    affiliation: 2
  - name: Kilian Vos
    orcid: 0000-0002-9518-1582
    affiliation: 3
affiliations:
  - name: Contracted to U.S. Geological Survey Pacific Coastal and Marine Science Center, Santa Cruz, California, United States.
    index: 1
  - name: U.S. Geological Survey Pacific Coastal and Marine Science Center, Santa Cruz, California, United States.
    index: 2
  - name: New South Wales Department of Planning and Environment, Sydney, Australia
    index: 3
date: 12 March 2024
bibliography: paper.bib
---

<!-- --------------------------------------- -->

# Summary

`CoastSeg` is an interactive browser-based program that aims to broaden the adoption of satellite-derived shoreline (SDS) detection workflows among coastal scientists and coastal resource management practitioners. SDS is a sub-field of coastal sciences that aims to detect and post-process a time-series of shoreline locations from publicly available satellite imagery [@turner2021satellite; @vitousek2023future; @luijendijk2018state]. `CoastSeg` is a Python package installed via pip into a `conda` environment that serves as an toolkit for building custom SDS workflows. `CoastSeg` also provides full SDS workflow implementations via Jupyter notebooks and Python scripts that call functions and classes in the core `CoastSeg` toolkit for specific workflows. CoastSeg provides two fully functioning SDS workflows, and its design allows for collaborators in the SDS software community to contribute additional workflows. All the code, notebooks, scripts, and documentation are hosted on the `CoastSeg` GitHub repository [@CoastSeg].

So-called 'instantaneous' SDS workflows, where shorelines are extracted from each individual satellite image rather than temporal composites [@pardopascual20121; @bishop2021mapping], follow a basic recipe, namely 1) waterline estimation, where the 2D (x,y) location of the land-sea interface is determined, and 2) water-level correction, where the waterline location is mapped onto a shore-perpendicular transect, converted to a linear distance along that transect, then corrected for water level, and referenced to a particular elevation contour on the beach [@vos2019coastsat]. The resulting measurement is called a 'shoreline' and it is the location that the waterline intersects a particular elevation datum. Water level corrections typically only account for tide [@vos2019coastsat], but recently SDS workflows have incorporated both wave setup and runup correction, which are a function of the instantaneous wave field at the time of image acquisition [@konstantinou2023satellite; @vitousek2023future; @vitousek2023model; @castelle2021satellite].

`CoastSeg` has three broad aims. The first aim is to be a toolkit consisting of functions that operate the core SDS workflow functionalities. This includes file input/output, image downloading, geospatial conversion, tidal model API handling, mapping 2D shorelines to 1D transect-based measurements, and numerous other functions common to a basic SDS workflow, regardless of a particular waterline estimation methodology. This waterline detection algorithm will be crucial to the success of any SDS workflow because it identifies the boundary between sea and land, which serves as the basis for shoreline mapping. The idea behind the design of `CoastSeg` is that users could extend or customize functionality using scripts and notebooks.

The second aim of `CoastSeg` is therefore to provide fully functioning SDS implementations in an accessible browser notebook format. Our principal objective to date has been to re-implement and improve upon a popular existing toolbox, `CoastSat` [@vos2019coastsat], allowing the user to carry out the well-established `CoastSat` SDS workflow with a well-supported literature [@castelle2021satellite; @castelle2022primary; @vos2023pacific; @vos2023benchmarking; @warrick2023large; @konstantinou2023satellite; @vitousek2023model; @mclean202350; @vandenhove2024secular], but in a more accessible and convenient way within the `CoastSeg` platform. In order to achieve this, we developed `CoastSat-package` [@voscoastsat], a Python package that is installed into the `CoastSeg` `conda` environment. `CoastSat-package` contains re-implemented versions of the original `CoastSat` codes, addresses the lack of pip or conda installability of `CoastSat`, and isolates the CoastSeg-specific enhancements from the original `CoastSat` code. These improvements include additional image download filtering, such as by cloud coverage in the scene, additional parameters to control shoreline extraction, and more accessible output formats, all while retaining the foundational elements of the original `CoastSat` code. The `CoastSeg` re-implementation of the `CoastSat` workflow is end-to-end within a single notebook. That notebook allows the user to, among other tasks: a) define a Region of Interest (ROI) on a webmap, and upload geospatial vector format files; b) define, download and post-process satellite imagery; c) identify waterlines in that imagery using the `CoastSat` method [@vos2019coastsat]; d) correct those waterlines to elevation-based shorelines using tidal elevation-datum corrections provided through interaction with the pyTMD [@tyler_sutterley_2024] API; and e) save output files in a variety of modern geospatial and other formats for subsequent analysis. Additionally, `CoastSeg's` toolkit-based design enables it to run as non-interactive scripts, catering to larger scale shoreline analysis projects.This flexibility ensures that `CoastSeg` can accommodate a wide range of research needs, from detailed, interactive exploration to extensive, automated analyses.

The third and final aim of `CoastSeg` is to implement a method to carry out SDS workflows in experimental and collaborative contexts, which aids both oversight and reproducibility, as well as practical needs based on division of labor. We do this using `sessions`, a mechanism for saving the current state of the application into a session's folder. This folder contains all necessary inputs, outputs, and references to downloaded data used to generate the results. `Sessions` allow users to iteratively experiment with different combinations of settings and make `CoastSeg` fully reproducible because everything needed to reproduce the session is saved to the folder. Users can share their `sessions` with others, enabling peers to replicate experiments, build upon previous work, or access data downloaded by someone else. This simplifies handovers to new users from existing users, simplifies teaching of the program, and encourages collective experimentation, which may result in better shoreline data. Users might expect to adjust settings across several sessions to find the optimal configuration for each site, typically requiring two to five adjustments to achieve the best quality shorelines.

`CoastSeg` is also designed to be extendable, serving as a hub that hosts alternative SDS workflows and similar workflows that can be encoded in a Jupyter notebook built upon the `CoastSeg` and `CoastSat-package` core functionalities. Additional notebooks can be designed to carry out shoreline extraction using alternative methods. We provide an example of an alternative SDS workflow based on a deep-learning based semantic segmentation model that is briefly summarized at the end of this paper. To implement a custom waterline detection workflow the originator of that workflow would contribute a new Jupyter notebook, and add their specific waterline detection algorithm to the `CoastSeg` source code, so it could be used in their notebook's implementation.

<!-- --------------------------------------- -->

# Statement of Need

Coastal scientists and resource managers now have access to extensive collections of satellite data spanning more than four decades. However, it's only in recent years that advancements in algorithms, machine learning, and deep learning have enabled the automation of processing this satellite imagery to accurately identify and map shorelines from imagery, a process known as Satellite-Derived Shorelines, or SDS. SDS workflows [@garcia2015evaluating; @almonacid2016evaluation] are gaining rapidly in popularity, particularly since the publication of the open-source implementation of the `CoastSat` workflow [@vos2019coastsat] for instantaneous SDS in 2018 [@vos2019coastsat]. Existing open-source software for SDS often requires the user to navigate between platforms (non-reproducible elements), develop custom code, and/or engage in substantial manual effort.

We built `CoastSeg` with the aim of enhancing the CoastSat workflow. Our design streamlines the entire shoreline extraction process, thus facilitating a more efficient experimental approach to determine the optimal combination of settings to extract the greatest number of accurate shorelines. `CoastSeg` achieves these improvements through several key advancements: it ensures reproducible sessions for consistent comparison and analysis; introduces additional filtering mechanisms to refine results; and provides an interactive user webmap that allows users to view the quality of the extracted shorelines. Further, `CoastSeg` has been designed specifically to host alternative SDS workflows, recognizing that it is a nascent field of coastal science, and the optimal methodologies for all coastal environments and sources of imagery are yet to be established. Therefore, `CoastSeg` provides a means with which to extract shorelines using multiple methods and adopt the one that most suits their needs, or implement new methods.

We summarize the needs met by the `CoastSeg` project as follows:

- A re-implementation of (and improvement of) the `CoastSat` workflow with pip-installable APIs and `coastsat-package`.

- A browser-based workflow and an interactive mapping interface provided by Leafmap [@wu2021leafmap].

- A more accessible, entirely graphical and menu-based SDS workflow, with no (mandatory) exposure of source code to the user.

- A session system that streamlines the experimentation process to find the settings that extract optimal shorelines from satellite imagery.

- Improved core SDS workflow components, such as a faster and more seamless tidal correction workflow, and faster image downloading.

- Consolidation of workflows in a single platform and reusable codebase.

- An extendable hub of alternative SDS workflows in one location.

<!-- --------------------------------------- -->

# Implementation of core SDS workflow

## Architecture & Design

At a high level, `CoastSeg` is designed to be an accessible and extendable hub for both `CoastSat`-based and alternate workflows, each of which are implemented in a single notebook. The user is therefore presented with a single menu of notebooks, each of which calls on a common set of core functionalities provided by `CoastSeg` and `coastsat-package`, and export data to common file formats and conventions.

`CoastSeg` is installable as a package into a `conda` environment. `CoastSeg` notebooks are accessed from GitHub. We also created a pip package for the `CoastSat` workflow we named `CoastSat-package` in order to: a) improve the `CoastSat` method's software implementation without affecting the parent repository, and b) install it as a package into a `conda` environment, rather than duplicate code from CoastSat.

`CoastSeg` is built with an object-oriented architecture, where elements required by the `CoastSat` workflow such as regions of interest, reference shorelines, and transects are represented as distinct objects on the map. Each class stores data specific to that feature type as well as encompassing methods for styling the feature on the map, downloading default features, and executing various post-processing functions.

## Sessions

SDS workflows require manipulating various settings in order to extract optimal shorelines. There are numerous settings in the `CoastSat` workflow, and sometimes determining optimal shorelines can be an iterative process requiring experimentation with settings. Sub-optimal shoreline extraction may result merely through user fatigue or a combination of misconfigured settings. Therefore, `CoastSeg` employs a `session`-based system that enables users to iteratively experiment with different combinations of settings. Each time the user makes adjustments to the settings used to extract shorelines from the imagery a new session folder is saved with the updated settings. This session system is what makes `CoastSeg` fully reproducible because all the settings, inputs, and outputs are stored within each session, as well as a reference to the downloaded data used to generate the extracted shorelines in the session. Moreover, the session system in `CoastSeg` fosters a collaborative environment. Users can share their sessions with others, enabling peers to replicate experiments, build upon previous work, or access data downloaded by someone else. This simplifies the process for new users and encourages collective experimentation and data sharing. This reproducibility and collaboration are beneficial in research contexts.

## Improvements to the `CoastSat` workflow

### Accessibility

`CoastSeg` facilitates entirely browser-based workflows with an interactive webmap and `ipywidget` controls. It interfaces with the Zenodo API to download reference shorelines [@sayreEtAl2019] for any location in the world, organized into 5x5 degree chunks in GeoJSON format [@buscombe_2023_7786276]. `CoastSeg` also provides transects for specific locations, offering beachface slope metadata [@buscombe_2023_8187949] that is available when users hover over each transect with their cursor. We have improved the reliability of `CoastSeg` through rigorous error handling, which includes developer log files for in-depth diagnostics, user report files for transparency, and detailed error messages that provide guidance for troubleshooting and problem resolution. We have also provided a set of utility scripts for common data input/output tasks, often the result of specific requests from our software testers (see Acknowledgments). In addition to a project wiki and improved documentation, we have researched minimum, maximum, and recommended values for all settings, set suggested default values, and have provided visual project management aids.

### Performance

`CoastSeg` improves upon the Google Earth Engine-based image retrieval process adopted by `CoastSat` by offering a more reliable and efficient download mechanism. Like `CoastSat`, we limit image sources to only the Landsat and Sentinel missions, which are publicly available to all. `CoastSeg` supports downloading multiple regions of interest in a single session, and ensures downloads persist even over an unstable internet connection. This is important because SDS users typically download all available imagery from an ROI, which may amount to several hundred to thousand individual downloaded scenes. Should a download error occur, `CoastSeg` briefly pauses before reconnecting to Google Earth Engine, ensuring that the process does not halt completely. In cases where image downloading fails repeatedly, the filename is logged to a report file located within the downloaded data folder. This report file tracks the status of all requested images from Google Earth Engine. `CoastSeg`'s reliable image retrieval process enhances coastal monitoring by facilitating easier data management and collaboration.

We added helpful workflow components such as image filtering options; for example, users can now filter their imagery based on image size and the proportion of no data pixels in an image. Additionally, the user can decide to turn off cloud masking, which is necessary when the cloud masking process fails and obscures non-cloudy regions such as bright pixels of sand beaches. Finally, we replaced non-cross-platform components of the original workflow; for example, the pickle format was replaced with JSON or geoJSON formats which are both human-readable and compatible with GIS and webGIS.

![Schematic of the tidal correction workflow used by a) ``CoastSat`` and b) ``CoastSeg``.](figs/coastseg_figure_1.png)

### Tide

The CoastSat methodology for applying tide correction to shoreline positions involved a multi-step process. First, the user would need to independently download and configure the FES2014 [@lyard2021fes2014] tide model, a widely recognized tidal model. After configuring the tide model, users would then generate tide estimates at 15-minute intervals for a single location within their ROI across the entire satellite imagery time series. The tide estimate closest to the time of shoreline detection was used to adjust the shoreline position. This method, while comprehensive, was time-consuming, potentially requiring hours to generate all necessary tide estimates.

In contrast, `CoastSeg` introduces a significant improvement to this process by leveraging the pyTMD API [@tyler_sutterley_2024] for a more streamlined and accurate approach to tidal correction (Figure 1). pyTMD facilitates downloading a variety of tide models, including FES2014 and models specific to polar regions, and automates tide estimations. We provide an automated workflow that downloads and subdivides the FES2014 model data into 11 global regions (an idea adopted from [@krause2021dea]). This subdivision allows the program to access only relevant subsets of data, drastically reducing the time required to estimate tides—from hours to minutes for multi-decadal satellite time series. Furthermore, `CoastSeg` calculates tide estimates for each transect corresponding to the times shorelines were detected. This ensures tide corrections are based on temporal and spatial matches, enhancing the accuracy of shoreline position adjustments.

![Schematic of the SDS workflows currently available in ``CoastSeg``. a) ``CoastSat`` workflow; b) ``Zoo`` workflow. Each session has distinct settings that influence the quality of the extracted shoreline. In this example, the reference shoreline buffer size varies between sessions in both the CoastSat and Zoo workflows.](figs/coastseg_figure_2.png)

<!-- --------------------------------------- -->

# Implementation of an Alternative Deep-Learning-Based SDS Workflow

As we noted above, we have developed a notebook that carries out an alternative SDS workflow based on deep-learning based semantic segmentation models. The name 'CoastSeg' is derived from this functionality—using semantic segmentation models for the precise classification of coastal geomorphological features. This advanced classification refines the extraction of shoreline data from satellite imagery. To implement this custom workflow, we created a new Jupyter notebook, and added source code to the `CoastSeg` codebase. The changes ensured that the inputs and outputs were those expected by the core functions in the `CoastSeg` toolkit. We call this alternative workflow the `Zoo` workflow, in reference to the fact that the deep learning models implemented originate from the `Segmentation Zoo` GitHub repository and result from the `Segmentation Gym` deep-learning based image segmentation model training package [@buscombe2022reproducible]. The name `Zoo` has become a standard for online trained ML models, and the repository contains both SDS models and others. Figure 2 describes in detail how the two workflows differ. While the optimal SDS workflow adopted for waterline detection, as determined against field validation data, will be the subject of a future manuscript, it is important to note that these models have not been thoroughly tested yet. We are currently benchmarking these models across various coastal environments, with the results to be documented in a separate repository and linked to `CoastSeg` upon conclusion.

<!-- --------------------------------------- -->

# Project Roadmap

We intend `CoastSeg` to be a collaborative research project and encourage contributions from the SDS community. As well as implementing alternative SDS waterline detection workflows, other improvements that could continue to be made include more (or more refined) outlier detection methods, image filtering procedures, and other basic image pre- or post-processing routines, especially image restoration on degraded imagery [@vitousek2023future]. Such additions would all be possible without major changes to the existing `CoastSeg` toolkit.

Integration of new models for the deep-learning workflow are planned, based on Normalized Difference Water Index (NDWI) and Modified Normalized Difference Water Index (MNDWI) spectral indices, as is a new `CoastSeg` toolbox extension for daily 3-m Planetscope imagery [@doherty2022python] from Planet Labs. Docker may be adopted in the future to manage dependencies in the `conda` virtual environment required to run the program. Other sources of imagery and other spectral indices may have value in SDS workflows, and we encourage SDS users to contribute their advances through a `CoastSeg` Jupyter notebook implementation.

It would also be possible to incorporate automated satellite image subpixel co-registration in `CoastSeg` using the AROSICS package [@scheffler2017arosics]. This would co-register all available imagery to the nearest-in-time LandSat image. Furthermore, future work could include accounting for the contributions of runup and setup to total water level [@vitousek2023model; @vos2023benchmarking]. In practice, this would merely add/subtract a height from the instantaneous predicted tide, then apply horizontal correction. However, the specific methods used to estimate runup or setup from the prevailing wave field would require integration with observed or hindcasted databases of wave conditions.

<!-- --------------------------------------- -->

# Acknowledgments

The authors would like to thank Qiusheng Wu, developer of `Leafmap`, which adds a lot of functionality to `CoastSeg`. Thanks also to the developers and maintainers of `pyTMD`, `DEA-tools`, `xarray`, and `GDAL`, without which this project would be impossible. We would also like to thank Freya Muir and Floris Calkoen for reviewing `CoastSeg`. We acknowledge contributions from Robbi Bishop-Taylor, Evan Goldstein, Venus Ku, software testing and suggestions from Catherine Janda, Eli Lazarus, Andrea O'Neill, Ann Gibbs, Rachel Henderson, Emily Himmelstoss, Kathryn Weber, and Julia Heslin, and support from USGS Coastal Hazards and Resources Program, and USGS Merbok Supplemental. Any use of trade, firm, or product names is for descriptive purposes only and does not imply endorsement by the U.S. Government.

<!-- --------------------------------------- -->

# References
