
====
Itzï
====

Itzï is a dynamic, fully distributed hydrologic and hydraulic model that
simulates 2D surface flows on a regular raster grid and drainage flows through the SWMM model.
It uses GRASS GIS as a back-end for entry data and result writing.

Website: http://www.itzi.org/

Documentation: http://itzi.rtfd.io/



Description
===========

Itzï allows the simulation of surface flows from direct rainfall or user-given point inflows.
Its GIS integration simplifies the pre- and post-processing of data.
Notably, to change the resolution or the extend of the computational domain,
the user just need to change the GRASS computational region and the raster mask, if applicable.
This means that map export or re-sampling is not needed.

Itzï uses raster time-series as entry data, allowing the use of rainfall from weather radars.


Model description
=================

The surface model of Itzï is described in details in the following open-access article:

    Courty, L. G., Pedrozo-Acuña, A., & Bates, P. D. (2017).
    Itzï (version 17.1): an open-source, distributed GIS model for dynamic flood simulation.
    Geoscientific Model Development, 10(4), 1835–1847.
    http://doi.org/10.5194/gmd-10-1835-2017


Changelog
=================
new in Version 19.0 By P.Borer
Bugs:

- Corrected bug in Green-Ampt infiltration. Infiltration capping acounts also for flow from neigbooring cells
- Corrected bug when reading Raster maps which causes memory leak and "out of memory" exception for large models.
- Corrected error in 2D Flow stencil in numerical scheme implementation.

Enhancements:
- 1D-2D coupling code rewritten. The amount of water entering the SWMM nodes is limited at each 2D timestep and cumulated.   
  The cumulative water quantity is then added to the SWMM Model at each 1D Timestep.
  This prevents instability issues when 2D timestep is much smaller than 1D timestep.
- 2D timestep accounts now for waterlevel at 1D nodes. This prevents instability whith large outflows from 1D to 2D surface.
- weir, submerged weir and orifice coupling functions modified to ensure smooth transition and ensure stability.
- Masked areas or with bc_type = 0 are ignored during calculation. Speed-up of simulation.


New Features
- Added Initial Losses. The initial losses depth must be filled up in order that runoff occurs.
- Raingage from SWMM File can be used for 2D Precipitation. 
- SWMM output file .out is written at the end of simulation
- 1D Nodes outside the domain are set to ponding depth 1000x the specified ponded_area. This prevents excessive Waterheads downstream the calculation Domain. It is no longer necessary to specify a different value for those nodes when changing the mask.
- The domain must now be defined with bc_type=1. 
- Open Boundary has been implemented with bc_type=2. Open-Boundary only permits outflow of domain. Instead of masking the domain, one can specify a bc_type value of 2 for outside the domain to allow free outflow of water. 

