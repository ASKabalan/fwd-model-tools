# PRIVATE TODOs

## Check if _compute.py works

I need it to have a single mesh usage for flat_cl spherical_cl and 3d pk
able to be traced

compute.py should just call it

## density.py refactor

should be able to

- call paint in a lax.map
- call compute spectra in a lax.map
- have easy to use felixble plotting functions

## NBody function

has to be able to accept

- ts=None will infer from nb_shells
- ts=[t0,t1,...] explicit times and no width given by users (compute widths from ts)

## Save lightcone function

dumps light cone to disk in a npz with parquet tables with

lighcone + cosmology + near_z + far_z



## add two slides to
