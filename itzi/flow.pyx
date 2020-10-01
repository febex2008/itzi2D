# coding=utf8
"""
Copyright (C) 2015-2016 Laurent Courty. Green-Ampt infiltration and Capping of infiltration losses corrected by Pirmin Borer

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""
from __future__ import division
cimport numpy as np
cimport cython
from cython.parallel cimport prange
import numpy as np
from libc.math cimport pow as c_pow
from libc.math cimport sqrt as c_sqrt
from libc.math cimport fabs as c_abs
from libc.math cimport atan2 as c_atan
from libc.math cimport fmax as c_max
from libc.math cimport fmin as c_min

ctypedef np.float32_t DTYPE_t
cdef float PI = 3.1415926535898


@cython.wraparound(False)  # Disable negative index check
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def arr_sum(DTYPE_t [:, :] arr):
    '''Return the sum of an array using parallel reduction'''
    cdef int rmax, cmax, r, c
    cdef float asum = 0.
    rmax = arr.shape[0]
    cmax = arr.shape[1]
    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            asum += arr[r, c]
    return asum


@cython.wraparound(False)  # Disable negative index check
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def arr_add(DTYPE_t [:, :] arr1, DTYPE_t [:, :] arr2):
    '''Add arr1 to arr2'''
    cdef int rmax, cmax, r, c
    rmax = arr1.shape[0]
    cmax = arr1.shape[1]
    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            arr2[r, c] += arr1[r, c]


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def apply_hydrology(DTYPE_t [:, :] arr_rain, DTYPE_t [:, :] arr_inf,
                    DTYPE_t [:, :] arr_etp, DTYPE_t [:, :] arr_capped_losses, DTYPE_t [:, :] arr_init_losses, DTYPE_t[:,:] arr_cum_init_losses,
                    DTYPE_t [:, :] arr_h, DTYPE_t [:, :] arr_bct,float dt ):
    '''Add rain, infiltration etc. to the domain depth for the current time-step
    rain and inf in mm/h, deph in m, dt in seconds, initial losses in mm'''
    cdef int rmax, cmax, r, c
    cdef float rain, etp, inf, capped_losses, dt_h, h, h_new, dh, init_losses, cum_init_losses
    rmax = arr_rain.shape[0]
    cmax = arr_rain.shape[1]
    dt_h = dt / 3600.  # dt from sec to hours
    
    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            if arr_bct[r,c]==0 or arr_bct[r,c]==2: #Do not compute outside domain (closed or open boundary regions). 
                arr_rain[r,c]=0.
                continue
            if arr_bct[r,c]==5: #if bc type is 5, do not apply rain (E.g. when using SWMM and using catchment). Setting rain to zero allows water flowing from neighboring areas to infiltrate. 
                rain = 0.

            rain = arr_rain[r, c]

            inf = arr_inf[r, c]
            etp = arr_etp[r, c]
            capped_losses = arr_capped_losses[r, c]
            init_losses = arr_init_losses[r, c] #user-specified initial losses height
            cum_init_losses = arr_cum_init_losses[r , c] #current water height in mm in initial losses
            h = arr_h[r, c] #current water height on cell

            dh = (rain - inf - etp - capped_losses) * dt_h #water height in mm added to cell. 

            #Update waterheight
            h_new = c_max(h + dh/1000. - (init_losses-cum_init_losses) / 1000. , 0) #update water height minus difference between inital_losses and cumulated inital losses. 

            #Only update initial losses if they are not filled or water_height is zero
            if cum_init_losses < init_losses or h == 0 :
                cum_init_losses = c_min(c_max(cum_init_losses + dh + 1000 * h, 0), init_losses) #initial losses are filled up by rain and water on cell. Minimum value is 0 and maximum value is user_specified init_losses

            #Update arrays
            arr_cum_init_losses[r,c] = cum_init_losses
            arr_h[r, c] = h_new


@cython.wraparound(False)  # Disable negative index check
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def flow_dir(DTYPE_t [:, :] arr_max_dz, DTYPE_t [:, :] arr_dz0,
             DTYPE_t [:, :] arr_dz1, DTYPE_t [:, :] arr_dir):
    '''Populate arr_dir with a rain-routing direction:
    0: the flow is going dowstream, index-wise
    1: the flow is going upstream, index-wise
    -1: no routing happening on that face
    '''
    cdef int rmax, cmax, r, c
    cdef float max_dz, dz0, dz1, qdir
    rmax = arr_max_dz.shape[0]
    cmax = arr_max_dz.shape[1]
    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            max_dz = arr_max_dz[r, c]
            dz0 = arr_dz0[r, c]
            dz1 = arr_dz1[r, c]
            qdir = arr_dir[r, c]
            if max_dz > 0:
                if max_dz == dz0:
                    qdir = 0
                elif max_dz == dz1:
                    qdir = 1
                else:
                    qdir = -1
            else:
                qdir = -1
            # update results array
            arr_dir[r, c] = qdir


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def solve_q(DTYPE_t [:, :] arr_dire, DTYPE_t [:, :] arr_dirs,
            DTYPE_t [:, :] arr_z, DTYPE_t [:, :] arr_n, DTYPE_t [:, :] arr_h,
            DTYPE_t [:, :] arrp_qe, DTYPE_t [:, :] arrp_qs,
            DTYPE_t [:, :] arr_hfe, DTYPE_t [:, :] arr_hfs,
            DTYPE_t [:, :] arr_qe_new, DTYPE_t [:, :] arr_qs_new,
            DTYPE_t [:, :] arr_bct, DTYPE_t [:, :] arr_bcv,
            float dt, float dx, float dy, float g,
            float theta, float hf_min, float v_rout, float sl_thres):
    '''Calculate hflow in m, flow in m2/s
    '''
    # Stencils corrected by P.Borer
    cdef int rmax, cmax, r, c, rp, cp, isboundary
    cdef float wse_e, wse_s, wse0, z0, ze, zs, n0, n, ne, ns
    cdef float qe_st, qs_st, qe_vect, qs_vect, qdire, qdirs
    cdef float qe_new, qs_new, hf_e, hf_s, h0, h_e, h_s,hf

    rmax = arr_z.shape[0]
    cmax = arr_z.shape[1]
    for r in prange(rmax, nogil=True):
        for c in xrange(cmax):
            rp = r + 1
            cp = c + 1

            qe_new = 0
            qs_new = 0


            if arr_bct[r,c]!=0: #Do not compute cells outside of domain
                # values at the current cell
                z0 = arr_z[r, c]
                h0 = arr_h[r, c]
                wse0 = z0 + h0
                n0 = arr_n[r,c]
                # x dimension, flow at E cell boundary
                # prevent calculation of domain boundary
                # range(10) is from 0 to 9, so last cell is max - 1
                # x dimension, flow at E cell boundary
                
                if c < cmax and arr_bct[r,c+1]!=0 and (arr_bct[r,c]==2 and arr_bct[r,c+1]==2)!=1 : #Only compute where next cell is inside domain.

                    # flow routing direction
                    qdire = arr_dire[r, c]
                    # water surface elevation
                    ze = arr_z[r, c+1]
                    h_e = arr_h[r, c+1]
                    wse_e = ze + h_e

                    # hflow
                    hf_e = hflow(z0=z0, z1=ze, wse0=wse0, wse1=wse_e)
                    arr_hfe[r, c] = hf_e
                    hf = hf_e + arrp_qe[rp,cp]/hf_e * arrp_qe[rp,cp]/hf_e /2/9.81 #P.Borer :account for flow speed to avoid rain_routing where high velocities but flow height is low.
                    # flow and velocity
                    if hf_e <= 0:
                        qe_new = 0
                    elif hf > hf_min:
                        # average friction
                        ne = 0.5 * (n0 + arr_n[r,c+1])
                        # qnorm
                        #if next cell is open-boundary
                        if arr_bct[r,c]==1 and arr_bct[r,c+1]==2 :
                            arrp_qe[rp,cp+1]= c_max(arrp_qe[rp, cp],0) #flow must be positive, no inflow permitted
                            arrp_qs[rp,cp+1]= arrp_qs[rp,cp]
                            arrp_qs[rp-1,cp+1]=arrp_qs[rp-1,cp]
                            wse_e = ze+h0
                        #if current cell is open boundary and next cell domain
                        if arr_bct[r,c+1]==1 and arr_bct[r,c]==2: 
                            arrp_qe[rp,cp-1]= c_min(arrp_qe[rp, cp],0) #flow must be negative, no inflow permitted
                            arrp_qs[rp,cp]= arrp_qs[rp,cp+1]
                            arrp_qs[rp-1,cp]=arrp_qs[rp-1,cp+1]
                            wse0 = z0+h_e
                        # calculate average flow from stencil
                        qe_st = .25 * (arrp_qs[rp-1,cp] + arrp_qs[rp,cp] + arrp_qs[rp-1,cp+1] + arrp_qs[rp,cp+1])
                        # calculate qnorm
                        qe_vect = c_sqrt(arrp_qe[rp,cp] * arrp_qe[rp,cp] + qe_st * qe_st)
                        qe_new = almeida2013(hf=hf_e, wse0=wse0, wse1=wse_e, n=ne,
                                             qm1=arrp_qe[rp,cp-1], q0=arrp_qe[rp,cp],
                                             qp1=arrp_qe[rp,cp+1], q_norm=qe_vect,
                                             theta=theta, g=g, dt=dt, cell_len=dx)
                    # flow going W, i.e negative
                    elif hf <= hf_min  and wse_e > wse0: #and qdire == 0
                        qe_new = - rain_routing(h_e, wse_e, wse0,
                                                dt, dx, v_rout)
                    # flow going E, i.e positive
                    elif hf <= hf_min  and wse0 > wse_e: #and  qdire == 1
                        qe_new = rain_routing(h0, wse0, wse_e,
                                              dt, dx, v_rout)
                    else:
                        qe_new = 0
                    # udpate array
                    arr_qe_new[r, c] = qe_new

                # y dimension, flow at S cell boundary
                if r < rmax and arr_bct[r+1,c]!=0 and (arr_bct[r,c]==2 and arr_bct[r+1,c]==2)!=1 : #Only compute where next cell is inside domain.
                    
                    # flow routing direction
                    qdirs = arr_dirs[r, c]
                    # water surface elevation
                    zs = arr_z[r+1, c]
                    h_s = arr_h[r+1, c]
                    wse_s = zs + h_s

                    # hflow
                    hf_s = hflow(z0=z0, z1=zs, wse0=wse0, wse1=wse_s)
                    arr_hfs[r, c] = hf_s
                    hf = hf_s + arrp_qs[rp,cp]/hf_s * arrp_qs[rp,cp]/hf_s /2./9.81 #P.Borer :account for flow speed to avoid rain_routin where high velocities but flow height is low.
                    if hf_s <= 0:
                        qs_new = 0
                    elif hf > hf_min:
                        # average friction
                        ns = 0.5 * (n0 + arr_n[r+1,c])
                        #if next cell is open-boundary
                        if arr_bct[r+1,c]==2 :
                            arrp_qs[rp+1,cp]= c_max(arrp_qs[rp, cp],0)
                            arrp_qe[rp+1,cp]= arrp_qe[rp,cp]
                            arrp_qe[rp+1,cp-1]=arrp_qe[rp,cp-1]
                            wse_s = zs+h0
                        #if current cell is open boundary and next cell domain
                        if arr_bct[r+1,c]==1 and arr_bct[r,c]==2: 
                            arrp_qs[rp-1,cp]= c_min(arrp_qs[rp, cp],0)
                            arrp_qe[rp,cp]= arrp_qe[rp+1,cp]
                            arrp_qe[rp,cp-1]=arrp_qe[rp+1,cp-1]
                            wse0 = z0+h_s
                        # qnorm
                        # calculate average flow from stencil

                        qs_st = .25 * (arrp_qe[rp,cp-1] + arrp_qe[rp,cp] + arrp_qe[rp+1,cp-1] + arrp_qe[rp+1,cp])
                        # calculate qnorm
                        qs_vect = c_sqrt(arrp_qs[rp,cp] * arrp_qs[rp,cp] +
                                     qs_st * qs_st)
                        qs_new = almeida2013(hf=hf_s, wse0=wse0, wse1=wse_s, n=ns,
                                             qm1=arrp_qs[rp-1,cp], q0=arrp_qs[rp,cp],
                                             qp1=arrp_qs[rp+1,cp], q_norm=qs_vect,
                                             theta=theta, g=g, dt=dt, cell_len=dy)
                    # flow going N, i.e negative
                    elif hf <= hf_min and  wse_s > wse0:# and qdirs == 0 :
                        qs_new = - rain_routing(h_s, wse_s, wse0,
                                                dt, dy, v_rout)
                    # flow going S, i.e positive
                    elif hf <= hf_min  and wse0 > wse_s: #and  qdirs == 1:
                        qs_new = rain_routing(h0, wse0, wse_s,
                                              dt, dy, v_rout)
                    else:
                        qs_new = 0
                    # udpate array
                    arr_qs_new[r, c] = qs_new




cdef float hflow(float z0, float z1, float wse0, float wse1) nogil:
    """calculate flow depth
    """
    return max(wse1, wse0) - max(z1, z0)


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
cdef float almeida2013(float hf, float wse0, float wse1, float n,
                       float qm1, float q0, float qp1,
                       float q_norm, float theta,
                       float g, float dt, float cell_len) nogil:
    '''Solve flow using q-centered scheme from Almeida et Al. (2013)
    '''
    cdef float term_1, term_2, term_3, slope

    slope = (wse0 - wse1) / cell_len

    term_1 = theta * q0 + (1 - theta) * (qm1 + qp1) * 0.5
    term_2 = g * hf * dt * slope
    term_3 = 1 + g * dt * (n*n) * q_norm / c_pow(hf, 7./3.)
    # If flow direction is not coherent with surface slope,
    # use only previous flow, i.e. ~switch to Bates 2010
    if term_1 * term_2 < 0:
        term_1 = q0
    return (term_1 + term_2) / term_3


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
cdef float rain_routing(float h0, float wse0, float wse1, float dt,
                        float cell_len, float v_routing) nogil:
    """Calculate flow routing at a face in m2/s
    Cf. Sampson et al. (2013)
    """
    cdef float maxflow, q_routing, dh
    # fraction of the depth to be routed
    dh = wse0 - wse1
    # make sure it's positive (should not happend, checked before)
    dh = c_max(dh, 0)
    # if WSE of destination cell is below the dem of the drained cell, set to h0
    dh = c_min(dh, h0)
    # maxflow = cell_len * dh / dt
    # q_routing = min(dh * v_routing, maxflow)
    q_routing = dh * v_routing
    return q_routing


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def solve_h(DTYPE_t [:, :] arr_ext,
            DTYPE_t [:, :] arr_qe, DTYPE_t [:, :] arr_qw,
            DTYPE_t [:, :] arr_qn, DTYPE_t [:, :] arr_qs,
            DTYPE_t [:, :] arr_bct, DTYPE_t [:, :] arr_bcv,
            DTYPE_t [:, :] arr_h, DTYPE_t [:, :] arr_hmax,
            DTYPE_t [:, :] arr_hfix, DTYPE_t [:, :] arr_herr,
            DTYPE_t [:, :] arr_hfe, DTYPE_t [:, :] arr_hfw,
            DTYPE_t [:, :] arr_hfn, DTYPE_t [:, :] arr_hfs,
            DTYPE_t [:, :] arr_v, DTYPE_t [:, :] arr_vdir,
            DTYPE_t [:, :] arr_vmax, DTYPE_t [:, :] arr_fr,
            DTYPE_t [:, :] arr_celerity,
            float dx, float dy, float dt, float g):
    '''Update the water depth and max depth
    Adjust water depth according to in-domain 'boundary' condition
    Calculate vel. magnitude in m/s, direction in degree and Froude number.
    '''
    cdef int rmax, cmax, r, c
    cdef float qext, qe, qw, qn, qs, h, q_sum, h_new, hmax, bct, bcv
    cdef float hfe, hfs, hfw, hfn, ve, vw, vn, vs, vx, vy, v, vdir, v_square

    rmax = arr_qe.shape[0]
    cmax = arr_qe.shape[1]
    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            bct = arr_bct[r, c]
            bcv = arr_bcv[r, c]
 
            if bct ==1 or bct ==4 or bct ==5:

                qext = arr_ext[r, c]
                qe = arr_qe[r, c]
                qw = arr_qw[r, c]
                qn = arr_qn[r, c]
                qs = arr_qs[r, c]

                h = arr_h[r, c]
                hmax = arr_hmax[r, c]
                # Sum of flows in m/s
                q_sum = (qw - qe) / dx + (qn - qs) / dy

                # calculate new flow depth
                h_new = h + (qext + q_sum) * dt
                if h_new < 0.:
                    # Write error. Always positive (mass creation)
                    arr_herr[r, c] += - h_new
                    h_new = 0.
                # Apply fixed water level
                if bct == 4:
                    # Positive if water enters the domain
                    arr_hfix[r, c] += bcv - h_new
                    h_new = bcv

                # Update max depth array
                arr_hmax[r, c] = max(h_new, hmax)
                # Update depth array
                arr_h[r, c] = h_new

                ## Velocity and Froude ##
                # Do not accept NaN
                hfe = arr_hfe[r, c]
                hfw = arr_hfw[r, c]
                hfn = arr_hfn[r, c]
                hfs = arr_hfs[r, c]
                if hfe <= 0.:
                    ve = 0.
                else:
                    ve = qe / hfe
                if hfw <= 0.:
                    vw = 0.
                else:
                    vw = qw / hfw
                if hfs <= 0.:
                    vs = 0.
                else:
                    vs = qs / hfs
                if hfn <= 0.:
                    vn = 0.
                else:
                    vn = qn / hfn

                vx = .5 * (ve + vw)
                vy = .5 * (vs + vn)

                # velocity magnitude and direction
                
                v = c_sqrt(vx*vx + vy*vy)
                arr_celerity[r,c]= v + c_sqrt(9.81*h_new)
                arr_v[r, c] = v
                arr_vmax[r, c] = c_max(v, arr_vmax[r, c])
                vdir = c_atan(-vy, vx) * 180. / PI
                if vdir < 0:
                    vdir = 360 + vdir
                arr_vdir[r, c] = vdir

                # Froude number
                arr_fr[r, c] = v / c_sqrt(g * h_new)


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def set_ext_array(DTYPE_t [:, :] arr_qext, DTYPE_t [:, :] arr_drain,
                  DTYPE_t [:, :] arr_ext):
    '''Calculate the new ext_array to be used in depth update
    '''
    cdef int rmax, cmax, r, c
    cdef float qext, rain, inf, qdrain

    rmax = arr_qext.shape[0]
    cmax = arr_qext.shape[1]
    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            arr_ext[r, c] = arr_qext[r, c] + arr_drain[r, c]


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def inf_user(DTYPE_t [:, :] arr_h,
        DTYPE_t [:, :] arr_inf_in, DTYPE_t [:, :] arr_inf_out,
        DTYPE_t [:, :] arr_rain,
        DTYPE_t [:, :] arr_losses,
        DTYPE_t [:, :] arr_cum_init_losses,
        float dt):
    '''Calculate infiltration rate using a user-defined fixed rate
    '''
    cdef int rmax, cmax, r, c
    cdef float dt_h

    rmax = arr_h.shape[0]
    cmax = arr_h.shape[1]
    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            dt_h = dt / 3600.  # dt from sec to hours
            # cap the rate
            arr_inf_out[r,c] = cap_inf_rate(dt_h, arr_h[r,c], arr_inf_in[r,c], arr_rain[r,c], arr_losses[r,c], arr_cum_init_losses[r,c])


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def inf_ga(DTYPE_t [:, :] arr_h, DTYPE_t [:, :] arr_eff_por,
        DTYPE_t [:, :] arr_pressure, DTYPE_t [:, :] arr_conduct,
        DTYPE_t [:, :] arr_inf_amount, DTYPE_t [:, :] arr_water_soil_content,
        DTYPE_t [:, :] arr_inf_out, DTYPE_t [:, :] arr_rain, DTYPE_t [:, :] arr_losses, DTYPE_t [:, :] arr_cum_init_losses, float dt):
    '''Calculate infiltration rate using the Green-Ampt formula
    '''
    cdef int rmax, cmax, r, c
    cdef float dt_h, infrate, avail_porosity, poros_cappress, conduct
    rmax = arr_h.shape[0]
    cmax = arr_h.shape[1]
    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            dt_h = dt / 3600.  # dt from sec to hours
            conduct = arr_conduct[r, c]
            avail_porosity = arr_eff_por[r, c] - arr_water_soil_content[r, c]
            poros_cappress = avail_porosity * arr_pressure[r, c]
            infrate = conduct * (1 +(poros_cappress / arr_inf_amount[r, c]))
            # cap the rate
            infrate = cap_inf_rate(dt_h, arr_h[r,c], infrate, arr_rain[r,c], arr_losses[r,c], arr_cum_init_losses[r,c])
            # update total infiltration amount
            arr_inf_amount[r, c] += infrate * dt_h
            # populate output infiltration array
            arr_inf_out[r, c] = infrate


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
cdef float cap_inf_rate(float dt_h, float h, float infrate, float rain, float losses, float init_losses) nogil:
    '''Cap the infiltration rate. The maximum infiltration accounts for waterheight, rain and losses. Losses are considered substracted first.
    '''
    cdef float h_mm, max_rate
    # calculate the max_rate
    h_mm = h * 1000.
    max_rate = (h_mm + init_losses) / dt_h + rain - losses
    return min(max_rate, infrate)

@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def cap_losses(float dt, DTYPE_t [:, :] arr_h, DTYPE_t [:, :] arr_rain, DTYPE_t [:, :] arr_in_losses, DTYPE_t [:, :] arr_out_losses ):
    '''Cap the losses. The losses are substracted from rain and water depth. 
    '''
    cdef int rmax, cmax, r, c
    cdef float dt_h, max_rate
    rmax = arr_h.shape[0]
    cmax = arr_h.shape[1]

    dt_h = dt / 3600.
    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            max_rate = arr_h[r,c] * 1000. / dt_h + arr_rain[r,c]
            arr_out_losses[r,c]= min(max_rate, arr_in_losses[r,c])


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def populate_stat_array(DTYPE_t [:, :] arr, DTYPE_t [:, :] arr_stat,
                        float conv_factor, float time_diff):
    '''Populate an array of statistics
    '''
    cdef int rmax, cmax, r, c
    rmax = arr.shape[0]
    cmax = arr.shape[1]
    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            arr_stat[r, c] += arr[r, c] * conv_factor * time_diff

