# coding=utf8

"""
Copyright (C) 2017 Laurent Courty. drainage_flow() and get_linkage_flow() rewritten by Pirmin Borer

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
import os
cimport numpy as np
cimport cython
from cython.parallel cimport prange
import numpy as np
from libc.math cimport pow as c_pow
from libc.math cimport sqrt as c_sqrt
from libc.math cimport fabs as c_abs
from libc.math cimport fmin as c_min
from libc.math cimport fmax as c_max

from libc.math cimport copysign as c_copysign

cdef float PI = 3.1415926535898
cdef float FOOT = 0.3048

ctypedef np.float32_t F32_t
ctypedef np.int32_t I32_t
ctypedef np.uint8_t uint8_t

ctypedef struct link_struct:
    I32_t idx
    F32_t flow
    F32_t depth
    F32_t velocity
    F32_t volume
    I32_t type
    F32_t offset1
    F32_t offset2
    F32_t full_depth
    F32_t froude

ctypedef struct node_struct:
    I32_t idx
    I32_t linkage_type
    F32_t inflow
    F32_t outflow
    F32_t linkage_flow
    F32_t acc_linkage_flow
    F32_t head
    F32_t crest_elev
    I32_t type
    I32_t sub_index
    F32_t invert_elev
    F32_t init_depth
    F32_t full_depth
    F32_t sur_depth
    F32_t ponded_area
    F32_t weir_area
    I32_t degree
    F32_t crown_elev
    F32_t losses
    F32_t volume
    F32_t full_volume
    F32_t overflow
    F32_t depth
    F32_t lat_flow
    I32_t row
    I32_t col

cdef extern from "source/headers.h" nogil:
    ctypedef struct linkData:
        double flow
        double depth
        double velocity
        double volume
        int    type
        double offset1
        double offset2
        double yFull
        double froude
    ctypedef struct nodeData:
        double inflow
        double outflow
        double head
        double crestElev
        int    type
        int    subIndex
        double invertElev
        double initDepth
        double fullDepth
        double surDepth
        double pondedArea
        int    degree
        char   updated
        double crownElev
        double losses
        double newVolume
        double fullVolume
        double overflow
        double newDepth
        double newLatFlow

    # functions
    double node_getSurfArea(int j, double d)
    double node_getPondedArea(int j, double d)
    int    project_findObject(int type, char* id)
    # exported values
    double MinSurfArea

cdef extern from "source/swmm5.h" nogil:
    int swmm_getNodeID(int index, char* id)
    int swmm_getLinkID(int index, char* id)
    int swmm_getNodeData(int index, nodeData* data)
    int swmm_getLinkData(int index, linkData* data)
    int swmm_addNodeInflow(int index, double inflow)
    int swmm_setNodeFullDepth(int index, double depth)
    int swmm_setNodeSurDepth(int index, double depth)
    int swmm_setNodePondedArea(int index, double area)
    double swmm_getRaingage(int index) #P. Borer

cdef enum linkage_types:
    NOT_LINKED
    NO_LINKAGE
    FREE_WEIR
    SUBMERGED_WEIR
    ORIFICE
def set_NodeFullDepth(int idx, float full_depth):
    swmm_setNodeFullDepth(idx, full_depth / FOOT )

def get_object_index(int obj_type_code, bytes object_id):
    """return the index of an object for a given ID and type
    """
    cdef char* c_obj_id
    c_obj_id = object_id
    return project_findObject(obj_type_code, c_obj_id)


def set_ponding_area(int node_idx, area):
    """Set the ponded area equal to node area.
    SWMM internal ponding don't have meaning any more with the 2D coupling
    The ponding depth is used to keep the node head consistent with
    the WSE of the 2D model.
    """

    swmm_setNodePondedArea(node_idx, area / FOOT ** 2)

def get_Raingage(int gage_idx): #P. Borer
    return swmm_getRaingage(gage_idx)



@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def update_links(link_struct[:] arr_links):
    """get current values from the C objects and put them in the
    corresponding array
    """
    cdef int r, rmax, link_idx
    cdef linkData link_data
    cdef char* link_id
    cdef link_struct link
    rmax = arr_links.shape[0]
    # those operations are not thread safe
    for r in range(rmax):
        # get values
        link = arr_links[r]
        link_idx = link.idx
        swmm_getLinkData(link_idx, &link_data)
        # data type
        link.type = link_data.type

        # assign values
        link.flow = link_data.flow * FOOT ** 3
        link.depth = link_data.depth * FOOT
        link.velocity = link_data.velocity
        link.volume = link_data.volume * FOOT ** 3
        link.offset1 = link_data.offset1 * FOOT
        link.offset2 = link_data.offset2 * FOOT
        link.full_depth = link_data.yFull * FOOT
        link.froude = link_data.froude

        # update array
        arr_links[r] = link


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def update_nodes(node_struct[:] arr_node):

    """get current values from the C objects and put them in the
    corresponding array
    """
    cdef int r, rmax
    cdef nodeData node_data
    cdef char* node_id
    cdef node_struct node
    rmax = arr_node.shape[0]
    # those operations are not thread safe
    for r in range(rmax):
        # get values
        node = arr_node[r]
        swmm_getNodeData(node.idx, &node_data)
        # data type
        node.type = node_data.type

        # assign values
        node.sub_index = node_data.subIndex
        node.degree = node_data.degree

        # translate to SI units
        node.invert_elev = node_data.invertElev * FOOT
        node.init_depth  = node_data.initDepth * FOOT
        node.full_depth  = node_data.fullDepth * FOOT
        node.depth       = node_data.newDepth * FOOT
        node.sur_depth   = node_data.surDepth * FOOT
        node.crown_elev  = node_data.crownElev * FOOT
        node.head        = node_data.head * FOOT
        node.crest_elev  = node_data.crestElev * FOOT
        node.ponded_area = node_data.pondedArea * FOOT ** 2
        node.volume      = node_data.newVolume * FOOT ** 3
        node.full_volume = node_data.fullVolume * FOOT ** 3

        node.inflow      = node_data.inflow * FOOT ** 3
        node.outflow     = node_data.outflow * FOOT ** 3
        node.losses      = node_data.losses * FOOT ** 3
        node.overflow    = node_data.overflow * FOOT ** 3
        node.lat_flow    = node_data.newLatFlow * FOOT ** 3
        # update array
        arr_node[r] = node


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def drainage_flow(node_struct[:] arr_node,
                       F32_t[:,:] arr_h, F32_t[:,:] arr_z, F32_t[:,:] arr_qdrain, F32_t[:,:] arr_qw, F32_t[:,:] arr_qe, F32_t[:,:] arr_qn, F32_t[:,:] arr_qs, F32_t[:,:] arr_q_in, uint8_t[:,:] arr_mask,
                       float dx, float dy, float dt2d, float dt1d, float g, 
                       float orifice_coeff, float free_weir_coeff, float submerged_weir_coeff):
    '''select the linkage type then calculate the flow between
    surface and drainage models (Pirmin Borer)
    flow sign is :
     - negative when entering the drainage (leaving the 2D model)
     - positive when leaving the drainage (entering the 2D model)
    cell_surf: area in m² of the cell above the node
    dt2d: time-step of the 2d model in seconds
    dt1d: time-step of the drainage model in seconds
    '''
    cdef int i, imax, linkage_type, row, col
    cdef float crest_elev, weir_width, overflow_area, 
    cdef float linkage_flow, maxflow, q_sum, qw, qe, qn, qs, q_in
    cdef float wse, z , dh,  h,  full_depth, hmax, qtot, head_estim, max_head
    cdef node_struct node

    cell_surf = dx * dy
    hmax = 0
    imax = arr_node.shape[0]
    for i in range(imax,):
        node = arr_node[i]
        # don't do anything if the node is not linked
        if node.linkage_type == linkage_types.NOT_LINKED:
            continue

        # corresponding grid coordinates at drainage node
        row = node.row
        col = node.col
        # if node is masked don't do anything
        if arr_mask[row, col] :
            continue

        # values on the surface
        qw = arr_qw[row, col]
        qe = arr_qe[row, col]
        qn = arr_qn[row, col]
        qs = arr_qs[row, col]
        q_in = arr_q_in[row, col]

        h = arr_h[row, col]

        # crest elevation has been set by _set_linkable. It assumes DEM is on top of node crest_elevation.
        wse = node.crest_elev + h
        #get either node surface_area or node_ponded area if node is full
        overflow_area = get_overflow_area(node.idx,node.depth)

        # weir width is the circumference (node considered circular)
        weir_width = PI * 2. * c_sqrt(node.weir_area / PI)

        head_estim =min( node.head + (node.inflow-node.outflow)*dt1d / overflow_area , c_pow(0.7*dx/dt2d,2)/9.81 )
        ## calculate drainage flow and set linkage type##
        linkage_flow = get_linkage_flow(node, head_estim, wse, weir_width, node.ponded_area, g, orifice_coeff, free_weir_coeff, submerged_weir_coeff, dt2d)


        if linkage_flow < 0 :
            #Cap the drainage flow such as not to generate negative depths. 
            #flow from neighbor cells
            q_sum = (qw - qe) / dx + (qn - qs) / dy #flow from neighboring cells in m/s
            
            #limit flow to waterheight on cell plus flow from neighbor cells. Also limit flow to prevent flow inversion until next 1d step
            maxflow = min(cell_surf *(h / dt2d + q_sum + q_in), (wse-node.head)*overflow_area/dt1d) #maxflow in m3/s
            linkage_flow = - c_min(maxflow, -linkage_flow)+node.overflow

        # prevent flow inversion when node is surcharging
        if linkage_flow > 0:
            maxflow = (node.head-wse)*overflow_area/dt1d
            linkage_flow = min(linkage_flow, maxflow)

        # if 2D Step is smaller than 1D Step limit the accumulated drainage flow to prevent flow inversion until next 1D Step. 
        # Else increment accumulated_flow which will be added at next 1D Step to SWMM Model.
        maxflow = (wse-node.head)*overflow_area #maximum volume. Variable maxflow is reused.
        if abs(node.acc_linkage_flow)>abs(maxflow):
            node.acc_linkage_flow = maxflow
            linkage_flow = 0.
        else:
            #accumulate drainage flow for SWMM Model
            node.acc_linkage_flow -= linkage_flow * dt2d

        #apply linkage flow to 2D model
        arr_qdrain[row, col] = linkage_flow /cell_surf

        # update node array
        arr_node[i] = node
        #update maximum node water height for courant criterion
        max_head = node.head - node.crest_elev
        if max_head > hmax :
           hmax = max_head

    return hmax
    
@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def apply_linkage_flow_SWMM(node_struct[:] arr_node, F32_t[:,:] arr_qdrain, float dt1d):
    cdef int i, imax, row, col
    cdef float linkage_flow
    #Add drainage flow since last 1D Step. Accumulated drainage flow is added at 1D steps to SWMM Model
    imax = arr_node.shape[0]

    for i in range(imax,):

        row = arr_node[i].row
        col = arr_node[i].col
        
        linkage_flow = arr_node[i].acc_linkage_flow / dt1d 
        swmm_addNodeInflow(arr_node[i].idx, linkage_flow/ FOOT ** 3) 
        arr_node[i].acc_linkage_flow = 0 #reset the accumulated flow
        arr_node[i].linkage_flow = linkage_flow


cdef float get_linkage_flow(node_struct node, float head, float wse, float weir_width,
                            float overflow_area,
                            float g, float orifice_coeff, float free_weir_coeff,
                            float submerged_weir_coeff, float dt2d):
    """flow sign is :
            - negative when entering the drainage (leaving the 2D model)
            - positive when leaving the drainage (entering the 2D model)
    """
    cdef float hs, ha, weir_flow, sub_weir_flow, orifice_flow, linkage_flow
    cdef int drainage, overflow

    hs = wse - node.crest_elev

    drainage = head < wse and wse > node.crest_elev 
    overflow = head > wse + node.sur_depth and head > node.crest_elev+node.sur_depth
    
    # calculate the flow
    if drainage:
        weir_flow = (2./3.) * free_weir_coeff * weir_width * c_pow(hs, 3/2.) * c_sqrt(2. * g)
        sub_weir_flow = (2./3.) * submerged_weir_coeff * weir_width *  c_sqrt(2. * g ) * hs * c_sqrt(wse - head)
        orifice_flow = (orifice_coeff * overflow_area *c_sqrt(2. * g)* c_sqrt(wse - head))


        # take the minimum flow between possible flow regimes (velocity is minimised and thus corresponds to entropy principle. Also ensures smooth transitions

        if head < node.crest_elev and weir_flow <= sub_weir_flow and weir_flow <= orifice_flow:
           node.linkage_type = linkage_types.FREE_WEIR
           linkage_flow = - weir_flow
        elif sub_weir_flow < weir_flow and sub_weir_flow <= orifice_flow:
           node.linkage_type = linkage_types.SUBMERGED_WEIR
           linkage_flow = - sub_weir_flow
        else :
           node.linkage_type = linkage_types.ORIFICE
           linkage_flow = - orifice_flow

    elif overflow:
        orifice_flow = orifice_coeff * overflow_area *c_sqrt(2. * g)*c_sqrt(head - node.sur_depth - wse)
        node.linkage_type = linkage_types.ORIFICE
        linkage_flow = orifice_flow
        
    else:
        node.linkage_type = linkage_types.NO_LINKAGE
        linkage_flow = 0.


    return linkage_flow


cdef float get_overflow_area(int node_idx, float node_depth):
    """return overflow area defauting to MinSurfArea
    """
    cdef float overflow_area, surf_area
    surf_area = node_getPondedArea(node_idx, node_depth)
    if surf_area <= 0.:
        surf_area = MinSurfArea
    return surf_area * FOOT ** 2
