# coding=utf8
"""
Copyright (C) 2016-2017 Laurent Courty. drainage_flow() and step() rewritten by Pirmin Borer

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
from __future__ import absolute_import
import os
from datetime import datetime, timedelta
from collections import namedtuple
import numpy as np
import networkx as nx

from itzi.swmm import swmm
from itzi.itzi_error import DtError
import itzi.messenger as msgr

class DrainageSimulation(object):
    """manage simulation of the pipe network
    write results to RasterDomain object
    """

    # define namedtuples
    LayerDescr = namedtuple('LayerDescr', ['table_suffix', 'cols', 'layer_number'])
    GridCoords = namedtuple('GridCoords', ['row', 'col'])

    def __init__(self, domain, drainage_params, igis, g):
        self.dom = domain
        self.g = g
        inp_file = drainage_params['swmm_inp']
        self.swmm_gage = drainage_params['swmm_gage']
        # create swmm object, open files and start simulation
        self.swmm5 = swmm.Swmm5()
        self.swmm5.swmm_open(input_file=inp_file,
                             report_file=inp_file[:len(inp_file)-4] + '.rpt',
                             output_file=inp_file[:len(inp_file)-4] + '.out')
        self.swmm5.swmm_start()
        # allow ponding
        self.swmm5.allow_ponding()
        # geo information
        self.cell_surf = igis.dx * igis.dy
        self.gis = igis
        self.hmax = 0. #initialize max head of 1D network
        # definition of linking_elements (used for GRASS vector writing)
        node_col_def = swmm.SwmmNode.sql_columns_def
        link_col_def = swmm.SwmmLink.sql_columns_def
        self.linking_elements = {'node': self.LayerDescr(table_suffix='_node',
                                                         cols=node_col_def,
                                                         layer_number=1),
                                 'link': self.LayerDescr(table_suffix='_link',
                                                         cols=link_col_def,
                                                         layer_number=2)}

        swmm_inp = swmm.SwmmInputParser(drainage_params['swmm_inp'])
        self.node_id_dict = swmm_inp.get_nodes_id_as_dict()
        self.link_id_dict = swmm_inp.get_links_id_as_dict()
        # create SwmmNetwork object
        self.swmm_net = swmm.SwmmNetwork(self.node_id_dict, self.link_id_dict,
                                         self.gis, self.dom, self.g, self.cell_surf,
                                         drainage_params['orifice_coeff'],
                                         drainage_params['free_weir_coeff'],
                                         drainage_params['submerged_weir_coeff'])

        # create a graph made of drainage nodes and links objects
        node_obj_dict = self.get_node_object_dict(self.node_id_dict)
        link_obj_list = self.get_link_object_list(self.link_id_dict)
        self.create_drainage_network_graph(node_obj_dict, link_obj_list)



    def __del__(self):
        """Make sure the swmm simulation is ended and closed properly.
        """
        self.swmm5.swmm_end()
        self.swmm5.swmm_close()

    def solve_dt(self):
        """Get the time-step from swmm object
        """
        new = self.swmm5.get_NewRoutingTime()
        old = self.swmm5.get_OldRoutingTime()
        self._dt = new - old
        if self._dt <= 0:
            self._dt = self.swmm5.routing_getRoutingStep()

        if self.hmax > 0:
            dt2d = 0.7 * (min(self.gis.dx, self.gis.dy) / (np.sqrt(self.g * self.hmax)))
        else:
            dt2d = 86399.
            
        self.dt2d = timedelta(seconds = dt2d)

    @property
    def dt(self):
        return timedelta(seconds=self._dt)

    @dt.setter
    def dt(self, value):
        raise DtError("Can't set time-step of a SWMM simulation")

    def get_node_object_dict(self, n_dict):
        """create dict id:SwmmNode object
        """
        drain_nodes = {}
        for k, coords in n_dict.iteritems():
            # create Node object
            node = swmm.SwmmNode(swmm_network=self.swmm_net, node_id=k,
                                 coordinates=coords)
            # populate dict
            drain_nodes[k] = node
        return drain_nodes

    def get_link_object_list(self, lnk_dict):
        """Take a dict of objects definition as input.
        Return a list of SwmmLink objects
        """
        drain_links = []
        for k, values in lnk_dict.iteritems():
            drain_link = swmm.SwmmLink(swmm_network=self.swmm_net, link_id=k)
            drain_link.vertices = values.vertices
            drain_link.start_node_id = values.in_node
            drain_link.end_node_id = values.out_node
            drain_links.append(drain_link)
        return drain_links

    def create_drainage_network_graph(self, node_dict, link_list):
        """create a networkx object using given links and nodes lists
        """
        self.drainage_network = nx.MultiDiGraph()
        self.drainage_network.add_nodes_from(node_dict.values())
        for link in link_list:
            in_node = link.start_node_id
            out_node = link.end_node_id
            self.drainage_network.add_edge(node_dict[in_node],
                                           node_dict[out_node],
                                           object=link)
        return self

    def set_linkable(self):
        self.swmm_net.update_nodes()
        self.swmm_net._set_linkable(self.node_id_dict)

    def step(self):
        """Run a swmm time-step
        calculate the exchanges with raster domain
        """
        # calculate when will happen the next time-step



        self.swmm5.swmm_step()
        self.solve_dt()
        self.swmm_net.update_nodes()
        self.swmm_net.update_links()


        self.get_rainfall()
        return self

    def drainage_flow(self, dt2d):
        """For each linked node,
        calculate the flow entering or leaving the drainage network
        Apply the flow to the node and to the relevant raster cell"""

        self.su = slice(None, -2)
        self.ss = slice(1, -1)
        arr_h = self.dom.get('h')
        arr_z = self.dom.get('z')
        arr_qd = self.dom.get('n_drain')
        arr_qw = self.dom.get_padded('qe_new')[self.ss, self.su]
        arr_qe = self.dom.get('qe_new')
        arr_qn = self.dom.get_padded('qs_new')[self.su, self.ss]
        arr_qs = self.dom.get('qs_new')
        dx = self.dom.dx
        dy = self.dom.dy
        arr_q_in = self.dom.get('in_q')
        mask = self.dom.mask
        self.hmax = self.swmm_net.drainage_flow(arr_h, arr_z, arr_qd, arr_qw, arr_qe, arr_qn, arr_qs, arr_q_in, mask, dx, dy, dt2d, self._dt)

        return self

    def apply_drainage_flow(self):
        arr_qdrain = self.dom.get('n_drain')
        self.swmm_net.apply_linkage_flow_SWMM(arr_qdrain, self._dt)

    def get_rainfall(self):
        if self.swmm_gage == None :
            return self
        new_rainfall = self.swmm_net.get_rainfall(self.swmm_gage)

        if  new_rainfall > 0.:
            self.dom.arr['rain'][:] = self.dom.swmm_rainfall
            self.dom.mask_array(self.dom.arr['rain'], 0.)
            self.dom.swmm_rainfall = new_rainfall
            #msgr.message("swmm rain: " + str(self.dom.swmm_rainfall))
        elif new_rainfall == 0. and self.dom.swmm_rainfall > 0. : #stop using swmm rainfall if it is not raining
            self.dom.arr['rain'][:] = 0.
            self.dom.swmm_rainfall = 0.
            
        return self