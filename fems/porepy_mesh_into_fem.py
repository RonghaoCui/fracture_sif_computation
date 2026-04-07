from __future__ import annotations

import numpy as np

class FemMeshFromPorePy:

    def split_fracture_nodes(
            grid2d, 
            tip_face_centers, 
            frac_gap,
            theta_rad
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the nodes on the fracture surfaces into upper and lower parts. 
        The tip nodes are not included in both upper and lower parts. 

        This method is modified from def adjustmesh(g, tips, gap): in Dang et al. (2022)

        Parameters: 
        g : 2d grid dictionary
            g = gb.grids_of_dimension(2)[0]
        tips : np.array(shape = (n, 2), dtype = float)
            each row is coordiante of each tip.
        gap : float
            gap between fractures' faces.

        Returns: 
        nodes_coords : np.array(shape = (n, 2), dtype = float)
            each row is coordiante of each node.
        elements : np.array(shape = (m, 3), dtype = float)
            each row is index of each cell.
        upper_frac_edges: node indices on the upper fracture surface.
        lower_frac_edges: node indices on the lower fracture surface.
        """
        nodes_coords = grid2d.nodes[[0, 1], :].T.copy()
        elements = grid2d.cell_nodes().indices.reshape((3, grid2d.num_cells), order='F').T

        face_ind = np.reshape(grid2d.face_nodes.indices, (grid2d.dim, -1), order='F').T
        frac_faces = np.where(grid2d.tags['fracture_faces'])[0]

        t_vec = np.array([np.sin(theta_rad), np.cos(theta_rad)], dtype=float)
        t_vec = t_vec / np.linalg.norm(t_vec)
        n_vec = np.array([-np.cos(theta_rad), np.sin(theta_rad)], dtype=float)
        n_vec /= np.linalg.norm(n_vec)

        upper_frac_edges = []
        lower_frac_edges = []

        node_side = dict()

        for i in frac_faces:
            nodes = face_ind[i] 

            cells = np.intersect1d(
                np.where(elements == nodes[0])[0],
                np.where(elements == nodes[1])[0]
            )

            # There is only one cell that has both nodes of the fracture face as its vertices.
            cell = cells[0]

            third_node = np.setdiff1d(elements[cell], nodes)[0]

            A, B = nodes_coords[nodes]
            M = nodes_coords[third_node]

            AB = B - A
            AM = M - A
            proj = A + np.dot(AM, AB) / np.dot(AB, AB) * AB
            
            # Here we only consider the fracture orientation from 0 to 90 degree. 
            # For other orientations, the sign of the normal vector may need to be carefully determined.
            # According to the current definition of the normal vector, the upper face is on the left-hand 
            # side of the fracture and the lower face is on the right-hand side of the fracture.
            norm_on_frac_face = M - proj
            norm_on_frac_face /= np.linalg.norm(norm_on_frac_face)
            sign = np.sign(np.dot(norm_on_frac_face, n_vec))

            if sign > 0:
                upper_frac_edges.append(tuple(nodes))
                side = +1
            else:
                lower_frac_edges.append(tuple(nodes))
                side = -1

            for n in nodes:
                if n not in node_side:
                    node_side[n] = side

        for n, side in node_side.items():
            nodes_coords[n] = nodes_coords[n] + side * n_vec * frac_gap / 2

        upper_frac_edges = np.array(upper_frac_edges, dtype=int)
        lower_frac_edges = np.array(lower_frac_edges, dtype=int)

        return nodes_coords, elements, upper_frac_edges, lower_frac_edges



        