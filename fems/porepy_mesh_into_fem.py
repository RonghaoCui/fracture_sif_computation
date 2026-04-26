from __future__ import annotations

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from fems import mmfa

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

        edge_to_cells = {}
        for cell_id, tri in enumerate(elements): 
            a, b, c = tri
            for key in (min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(a, c), max(a, c)):
                edge_to_cells.setdefault(key, []).append(cell_id)

        upper_frac_edges = []
        lower_frac_edges = []
        node_side = dict()
        for i in frac_faces:
            nodes = face_ind[i] 
            n0, n1 = nodes
            # cells = np.intersect1d(
            #     np.where(elements == nodes[0])[0],
            #     np.where(elements == nodes[1])[0]
            # )
            cells = edge_to_cells[(min(n0, n1), max(n0, n1))]
            
            # Note: there is only one cell that has both nodes of the fracture face as its vertices.
            cell = cells[0]

            # third_node = np.setdiff1d(elements[cell], nodes)[0]
            tri = elements[cell]

            # if tri[0] != n0 and tri[0] != n1:
            #     third_node = tri[0]
            # elif tri[1] != n0 and tri[1] != n1:
            #     third_node = tri[1]
            # else:
            #     third_node = tri[2]
            third_node = tri[(tri != n0) & (tri != n1)][0]

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

        # for n, side in node_side.items():
        #     nodes_coords[n] = nodes_coords[n] + side * n_vec * frac_gap / 2

        upper_frac_edges = np.array(upper_frac_edges, dtype=int)
        lower_frac_edges = np.array(lower_frac_edges, dtype=int)

        return nodes_coords, elements, upper_frac_edges, lower_frac_edges

    def stiffness(
            p6: np.ndarray, 
            t6: np.ndarray, 
            material: np.ndarray, 
            qpe: list,
    ) -> sps.csr_matrix:
        """
        Assemble the global stiffness matrix for a 2D quadratic finite element mesh
        with optional quarter-point elements (QPE) for fracture tips. 

        The standard element stiffness matrix is computed as: 

        Ke = ∫_Ωe B^T D B dΩ,

        where B is the strain-displacement matrix, D is the material matrix, and
        Ωe is the current element domain. Numerical quadrature is used for the
        integration.

        Assembly is performed by sparse matrix construction using coordinate format (COO). 

        Parameters: 
            p6: coordinates of the T6 mesh nodes. 
            t6: node connection of the T6 mesh elements. Each row corresponds 
                to an element and contains the indices of its 6 nodes.
            material: the constitutive material matrix. 
            qpe: a list of quarter-point element (QPE) element indices (cell indices) at different tips. 
        
        Returns: 
            csr_matrix: the assembled global stiffness matrix. 
        """
        # Collect all QPE element indices from all tips. 
        CQPE = []
        for qi in qpe:
            if len(qi) > 0:
                CQPE.extend(qi)
        CQPE = set(np.unique(CQPE).tolist())

        poi3, wei3 = mmfa.gausspoint(7, 'T3')
        poi4, wei4 = mmfa.gausspoint(7, 'Q4')
        ne = t6.shape[0]        # The number of elements. 
        sdof = p6.shape[0] * 2  # Total DOFs in the system. 
        edof = 12               # DOFs per element for T6 elements. 

        # Sparse assembly triplets
        rows = []
        cols = []
        data = []
        for e in range(ne):
            if len(material) == 3:
                D = material
            else:
                D = material[e]

            # Use a local copy to avoid modifying input t6 in-place
            tri = t6[e, :].copy()
            X = p6[tri, 0]
            Y = p6[tri, 1]

            # Orientation check for node ordering in the element. 
            if (X[1] - X[0]) * (Y[-1] - Y[0]) - (X[-1] - X[0]) * (Y[1] - Y[0]) < 0:
                indinv = tri[::-1]
                tri[1:] = indinv[:-1]
                X = p6[tri, 0]
                Y = p6[tri, 1]
            
            # The index mapping from local element DOFs to global DOFs.
            index = np.array([
                tri[0] * 2, tri[0] * 2 + 1,
                tri[1] * 2, tri[1] * 2 + 1,
                tri[2] * 2, tri[2] * 2 + 1,
                tri[3] * 2, tri[3] * 2 + 1,
                tri[4] * 2, tri[4] * 2 + 1,
                tri[5] * 2, tri[5] * 2 + 1,
            ], dtype=np.int32)

            Ke = np.zeros((edof, edof), dtype=float)
            # For elements that are not quarter-point elements, use the standard T6 formulation.
            if e not in CQPE:
                xis = poi3[:, 0]
                eta = poi3[:, 1]
                N, dNdx, dNdy, detJac = mmfa.T6element(X, Y, xis, eta)

                for i in range(poi3.shape[0]):
                    B = np.zeros((3, edof), dtype=float)
                    B[0, 0::2] = dNdx[i, :]
                    B[1, 1::2] = dNdy[i, :]
                    B[2, 0::2] = dNdy[i, :]
                    B[2, 1::2] = dNdx[i, :]

                    Ke += (B.T @ D @ B) * detJac[i] * wei3[i]
            # For quarter-point elements, use the special formulation for Q8 elements with the quarter-point nodes.
            else:
                xis = poi4[:, 0]
                eta = poi4[:, 1]
                N, dNdx, dNdy, detJac = mmfa.Q8element(X, Y, xis, eta)

                for i in range(poi4.shape[0]):
                    dx0 = dNdx[i, 0] + dNdx[i, 6] + dNdx[i, 7]
                    dy0 = dNdy[i, 0] + dNdy[i, 6] + dNdy[i, 7]

                    B = np.zeros((3, edof), dtype=float)
                    B[0, 0::2] = [dx0, dNdx[i, 1], dNdx[i, 2], dNdx[i, 3], dNdx[i, 4], dNdx[i, 5]]
                    B[1, 1::2] = [dy0, dNdy[i, 1], dNdy[i, 2], dNdy[i, 3], dNdy[i, 4], dNdy[i, 5]]
                    B[2, 0::2] = [dy0, dNdy[i, 1], dNdy[i, 2], dNdy[i, 3], dNdy[i, 4], dNdy[i, 5]]
                    B[2, 1::2] = [dx0, dNdx[i, 1], dNdx[i, 2], dNdx[i, 3], dNdx[i, 4], dNdx[i, 5]]

                    Ke += (B.T @ D @ B) * detJac[i] * wei4[i]

            # Add local Ke into sparse triplets
            ii, jj = np.meshgrid(index, index, indexing="ij")
            rows.extend(ii.ravel())
            cols.extend(jj.ravel())
            data.extend(Ke.ravel())

        # Build sparse matrix
        K = sps.coo_matrix((data, (rows, cols)), shape=(sdof, sdof))
        K.sum_duplicates()
        return K.tocsr()

    def linearsolverI(
            K: sps.spmatrix, 
            F: np.ndarray, 
            udof:np.ndarray, 
            uval:np.ndarray, 
            sdof:int
    ) -> np.ndarray:
        """
        Solve the linear system KU=F with specified Dirichlet boundary conditions. 
        This method only processes the stiffness matrix K to include the Dirichlet 
        BCs before solving.

        Notes: 
            The matrix K in this method during solving is not symmetric anymore. 
        """
        # Dense RHS vector: (n,)
        F = F.reshape(-1)

        # Apply Dirichlet BCs
        K[udof, :] = 0.0
        K[udof, udof] = 1.0
        F[udof] = uval

        # Convert to CSR for solving
        K = K.tocsr()

        # Solve
        U = spla.spsolve(K, F)

        return U[:sdof]
    
    def linearsolverII(
            K: sps.spmatrix, 
            F: np.ndarray, 
            udof:np.ndarray, 
            uval:np.ndarray, 
            sdof:int
    ) -> np.ndarray:
        """
        Solve the linear system KU=F with specified Dirichlet boundary conditions. 
        This method processes the stiffness matrix K and the RHS vector F to 
        include the Dirichlet BCs before solving.

        Notes: 
            The matrix K in this method during solving is symmetric. 
        """
        # Dense RHS vector: (n,)
        F = F.reshape(-1)

        # Convert to CSC (compressed sparse column) for efficient column slicing
        K = K.tocsc() 

        # RHS correction: move known Dirichlet contribution to RHS
        F = F - K[:, udof] @ uval
         
        # Modify matrix
        K = K.tolil()  # Convert to LIL (list of lists) for efficient row modification
        K[udof, :] = 0.0
        K[:, udof] = 0.0
        K[udof, udof] = 1.0
        F[udof] = uval

        # Convert to CSR (compressed sparse row) for solving
        K = K.tocsr()
        # Solve
        # K has to be sparse, and F is dense (recommended). 
        # The shape of U is (n,).
        U = spla.spsolve(K, F)
        
        return U[:sdof]

    def loadsegment(p6, NeuDof=None, NeuVal=None):
        """
        Assemble consistent nodal loads on quadratic boundary edges.

        This version supports both standard T6 edges and quarter-point edges,
        because the edge Jacobian is computed from the actual coordinates of
        the three edge nodes.
        """
        sdof = p6.shape[0] * 2
        F = np.zeros((sdof, 1))

        if NeuDof is None:
            return F

        # Three-point Gauss rule on [-1, 1].
        # Two-point is exact for straight quarter-point edges, but three-point is safer.
        gauss_xi = np.array([
            -np.sqrt(3.0 / 5.0),
            0.0,
            np.sqrt(3.0 / 5.0),
        ])
        gauss_w = np.array([
            5.0 / 9.0,
            8.0 / 9.0,
            5.0 / 9.0,
        ])

        for e in range(NeuDof.shape[0]):
            index = NeuDof[e, :].astype(int)   # [n0, nmid, n1]
            coords = p6[index, :2]             # shape = (3, 2)
            traction = NeuVal[e, :]            # [tx, ty]

            fe = np.zeros((3, 2))

            for xi, w in zip(gauss_xi, gauss_w):
                # Quadratic edge shape functions:
                # node order: [-1, 0, +1] -> [n0, nmid, n1]
                N = np.array([
                    0.5 * xi * (xi - 1.0),
                    1.0 - xi**2,
                    0.5 * xi * (xi + 1.0),
                ])

                dN_dxi = np.array([
                    xi - 0.5,
                    -2.0 * xi,
                    xi + 0.5,
                ])

                dx_dxi = dN_dxi @ coords
                J = np.linalg.norm(dx_dxi)

                fe += np.outer(N, traction) * J * w

            F[index * 2, 0] += fe[:, 0]
            F[index * 2 + 1, 0] += fe[:, 1]

        return F

