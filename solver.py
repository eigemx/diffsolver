import numpy as np

import boundary
import mesh

from common import info


class StructuredSolver:
    def __init__(self, _mesh):
        self.mesh = _mesh
        self.cells_count = len(self.mesh.cells)

        # Temperature of each cell, located at the centroid.
        self.T = np.ones((self.cells_count, 1)) * 500.0

        # The discretized diffusion equation is in the form: a_c*T_c + Sigma(a_F*T_F) = b
        # The following variables will hold a_c and b coefficients for the discretized equation applied to each cell.
        self.a_c = 0.0
        self.afTf = 0.0
        self.b = 0.0

    # Gauss-Seidel solver.
    def solve(self, it=100, eps=0.001):
        for i in range(it):
            # Copy the temperature vector before each iteration to do the following:
            #   1) Calculate the average temperature residuals.
            #   2) Check the stoping criteria using np.allclose()
            T_old = self.T.copy()

            self.visit_cells()

            info(f'Iteration no.: {i}, Average temperature residuals = {np.average(np.abs(T_old - self.T)):e}')

            # Stopping criteria, if `eps` is acheived.
            if np.allclose(self.T, T_old, eps):
                return

    # Loops over every cell in the mesh and initializes discretized equation coeffecients
    def visit_cells(self):
        for cell in self.mesh.cells:
            self.a_c = 0.0
            self.b = 0.0
            self.afTf = 0.0

            # Check if face is an interior or boundary face.
            for face in cell.faces:
                if face.is_boundary:
                    self.handle_boundary_face(cell, face)
                else:
                    self.handle_interior_face(cell, face)

            self.apply_discretize_eqn(cell.cell_id)

    # Applies the discretized equation to each cell and updates cell temperature value.
    def apply_discretize_eqn(self, cell_id):
        self.T[cell_id] = (self.afTf + self.b) / self.a_c

    def handle_interior_face(self, cell: mesh.Cell, face: mesh.Face):
        # Get the adjacent cell sharing the face
        adj_cell_id, adj_cell = self.mesh.adjacent_cell(cell, face)

        sf_mag = face.area
        cf_vec = cell.center - adj_cell.center  # Vector joining current cell and adjacent cell centroids.
        d_cf = np.linalg.norm(cf_vec)

        gDiff = sf_mag / d_cf

        self.afTf += gDiff * self.T[adj_cell_id]
        self.a_c += gDiff

    def handle_boundary_face(self, cell, face):
        if face.boundary.btype == boundary.BoundaryType.EMPTY:
            # if grid is two-dimensional, front and back faces are ignored.
            return

        elif face.boundary.btype == boundary.BoundaryType.INSULATED:
            # Neumann boundary condition
            # Walls are insulated, no flux going out through the faces. Nothing to do here
            return

        elif face.boundary.btype == boundary.BoundaryType.FIXED:
            self.apply_fixed_bc(cell, face)
            return

        else:
            raise boundary.BoundaryError(f"handle_boundary_face() invalid boundary type. "
                                         f"Boundary name: f{face.boundary.name}")

    def apply_fixed_bc(self, cell, face):
        # Drichlet boundary condition
        # Calculate distance between cell centroid and face centroid
        face_center, sf_mag = face.center, face.area

        cf_vec = cell.center - face_center
        d_cf = np.linalg.norm(cf_vec)
        gDiff = sf_mag / d_cf

        self.b += gDiff * face.boundary.T
        self.a_c += gDiff


class UnstructuredSolver(StructuredSolver):
    def __init__(self, _mesh):
        super().__init__(_mesh)
        self.gradients = np.zeros((self.cells_count, 3))

    def handle_interior_face(self, cell: mesh.Cell, face: mesh.Face):
        # ** Orthogonal-like contribution.
        # Get the adjacent cell sharing the face
        adj_cell_id, adj_cell = self.mesh.adjacent_cell(cell, face)

        Sf, Ef, Ef_mag, _, e_mag = self.over_relaxed_correction(cell, face, adj_cell)

        gDiff = Ef_mag / e_mag

        self.afTf += gDiff * self.T[adj_cell_id]
        self.a_c += gDiff

        # ** Non-orthogonal contribution.
        # Compute gradient at face center f
        # first, compute gradient of temperature at cell center
        grad_t_C = self.grad_t(cell)

        # compute gradient of temperature at adjacent cell center
        grad_t_F = self.grad_t(adj_cell)

        # interpolation weight factors
        gc, gf = mesh.Mesh.weight_factors(cell, adj_cell, face)
        grad_t_f = (gc*grad_t_C) + (gf*grad_t_F)

        self.b += np.dot(grad_t_f, Sf - Ef)

    def apply_fixed_bc(self, cell, face):
        # Drichlet boundary condition
        Sf, Ef, Ef_mag, e, e_mag = self.over_relaxed_correction(cell, face)
        gDiff = Ef_mag / e_mag
        e_hat = e / e_mag

        grad_t_face = ((face.boundary.T - self.T[cell.cell_id]) / e_mag) * e_hat

        self.b += (gDiff * face.boundary.T) + np.dot(grad_t_face, Sf - Ef)
        self.a_c += gDiff

    # Computes gradient of temperature at cell centroid, this method is used when the gradient term
    # cannot be linearized due to non-orthogonality, such as the case when computing cross diffusion term.
    def grad_t(self, cell: mesh.Cell):
        grad = np.zeros((3,))
        cell_volume = cell.volume

        for face in cell.faces:
            Sf = face.normal if face in cell.owner else -face.normal
            if face.is_boundary:
                # Boundary face
                if face.boundary.btype == boundary.BoundaryType.EMPTY:
                    continue
                if face.boundary.btype == boundary.BoundaryType.FIXED:
                    grad += face.boundary.T * Sf
                    continue
                if face.boundary.btype == boundary.BoundaryType.INSULATED:
                    grad += self.T[cell.cell_id] * Sf
                    continue
            else:
                # Interior face
                # calculate the temperature of interior face by interpolating adjacent cells temperatures.
                # get adjacent cell sharing the face
                adj_cell_id, adj_cell = self.mesh.adjacent_cell(cell, face)
                gc, gf = self.mesh.weight_factors(cell, adj_cell, face)
                face_temp = (gc * self.T[cell.cell_id]) + (gf * self.T[adj_cell_id])

                grad += face_temp * Sf

        grad = grad / cell_volume
        self.gradients[cell.cell_id] = grad
        return grad

    # applies over-relaxed correction due to grid non-orthogonality
    @staticmethod
    def over_relaxed_correction(cell: mesh.Cell, face: mesh.Face, adj_cell: mesh.Cell=None):
        Sf = face.normal if face in cell.owner else -face.normal

        if adj_cell is None:
            e = face.center - cell.center
        else:
            e = adj_cell.center - cell.center

        # make sure that e (vector joining cells centroids or cell and boundary face centroids) isn't in the opposite
        # direction to Sf (face normal).
        if np.dot(Sf, e) < 0:
            e = -e

        e_mag = np.linalg.norm(e)
        e_hat = e / e_mag

        Ef = (np.dot(Sf, Sf) / np.dot(e_hat, Sf)) * e_hat
        Ef_mag = np.linalg.norm(Ef)

        return Sf, Ef, Ef_mag, e, e_mag

    # writes temperature distribution to OpenFOAM format file.
    def results_to_foam(self):
        with open('T', 'w') as fd:
            fd.write('dimensions      0 0 0 1 0 0 0;\n')
            fd.write('internalField   nonuniform List<scalar>\n')
            fd.write(f'{self.cells_count}\n')
            fd.write('(\n')
            for T in self.T:
                fd.write(f'{str(T[0])}\n')
            fd.write(')\n')
