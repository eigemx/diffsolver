import numpy as np


class Quality:
    def __init__(self, mesh):
        self.mesh = mesh
        self.thetas = list()        # ِ A list of angles of non-orthogonality
        self.vol = list()           # Cell volumes
        self.face_areas = list()    # Face areas
        self.face_aspect = list()   # A list of faces aspect ratio.
        self.cell_aspect = list()   # A list of cell aspect ratio.

        for cell in mesh.cells:
            # Cell volume
            self.vol.append(cell.volume)

            # Cell aspect ratio
            self.cell_aspect.append(self.calc_cell_aspect_ratio(cell))

            for face in cell.faces:
                # Face area
                self.face_areas.append(face.area)

                # Face aspect ratio
                self.face_aspect.append(self.calc_face_aspect_ratio(face))

                # Get adjacent cell of the current face.
                if face in cell.owner:
                    adj_cell_id = self.mesh.get_neighbor_cell_id(face.face_id)
                else:
                    adj_cell_id = self.mesh.get_owner_cell_id(face.face_id)

                if adj_cell_id is None:
                    continue

                adj_cell = self.mesh.cells[adj_cell_id]

                # Calculate non-orthogonality
                self.thetas.append(self.calc_nonortho(face, cell, adj_cell))

    # Calculate non-orthogonality
    @staticmethod
    def calc_nonortho(face, cell, adj_cell):
        centroid = cell.center
        adj_centroid = adj_cell.center

        # Line connecting current and adjacent cells centroids.
        ef = adj_centroid - centroid
        # Straddling face normal vector.
        sf = face.normal if face in cell.owner else -face.normal

        if np.dot(sf, ef) < 0:
            ef = -ef

        # Angle between ef and sf.
        costheta = np.dot(ef, sf) / (np.linalg.norm(ef) * np.linalg.norm(sf))
        theta = np.arccos(costheta) * (180 / np.pi)

        return theta

    # Calculate face aspect ratio
    # Ansys FLUENT defines aspect ratio as the ratio of the longest edge length to the shortest edge length.
    @staticmethod
    def calc_face_aspect_ratio(face):
        edges_length = list()
        for i in range(face.size):
            # Set the subface points
            p1 = face.vertices[i]
            p2 = face.vertices[(i + 1) % face.size]
            edges_length.append(np.linalg.norm(p2 - p1))

        return max(edges_length) / min(edges_length)

    # Calculate cell aspect ratio.
    # Cell aspect ration is the ratio between max. to min. face area of the cell bounding box.
    @staticmethod
    def calc_cell_aspect_ratio(cell):
        # Get min. and max. points of the cell
        x = np.array([np.inf, -np.inf])
        y = np.array([np.inf, -np.inf])
        z = np.array([np.inf, -np.inf])

        for face in cell.faces:
            for vertix in face.vertices:
                x[0] = np.min([x[0], vertix[0]])
                x[1] = np.max([x[1], vertix[0]])

                y[0] = np.min([y[0], vertix[1]])
                y[1] = np.max([y[1], vertix[1]])

                z[0] = np.min([z[0], vertix[2]])
                z[1] = np.max([z[1], vertix[2]])

        length = np.linalg.norm(np.array([x[1], 0, 0]) - np.array([x[0], 0, 0]))
        width = np.linalg.norm(np.array([y[1], 0, 0]) - np.array([y[0], 0, 0]))
        height = np.linalg.norm(np.array([z[1], 0, 0]) - np.array([z[0], 0, 0]))
        bbox_surface_areas = [length * width, length * height, width * height]
        return max(bbox_surface_areas) / min(bbox_surface_areas)

    def print_stats(self):
        print('Mesh Statistics:')
        print(f'\tPoints:   {len(self.mesh.points)}')
        print(f'\tFaces:    {len(self.mesh.faces)}')
        print(f'\tCells:    {len(self.mesh.cells)}')

        print('Mesh Quality:')
        print(f'\tFace aspect ratio:    max = {max(self.face_aspect):.4f} min = {min(self.face_aspect):.4f}')
        print(f'\tCell aspect ratio:    max = {max(self.cell_aspect):.4f} min = {min(self.cell_aspect):.4f}')
        print(f'\tFace area:            max = {max(self.face_areas):e} min = {min(self.face_areas):e}')
        print(f'\tCell volume:          '
              f'max = {max(self.vol):e} avg = {np.average(self.vol):e} min = {min(self.vol):e}')
        print(f'\tMesh non-orthogonality:   '
              f'max = {max(self.thetas):.2f}, avg = {np.average(self.thetas):.2f}, min = {min(self.thetas):.2f}')
