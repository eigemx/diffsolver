import numpy as np

from boundary import *
from common import *
from quality import *


class MeshError(Exception):
    pass


# Construct a face from a set of vertices
class Face:
    def __init__(self, vertices):
        # a vertex is an np.ndarray of shape (3,)
        self.vertices = vertices
        self.size = len(vertices)
        self.face_id = 0

        # Is this a boundary face?
        self.is_boundary = False
        self.boundary = BoundaryFace()

        self.center = None
        self.area = None
        self.normal = None

    def center_and_area(self):
        # return members if already calculated.
        if self.center is not None:
            return self.center, self.area, self.normal

        # calculate the geometric centre of k points forming a polygonal face
        geo_center = np.sum(self.vertices, axis=0) / self.size

        face_total_area = 0.0
        face_normal_vector = np.zeros((3,))
        face_centroid = np.zeros((3,))

        # form the traingular subfaces.
        # each face is constructed using an edge as the base of the triangle and geometric center as its apex.
        for i in range(self.size):
            # Set the subface points
            p1 = self.vertices[i]
            p2 = self.vertices[(i + 1) % self.size]
            p3 = geo_center

            # calculate the subface geometric center
            subface_geo_center = np.sum([p1, p2, p3], axis=0) / 3.0

            # calculate the area and normal vector 'sf' for the subface
            sf = np.cross((p2 - p1), (p3 - p1))
            area = np.linalg.norm(sf) / 2.0

            face_normal_vector += sf
            face_total_area += area
            face_centroid = face_centroid + (area * subface_geo_center)

        face_centroid /= face_total_area
        face_normal_vector /= 2.0

        self.center, self.area, self.normal = face_centroid, face_total_area, face_normal_vector
        return face_centroid, face_total_area, face_normal_vector


# construct a volumne element/cell from a set of faces
class Cell:
    def __init__(self, faces_list=None):
        self.faces = faces_list
        self.cell_id = 0
        self.size = 0

        if self.faces:
            self.size = len(faces_list)

        self.owner = list()
        self.neighbour = list()

        self.center = None
        self.volume = None

    # after reading owner & neighbour faces, append them to self.faces
    def set_faces(self):
        if self.faces is not None:
            raise MeshError("Attempting to re-construct a cell that's already defined")

        if len(self.owner) + len(self.neighbour) == 0:
            raise MeshError(f'Cannot set faces for a cell from empty owner & neighbour lists: cell id: {self.cell_id}')

        self.faces = list()
        self.faces.extend(self.owner)
        self.faces.extend(self.neighbour)

    def center_and_volume(self):
        # return members if already calculated.
        if self.center is not None:
            return self.center, self.volume

        # calculate the geometric center of the element
        # which is simply the average of all the vertices defining the volume element.
        # OpenFOAM calculates geometric center as average of face centers.
        geo_centeroid = np.zeros((3,))

        for face in self.faces:
            geo_centeroid += face.center
        geo_centeroid /= len(self.faces)

        # construct a pyramid with each cell face as the base and geometric center as the apex.
        element_volume = 0.0
        element_centroid = np.zeros((3,))

        for face in self.faces:
            face_centroid, face_area, _ = face.center_and_area()

            # pyramid centroid is located on 0.25 the distance between the base and the apex.
            pyramid_centroid = (0.75 * face_centroid) + (0.25 * geo_centeroid)
            pyramid_volume = (1.0 / 3.0) * face_area * np.linalg.norm(face_centroid - geo_centeroid)

            element_volume += pyramid_volume
            # element_centroid will be divided by total volume after the end of faces loop
            element_centroid += (pyramid_volume * pyramid_centroid)

        # Finally, calculate the volume weighted element centre.
        element_centroid = element_centroid / element_volume

        self.center, self.volume = element_centroid, element_volume

        return element_centroid, element_volume


class Mesh:
    def __init__(self, mesh_dir, boundary_dict):
        self.points = None
        self.faces = None
        self.owner = None
        self.neighbour = None
        self.cells = None

        self.POINTS_DIR = mesh_dir + '\\points'
        self.FACES_DIR = mesh_dir + '\\faces'
        self.OWNER_DIR = mesh_dir + '\\owner'
        self.NEIGHBOUR_DIR = mesh_dir + '\\neighbour'

        self.boundary_dict = boundary_dict

        self.read_points()
        self.read_faces()
        self.read_owner_neighbour()

    # Read `points` file and return a list of 3D coordinates
    def read_points(self):
        points = list()

        info('Reading points...')

        with open(self.POINTS_DIR, 'r') as fd:
            lines = fd.readlines()
            for line in lines:
                p1, p2, p3 = line.split(' ')
                vertex = np.array([float(p1), float(p2), float(p3)])
                points.append(vertex)

        self.points = points

    # `faces` file contain a list of list of vertices labels
    # each entry of the file defines a face of 4 vertices
    # read_faces reads the file and returns a list of a list of vertices labels.
    def read_faces(self):
        info('Reading faces...')

        with open(self.FACES_DIR, 'r') as fd:
            lines = fd.readlines()
            faces = [None] * len(lines)
            i = 0
            for line in lines:
                vertices_idx = [int(i) for i in line.split(' ')]

                # In order to construct a face, we need an array of points.
                face = Face(np.array([self.points[i] for i in vertices_idx]))
                face.face_id = i

                # check if boundary face
                face_boundary = self.boundary_dict.get_face_boundary(face.face_id)

                if face_boundary is not None:
                    face.is_boundary = True
                    face.boundary = face_boundary

                face.center_and_area()
                faces[i] = face
                i += 1

        self.faces = faces

    def read_owner_neighbour(self):
        info('Reading `owner` file...')

        cells = dict()
        owner = list()
        neighbor = list()

        # read `owner` faces.
        with open(self.OWNER_DIR, 'r') as fd:
            lines = fd.readlines()
            i = 0  # marks the current line number we are in

            for line in lines:
                # The line string (example: 23) we just read is the number of the cell that owns the face
                # that face have an id of current line number.
                # Example: cell 23 (line string) owns faces 47 (line number)
                cell_number = int(line)
                owner.append(cell_number)

                if cell_number not in cells:
                    # cell does not exist yet.
                    cells[cell_number] = Cell()
                    cells[cell_number].cell_id = cell_number

                # append the face owned by the cell to its 'owner' list
                cells[cell_number].owner.append(self.faces[i])
                i += 1
        info('Reading `neighbour` file...')

        # read `neighbour` faces.
        with open(self.NEIGHBOUR_DIR, 'r') as fd:
            lines = fd.readlines()
            i = 0  # marks the current line we are in

            for line in lines:
                cell_number = int(line)
                neighbor.append(cell_number)

                # check if cell does not exist.
                if cell_number not in cells:
                    # cell does not exist yet.
                    cells[cell_number] = Cell()
                    cells[cell_number].cell_id = cell_number

                # append the face owned by the cell to its 'neighbour' list
                cells[cell_number].neighbour.append(self.faces[i])
                i += 1

        self.cells = [cells.get(k) for k in range(len(cells)) if k in cells]
        self.owner = owner
        self.neighbour = neighbor

        info('Constructing owner-neighbour relationships & calculating cells volumes and centroids...')

        for cell in self.cells:
            cell.set_faces()
            cell.center_and_volume()

    def get_owner_cell_id(self, face_id):
        return self.owner[face_id]

    def get_neighbor_cell_id(self, face_id):
        if face_id >= len(self.neighbour):
            return None
        return self.neighbour[face_id]

    def adjacent_cell(self, cell, face):
        if face in cell.owner:
            adj_cell_id = self.get_neighbor_cell_id(face.face_id)
        else:
            adj_cell_id = self.get_owner_cell_id(face.face_id)

        if adj_cell_id is None:
            return None, None

        return adj_cell_id, self.cells[adj_cell_id]

    @staticmethod
    def weight_factors(cell, adj_cell, face):
        d_Cf = np.linalg.norm(cell.center - face.center)
        d_fF = np.linalg.norm(face.center - adj_cell.center)
        gc = d_fF / (d_Cf + d_fF)
        gf = 1.0 - gc

        assert(gc + gf <= 1.0)
        return gc, gf
