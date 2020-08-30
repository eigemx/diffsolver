import json


class BoundaryError(Exception):
    pass


# Types of heat equation boundary conditions.
class BoundaryType:
    FIXED = 0
    INSULATED = 1
    EMPTY = 2


# If a face is located in boundary region, `Face` object will contain these data in `Face.boundary` field.
class BoundaryFace:
    def __init__(self):
        self.name = None
        self.nFaces = 0
        self.startFace = 0
        self.btype = None   # BoundaryType() instance
        self.T = None       # For BoundaryType.FIXED boundary condition


# Reads boundary dictionary JSON file
class BoundaryDict:
    boundaries = list()

    def __init__(self, b_dict_file):
        with open(b_dict_file, 'r') as fd:
            text = fd.read()
            json_data = json.loads(text)

        for boundary_name in json_data.keys():
            boundary = BoundaryFace()
            boundary.name = boundary_name
            boundary.nFaces = json_data[boundary_name]['nFaces']
            boundary.startFace = json_data[boundary_name]['startFace']

            btype = json_data[boundary_name]['type']
            if btype == 'fixed':
                boundary.T = json_data[boundary_name]['T']
                boundary.btype = BoundaryType.FIXED

            elif btype == 'insulated':
                boundary.btype = BoundaryType.INSULATED

            elif btype == 'empty':
                boundary.btype = BoundaryType.EMPTY

            else:
                raise BoundaryError(f'Unknown boundary type: {boundary_name} '
                                    f'in boundary dictionary file: {b_dict_file}.')

            self.boundaries.append(boundary)

    # Given a face id, if face is in boundary region this will return the corresponding `FaceBoundary()` object.
    # returns None if face is an internal face.
    def get_face_boundary(self, face_id):
        boundary = None

        for item in self.boundaries:
            if item.startFace <= face_id < item.startFace + item.nFaces:
                boundary = item

        return boundary
