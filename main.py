import boundary
import solver
import mesh
import quality

if __name__ == '__main__':
    MESH_DIR = 'DiscMesh/'
    BOUNDARY_DICT = boundary.BoundaryDict(MESH_DIR + 'boundary.JSON')

    mesh = mesh.Mesh(MESH_DIR, BOUNDARY_DICT)
    q = quality.Quality(mesh)
    q.print_stats()

    s = solver.UnstructuredSolver(mesh)
    s.solve(it=250, eps=0.00001)

    s.results_to_foam()
