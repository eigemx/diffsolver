# diffsolver
Finite volume diffusion equation solver for structured/unstructured modified-OpenFOAM meshes.

diffsolver uses a modified version of OpenFOAM mesh (as can be found in 'test_mesh' folder), and a `boundary.JSON` file that describes boundary regions. The solver can work with orthogonal structured meshes using `solver.StructuredSolver()` and non-orthogonal unstructured meshes using `solver.UnstructuredSolver()`

Currently supported boundary types: fixed (Drichlet boundary condition) and insulated (zero-gradient Neumann boundary condition).

diffsolver is based on "Chapter 8 - Spatial discretization of diffusion term" in "The Finite Volume Method in Computational Fluid Dynamics" by Moukalled et al.

## Example
    import boundary
    import solver
    import mesh
    import quality

    if __name__ == '__main__':
        # Directory containing modified-OpenFOAM mesh.
        MESH_DIR = 'DiscMesh/'
    
        # Boundary JSON dictionary.
        BOUNDARY_DICT = boundary.BoundaryDict(MESH_DIR + 'boundary.JSON')
    
        # Read mesh.
        mesh = mesh.Mesh(MESH_DIR, BOUNDARY_DICT)
      
        # Report mesh quality.
        q = quality.Quality(mesh)
        q.print_stats()
      
        # Run solver.
        s = solver.UnstructuredSolver(mesh)
        s.solve(it=250, eps=0.00001)

        s.results_to_foam()
