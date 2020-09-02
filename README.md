# diffsolver
Finite volume diffusion equation solver for structured/unstructured modified-OpenFOAM meshes.

diffsolver uses a modified version of OpenFOAM mesh (as can be found in `Examples` folder), and a `boundary.JSON` file that describes boundary regions. The solver can work with orthogonal structured meshes using `solver.StructuredSolver()` and non-orthogonal unstructured meshes using `solver.UnstructuredSolver()`

Currently supported boundary types: fixed (Drichlet boundary condition) and insulated (zero-gradient Neumann boundary condition).

diffsolver is based on "Chapter 8 - Spatial discretization of diffusion term" in "The Finite Volume Method in Computational Fluid Dynamics" by Moukalled et al.

## Selected Solver Results

Unstructured non-orthogonal rectangular mesh with two fixed temperature boundary conditions and two insulated walls (left, right), mesh can be found in `Examples\usSmall`

![Image of rectangular mesh](https://github.com/EigenEmara/diffsolver/blob/master/Examples/recatgular_mesh.png)


Unstructured non-orthogonal disc mesh with two fixed temperature boundary conditions (inner and outer rings), mesh can be found in `Examples\DiscMesh`

![Image of rectangular mesh](https://github.com/EigenEmara/diffsolver/blob/master/Examples/disc_mesh.png)


## Code Example
    import boundary
    import solver
    import mesh
    import quality

    if __name__ == '__main__':
        # Directory containing modified-OpenFOAM mesh.
        MESH_DIR = 'Examples/DiscMesh/'
    
        # Boundary JSON dictionary.
        BOUNDARY_DICT = boundary.BoundaryDict(MESH_DIR + 'boundary.JSON')
    
        # Read mesh.
        mesh = mesh.Mesh(MESH_DIR, BOUNDARY_DICT)
      
        # Report mesh quality.
        q = quality.Quality(mesh)
        q.print_stats()
      
        # Run solver.
        s = solver.UnstructuredSolver(mesh)
        s.solve(it=2500, eps=0.00001)

        s.results_to_foam()
