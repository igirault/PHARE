You need to test mhd cases with the following combinations of parameters:

- the Riemann solvers: Rusanov, HLL, HLLD
- the pair of timesteppers and spatial reconstructions: Euler/Constant, TVDRK2/Linear (with VanLeer limiter), TVDRK3/WENO3, SSPRK4_5/WENOZ, SSPRK4_5/MP5

For the limiter, it should always be none except for the TVDRK2/Linear case.

For the trio of parameter Hall, Resisitivity, and hyper resistivity, you can always keep True, False, True.

The tests should be modified, but you also need to generate the necessary combinations of parameters in all.txt, and re-configure the build and compile the code. Please mind that we will add 3d cases later, so please generate combinations starting wih 3 in all.txt.
