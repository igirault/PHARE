# Goal : Enhance flexibility of magnetic field initialization

Currently user can specify B0, and if nothing is provided it must default to zero. The user can speciefy total initial magnetic field B, and if nothing is provided total magnetic field must default to zero. Then B1 is internally initialized as the B - B0.

I want to enable the user to directly specify B1. As B, B0, and B1 are related by the equation B = B0 + B1, multiple initialization scenarios are possible that must be handled correctly.

# Expected behavior:

The user can specify B0. If B0 is not specified, it defaults to zero. Then:
   - The user can specify B. Then B1 is inizialized as B - B0.
   - The user can specify B1. Then B is initialized as B0 + B1.
   - The user cannot specify both B and B1. If they do, an error is raised.
   - The can specify neither B nor B1. Then B1 defaults to zero, so B is initialized as B0.

# Implementation requirements

- the code should ionly modify initialization related files.
- the different initialization scenarios should be tested to ensure correct behavior.
