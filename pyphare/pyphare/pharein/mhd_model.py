from . import global_vars


class MHDModel(object):
    def defaulter(self, input, value):
        if input is not None:
            import inspect

            params = list(inspect.signature(input).parameters.values())
            assert len(params)
            param_per_dim = len(params) == self.dim
            has_vargs = params[0].kind == inspect.Parameter.VAR_POSITIONAL
            assert param_per_dim or has_vargs
            return input
        if self.dim == 1:
            return lambda x: value + x * 0
        if self.dim == 2:
            return lambda x, y: value
        if self.dim == 3:
            return lambda x, y, z: value

    def has_user_input(self, *components):
        return any(component is not None for component in components)

    def combine_components(self, left, right, operation):
        return lambda *xyz: operation(left(*xyz), right(*xyz))

    def __init__(
        self,
        density=None,
        vx=None,
        vy=None,
        vz=None,
        bx=None,
        by=None,
        bz=None,
        b1x=None,
        b1y=None,
        b1z=None,
        p=None,
        b0x=None,
        b0y=None,
        b0z=None,
        a0z=None,
        a1z=None,
    ):
        if global_vars.sim is None:
            raise RuntimeError("A simulation must be declared before a model")

        if global_vars.sim.model is not None:
            raise RuntimeError("A model is already created")

        self.dim = global_vars.sim.ndim

        # --- vector-potential init (2D only) -------------------------------------
        # B = curl(A_z z_hat): Bx = dA_z/dy, By = -dA_z/dx, Bz = 0. Computed on the C++
        # side with the discrete curl so that div B = 0 to machine precision. a0z drives
        # B0, a1z drives B1; either, both, or neither may be given (independent modes).
        b0_from_potential = self.has_user_input(a0z)
        b1_from_potential = self.has_user_input(a1z)
        if (b0_from_potential or b1_from_potential) and self.dim != 2:
            raise ValueError(
                "MHDModel vector-potential init (a0z/a1z) is only supported in 2D"
            )
        if b0_from_potential and self.has_user_input(b0x, b0y):
            raise ValueError(
                "MHDModel: a0z (B0 from vector potential) is exclusive with b0x/b0y"
            )

        has_total_magnetic = self.has_user_input(bx, by, bz)
        has_perturbation_magnetic = self.has_user_input(b1x, b1y, b1z)
        if has_total_magnetic and has_perturbation_magnetic:
            raise ValueError("MHDModel accepts either total magnetic field B or perturbation B1, not both")
        if b1_from_potential and (has_total_magnetic or has_perturbation_magnetic):
            raise ValueError(
                "MHDModel: a1z (B1 from vector potential) is exclusive with bx/by and b1x/b1y/b1z"
            )

        density = self.defaulter(density, 1.0)
        vx = self.defaulter(vx, 0.0)
        vy = self.defaulter(vy, 0.0)
        vz = self.defaulter(vz, 0.0)
        p = self.defaulter(p, 1.0)
        b0x = self.defaulter(b0x, 0.0)
        b0y = self.defaulter(b0y, 0.0)
        b0z = self.defaulter(b0z, 0.0)
        a0z = self.defaulter(a0z, 0.0)
        a1z = self.defaulter(a1z, 0.0)

        if has_total_magnetic:
            bx = self.defaulter(bx, 0.0)
            by = self.defaulter(by, 0.0)
            bz = self.defaulter(bz, 0.0)
            b1x = self.combine_components(bx, b0x, lambda total, external: total - external)
            b1y = self.combine_components(by, b0y, lambda total, external: total - external)
            b1z = self.combine_components(bz, b0z, lambda total, external: total - external)
        else:
            b1x = self.defaulter(b1x, 0.0)
            b1y = self.defaulter(b1y, 0.0)
            b1z = self.defaulter(b1z, 0.0)
            bx = self.combine_components(b0x, b1x, lambda external, perturbation: external + perturbation)
            by = self.combine_components(b0y, b1y, lambda external, perturbation: external + perturbation)
            bz = self.combine_components(b0z, b1z, lambda external, perturbation: external + perturbation)

        self.model_dict = {}

        self.model_dict.update(
            {
                "density": density,
                "vx": vx,
                "vy": vy,
                "vz": vz,
                "bx": bx,
                "by": by,
                "bz": bz,
                "b1x": b1x,
                "b1y": b1y,
                "b1z": b1z,
                "p": p,
                "b0x": b0x,
                "b0y": b0y,
                "b0z": b0z,
                "a0z": a0z,
                "a1z": a1z,
                "b0_init_mode": "potential" if b0_from_potential else "components",
                "b1_init_mode": "potential" if b1_from_potential else "components",
            }
        )

        global_vars.sim.set_model(self)
