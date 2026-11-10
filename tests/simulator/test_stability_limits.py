import unittest

from pyphare.pharein.stability_limits import (
    diffusive_limit,
    hyper_resistive_limit,
    hyperbolic_limit,
    whistler_limit,
)

INTEGRATORS = ("Euler", "TVDRK2", "TVDRK3", "SSPRK4_5")

HYPERBOLIC = {
    "Constant": (1.00, 1.00, 1.25, 2.19),
    "Linear": (0.5, 1.00, 1.17, 1.73),
    "WENO3": (None, 0.873, 1.62, 2.10),
    "WENOZ": (None, 1.00, 1.43, 1.98),
    "MP5": (None, 1.00, 1.43, 1.98),
}


class StabilityLimits(unittest.TestCase):
    def test_hyperbolic(self):
        for reconstruction, values in HYPERBOLIC.items():
            for integrator, expected in zip(INTEGRATORS, values):
                with self.subTest(reconstruction=reconstruction, integrator=integrator):
                    self.assertEqual(
                        hyperbolic_limit(reconstruction, integrator), expected
                    )

    def test_whistler(self):
        for integrator, expected in zip(INTEGRATORS, (None, None, 1.73, 3.27)):
            with self.subTest(integrator=integrator):
                self.assertEqual(whistler_limit(integrator), expected)

    def test_diffusive(self):
        for integrator, expected in zip(INTEGRATORS, (2.00, 2.00, 2.51, 5.33)):
            with self.subTest(integrator=integrator):
                self.assertEqual(diffusive_limit(integrator), expected)

    def test_hyper_resistive(self):
        for integrator, expected in zip(INTEGRATORS, (2.00, 2.00, 2.51, 5.33)):
            with self.subTest(integrator=integrator):
                self.assertEqual(hyper_resistive_limit(integrator), expected)

    def test_unknown_names_raise(self):
        with self.assertRaises(ValueError):
            hyperbolic_limit("Quadratic", "TVDRK3")
        with self.assertRaises(ValueError):
            hyperbolic_limit("Linear", "RK4")
        with self.assertRaises(ValueError):
            whistler_limit("RK4")
        with self.assertRaises(ValueError):
            diffusive_limit("RK4")
        with self.assertRaises(ValueError):
            hyper_resistive_limit("RK4")


if __name__ == "__main__":
    unittest.main()
