from typing import Tuple
import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt

from fiber import Fiber

mi.set_variant('llvm_ad_rgb')


def brdf(intersection: mi.SurfaceInteraction3f, rand: mi.Float, active: mi.Mask) -> Tuple[mi.Vector3f, mi.Point3f, mi.Float]:
    pass



def point_outside_fiber_from_ray(fiber: Fiber, pos: mi.Point3f, dir: mi.Vector3f) -> mi.Point3f:
    """
    Get a point on the boundary of the fiber in the defined ray
    """
    pass


class TabulatedBCRDF():
    def __init__(self, files: list[str], lambda_min=400, lamda_max=700):
        self.theta_range = 450
        self.phi_range = 880
        
        self.layers = len(files)
        self.lambda_min = lambda_min
        self.lambda_max = lamda_max
        self.files = files

        # Load all the files into numpy arrays
        self.tables = []
        for filename in files:
            intensities = np.fromfile(filename, dtype="float32")
            intensities = intensities.reshape(self.theta_range, self.phi_range)
            self.tables.append(intensities)

        
        
    def interpolate_tables(wavelength: mi.Float, theta: mi.Float, phi: mi.Float, interpolator: str = "linear"):
        if interpolator == "linear":
            
            pass

    def brdf(self, intersection: mi.SurfaceInteraction3f, rand: mi.Float, active: mi.Mask) -> Tuple[mi.Vector3f, mi.Point3f, mi.Float]:
        pass

    def show_layers(self):
        # The base paper for this uses 24 layers of wavelengths
        fig, ax = plt.subplots(self.layers//4, 4)
        for (n, img) in enumerate(self.tables):
            ax[n//4, n%4].imshow(img)
        plt.show()


test = TabulatedBCRDF(["fiber_0/fiber_0_lambda" + str(i) + "_TM_depth6.binary" for i in range(24)])
test.show_layers()