from typing import Tuple
import math


import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
import drjit as dr

from fiber import Fiber

mi.set_variant('llvm_ad_rgb')


def point_outside_fiber_from_ray(fiber_radius: mi.Float, fiber_dir: mi.Vector3f, pos: mi.Point3f, dir: mi.Vector3f, epsilon: float = 0.0001) -> mi.Point3f:
    """
    Get a point on the boundary of the fiber in the defined ray
    """
    # distance_to_outside = radius * tan(dot(dir, fiber_dir)* 2pi)
    length: mi.Float = fiber_radius * dr.tan(dr.dot(fiber_dir, dir) * dr.pi) + epsilon
    point: mi.Point3f = pos + length * dir
    return point

def cartesian_to_polar(direction: mi.Vector3f) -> tuple[mi.Float, mi.Float]:
    theta = dr.atan2(direction[1], direction[0])
    phi = dr.acos(direction[2] / dr.norm(direction))
    return (theta, phi)

class TabulatedBCRDF():
    def __init__(self, files: list[str], lambda_min=400., lamda_max=700., theta_table_size=450, phi_table_size=880):
        # The size of the table. The default is the fiber_0 from the paper
        self.theta_range: int = theta_table_size
        self.phi_range: int = phi_table_size
        
        self.layers: int = len(files)
        self.lambda_min: float = lambda_min
        self.lambda_max: float = lamda_max
        self.files: list[str] = files

        # Load all the files into numpy arrays
        self.tables = []
        for filename in files:
            intensities = np.fromfile(filename, dtype="float32")
            intensities = intensities.reshape(self.theta_range, self.phi_range)
            self.tables.append(intensities)
        numpy_tables = np.stack(self.tables, axis=0)[:,:,:,None]

        self.tables_tensor = mi.TensorXf(numpy_tables)
        self.tables_texture = mi.Texture3f(self.tables_tensor)
        # self.tables_tensor = mi.TensorXf(mi.Float(0), shape=[self.layers, self.theta_range, self.phi_range, ])

    def interpolate_tables(self, wavelength: mi.Float, theta: mi.Float, phi: mi.Float, interpolator: str = "linear") -> mi.Float:
        # wavelength 0-amount of tables, theta 0-theta_range, phi 0-phi_range
        # phi 360°
        # theta 180°
        coord = mi.Vector3f(
            ((wavelength - self.lambda_min) / (self.lambda_max - self.lambda_min)),
            (theta / 180.),
            (phi / 360.)
            )
        
        return self.tables_texture.eval(coord)[0]


    def brdf(self, intersection: mi.SurfaceInteraction3f, active: mi.Mask, fiber_dir: mi.Vector3f, fiber_radius: mi.Float, sampler: mi.Sampler, wavelength: float) -> Tuple[mi.Vector3f, mi.Point3f, mi.Float]:
        #####################
        ####################
        # TODO
        WRONG_FIBER_RADIUS = mi.Float(1.)
        
        rand_1 = sampler.next_1d()
        rand_2 = sampler.next_1d()
        rand_3 = sampler.next_1d()
        rand_4 = sampler.next_1d()
        
        direction = mi.Vector3f(rand_1, rand_2, rand_3)

        out_theta, out_phi = cartesian_to_polar(direction)
        fiber_theta, fiber_phi = cartesian_to_polar(fiber_dir)

        # TODO: Is this correct?
        out_theta -= fiber_theta
        out_phi -= fiber_phi

        magnitude = self.interpolate_tables(wavelength, out_theta, out_phi, "linear")
        
        # position = intersection.shape.get_out_pos(intersection, 0.0001, direction)
        position = point_outside_fiber_from_ray(fiber_radius, fiber_dir, intersection.p, direction)

        # Shift the position along the side vector of the fiber
        side = dr.cross(direction, fiber_dir)
        shift_amt = rand_4 - 0.5
        position += shift_amt * fiber_radius * side

        return (position, direction, magnitude)
    


    def show_layers(self):
        # The base paper for this uses 24 layers of wavelengths
        fig, ax = plt.subplots(self.layers//4, 4)
        for (n, img) in enumerate(self.tables):
            ax[n//4, n%4].imshow(img)
        plt.show()


# test = TabulatedBCRDF(["fiber_0/fiber_0_lambda" + str(i) + "_TM_depth6.binary" for i in range(24)])
# test.show_layers()
# print(test.interpolate_tables(500, 90, 180))