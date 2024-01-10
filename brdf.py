from typing import Tuple
import math

import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
import drjit as dr

from util import mitsuba_cartesian_to_polar

mi.set_variant('llvm_ad_rgb')

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
            intensities /= np.sum(intensities)
            print(f'Sum:{np.sum(intensities)}, mean:{np.mean(intensities)}')
            # test phi
            # phi = np.linspace(0.,1.,phi_table_size).reshape(1,self.phi_range)
            # intensities = np.ones(intensities.shape) * phi
            # test theta
            # theta = np.linspace(0.,1.,theta_table_size).reshape(self.theta_range, 1)
            # intensities = np.ones(intensities.shape) * theta
            # intensities = np.zeros_like(intensities)
            # intensities[0,0] = 1

            # intensities = np.ones(intensities.shape)
            self.tables.append(intensities)
        self.show_single_layer(16)
        numpy_tables = np.stack(self.tables, axis=0)[:,:,:,None]
        print(f'Total Sum: {np.sum(numpy_tables)}')

        self.tables_tensor = mi.TensorXf(numpy_tables)
        print(self.tables_tensor)
        # shape: (wavelength, theta_o, phi_o, 1)
        # TODO: shape: (wavelength, theta_o, phi_o, theta_i, phi_i, 1)
        self.tables_texture = mi.Texture3f(self.tables_tensor, filter_mode=dr.FilterMode.Nearest)
        mi.Texture

    def interpolate_tables(self, wavelength: mi.Float, out_theta: mi.Float, out_phi: mi.Float) -> mi.Float:
        # wavelength 0-amount of tables, theta 0-theta_range, phi 0-phi_range
        # phi 360°
        # theta 180°
        out_phi[out_phi<0] += dr.pi*2
        out_phi[out_phi>dr.pi*2] -= dr.pi*2
        coord = mi.Vector3f(
            (out_phi / (2*dr.pi)), 
            ((out_theta + dr.pi/2) / dr.pi),
            ((wavelength - self.lambda_min) / (self.lambda_max - self.lambda_min)),
            )

        
        # dr.printf_async("phi: %f -> coord_phi: %f\n", phi, coord.x)
        # dr.printf_async("coord: (%f,%f,%f)\n", coord.x, coord.y, coord.z)
        return self.tables_texture.eval(coord)[0]
    
    def interpolate_tables_new(self, wavelength: mi.Float, out_theta: mi.Float, out_phi: mi.Float, in_theta: mi.Float, in_phi: mi.Float) -> mi.Float:
        # wavelength 0-amount of tables, theta 0-theta_range, phi 0-phi_range
        # phi 360°
        # theta 180°
        out_phi[out_phi<0] += dr.pi*2
        out_phi[out_phi>dr.pi*2] -= dr.pi*2
        # coord = mi.Vector3f(
        #     (out_phi / (2*dr.pi)), 
        #     ((out_theta + dr.pi/2) / dr.pi),
        #     ((wavelength - self.lambda_min) / (self.lambda_max - self.lambda_min)),
        #     )

        # Get the nearest neighbour position in the tensor
        wavelength_coord = mi.UInt((wavelength - self.lambda_min) / (self.lambda_max - self.lambda_min) * self.layers)
        theta_o_coord = mi.UInt((out_theta + dr.pi/2) / dr.pi * self.theta_range)
        phi_o_coord = mi.UInt((out_phi / (2*dr.pi)) * self.phi_range)
        theta_i_coord = dr.uint32_array_t((in_theta + dr.pi/2) / dr.pi * self.theta_range)
        phi_i_coord = dr.uint32_array_t((in_phi / (2*dr.pi)) * self.phi_range)
        # wavelength_coord = ((wavelength - self.lambda_min) / (self.lambda_max - self.lambda_min))
        # theta_o_coord = ((out_theta + dr.pi/2) / dr.pi)
        # phi_o_coord = (out_phi / (2*dr.pi))
        # theta_i_coord = ((in_theta + dr.pi/2) / dr.pi)
        # phi_i_coord = (in_phi / (2*dr.pi))
        # #(wavelength, theta_o, phi_o, theta_i, phi_i, 1)
        print(wavelength_coord, theta_o_coord, phi_o_coord, theta_i_coord, phi_i_coord)

        # TODO: Maybe something better than nearest neighbour interpolation

        
        # dr.printf_async("phi: %f -> coord_phi: %f\n", phi, coord.x)
        # dr.printf_async("coord: (%f,%f,%f)\n", coord.x, coord.y, coord.z)
        # return self.tables_texture.eval(coord)[0]
        # array = 
        # array = array.array
        # print(type(array))
        # return dr.gather(mi.Float, self.tables_tensor.array[wavelength_coord][theta_o_coord], phi_o_coord)
        print(type(self.tables_tensor.array))
        return self.tables_tensor[wavelength_coord][theta_o_coord][phi_o_coord].array


    def brdf(self, intersection: mi.SurfaceInteraction3f, direction: mi.Vector3f, sampler: mi.Sampler, wavelength: float, additional_phi_rotation: mi.Float) -> Tuple[mi.Vector3f, mi.Point3f, mi.Float]:
        # Uniform sphere point

        rand_1 = sampler.next_1d()
        rand_2 = sampler.next_1d()

        # position is already returned in world coordinates
        position = intersection.shape.get_out_pos(intersection, 0.01, direction)

        # TODO: Find the correct method to find the local direction in world space for rotated fibers
        out_theta, out_phi = mitsuba_cartesian_to_polar(direction)
        # dr.printf_async("Ori: (%f,%f,%f)\n", position.x, position.y, position.z)
        # dr.printf_async("Local Dir: (%f,%f,%f)\n", local_direction.x, local_direction.y, local_direction.z)
        magnitude = self.interpolate_tables_new(wavelength, out_theta, out_phi - additional_phi_rotation, mi.Float(0.), mi.Float(0.))
        # dr.printf_async("Mag: %f \n", magnitude)
        # Shift the position along the side vector of the fiber
        # side = dr.cross(direction, fiber_dir)
        # shift_amt = rand_3 - 0.5
        # position += shift_amt * fiber_radius * side
        # direction = mi.Vector3f(1,0,0.)
        return (position, direction, magnitude)
    
    def set_test_intensities(self, intensities):
        for i in range(len(self.tables)):
            self.tables[i] = intensities
        numpy_tables = np.stack(self.tables, axis=0)[:,:,:,None]

        self.tables_tensor = mi.TensorXf(numpy_tables)
        # shape: (wavelength, theta, phi, 1)
        self.tables_texture = mi.Texture3f(self.tables_tensor)

    def show_layers(self):
        # The base paper for this uses 24 layers of wavelengths
        fig, ax = plt.subplots(self.layers//4, 4)
        for (n, img) in enumerate(self.tables):
            ax[n//4, n%4].imshow(img)
        plt.show()

    def show_single_layer(self, layer):
        fig, ax = plt.subplots(1)
        fig.set_figheight(10)
        fig.set_figwidth(10)
        ax.set_xlabel("phi")
        ax.set_ylabel("theta")

        theta_labels = np.array([90, 45, 0, -45, -90])
        theta_tick_positions = np.arange(0,self.theta_range+1, (self.theta_range+1) // 4)
        ax.set_yticks(theta_tick_positions, theta_labels)

        phi_labels = np.array([-180, -90, 0, 90, 180])
        phi_tick_positions = np.arange(0,self.phi_range+1, (self.phi_range+1) // 4)
        ax.set_xticks(phi_tick_positions, phi_labels)


        ax.imshow(self.tables[layer][:,::2])
        plt.show()

brdf = TabulatedBCRDF(["fiber_0/fiber_0_lambda" + str(i) + "_TM_depth6.binary" for i in range(24)])
print(brdf.interpolate_tables(mi.Float(600.), mi.Float(0.), mi.Float(0.)))
print(brdf.interpolate_tables_new(mi.Float(600.), mi.Float(0.), mi.Float(0.), mi.Float(0.), mi.Float(0.)))
