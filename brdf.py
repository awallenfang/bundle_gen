from typing import Tuple
import math
from glob import glob

import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
import drjit as dr

from util import mitsuba_cartesian_to_polar

mi.set_variant('llvm_ad_rgb')

class TabulatedBCRDF():
    def __init__(self, model_folder: str, lambda_num, lambda_min=400., lamda_max=700., theta_table_size=450, phi_table_size=880):
        # The size of the table. The default is the fiber_0 from the paper
        self.shape_theta: int = theta_table_size
        self.shape_phi: int = phi_table_size
        self.theta_file_amt = 13

        self.shape_wavelength: int = 25
        self.lambda_min: float = lambda_min
        self.lambda_max: float = lamda_max
        self.model_folder: str = model_folder
        self.shapes = (self.shape_wavelength, self.theta_file_amt, self.shape_theta, self.shape_phi)
        self.lambda_num = lambda_num

        # intensities = np.fromfile("/home/marvin/code/Python/bundle_gen_2/fiber_model/fiber_0_lambda1_TM_depth6.binary", dtype="float32")
        # intensities = intensities.reshape(self.shape_theta, self.shape_phi)
        # plt.imshow(intensities)
        # plt.show()
        self.tables = np.zeros((self.theta_file_amt, self.shape_theta, self.shape_phi))
        # Go through all the model data which is in the fiber_model folder and then in folders called theta-x-phi-y
        # Get all the folder names
        folders = glob(self.model_folder + "/*")
        print(folders)
        for folder in folders:
            # Get the theta and phi values from the folder names
            theta = float(folder.split("-")[1])
            phi = float(folder.split("-")[3])
            # Get the theta number for the array
            theta_num = int(theta / 15)

            phi_num = int(phi / 45)

            # Get all the files in the folder
            # print(folder + "/*")
            files = glob(folder + "/*")
            files = sorted(files)
            # print(files)
            for (i, filename) in enumerate(files):
                # Find the wavelength index
                wavelength = int(filename.split("lambda")[1].split("_")[0])
                if wavelength == self.lambda_num:
                    intensities = np.fromfile(filename, dtype="float32")
                    intensities = intensities.reshape(self.shape_theta, self.shape_phi)
                    intensities /= np.sum(intensities)
                    # print(f'Sum:{np.sum(intensities)}, mean:{np.mean(intensities)}')
                    # plt.imshow(intensities)
                    # plt.show()
                    # Add the intensities to the correct position in the array
                    self.tables[theta_num] = intensities
                # print(f'Added {filename} to {wavelength, theta_num, phi_num} with sum {np.sum(intensities)}')
                if np.sum(intensities) == 0.:
                    print(filename)
                    # plt.imshow(intensities)
                    # plt.show()
        

        self.tables_tensor = mi.TensorXf(self.tables[:,:,:,None])
        print(self.tables_tensor)
        # shape: (wavelength, theta_o, phi_o, 1)
        # TODO: shape: (theta_i, theta_o, phi_o, 1)
        self.tables_texture = mi.Texture3f(self.tables_tensor, filter_mode=dr.FilterMode.Linear)


    def interpolate_tables(self, in_theta: mi.Float , out_theta: mi.Float, out_phi: mi.Float) -> mi.Float:
        # theta_i 0-180°
        # phi 360°
        # theta 180°
        out_phi[out_phi<0] += dr.pi*2
        out_phi[out_phi>dr.pi*2] -= dr.pi*2
        coord = mi.Vector3f(
            (out_phi / (2*dr.pi)), 
            ((out_theta + dr.pi/2) / dr.pi),
            ((in_theta + dr.pi/2) / dr.pi),
            )

        return self.tables_texture.eval(coord)[0]
    
    def interpolate_tables_new(self, wavelength: mi.Float, out_theta: mi.Float, out_phi: mi.Float, in_theta: mi.Float) -> mi.Float:
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
        wavelength_coord = mi.UInt((wavelength - self.lambda_min) / (self.lambda_max - self.lambda_min) * self.shape_wavelength)
        theta_o_coord = mi.UInt((out_theta + dr.pi/2) / dr.pi * self.shape_theta)
        phi_o_coord = mi.UInt((out_phi / (2*dr.pi)) * self.shape_phi)
        theta_i_coord = mi.UInt((in_theta + dr.pi/2) / dr.pi * self.shape_theta)
        # phi_i_coord = dr.uint32_array_t((in_phi / (2*dr.pi)) * self.shape_phi)
        # wavelength_coord = ((wavelength - self.lambda_min) / (self.lambda_max - self.lambda_min))
        # theta_o_coord = ((out_theta + dr.pi/2) / dr.pi)
        # phi_o_coord = (out_phi / (2*dr.pi))
        # theta_i_coord = ((in_theta + dr.pi/2) / dr.pi)
        # phi_i_coord = (in_phi / (2*dr.pi))
        # #(wavelength, theta_o, phi_o, theta_i, phi_i, 1)
        # dr.printf_async("wavelength: %f, theta_o: %f, phi_o: %f, theta_i: %f\n", wavelength_coord, theta_o_coord, phi_o_coord, theta_i_coord)



        # TODO: Maybe something better than nearest neighbour interpolation

        
        # dr.printf_async("phi: %f -> coord_phi: %f\n", phi, coord.x)
        # dr.printf_async("coord: (%f,%f,%f)\n", coord.x, coord.y, coord.z)
        # return self.tables_texture.eval(coord)[0]
        # array = 
        # array = array.array
        # print(type(array))
        # return dr.gather(mi.Float, self.tables_tensor.array[wavelength_coord][theta_o_coord], phi_o_coord)
        # print(type(self.tables_tensor.array))
        pos = self.flat_index(wavelength_coord, theta_o_coord, phi_o_coord, theta_i_coord)
        return dr.gather(mi.Float, self.tables_tensor.array, pos)

    def flat_index(self, wavelength: mi.UInt, theta_o: mi.UInt, phi_o: mi.UInt, theta_i: mi.UInt) -> mi.UInt:
        index = dr.fma(
                dr.fma(dr.fma(theta_i, self.shape_wavelength,wavelength), self.shape_theta, theta_o),
            self.shape_phi, phi_o
        )
        return index
    
    def flat_index_3d(self, wavelength: mi.UInt, theta_o: mi.UInt, phi_o: mi.UInt, theta_i: mi.UInt) -> mi.UInt:
        index = dr.fma(
            dr.fma(wavelength, self.shape_theta, theta_o),
            self.shape_phi, phi_o
        )
        return index

    def brdf(self, intersection: mi.SurfaceInteraction3f, direction: mi.Vector3f, old_direction: mi.Vector3f, sampler: mi.Sampler, wavelength: float, additional_phi_rotation: mi.Float) -> Tuple[mi.Vector3f, mi.Point3f, mi.Float]:
        # Uniform sphere point

        rand_1 = sampler.next_1d()

        position = intersection.shape.get_out_pos(intersection, 0.01, direction, rand_1)

        out_theta, out_phi = mitsuba_cartesian_to_polar(direction)


        in_theta, _ = mitsuba_cartesian_to_polar(old_direction)

        magnitude = self.interpolate_tables(in_theta, out_theta, out_phi - additional_phi_rotation)

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
        fig, ax = plt.subplots(self.shape_wavelength//4, 4)
        for (n, img) in enumerate(self.tables):
            ax[n//4, n%4].imshow(img)
        plt.show()

    def show_single_layer(self, layer):
        fig, ax = plt.subplots(1)
        fig.set_figheight(10)
        fig.set_figwidth(10)
        ax.set_xlabel("phi")
        ax.set_ylabel("theta")

        # theta_labels = np.array([90, 45, 0, -45, -90])
        # theta_tick_positions = np.arange(0,self.shape_theta+1, (self.shape_theta+1) // 4)
        # ax.set_yticks(theta_tick_positions, theta_labels)

        # phi_labels = np.array([-180, -90, 0, 90, 180])
        # phi_tick_positions = np.arange(0,self.shape_phi+1, (self.shape_phi+1) // 4)
        # ax.set_xticks(phi_tick_positions, phi_labels)


        ax.imshow(self.tables[layer, :,::2])
        plt.show()

# brdf = TabulatedBCRDF("./fiber_model", 0)


# print(brdf.interpolate_tables(mi.Float(0.), mi.Float(0.), mi.Float(dr.pi/2.)))
# # print(brdf.interpolate_tables_new(mi.Float(600.), mi.Float(0.), mi.Float(0.), mi.Float(0.)))
