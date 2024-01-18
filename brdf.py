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
    def __init__(self, model_folder: str, lambda_num, theta_table_size=450, phi_table_size=880):
        # The size of the table. The default is the fiber_0 from the paper
        self.shape_theta: int = theta_table_size
        self.shape_phi: int = phi_table_size
        self.theta_file_amt = 13

        self.model_folder: str = model_folder
        self.lambda_num = lambda_num

        self.tables = np.zeros((self.theta_file_amt, self.shape_theta, self.shape_phi))
        
        folders = glob(self.model_folder + "/*")

        for folder in folders:
            # Get the theta and phi values from the folder names
            theta = float(folder.split("-")[1])
            _phi = float(folder.split("-")[3])
            # Get the theta number for the array
            theta_num = int(theta / 15)


            files = glob(folder + "/*")

            for filename in files:
                # Find the wavelength index
                wavelength = int(filename.split("lambda")[1].split("_")[0])
                if wavelength == self.lambda_num:
                    intensities = np.fromfile(filename, dtype="float32")
                    intensities = intensities.reshape(self.shape_theta, self.shape_phi)
                    intensities /= np.sum(intensities)
                    

                    self.tables[theta_num] = intensities

        
        # plt.imshow(self.tables[10])
        # plt.show()
        self.tables_tensor = mi.TensorXf(self.tables[:,:,:,None])
        # shape: (theta_i, theta_o, phi_o, 1)

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
    
    def brdf(self, intersection: mi.SurfaceInteraction3f, new_direction: mi.Vector3f, old_direction: mi.Vector3f, sampler: mi.Sampler, additional_phi_rotation: mi.Float) -> Tuple[mi.Vector3f, mi.Float]:
        # Uniform sphere point

        rand_1 = sampler.next_1d()

        position = intersection.shape.get_out_pos(intersection, 0.01, new_direction, rand_1)

        out_theta, out_phi = mitsuba_cartesian_to_polar(new_direction)


        in_theta, _ = mitsuba_cartesian_to_polar(old_direction)

        magnitude = self.interpolate_tables(in_theta, out_theta, out_phi - additional_phi_rotation)

        return (position,  magnitude)
    
    def set_test_intensities(self, intensities):
        for i in range(len(self.tables)):
            self.tables[i] = intensities
        numpy_tables = np.stack(self.tables, axis=0)[:,:,:,None]

        self.tables_tensor = mi.TensorXf(numpy_tables)
        # shape: (theta_i, theta_o, phi_o, 1)
        self.tables_texture = mi.Texture3f(self.tables_tensor)


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
