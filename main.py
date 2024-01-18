import time
import os

import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
import drjit as dr
import scipy.stats.qmc as qmc

import fiber
from brdf import TabulatedBCRDF
from util import plot_results
from render import Renderer

def input_dir_from_theta_phi(theta: float, phi: float) -> mi.Vector3f:
    theta += np.pi / 2.
    return mi.Vector3f(np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))

def arc_to_degree(arc: float) -> float:
    return arc * 180. / np.pi

def degree_to_arc(degree: float) -> float:
    return degree * np.pi / 180.

TEST = True

THETA_TABLE_SIZE=450
PHI_TABLE_SIZE=880

FIBER_RADIUS = 20.
BUNDLE_RADIUS = 500.

mi.set_variant('llvm_ad_rgb')


fibers = []

# fibers = fiber.generate_single(2.)

fibers = fiber.generate_random(FIBER_RADIUS, BUNDLE_RADIUS, show_structure=True)

radius, center_x, center_y = fiber.get_bounds(fibers)

for theta in range(0, 180, 10):
    phi = 0.
    phi_deg = 0.
    theta_deg = degree_to_arc(theta)
    # Create the folder output/theta-{theta}-phi-{phi} if it doesn't exist yet
    if not os.path.exists("output/theta-" + str(int(theta)) + "-phi-" + str(int(phi))):
        os.makedirs("output/theta-" + str(int(theta)) + "-phi-" + str(int(phi)))
    
    for w in range(25):
        brdf = TabulatedBCRDF("./fiber_model", w)
        in_dir = input_dir_from_theta_phi(theta_deg,phi_deg)

        in_pos = mi.Point3f(center_x, center_y, 0.) - dr.normalize(in_dir) * (radius * 1.1)
        print_in_dir = in_dir.numpy()
        bounces = 100000
        samples = 1000000
        renderer = Renderer(fibers, brdf, samples=samples, bounces=bounces, in_dir=in_dir, in_pos=in_pos,out_size_phi=200, out_size_theta=200)

        out_model: np.array = renderer.render_structure()
        # plot_results(out_model, out_theta=200, out_phi=200)

        file_name = "lambda_" + str(w) + "_intensities"
        # write to the file with the name date_time_depth_samples_in_dir
        # np.save("output_" + str(int(time.time())) + "_" + str(bounces) + "_" + str(samples) + "_" + str(print_in_dir[0,0]) + "_" + str(print_in_dir[0,1]) + "_" + str(print_in_dir[0,2]), out_model)
        np.save(file_name, out_model)
        # Move the file into the correct folder
        os.rename(file_name + ".npy", "output/theta-" + str(int(theta)) + "-phi-" + str(int(phi)) + "/" + file_name + ".npy")


