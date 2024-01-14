import time

import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
import drjit as dr
import scipy.stats.qmc as qmc

import fiber
from brdf import TabulatedBCRDF
from util import plot_results
from render import Renderer

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

for w in range(25):
    brdf = TabulatedBCRDF("./fiber_model", w)
    in_dir = mi.Vector3f(1.,0.,0.)

    in_pos = mi.Point3f(center_x, center_y, 0.) - dr.normalize(in_dir) * (radius * 1.1)
    print_in_dir = in_dir.numpy()
    bounces = 100000
    samples = 100
    renderer = Renderer(fibers, brdf, samples=samples, bounces=bounces, in_dir=in_dir, in_pos=in_pos,out_size_phi=200, out_size_theta=200)

    out_model: np.array = renderer.render_structure()
    print("A")
    plot_results(out_model, out_theta=200, out_phi=200)

    # write to the file with the name date_time_depth_samples_in_dir
    np.save("output_" + str(int(time.time())) + "_" + str(bounces) + "_" + str(samples) + "_" + str(print_in_dir[0,0]) + "_" + str(print_in_dir[0,1]) + "_" + str(print_in_dir[0,2]), out_model)

exit()
