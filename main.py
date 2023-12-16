import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
import drjit as dr
import scipy.stats.qmc as qmc

import fiber
from brdf import TabulatedBCRDF
from util import plot_results
from render import Renderer

TEST = False

THETA_TABLE_SIZE=450
PHI_TABLE_SIZE=880

FIBER_RADIUS = 2.
BUNDLE_RADIUS = 100.

mi.set_variant('llvm_ad_rgb')

# Get the input rotation to make a difference
# Output offsets in fiber

# Poisson disk sampling in circle
poisson_engine = qmc.PoissonDisk(d=2, radius=(2.*FIBER_RADIUS)/BUNDLE_RADIUS)
samples = poisson_engine.random(100000)
samples -= 0.5
samples *= 2.*BUNDLE_RADIUS
samples = samples[np.sqrt(samples[:,0]*samples[:,0] + samples[:,1] * samples[:,1]) <= BUNDLE_RADIUS]

plt.scatter(samples[:,0], samples[:,1])
plt.show()

fibers = []
fibers.append(fiber.Fiber(0.,0.,2., [0.,0.,1.]))
fibers.append(fiber.Fiber(0.,5.,2., [0.,0.,1.]))
# for elem in samples:
    # fibers.append(fiber.Fiber(elem[0], elem[1], FIBER_RADIUS, [0.,0.,1.]))


radius, center_x, center_y = fiber.get_bounds(fibers)

# Check the interactions
if not TEST:
    brdf = TabulatedBCRDF(["fiber_0/fiber_0_lambda" + str(i) + "_TM_depth6.binary" for i in range(24)])

    # phi = np.linspace(0.,1.,THETA_TABLE_SIZE).reshape(THETA_TABLE_SIZE,1)
    # intensities = np.ones((THETA_TABLE_SIZE, PHI_TABLE_SIZE)) * phi
    # phi = np.linspace(0.,1.,PHI_TABLE_SIZE).reshape(1,PHI_TABLE_SIZE)
    # intensities = np.ones((THETA_TABLE_SIZE, PHI_TABLE_SIZE)) * phi
    # intensities = np.zeros((THETA_TABLE_SIZE, PHI_TABLE_SIZE))
    # intensities[0:50,:] = 1
    # intensities[100:150,:] = 1
    # intensities[200:250,:] = 1
    # intensities[300:350,:] = 1
    # intensities[400:450,:] = 1

    # intensities[:,0:50] = 1
    # intensities[:,100:150] = 1
    # intensities[:,200:250] = 1
    # intensities[:,300:350] = 1
    # intensities[:,400:450] = 1
    # intensities[:,500:550] = 1
    # intensities[:,600:650] = 1
    # intensities[:,700:750] = 1
    # intensities[:,800:850] = 1

    # brdf.set_test_intensities(intensities)
    # brdf.show_layers()

    in_dir = mi.Vector3f(1.,0.,0.)

    in_pos = mi.Point3f(center_x, center_y, 0.) - dr.normalize(in_dir) * (radius * 1.1)

    renderer = Renderer(fibers, brdf, samples=1000000, bounces=1000, in_dir=in_dir, in_pos=in_pos)

    out_model: np.array = renderer.render_structure()

    plot_results(out_model)

    out_model.tofile("test")

else:
    rend_scene = fiber.preview_render_dict_from_fibers(fibers)

    img = mi.render(mi.load_dict(rend_scene))
    plt.imshow(img)
    plt.show()
