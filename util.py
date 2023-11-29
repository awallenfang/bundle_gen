import mitsuba as mi
import drjit as dr
import numpy as np
import matplotlib.pyplot as plt

mi.set_variant('llvm_ad_rgb')

def mitsuba_cartesian_to_polar(direction: mi.Vector3f) -> tuple[mi.Float, mi.Float]:
    """
    Cartesian to fiber polar coordinates  conversion with mitsuba Vectors

    Returns (theta, phi)
    """
    phi = dr.atan2(direction[1], direction[0])
    phi[phi < 0] += dr.pi * 2
    theta = dr.asin(direction[2] / dr.norm(direction))

    return (theta, phi)

def py_cartesian_to_polar(x,y,z):
    """
    Convert the vector from cartesian to fiber polar coordinates as defined by Xia's paper

    Returns (theta, phi)
    """
    phi = np.arctan2(y,x)
    phi += (phi<0)*np.pi*2
    theta = np.arcsin(z / np.sqrt(x*x + y*y + z*z))

    return (theta, phi)

def plot_results(out_model:np.array, out_theta = 100, out_phi = 100):
    """
    Plot the results of the renderer. This requires the shape of the resulting model
    """
    # out_model = np.zeros((out_theta+1, out_phi+1))
    out_model = out_model.reshape(out_theta,out_phi)
    # phis = np.arctan2(directions[:,1], directions[:,0])
    # thetas = np.arcsin(directions[:,2] / np.linalg.norm(directions, axis=1))

    # # phis[phis<0] += np.pi*2
    # # phi_coords = np.floor((phis / (2*np.pi)) * OUT_PHI)
    # # phi_coords = phi_coords.astype(int)
    # # print(np.max(phi_coords))

    # # print(f'phi:0 -> {np.floor((0 / (2*np.pi)) * OUT_PHI)}')
    # # print(f'theta:0 -> {np.floor((0 / np.pi) * OUT_THETA)}')

    # # theta_coords = np.floor((thetas / np.pi) * OUT_THETA)
    # # theta_coords = theta_coords.astype(int)
    # # # print(np.max(theta_coords))
    # # print(np.max(magnitudes))
    # # print(thetas)
    # # TODO: drjit.scatter_reduce
    # for i in range(0, magnitudes.shape[0]):
    #     phi, theta = (phis[i], thetas[i])
    #     pos_x = int(np.floor(((phi + 2*np.pi * (phi < 0)) / (np.pi*2)) * out_phi))
    #     pos_y = int(np.floor(((theta + (np.pi/2)) / (np.pi)) * out_theta))
   
    #     mag = magnitudes[i]
        

    #     out_model[pos_y, pos_x] += mag / (np.abs(np.cos(theta)) + 0.000001)


    fig, ax = plt.subplots(1)
    ax.set_xlabel("phi")
    ax.set_ylabel("theta")

    theta_labels = np.array([90, 45, 0, -45, -90])
    theta_tick_positions = np.arange(0,out_theta+1, (out_theta+1) // 4)
    ax.set_yticks(theta_tick_positions, theta_labels)

    phi_labels = np.array([-180, -90, 0, 90, 180])
    phi_tick_positions = np.arange(0,out_phi+1, (out_phi+1) // 4)
    ax.set_xticks(phi_tick_positions, phi_labels)


    ax.imshow(out_model)
    plt.show()