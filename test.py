import numpy as np
import matplotlib.pyplot as plt

def test_intensity_gradiant_phi(theta_size, phi_size):
    phi = np.linspace(0.,1.,theta_size).reshape(theta_size,1)
    intensities = np.ones((theta_size, phi_size)) * phi
    return intensities

def test_intensity_gradiant_theta(theta_size, phi_size):
    theta = np.linspace(0.,1.,phi_size).reshape(1,phi_size)
    intensities = np.ones((theta_size, phi_size)) * theta
    return intensities

def test_intensity_grid(theta_size, phi_size):
    intensities = np.zeros((theta_size, phi_size))
    intensities[0:50,:] = 1
    intensities[100:150,:] = 1
    intensities[200:250,:] = 1
    intensities[300:350,:] = 1
    intensities[400:450,:] = 1

    intensities[:,0:50] = 1
    intensities[:,100:150] = 1
    intensities[:,200:250] = 1
    intensities[:,300:350] = 1
    intensities[:,400:450] = 1
    intensities[:,500:550] = 1
    intensities[:,600:650] = 1
    intensities[:,700:750] = 1
    intensities[:,800:850] = 1
    return intensities

def print_output_example(lambda_num):
    # Load the lambda_num table from each of the 18 theta_i folders and show it in a big subplot
    fig, axs = plt.subplots(3, 6, figsize=(12, 6))
    fig.tight_layout(pad=3.0)

    for theta_i in range(18):
        theta = theta_i * 10
        theta_i_table = np.load(f"./output/theta-{theta}-phi-0/lambda_{lambda_num}_intensities.npy")
        theta_i_table = theta_i_table.reshape(200,200)
        axs[theta_i // 6, theta_i % 6].imshow(theta_i_table)
        axs[theta_i // 6, theta_i % 6].set_title(f"theta_i={theta}")

    plt.show()