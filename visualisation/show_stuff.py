from matplotlib import pyplot as plt
import numpy as np
import test
import colour


def wavelength_to_rgb(wavelength, gamma=0.8):

    '''This converts a given wavelength of light to an 
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).
    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''

    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    R *= 255
    G *= 255
    B *= 255
    return (int(R) / 255., int(G) / 255., int(B) / 255.)

def array_to_rgb(w, array):
    if w > 780:
        w = 780

    if w < 360:
        w = 360
    # print(array.shape)
    rgb_array = np.zeros((array.shape[0], array.shape[1], 3))
    # print(rgb_array.shape)
    xyz = colour.wavelength_to_XYZ(w)
    rgb = colour.XYZ_to_RGB(xyz, colourspace=colour.models.RGB_COLOURSPACE_sRGB)

    # Normalize the array
    array = array / np.max(array)
    # print(rgb)
    # print( np.array([rgb[0], rgb[1], rgb[2]]) * array[100,100])
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            rgb_array[i,j] = np.array([rgb[0], rgb[1], rgb[2]]) * array[i,j]
    return rgb_array

def print_output_example_single_wavelength(lambda_num):
    # Load the lambda_num table from each of the 18 theta_i folders and show it in a big subplot
    fig, axs = plt.subplots(3, 6, figsize=(12, 6))
    fig.tight_layout(pad=3.0)

    for theta_i in range(13):
        # theta = theta_i * 10
        # theta_i_table = np.load(f"./output/theta-{theta}-phi-0/lambda_{lambda_num}_intensities.npy")
        # theta_i_table = theta_i_table.reshape(200,200)
        theta = theta_i * 15
        theta_i_table = np.load(f"./fiber_model/theta-{theta}-phi-0/fiber_0_lambda{lambda_num}_TM_depth6.binary")
        theta_i_table = theta_i_table.reshape(450,880)
        axs[theta_i // 6, theta_i % 6].imshow(array_to_rgb(400. + (400./24.) * lambda_num,theta_i_table))
        axs[theta_i // 6, theta_i % 6].set_title(f"theta_i={theta}")

    plt.show()

def get_output_colour_single_wavelength_single_theta(lambda_num, theta_i):
    theta = theta_i * 30
    theta_i_table = np.load(f"./elliptic_output/theta-{theta}-phi-0/lambda_{lambda_num}_intensities.npy")
    # theta = theta_i * 15
    # theta_i_table = np.fromfile(f"./fiber_model/theta-{theta}-phi-0/fiber_0_lambda{lambda_num}_TM_depth6.binary", dtype="float32")
    # theta_i_table = theta_i_table.reshape(450,880)
    return array_to_rgb(400. + (300./25.) * lambda_num,theta_i_table)

def get_output_colour_single_wavelength_single_theta_ell(lambda_num, theta_i):
    theta = theta_i * 30
    theta_i_table = np.load(f"./elliptic_output/theta-{theta}-phi-90/lambda_{lambda_num}_intensities.npy")
    # theta = theta_i * 15
    # theta_i_table = np.fromfile(f"./fiber_model/theta-{theta}-phi-0/fiber_0_lambda{lambda_num}_TM_depth6.binary", dtype="float32")
    # theta_i_table = theta_i_table.reshape(450,880)
    return array_to_rgb(400. + (300./25.) * lambda_num,theta_i_table)


def get_combined_color_single_theta(theta_i):
    combined_color = np.zeros((200,200,3))
    # combined_color = np.zeros((450,880,3))
    for lambda_num in range(25):
        combined_color += get_output_colour_single_wavelength_single_theta(lambda_num, theta_i) / 25.
    return combined_color
def get_combined_color_single_theta_ell(theta_i):
    combined_color = np.zeros((200,200,3))
    # combined_color = np.zeros((450,880,3))
    for lambda_num in range(25):
        combined_color += get_output_colour_single_wavelength_single_theta_ell(lambda_num, theta_i) / 25.
    return combined_color


# fig, ax = plt.subplots(1)
# ax.set_xlabel(r"$\varphi_o$")
# ax.set_ylabel(r"$\theta_o$")

# theta_labels = np.array([90, 45, 0, -45, -90])
# # theta_tick_positions = np.arange(0,450+1, (450+1) // 4)
# theta_tick_positions = np.arange(0,200+1, (200+1) // 4)
# ax.set_yticks(theta_tick_positions, theta_labels)

# phi_labels = np.array([-180, -90, 0, 90, 180])
# # phi_tick_positions = np.arange(0,440+1, (440+1) // 4)
# phi_tick_positions = np.arange(0,200+1, (200+1) // 4)
# ax.set_xticks(phi_tick_positions, phi_labels)


# ax.imshow((get_combined_color_single_theta(6) * 3.)[:,::2])
# ax.imshow(np.roll(get_combined_color_single_theta(2) * 3., 50, axis=1))
# ax.imshow(get_combined_color_single_theta(2) * 3.)
# plt.savefig('filename.png', dpi=500)
# plt.show()

# fig, axs = plt.subplots(3, 6, figsize=(12, 6))
# fig.tight_layout(pad=3.0)

# for theta_i in range(4):
#     print(theta_i)
#     axs[theta_i // 6, theta_i % 6].imshow(get_combined_color_single_theta(theta_i) * 3.)
#     axs[theta_i // 6, theta_i % 6].set_title(f"theta_i={theta_i}")

# plt.show()


fig, ax = plt.subplots(1,2, figsize=(12, 6))
ax[0].set_xlabel(r"$\varphi_o$")
ax[0].set_ylabel(r"$\theta_o$")
ax[1].set_xlabel(r"$\varphi_o$")
ax[1].set_ylabel(r"$\theta_o$")

theta_labels = np.array([90, 45, 0, -45, -90])
# theta_tick_positions = np.arange(0,450+1, (450+1) // 4)
theta_tick_positions = np.arange(0,200+1, (200+1) // 4)
ax[0].set_yticks(theta_tick_positions, theta_labels)
ax[1].set_yticks(theta_tick_positions, theta_labels)

phi_labels = np.array([-180, -90, 0, 90, 180])
# phi_tick_positions = np.arange(0,440+1, (440+1) // 4)
phi_tick_positions = np.arange(0,200+1, (200+1) // 4)
ax[0].set_xticks(phi_tick_positions, phi_labels)
ax[1].set_xticks(phi_tick_positions, phi_labels)

ax[0].imshow(get_combined_color_single_theta(2) * 3.)
ax[1].imshow(np.roll(get_combined_color_single_theta_ell(2) * 3., 50, axis=1))

plt.savefig('filename.png', dpi=500)


plt.show()

# chances = np.load("output/interaction_chances.npy")
# plt.imshow(chances[:,0,:])
# plt.show()