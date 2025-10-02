import numpy as np


def to_global(current_pos, dt, acc, vel, alpha, rates, attitude,):

    phi, theta, psi = attitude

    rotations_BodyToRef = np.array(
            [
                [
                    np.cos(theta) * np.cos(psi),
                    np.cos(theta) * np.sin(psi),
                    -np.sin(theta),
                ],
                [
                    (-np.cos(phi) * np.sin(psi) + np.sin(phi) * np.sin(theta) * np.cos(psi)),
                    (np.cos(phi) * np.cos(psi) + np.sin(phi) * np.sin(theta) * np.sin(psi)),
                    np.sin(phi) * np.cos(theta),
                ],
                [
                    (np.sin(phi) * np.sin(psi) + np.cos(phi) * np.sin(theta) * np.cos(psi)),
                    (-np.sin(phi) * np.cos(psi) + np.cos(phi) * np.sin(theta) * np.sin(psi)),
                    np.cos(phi) * np.cos(theta),
                ],
            ]
        )
    

    vel_ref = rotations_BodyToRef @ vel   # shape (3,1)


    # accx_com_body, accy_com_body, accz_com_body = np.einsum("ijk,jk->ik", rotations_BodyToRef, acc_com_ref
    current_pos[0] += vel_ref[0] * dt
    current_pos[1] += vel_ref[1] * dt
    current_pos[2] += vel_ref[2] * dt
    

    return current_pos, attitude