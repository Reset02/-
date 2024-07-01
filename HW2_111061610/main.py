###
### This homework is modified from CS231.
###


import sys
import numpy as np
import os
from scipy.optimize import least_squares
import math
from copy import deepcopy
from skimage.io import imread
from sfm_utils import *

'''
ESTIMATE_INITIAL_RT from the Essential Matrix, we can compute 4 initial
guesses of the relative RT between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
Returns:
    RT: A 4x3x4 tensor in which the 3x4 matrix RT[i,:,:] is one of the
        four possible transformations
'''
def estimate_initial_RT(E):
    # TODO: Implement this method!
    U, D, VT = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    Q_1 = U.dot(W.dot(VT))
    Q_2 = U.dot(W.T.dot(VT))
    
    if np.linalg.det(Q_1) < 0:
        Q_1 = - Q_1

    if np.linalg.det(Q_2) < 0:
        Q_2 = - Q_2
    
    # Create the two possible translation vectors (T1 and T2)
    T1 = U[:, 2] # Third column of U
    T2 = -U[:, 2] # Negative of the third column of U
    
    # Create the four possible transformation matrices
    RT = np.array([
        np.vstack([Q_1.T, T1]).T,
        np.vstack([Q_1.T, T2]).T,
        np.vstack([Q_2.T, T1]).T,
        np.vstack([Q_2.T, T2]).T
    ])

    return RT

'''
LINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point is the best linear estimate
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def linear_estimate_3d_point(image_points, camera_matrices):
    # TODO: Implement this method!
    M = len(image_points)

    if len(camera_matrices) != M:
        raise ValueError("Number of camera matrices must match the number of image points.")

    # Create an empty matrix to store the equations
    A = np.zeros((2 * M, 4))

    for i in range(M):
        # Extract the image point coordinates (u, v) and the camera matrix (P)
        u, v = image_points[i]
        P = camera_matrices[i]

        # Fill the rows of the matrix A
        A[2 * i] = u * P[2] - P[0]
        A[2 * i + 1] = v * P[2] - P[1]

    # Solve the linear system using SVD
    _, _, V = np.linalg.svd(A)

    # The 3D point is the right singular vector corresponding to the smallest singular value
    estimated_3d_point_homogeneous = V[-1]

    # Normalize the homogeneous coordinates (set the last element to 1)
    estimated_3d_point_homogeneous /= estimated_3d_point_homogeneous[3]

    # Extract the non-homogeneous 3D coordinates
    estimated_3d_point = estimated_3d_point_homogeneous[:3]

    return estimated_3d_point


'''
REPROJECTION_ERROR given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    error - the 2M reprojection error vector
'''
def reprojection_error(point_3d, image_points, camera_matrices):
    # TODO: Implement this method!
    M = len(image_points) # Determine the number of image points (M)
    P = np.hstack([point_3d, 1]) # P = [X, Y, Z, 1] is the 3D location of a point
    Mi = camera_matrices[:]
    # compute y = Mi*P
    # Mi is projection matrix
    y = np.matmul(Mi, P)

    # Normalize the projected points
    y = y.T
    projected_image_coordinate = y / y[-1, :]

    # Calculate the reprojection error
    reprojection_error = projected_image_coordinate[:-1, :].T - image_points

    # Convert to a 2Mx1 vector
    reprojection_error = reprojection_error.reshape(2 * M, )

    return reprojection_error

'''
JACOBIAN given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    jacobian - the 2Mx3 Jacobian matrix
'''
def jacobian(point_3d, camera_matrices):
    # TODO: Implement this method!
    # Initialize the Jacobian matrix with zeros
    num_cameras = camera_matrices.shape[0]
    jacobian = np.zeros((2 * num_cameras, 3))

    for i in range(num_cameras):
        # Extract the current camera matrix
        M = camera_matrices[i]

        # Homogeneous 3D point coordinates
        P = np.hstack([point_3d, 1])

        # Compute the denominator for the common factor
        denominator = (np.dot(M[2], P)) ** 2

        # Compute the elements of the Jacobian for the x and y components
        dx = np.array([
            M[0, 0] * np.dot(M[2, 1:], P[1:4]) - M[2, 0] * np.dot(M[0, 1:], P[1:4]),
            M[0, 1] * np.dot(M[2, [0, 2, 3]], P[[0, 2, 3]]) - M[2, 1] * np.dot(M[0, [0, 2, 3]], P[[0, 2, 3]]),
            M[0, 2] * np.dot(M[2, [0, 1, 3]], P[[0, 1, 3]]) - M[2, 2] * np.dot(M[0, [0, 1, 3]], P[[0, 1, 3]])
        ])

        dy = np.array([
            M[1, 0] * np.dot(M[2, 1:], P[1:4]) - M[2, 0] * np.dot(M[1, 1:], P[1:4]),
            M[1, 1] * np.dot(M[2, [0, 2, 3]], P[[0, 2, 3]]) - M[2, 1] * np.dot(M[1, [0, 2, 3]], P[[0, 2, 3]]),
            M[1, 2] * np.dot(M[2, [0, 1, 3]], P[[0, 1, 3]]) - M[2, 2] * np.dot(M[1, [0, 1, 3]], P[[0, 1, 3]])
        ])

        # Divide by the common denominator
        dx /= denominator
        dy /= denominator

        # Assign the computed values to the Jacobian matrix
        jacobian[2 * i] = dx
        jacobian[2 * i + 1] = dy

    return jacobian

'''
NONLINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point that iteratively updates the points
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def nonlinear_estimate_3d_point(image_points, camera_matrices):
    # TODO: Implement this method!
    pi = image_points.copy()
    Mi = camera_matrices.copy()

    max_iterations = 10
    tolerance = 1e-6

    estimated_3d_point = linear_estimate_3d_point(pi, Mi)
    prev_reprojection_error = float('inf')

    iteration = 0
    while iteration < max_iterations:
        J = jacobian(estimated_3d_point, Mi)
        reprojection_error_ = reprojection_error(estimated_3d_point, pi, Mi)
        estimated_3d_point -= np.linalg.lstsq(J, reprojection_error_, rcond=None)[0]

        current_reprojection_error = np.sum(reprojection_error_ ** 2)
        if abs(current_reprojection_error - prev_reprojection_error) < tolerance:
            break

        prev_reprojection_error = current_reprojection_error
        iteration += 1

    return estimated_3d_point

'''
ESTIMATE_RT_FROM_E from the Essential Matrix, we can compute  the relative RT 
between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
    image_points - N measured points in each of the M images (NxMx2 matrix)
    K - the intrinsic camera matrix
Returns:
    RT: The 3x4 matrix which gives the rotation and translation between the 
        two cameras
'''
def estimate_RT_from_E(E, image_points, K):
    # TODO: Implement this method!
    estimate_initial_RTs = estimate_initial_RT(E)

    M1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

    best_RT = None
    max_in_front_count = -1

    for RT in estimate_initial_RTs:
        M2 = K @ RT

        in_front_count = 0
        for i in range(image_points.shape[0]):
            X = linear_estimate_3d_point(image_points[i], np.array([M1, M2]))

            if X[2] > 0:
                in_front_count += 1

        if in_front_count > max_in_front_count:
            max_in_front_count = in_front_count
            best_RT = RT

    return best_RT

if __name__ == '__main__':
    run_pipeline = True

    # Load the data
    image_data_dir = 'data/statue/'
    unit_test_camera_matrix = np.load('data/unit_test_camera_matrix.npy')
    unit_test_image_matches = np.load('data/unit_test_image_matches.npy')
    image_paths = [os.path.join(image_data_dir, 'images', x) for x in
        sorted(os.listdir('data/statue/images')) if '.jpg' in x]
    focal_length = 719.5459
    matches_subset = np.load(os.path.join(image_data_dir,
        'matches_subset.npy'), allow_pickle=True, encoding='latin1')[0,:]
    dense_matches = np.load(os.path.join(image_data_dir, 'dense_matches.npy'), 
                               allow_pickle=True, encoding='latin1')
    fundamental_matrices = np.load(os.path.join(image_data_dir,
        'fundamental_matrices.npy'), allow_pickle=True, encoding='latin1')[0,:]

    # Part A: Computing the 4 initial R,T transformations from Essential Matrix
    print('-' * 80)
    print("Part A: Check your matrices against the example R,T")
    print('-' * 80)
    K = np.eye(3)
    K[0,0] = K[1,1] = focal_length
    E = K.T.dot(fundamental_matrices[0]).dot(K)
    im0 = imread(image_paths[0])
    im_height, im_width, _ = im0.shape
    example_RT = np.array([[0.9736, -0.0988, -0.2056, 0.9994],
        [0.1019, 0.9948, 0.0045, -0.0089],
        [0.2041, -0.0254, 0.9786, 0.0331]])
    print("Example RT:\n", example_RT)
    estimated_RT = estimate_initial_RT(E)
    print('')
    print("Estimated RT:\n", estimated_RT)

    # Part B: Determining the best linear estimate of a 3D point
    print('-' * 80)
    print('Part B: Check that the difference from expected point ')
    print('is near zero')
    print('-' * 80)
    camera_matrices = np.zeros((2, 3, 4))
    camera_matrices[0, :, :] = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
    camera_matrices[1, :, :] = K.dot(example_RT)
    unit_test_matches = matches_subset[0][:,0].reshape(2,2)
    estimated_3d_point = linear_estimate_3d_point(unit_test_matches.copy(),
        camera_matrices.copy())
    expected_3d_point = np.array([0.6774, -1.1029, 4.6621])
    print("Difference: ", np.fabs(estimated_3d_point - expected_3d_point).sum())

    # Part C: Calculating the reprojection error and its Jacobian
    print('-' * 80)
    print('Part C: Check that the difference from expected error/Jacobian ')
    print('is near zero')
    print('-' * 80)
    estimated_error = reprojection_error(
            expected_3d_point, unit_test_matches, camera_matrices)
    estimated_jacobian = jacobian(expected_3d_point, camera_matrices)
    expected_error = np.array((-0.0095458, -0.5171407,  0.0059307,  0.501631))
    print("Error Difference: ", np.fabs(estimated_error - expected_error).sum())
    expected_jacobian = np.array([[ 154.33943931, 0., -22.42541691],
         [0., 154.33943931, 36.51165089],
         [141.87950588, -14.27738422, -56.20341644],
         [21.9792766, 149.50628901, 32.23425643]])
    print("Jacobian Difference: ", np.fabs(estimated_jacobian
        - expected_jacobian).sum())

    # Part D: Determining the best nonlinear estimate of a 3D point
    print('-' * 80)
    print('Part D: Check that the reprojection error from nonlinear method')
    print('is lower than linear method')
    print('-' * 80)
    estimated_3d_point_linear = linear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    estimated_3d_point_nonlinear = nonlinear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    error_linear = reprojection_error(
        estimated_3d_point_linear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Linear method error:", np.linalg.norm(error_linear))
    error_nonlinear = reprojection_error(
        estimated_3d_point_nonlinear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Nonlinear method error:", np.linalg.norm(error_nonlinear))

    # Part E: Determining the correct R, T from Essential Matrix
    print('-' * 80)
    print("Part E: Check your matrix against the example R,T")
    print('-' * 80)
    estimated_RT = estimate_RT_from_E(E,
        np.expand_dims(unit_test_image_matches[:2,:], axis=0), K)
    print("Example RT:\n", example_RT)
    print('')
    print("Estimated RT:\n", estimated_RT)

    # Part F: Run the entire Structure from Motion pipeline
    if not run_pipeline:
        sys.exit()
    print('-' * 80)
    print('Part F: Run the entire SFM pipeline')
    print('-' * 80)
    frames = [0] * (len(image_paths) - 1)
    for i in range(len(image_paths)-1):
        frames[i] = Frame(matches_subset[i].T, focal_length,
                fundamental_matrices[i], im_width, im_height)
        bundle_adjustment(frames[i])
    merged_frame = merge_all_frames(frames)

    # Construct the dense matching
    camera_matrices = np.zeros((2,3,4))
    dense_structure = np.zeros((0,3))
    for i in range(len(frames)-1):
        matches = dense_matches[i]
        camera_matrices[0,:,:] = merged_frame.K.dot(
            merged_frame.motion[i,:,:])
        camera_matrices[1,:,:] = merged_frame.K.dot(
                merged_frame.motion[i+1,:,:])
        points_3d = np.zeros((matches.shape[1], 3))
        use_point = np.array([True]*matches.shape[1])
        for j in range(matches.shape[1]):
            points_3d[j,:] = nonlinear_estimate_3d_point(
                matches[:,j].reshape((2,2)), camera_matrices)
        dense_structure = np.vstack((dense_structure, points_3d[use_point,:]))

    np.save('results.npy', dense_structure)
    print ('Save results to results.npy!')
