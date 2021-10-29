# ComputerVision_3D-reconstruction

Constructs 3D point cloud on given point correspondences and their internal camera parameters.

Procedural steps:
1) Load data
2) Data normalization
3) Estimate Fundamental matrix using point correspondences and RANSAC algorithm
4) Estimate Essential matrix by Fundamental matrix and camera parameters
5) Decompose Essential matrix into 4 different pair projection matrices using SVD algorithm
6) For any point correspondence, select the least different 3D point estimated using all pairs and linear triangulation algorithm
7) Compute 3D point cloud for all point correspondences from the resultant pair of projection matrices
8) Save the result
