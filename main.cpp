#include <opencv2/highgui/highgui.hpp>
#include "MatrixReaderWriter.h"
#include "Normalization.h"
#include <iostream>

int getIterationNumber(int point_number_,
	int inlier_number_,
	int sample_size_,
	double confidence_);

void RansacFundamentalMatrix(
	const std::vector<cv::Point2d>& input_src_points_,
	const std::vector<cv::Point2d>& input_destination_points_,
	const std::vector<cv::Point2d>& normalized_input_src_points_,
	const std::vector<cv::Point2d>& normalized_input_destination_points_,
	const cv::Mat& T1_,
	const cv::Mat& T2_,
	cv::Mat& fundamental_matrix_,
	std::vector<size_t>& inliers_,
	double confidence_,
	double threshold_);

void getFundamentalMatrixLSQ(
	const std::vector<cv::Point2d>& source_points_,
	const std::vector<cv::Point2d>& destination_points_,
	cv::Mat& fundamental_matrix_);

void getProjectionMatrices(
	const cv::Mat& essential_matrix_,
	const cv::Mat& K1_,
	const cv::Mat& K2_,
	const cv::Mat& src_point_,
	const cv::Mat& dst_point_,
	cv::Mat& projection_1_,
	cv::Mat& projection_2_);

void linearTriangulation(
	const cv::Mat& projection_1_,
	const cv::Mat& projection_2_,
	const cv::Mat& src_point_,
	const cv::Mat& dst_point_,
	cv::Mat& point3d_);

int main(int argc, char** argv)
{
    //Reading camera parameters:
    MatrixReaderWriter* mrw = new MatrixReaderWriter("data/K.txt");
    cv::Mat K = cv::Mat(3, 3, CV_64F);
    for (int i = 0; i < mrw->rowNum; i++) {
        for (int j = 0; j < mrw->columnNum; j++) {
            K.at<double>(i, j) = (double)mrw->data[i * mrw->columnNum + j];
        }
    }

    //Reading features:
    mrw->load("data/2/features.mat");
    std::vector<cv::Point2d> p1, p2;
    for (int j = 0; j < mrw->columnNum; j++) {
        cv::Point2d temp1, temp2;
        temp1.x = mrw->data[mrw->columnNum * 0 + j];
        temp1.y = mrw->data[mrw->columnNum * 1 + j];
        temp2.x = mrw->data[mrw->columnNum * 2 + j];
        temp2.y = mrw->data[mrw->columnNum * 3 + j];
        p1.push_back(temp1);
        p2.push_back(temp2);
    }
	int ptsNum = mrw->columnNum;

    //Normalize points
    std::vector<cv::Point2d> p1_norm, p2_norm;
    cv::Mat T1, T2;
    NormalizeData(p1, p1_norm, T1);
    NormalizeData(p2, p2_norm, T2);

    //Estimate fundamental matrix
	cv::Mat F = cv::Mat(3, 3, CV_64F);
	std::vector<size_t> F_inliers;
	RansacFundamentalMatrix(p1, p2, p1_norm, p2_norm, T1, T2, F, F_inliers, 0.99, 1);

	//Estimate essential matrix
	cv::Mat E = K.t() * F * K;

	//Decompose essential matrix and select right projections
	cv::Mat P1, P2;
	getProjectionMatrices(E,
		K,
		K,
		(cv::Mat)p1[F_inliers[0]],
		(cv::Mat)p2[F_inliers[0]],
		P1,
		P2);

	//Compute 3D points
	cv::Mat pts3D = cv::Mat(ptsNum, 4, CV_64F);
	for (size_t i = 0; i < ptsNum; ++i) {
		cv::Mat point3d;
		linearTriangulation(P1,
			P2,
			(cv::Mat)p1[i],
			(cv::Mat)p2[i],
			point3d);
		point3d.push_back(1.0);
		for (size_t j = 0; j < 4; ++j) {
			pts3D.at<double>(i, j) = point3d.at<double>(j, 0);
		}
	}

	std::cout << pts3D << std::endl;

	//Write 3D points into xyz file
	int sizeXYZfile = ptsNum * 3;
	double* data = new double[sizeXYZfile];
	for (int i = 0; i < ptsNum; i++) {
		data[3 * i + 0] = pts3D.at<double>(i, 0); //x
		data[3 * i + 1] = pts3D.at<double>(i, 1); //y
		data[3 * i + 2] = pts3D.at<double>(i, 2); //z
	}

	mrw->data = data;
	mrw->rowNum = ptsNum;
	mrw->columnNum = 3;
	mrw->save("data/2/pts3D.txt");

}

int getIterationNumber(int point_number_,
	int inlier_number_,
	int sample_size_,
	double confidence_)
{
	const double inlier_ratio = static_cast<float>(inlier_number_) / point_number_;

	static const double log1 = log(1.0 - confidence_);
	const double log2 = log(1.0 - pow(inlier_ratio, sample_size_));

	const int k = log1 / log2;
	if (k < 0)
		return INT_MAX;
	return k;
}

void RansacFundamentalMatrix(
	const std::vector<cv::Point2d>& input_src_points_,
	const std::vector<cv::Point2d>& input_destination_points_,
	const std::vector<cv::Point2d>& normalized_input_src_points_,
	const std::vector<cv::Point2d>& normalized_input_destination_points_,
	const cv::Mat& T1_,
	const cv::Mat& T2_,
	cv::Mat& fundamental_matrix_,
	std::vector<size_t>& inliers_,
	double confidence_,
	double threshold_)
{
	// The so-far-the-best fundamental matrix
	cv::Mat best_fundamental_matrix;
	// The number of correspondences
	const size_t point_number = input_src_points_.size();

	// Initializing the index pool from which the minimal samples are selected
	std::vector<size_t> index_pool(point_number);
	for (size_t i = 0; i < point_number; ++i)
		index_pool[i] = i;

	// The size of a minimal sample
	constexpr size_t sample_size = 8;
	// The minimal sample
	size_t* mss = new size_t[sample_size];

	size_t maximum_iterations = std::numeric_limits<int>::max(), // The maximum number of iterations set adaptively when a new best model is found
		iteration_limit = 5000, // A strict iteration limit which mustn't be exceeded
		iteration = 0; // The current iteration number

	std::vector<cv::Point2d> source_points(sample_size),
		destination_points(sample_size);

	while (iteration++ < MIN(iteration_limit, maximum_iterations))
	{

		for (auto sample_idx = 0; sample_idx < sample_size; ++sample_idx)
		{
			// Select a random index from the pool
			const size_t idx = round((rand() / (double)RAND_MAX) * (index_pool.size() - 1));
			mss[sample_idx] = index_pool[idx];
			index_pool.erase(index_pool.begin() + idx);

			// Put the selected correspondences into the point containers
			const size_t point_idx = mss[sample_idx];
			source_points[sample_idx] = normalized_input_src_points_[point_idx];
			destination_points[sample_idx] = normalized_input_destination_points_[point_idx];
		}

		// Estimate fundamental matrix
		cv::Mat fundamental_matrix(3, 3, CV_64F);
		getFundamentalMatrixLSQ(source_points, destination_points, fundamental_matrix);
		fundamental_matrix = T2_.t() * fundamental_matrix * T1_; // Denormalize the fundamental matrix

		// Count the inliers
		std::vector<size_t> inliers;
		const double* p = (double*)fundamental_matrix.data;
		for (int i = 0; i < input_src_points_.size(); ++i)
		{
			// Symmetric epipolar distance   
			cv::Mat pt1 = (cv::Mat_<double>(3, 1) << input_src_points_[i].x, input_src_points_[i].y, 1);
			cv::Mat pt2 = (cv::Mat_<double>(3, 1) << input_destination_points_[i].x, input_destination_points_[i].y, 1);

			// TODO: calculate the error
			cv::Mat lL = fundamental_matrix.t() * pt2;
			cv::Mat lR = fundamental_matrix * pt1;

			// Calculate the distance of point pt1 from lL
			const double
				& aL = lL.at<double>(0),
				& bL = lL.at<double>(1),
				& cL = lL.at<double>(2);

			double tL = abs(aL * input_src_points_[i].x + bL * input_src_points_[i].y + cL);
			double dL = sqrt(aL * aL + bL * bL);
			double distanceL = tL / dL;

			// Calculate the distance of point pt2 from lR
			const double
				& aR = lR.at<double>(0),
				& bR = lR.at<double>(1),
				& cR = lR.at<double>(2);

			double tR = abs(aR * input_destination_points_[i].x + bR * input_destination_points_[i].y + cR);
			double dR = sqrt(aR * aR + bR * bR);
			double distanceR = tR / dR;

			double dist = 0.5 * (distanceL + distanceR);

			if (dist < threshold_)
				inliers.push_back(i);
		}

		// Update if the new model is better than the previous so-far-the-best.
		if (inliers_.size() < inliers.size())
		{
			// Update the set of inliers
			inliers_.swap(inliers);
			inliers.clear();
			inliers.resize(0);
			// Update the model parameters
			best_fundamental_matrix = fundamental_matrix;
			// Update the iteration number
			maximum_iterations = getIterationNumber(point_number,
				inliers_.size(),
				sample_size,
				confidence_);
		}

		// Put back the selected points to the pool
		for (size_t i = 0; i < sample_size; ++i)
			index_pool.push_back(mss[i]);
	}

	delete[] mss;

	fundamental_matrix_ = best_fundamental_matrix;
}

void getFundamentalMatrixLSQ(
	const std::vector<cv::Point2d>& source_points_,
	const std::vector<cv::Point2d>& destination_points_,
	cv::Mat& fundamental_matrix_)
{
	const size_t pointNumber = source_points_.size();
	cv::Mat A(pointNumber, 9, CV_64F);

	for (size_t pointIdx = 0; pointIdx < pointNumber; ++pointIdx)
	{
		const double
			& x1 = source_points_[pointIdx].x,
			& y1 = source_points_[pointIdx].y,
			& x2 = destination_points_[pointIdx].x,
			& y2 = destination_points_[pointIdx].y;

		A.at<double>(pointIdx, 0) = x1 * x2;
		A.at<double>(pointIdx, 1) = x2 * y1;
		A.at<double>(pointIdx, 2) = x2;
		A.at<double>(pointIdx, 3) = y2 * x1;
		A.at<double>(pointIdx, 4) = y2 * y1;
		A.at<double>(pointIdx, 5) = y2;
		A.at<double>(pointIdx, 6) = x1;
		A.at<double>(pointIdx, 7) = y1;
		A.at<double>(pointIdx, 8) = 1;
	}

	cv::Mat evals, evecs;
	cv::Mat AtA = A.t() * A;
	cv::eigen(AtA, evals, evecs);

	cv::Mat x = evecs.row(evecs.rows - 1); // x = [f1 f2 f3 f4 f5 f6 f7 f8 f9]
	fundamental_matrix_.create(3, 3, CV_64F);
	memcpy(fundamental_matrix_.data, x.data, sizeof(double) * 9);
}

void getProjectionMatrices(
	const cv::Mat& essential_matrix_,
	const cv::Mat& K1_,
	const cv::Mat& K2_,
	const cv::Mat& src_point_,
	const cv::Mat& dst_point_,
	cv::Mat& projection_1_,
	cv::Mat& projection_2_)
{
	// ****************************************************
	// Calculate the projection matrix of the first camera
	// ****************************************************
	projection_1_ = K1_ * cv::Mat::eye(3, 4, CV_64F);

	// ****************************************************
	// Calculate the projection matrix of the second camera
	// ****************************************************

	// Decompose the essential matrix
	cv::Mat rotation_1, rotation_2, translation;

	cv::SVD svd(essential_matrix_, cv::SVD::FULL_UV);
	// It gives matrices U D Vt

	if (cv::determinant(svd.u) < 0)
		svd.u.col(2) *= -1;
	if (cv::determinant(svd.vt) < 0)
		svd.vt.row(2) *= -1;

	cv::Mat w = (cv::Mat_<double>(3, 3) << 0, -1, 0,
		1, 0, 0,
		0, 0, 1);

	rotation_1 = svd.u * w * svd.vt;
	rotation_2 = svd.u * w.t() * svd.vt;
	translation = svd.u.col(2) / cv::norm(svd.u.col(2));

	// The possible solutions:
	// (rotation_1, translation)
	// (rotation_2, translation)
	// (rotation_1, -translation)
	// (rotation_2, -translation)

	cv::Mat P21 = K2_ * (cv::Mat_<double>(3, 4) <<
		rotation_1.at<double>(0, 0), rotation_1.at<double>(0, 1), rotation_1.at<double>(0, 2), translation.at<double>(0),
		rotation_1.at<double>(1, 0), rotation_1.at<double>(1, 1), rotation_1.at<double>(1, 2), translation.at<double>(1),
		rotation_1.at<double>(2, 0), rotation_1.at<double>(2, 1), rotation_1.at<double>(2, 2), translation.at<double>(2));
	cv::Mat P22 = K2_ * (cv::Mat_<double>(3, 4) <<
		rotation_2.at<double>(0, 0), rotation_2.at<double>(0, 1), rotation_2.at<double>(0, 2), translation.at<double>(0),
		rotation_2.at<double>(1, 0), rotation_2.at<double>(1, 1), rotation_2.at<double>(1, 2), translation.at<double>(1),
		rotation_2.at<double>(2, 0), rotation_2.at<double>(2, 1), rotation_2.at<double>(2, 2), translation.at<double>(2));
	cv::Mat P23 = K2_ * (cv::Mat_<double>(3, 4) <<
		rotation_1.at<double>(0, 0), rotation_1.at<double>(0, 1), rotation_1.at<double>(0, 2), -translation.at<double>(0),
		rotation_1.at<double>(1, 0), rotation_1.at<double>(1, 1), rotation_1.at<double>(1, 2), -translation.at<double>(1),
		rotation_1.at<double>(2, 0), rotation_1.at<double>(2, 1), rotation_1.at<double>(2, 2), -translation.at<double>(2));
	cv::Mat P24 = K2_ * (cv::Mat_<double>(3, 4) <<
		rotation_2.at<double>(0, 0), rotation_2.at<double>(0, 1), rotation_2.at<double>(0, 2), -translation.at<double>(0),
		rotation_2.at<double>(1, 0), rotation_2.at<double>(1, 1), rotation_2.at<double>(1, 2), -translation.at<double>(1),
		rotation_2.at<double>(2, 0), rotation_2.at<double>(2, 1), rotation_2.at<double>(2, 2), -translation.at<double>(2));

	std::vector< const cv::Mat* > Ps = { &P21, &P22, &P23, &P24 };
	double minDistance = std::numeric_limits<double>::max();

	for (const auto& P2ptr : Ps)
	{
		const cv::Mat& P1 = projection_1_;
		const cv::Mat& P2 = *P2ptr;

		// Estimate the 3D coordinates of a point correspondence
		cv::Mat point3d;
		linearTriangulation(P1,
			P2,
			src_point_,
			dst_point_,
			point3d);
		point3d.push_back(1.0);

		cv::Mat projection1 =
			P1 * point3d;
		cv::Mat projection2 =
			P2 * point3d;

		if (projection1.at<double>(2) < 0 ||
			projection2.at<double>(2) < 0)
			continue;

		projection1 = projection1 / projection1.at<double>(2);
		projection2 = projection2 / projection2.at<double>(2);

		// cv::norm(projection1 - src_point_)
		double dx1 = projection1.at<double>(0) - src_point_.at<double>(0);
		double dy1 = projection1.at<double>(1) - src_point_.at<double>(1);
		double squaredDist1 = dx1 * dx1 + dy1 * dy1;

		// cv::norm(projection2 - dst_point_)
		double dx2 = projection2.at<double>(0) - dst_point_.at<double>(0);
		double dy2 = projection2.at<double>(1) - dst_point_.at<double>(1);
		double squaredDist2 = dx2 * dx2 + dy2 * dy2;

		if (squaredDist1 + squaredDist2 < minDistance)
		{
			minDistance = squaredDist1 + squaredDist2;
			projection_2_ = P2.clone();
		}
	}
}

void linearTriangulation(
	const cv::Mat& projection_1_,
	const cv::Mat& projection_2_,
	const cv::Mat& src_point_,
	const cv::Mat& dst_point_,
	cv::Mat& point3d_)
{
	cv::Mat A(4, 3, CV_64F);
	cv::Mat b(4, 1, CV_64F);

	{
		const double
			& px = src_point_.at<double>(0),
			& py = src_point_.at<double>(1),
			& p1 = projection_1_.at<double>(0, 0),
			& p2 = projection_1_.at<double>(0, 1),
			& p3 = projection_1_.at<double>(0, 2),
			& p4 = projection_1_.at<double>(0, 3),
			& p5 = projection_1_.at<double>(1, 0),
			& p6 = projection_1_.at<double>(1, 1),
			& p7 = projection_1_.at<double>(1, 2),
			& p8 = projection_1_.at<double>(1, 3),
			& p9 = projection_1_.at<double>(2, 0),
			& p10 = projection_1_.at<double>(2, 1),
			& p11 = projection_1_.at<double>(2, 2),
			& p12 = projection_1_.at<double>(2, 3);

		A.at<double>(0, 0) = px * p9 - p1;
		A.at<double>(0, 1) = px * p10 - p2;
		A.at<double>(0, 2) = px * p11 - p3;
		A.at<double>(1, 0) = py * p9 - p5;
		A.at<double>(1, 1) = py * p10 - p6;
		A.at<double>(1, 2) = py * p11 - p7;

		b.at<double>(0) = p4 - px * p12;
		b.at<double>(1) = p8 - py * p12;
	}

	{
		const double
			& px = dst_point_.at<double>(0),
			& py = dst_point_.at<double>(1),
			& p1 = projection_2_.at<double>(0, 0),
			& p2 = projection_2_.at<double>(0, 1),
			& p3 = projection_2_.at<double>(0, 2),
			& p4 = projection_2_.at<double>(0, 3),
			& p5 = projection_2_.at<double>(1, 0),
			& p6 = projection_2_.at<double>(1, 1),
			& p7 = projection_2_.at<double>(1, 2),
			& p8 = projection_2_.at<double>(1, 3),
			& p9 = projection_2_.at<double>(2, 0),
			& p10 = projection_2_.at<double>(2, 1),
			& p11 = projection_2_.at<double>(2, 2),
			& p12 = projection_2_.at<double>(2, 3);

		A.at<double>(2, 0) = px * p9 - p1;
		A.at<double>(2, 1) = px * p10 - p2;
		A.at<double>(2, 2) = px * p11 - p3;
		A.at<double>(3, 0) = py * p9 - p5;
		A.at<double>(3, 1) = py * p10 - p6;
		A.at<double>(3, 2) = py * p11 - p7;

		b.at<double>(2) = p4 - px * p12;
		b.at<double>(3) = p8 - py * p12;
	}

	//cv::Mat x = (A.t() * A).inv() * A.t() * b;
	point3d_ = A.inv(cv::DECOMP_SVD) * b;
}