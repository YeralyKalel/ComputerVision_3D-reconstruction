#include "Normalization.h"
#include "math.h"

void NormalizeData(const std::vector<cv::Point2d>& inPts, std::vector<cv::Point2d>& outPts, cv::Mat& T) {
	
	outPts.clear();
	int ptsNum = inPts.size();
	double sqrt2 = sqrtf(2);

	//calculate means (they will be the center of coordinate systems)
	double meanX = 0.0, meanY = 0.0;
	for (int i = 0; i < ptsNum; i++) {
		meanX += inPts[i].x;
		meanY += inPts[i].y;
	}
	meanX /= ptsNum;
	meanY /= ptsNum;

	double spreadX = 0.0, spreadY = 0.0;

	for (int i = 0; i < ptsNum; i++) {
		spreadX += (inPts[i].x - meanX) * (inPts[i].x - meanX);
		spreadY += (inPts[i].y - meanY) * (inPts[i].y - meanY);
	}

	spreadX /= ptsNum;
	spreadY /= ptsNum;

	cv::Mat offs = cv::Mat::eye(3, 3, CV_64F);
	cv::Mat scale = cv::Mat::eye(3, 3, CV_64F);

	offs.at<double>(0, 2) = -meanX;
	offs.at<double>(1, 2) = -meanY;

	scale.at<double>(0, 0) = sqrt2 / sqrt(spreadX);
	scale.at<double>(1, 1) = sqrt2 / sqrt(spreadY);

	T = scale * offs;

	for (int i = 0; i < ptsNum; i++) {
		cv::Point2f p2D;

		p2D.x = sqrt2 * (inPts[i].x - meanX) / sqrt(spreadX);
		p2D.y = sqrt2 * (inPts[i].y - meanY) / sqrt(spreadY);

		outPts.push_back(p2D);
	}

}
