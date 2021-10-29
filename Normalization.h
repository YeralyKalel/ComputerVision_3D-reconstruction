#pragma once
#include <opencv2/opencv.hpp>  



void NormalizeData(const std::vector<cv::Point2d>& inPts, std::vector<cv::Point2d>& outPts, cv::Mat& T);
