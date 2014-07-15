//Theophile Dalens, INRIA & Telecom ParisTech 2014

#ifndef __METHOD_HPP__
#define __METHOD_HPP__

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

enum {
	METHOD_RANKING_SUM_DISTANCE,
	METHOD_RANKING_RATIO
};
struct Method
{
	std::string name;
	cv::FeatureDetector* detector;
	cv::DescriptorExtractor* extractor;
	cv::DescriptorMatcher* matcher;
	unsigned ranking;

	Method(std::string n,
		cv::FeatureDetector* d,
		cv::DescriptorExtractor*e,
		cv::DescriptorMatcher* m) :
	name(n), detector(d), extractor(e), matcher(m), ranking(METHOD_RANKING_RATIO) {}
};

std::vector<Method> generateMethods();

#endif //__METHOD_HPP__