//Theophile Dalens 2014

#include "method.hpp"
using namespace cv;

vector<Method> generateMethods()
{
	vector<Method> methods;
	//methods.push_back(Method("SIFT+SIFT", new SIFT(), new SIFT(), new BFMatcher(NORM_L2)));
	methods.push_back(Method("SIFT500+SIFT", new SIFT(500), new SIFT(), new BFMatcher(NORM_L2)));
	methods.push_back(Method("SURF475+SURF", new SURF(475), new SURF(), new BFMatcher(NORM_L2)));
	methods.push_back(Method("ORB+ORB", new ORB(), new ORB(), new BFMatcher(NORM_HAMMING)));
	//methods.push_back(Method("BRISK13+BRISK", new BRISK(13), new BRISK(), new BFMatcher(NORM_HAMMING)));
	//methods.push_back(Method("SURF475+BRISK", new SURF(475), new BRISK(), new BFMatcher(NORM_HAMMING)));
	methods.push_back(Method("ORB1000+ORB", new ORB(1000), new ORB(), new BFMatcher(NORM_HAMMING)));

	//methods.push_back(Method("SURF100+SURF", new SURF(100), new SURF(), new BFMatcher(NORM_L2)));
	//methods.push_back(Method("SURF100+ORB", new SURF(100), new ORB(), new BFMatcher(NORM_HAMMING)));
	//methods.push_back(Method("SURF100+BRISK", new SURF(100), new BRISK(), new BFMatcher(NORM_HAMMING)));
	//methods.push_back(Method("BRISK7+BRISK", new BRISK(7), new BRISK(), new BFMatcher(NORM_HAMMING)));
	//methods.push_back(Method("SURF475+BRISK", new SURF(475), new BRISK(), new BFMatcher(NORM_HAMMING)));
	//methods.push_back(Method("ORB1000+ORB", new ORB(1000), new ORB(), new BFMatcher(NORM_HAMMING)));

	return methods;
	
}