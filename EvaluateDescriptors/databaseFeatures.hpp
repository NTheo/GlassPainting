//Theophile Dalens, INRIA & Telecom ParisTech 2014

#ifndef __DATABASE_FEATURES_HPP__
#define __DATABASE_FEATURES_HPP__

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "method.hpp"
#include <dirent.h>

void computeFeatures(bool overrideFiles = false);

#endif //__DATABASE_FEATURES_HPP__