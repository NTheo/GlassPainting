//Theophile Dalens, INRIA & Telecom ParisTech 2014

#include "databaseFeatures.hpp"
using namespace std;
using namespace cv;

void computeFeatures(bool overrideFiles)
{
	string inputDirectory = "../../data/Louvre/FULL/";
	string outputDirectory = "../../descriptors/Louvre/FULL/";

	vector<Method> methods = generateMethods();

	DIR *dir;
	dir = opendir(inputDirectory.c_str());
	struct dirent *ent;
	while ((ent = readdir (dir)) != NULL)
	{		
		if(ent->d_name[0] == '.') //FIX THIS (check if file is not '.' nor '..')
			continue;

		cout << ent->d_name << endl;
		Mat input = imread(inputDirectory+(ent->d_name), CV_LOAD_IMAGE_GRAYSCALE);
		for(unsigned i = 0; i<methods.size(); i++)
		{
			string outputFileName = (outputDirectory+ent->d_name)+"."+methods[i].name+".yml";
			cout << outputFileName << " ...";
			struct stat s;
			if(!overrideFiles && (stat(outputFileName.c_str(), &s)==0))
			{
				cout << " file already exists." << endl;
				continue;
			}
			vector<KeyPoint> keypoints;
			methods[i].detector->detect(input, keypoints);

			Mat descriptors;
			methods[i].extractor->compute(input, keypoints, descriptors);

			FileStorage fs(outputFileName, FileStorage::WRITE);
			fs << "method" << methods[i].name << "keypoints" << keypoints << "descriptors" << descriptors;
			fs.release();
			cout << " done." << endl;
		}
	}
	closedir (dir);

}