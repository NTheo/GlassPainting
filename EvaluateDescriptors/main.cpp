//Theophile Dalens, INRIA & Telecom ParisTech 2014

#include "databaseFeatures.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
using namespace std;
using namespace cv;

typedef pair<string, float> ImageScore;

bool reversePairComparer (const ImageScore &ir1, const ImageScore &ir2)
{
	return (ir1.second<ir2.second);
}

int ranking(string imPath,
			string actualAnswer,
			char* imagesDirectory,
			string descriptorsDirectory,
			Method method)
{
	Mat x = imread(imPath);
	vector<KeyPoint> keypoints;
	method.detector->detect(x, keypoints);

	Mat descriptors;
	method.extractor->compute(x, keypoints, descriptors);

	vector<ImageScore> imageScores; 

	DIR *dir;
	dir = opendir(imagesDirectory);
	struct dirent *ent;
	while ((ent = readdir (dir)) != NULL)
	{
		if(ent->d_name[0] == '.') //FIX ME (code to avoid "." and "..")
			continue;
			
		string descriptorsFileName = (descriptorsDirectory+ent->d_name)+"."+method.name+".yml";
		FileStorage fs(descriptorsFileName, FileStorage::READ);
		Mat dbDesriptors;
		fs["descriptors"] >> dbDesriptors;
		fs.release();
		
		if(dbDesriptors.empty())
			continue;
		
		switch(method.ranking)
		{
		case METHOD_RANKING_SUM_DISTANCE:
		{
			vector<DMatch> matches;
			method.matcher->match(descriptors, dbDesriptors, matches);
			float distanceSum = 0.f;
			for(unsigned i=0; i<matches.size(); i++)
			{
				distanceSum += matches[i].distance;
			}
			imageScores.push_back(ImageScore(ent->d_name, distanceSum));
			break;
		}
		case METHOD_RANKING_RATIO:
		default:
		{
			vector<vector<DMatch>> matchess;
			method.matcher->knnMatch(descriptors, dbDesriptors, matchess, 2);
			float goodMatch = 0.f;
			for(unsigned i=0; i<matchess.size(); i++)
			{
				if(matchess[i].size() == 1 || matchess[i][0].distance < 0.8*matchess[i][1].distance)
					goodMatch++;
			}
			imageScores.push_back(ImageScore(ent->d_name, -goodMatch));
		}
		}
	}
	closedir (dir);

	sort(imageScores.begin(), imageScores.end(), reversePairComparer);
	unsigned index = 0;
	while(imageScores[index++].first!=actualAnswer);
	cout << "result for " << method.name << " : " << index << '\n'
		<< "with score " << imageScores[index-1].second << endl;
	for(int j=0; j<5; j++)
	{
		cout << imageScores[j].first << " " << imageScores[j].second << endl;
	}
	return index;
}

int main(int, char**)
{
	//computeFeatures();
	//return 0;
	vector<Method> methods = generateMethods();
	
	ofstream result("results4.csv");

	result << "input image;GT";
	
	for(unsigned i=0; i<methods.size(); i++)
	{
		result << ";" << methods[i].name;
	}

	result << "\n";
	const int numberOfInput = 10;
	string inputs[numberOfInput] = {
		"../input/cc.jpg",
		"../input/ex2.jpg",
		"../input/woman1.jpg",
		"../input/woman2.jpg",
		"../input/liberte-glass.jpg",
		"../input/chartres-input.jpg",
		"../input/corot-pearl-input.jpg", "../input/meduse.jpg",
		"../input/lebrun-input.jpg", "../input/lebrun2-input.jpg"};
	string gts[numberOfInput] = {"0013317N.BMP", "0013239N.BMP",
		"0039849N.BMP",
		"0079463N.BMP", "0079488N.BMP",
		"chartres-resized.jpg",
		"Camille_Corot_-_Woman_with_a_Pearl-resized.jpg", "l0meduse.jpg",
		"Lebrun-resized.jpg", "Lebrun2-resized.jpg"};

	for(unsigned n=0; n<numberOfInput; n++)
	{
		Mat x = imread(inputs[n]);
		Mat g = imread("../../data/Louvre/FULL/"+gts[n]);
		imshow("current input", x);
		imshow("current GT", g);
		waitKey(1000);
		result << inputs[n] << ";" << gts[n];
		for(unsigned i=0; i<methods.size(); i++)
		{
			cout << "ranking for " << methods[i].name << endl;
			result << ";" <<
				ranking(inputs[n],
				gts[n],
				"../../data/Louvre/samples/",
				"../../descriptors/Louvre/FULL/",
				methods[i]);
		}
		result << "\n";
	}	
	result.close();
	cout << "enter key to quit" << endl;
	cin.ignore();
	return 0;
}