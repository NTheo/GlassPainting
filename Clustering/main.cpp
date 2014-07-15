//Theophile Dalens & Jia Li, 2014

#define NOMINMAX
#include <iostream>
#include <numeric>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <dirent.h>
#include <ctime>

using namespace std;
using namespace cv;
using namespace cv::flann;

/* multiplication between sparse and classic matrix A=D*Q
	* hists: sparse matrix (typically matrix of database words)
	* nhist: matrix (typically matrix of query words)
	* return: hists*nhists (matrix)
*/
Mat mul(const SparseMat &hists, const Mat &nhist)
{
    Mat answer = Mat::zeros(hists.size(0),1,CV_32F);
    SparseMatConstIterator
    it = hists.begin(),
    it_end = hists.end();
    for (; it!=it_end;++it)
    {
		const SparseMat::Node* n = it.node();
		answer.at<float>(n->idx[0])+=nhist.at<float>(n->idx[1])*it.value<float>();
    }
    return answer;
}

bool irComparer2 (const pair<int, float> &ir1, const pair<int, float> &ir2)
{
	return (ir1.second>ir2.second);
}

inline bool compare_size(const vector<int> &a, const vector<int> &b) {return a.size() < b.size();}


//L2-normalise the norm of the histogram
Mat normalise(const vector<vector<int> > &hist)
{
	Mat histb;
	for (unsigned i=0; i<hist.size(); i++)
	{
		histb.push_back(float(hist[i].size()));
	}
	Mat nhist;
	normalize(histb, nhist);
	return nhist;
}

//computes x*x
inline double square(double x){return x*x;}

/* applies k means pp seeding
	* descs: descriptors (one descriptor per row)
	* k: number of seeds
	* rng: a random number gernerator
	* vocFileName: name of the output file of the seeds
*/
Mat kmeanspp(const Mat& descs, int k, RNG rng, string vocFileName)
{
	Mat dists;
	BFMatcher matcher(NORM_HAMMING);
	Mat centers = descs.row(rng.uniform(0, descs.rows));
	Mat allDistances;
	batchDistance(centers.row(centers.rows-1), descs, allDistances, CV_32S, noArray(), NORM_HAMMING);
	for(int i=1; i<k; i++)
	{
		//vector<DMatch> matches;
		//matcher.match(centers, descs, matches);
		Mat distances;
		batchDistance(centers.row(centers.rows-1), descs, distances, CV_32S, noArray(), NORM_HAMMING);
		allDistances = min(distances, allDistances);
		vector<double> distancesVec;
		allDistances.copyTo(distancesVec);
		transform(distancesVec.begin(), distancesVec.end(), distancesVec.begin(), square);
		partial_sum(distancesVec.begin(), distancesVec.end(), distancesVec.begin());
		double s = rng.uniform(0., distancesVec[distancesVec.size()-1]);
		int up = int(upper_bound(distancesVec.begin(), distancesVec.end(), s)-distancesVec.begin());
		centers.push_back(descs.row(up));
		//cout << "cluster " << i+1 << " of " << k << ", square distances:" << distancesVec[distancesVec.size()-1] << endl;
	}
	
	FileStorage fs(vocFileName, FileStorage::WRITE);
	fs << "centers" << centers;
	fs.release();
	return centers;
}

/* compute cluster "centers" with k means pp
* imagesDirectory: directory of database images
* descriptorsDirectory: directory with the pre-computed descripteurs
* vocFileName: path to the output file
*/
void computeCenters(char* imagesDirectory, string descriptorsDirectory, string vocFileName, int k)
{
	DIR *dir;
	dir = opendir(imagesDirectory);
	struct dirent *ent;
	Mat descs(0, 32, CV_8U);
	cout << descs.rows << " " << descs.cols << endl;
	while ((ent = readdir (dir)) != NULL)
	{
		if(ent->d_name[0] == '.') //FIX ME (code to avoid "." and "..")
			continue;
			
		string descriptorsFileName = descriptorsDirectory+ent->d_name+".ORB1000+ORB.yml";
		cout << descriptorsFileName << endl;
		FileStorage fs(descriptorsFileName, FileStorage::READ);
		Mat dbDesriptors;
		fs["descriptors"] >> dbDesriptors;
		fs.release();
		cout << "+ " << dbDesriptors.rows << " " << dbDesriptors.cols << endl;
		descs.push_back(dbDesriptors);
		cout << descs.rows << " " << descs.cols << endl;
	}
	RNG rng;
	closedir(dir);
	kmeanspp(descs, k, rng, vocFileName);
	cout << "done computing centers" << endl;
	cin.ignore();
	return;
}
/* iteration of k-means pp to have less descriptors
* upNumber: number of clusters in uplayer
* safe: k in k-nearest-neighbour for the mapping between layers
* bottomFileName: input path for the descriptors in the bottom layer
* upFileName: output path for the descriptors in the up layer
* matsFileName: output path for bottom descriptors for the matched up descriptors
* indexFileName: output path for up-to-bottom descriptors index matrix (also in return)
*/
Mat bottomUp(int upNumber, int safe, string bottomFileName, string upFileName, string matsFileName, string indexFileName)
{
	vector<Mat> mats;
	FileStorage fs(bottomFileName, FileStorage::READ);
	Mat bottom;
	fs["centers"] >> bottom;
	fs.release();
	RNG rng;
	Mat up = kmeanspp(bottom, upNumber, rng, upFileName);

	BFMatcher matcher(NORM_HAMMING);
	vector<vector<DMatch>> matchess;
	matcher.knnMatch(bottom, up, matchess, safe);

	vector<vector<int> > upToBottom;
	for(int u=0; u<upNumber; u++)
	{
		mats.push_back(Mat(0, 32, CV_8U));
		upToBottom.push_back(vector<int>());
	}
	for(unsigned i=0; i<matchess.size(); i++)
		for(unsigned j=0; j<matchess[i].size(); j++)
		{
			mats[matchess[i][j].trainIdx].push_back(bottom.row(matchess[i][j].queryIdx));
			upToBottom[matchess[i][j].trainIdx].push_back(matchess[i][j].queryIdx);
		}

	size_t mx = (*max_element(upToBottom.begin(), upToBottom.end(), compare_size)).size();
	cout << "max size " << int(mx) << endl;
	Mat indexesUB(upNumber, int(mx), CV_32S);
	for(int i=0; i<upNumber; i++)
		for(unsigned j=0; j<upToBottom[i].size(); j++)
			indexesUB.at<INT32>(i,j) = upToBottom[i][j];
	
	FileStorage ifs(indexFileName, FileStorage::WRITE);
	ifs << "index" << indexesUB;
	ifs.release();

	FileStorage mfs(matsFileName, FileStorage::WRITE);
	write(mfs, "mats", mats);
	mfs.release();

	return indexesUB;
}

/* Bag the words
* imagesDirectory: folder of the images
* descriptorsDirectory: directory of the descriptors to bag
* vocFileName: path to the file of the vocabulary
* histsFileName: path to the outputFile of the bags of words
*/
void bag(char* imagesDirectory, string descriptorsDirectory, string vocFileName, string histsFileName)
{
	DIR *dir;
	dir = opendir(imagesDirectory);
	struct dirent *ent;
	
	//The histograms
	Mat hists;
	
	//BOW
	DescriptorExtractor* de = new ORB(1000);
	Ptr<DescriptorExtractor> descex(de);
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	BOWImgDescriptorExtractor bow(descex, matcher);
	
	//ORB vocabulary
	Mat vocabulary;
	FileStorage voc(vocFileName, FileStorage::READ);
	voc["centers"] >> vocabulary;
	voc.release();
	bow.setVocabulary(vocabulary);
	cout << vocabulary.rows << endl;
	cin.ignore();
	//ORB
	OrbDescriptorExtractor ORBde(1000);
	while ((ent = readdir (dir)) != NULL)
	{
		if(ent->d_name[0] == '.') //FIX ME (code to avoid "." and "..")
			continue;
		cout << ent->d_name << endl;
		Mat input = imread(string(imagesDirectory)+(ent->d_name), CV_LOAD_IMAGE_GRAYSCALE);
		vector<vector<int> > words;
		vector<KeyPoint> keypoints;
		ORBde(input, Mat(), keypoints);
		bow.compute(input, keypoints, Mat(), &words);
		cout << words.size() << endl;
		Mat words2;
		transpose(Mat(normalise(words)), words2);
		hists.push_back(words2);
		cout << hists.rows << " " << hists.cols << endl;
	}
	SparseMat sparsehists(hists);
	FileStorage fs(histsFileName, FileStorage::WRITE);
	fs << "hists" << sparsehists;
	fs.release();
	cout << "with clustering" << endl;
	cin.ignore();
	return;
}
vector<pair<int, float> > paintingSearch(const Mat& QI,const SparseMat& hists, BOWImgDescriptorExtractor bow,
                     vector<KeyPoint>& kp,vector<vector<int> >& hist)
{
	ORB detect;
    vector<pair<int, float> > imageResponse;
    detect(QI, Mat(), kp);

	Mat imgDescriptor;
    bow.compute(QI, kp, imgDescriptor, &hist);
    Mat nhist = normalise(hist);

    Mat answer = mul(hists, nhist);

    for(int i=0; i<answer.rows; i++)
    {
        imageResponse.push_back(pair<int, float>(i, answer.at<float>(i)));
    }

    std::sort(imageResponse.begin(), imageResponse.end(), irComparer2);

    return imageResponse;
}
int main(int, char**)
{
	char* imagesDirectory = "../../data/Louvre/samples/";
	string descriptorsDirectory = "../../descriptors/Louvre/FULL/";
	string vocFileName = "../../descriptors/words/ORB+ORB+100000.yml";
	string histsFileName = "../../descriptors/words/samples-hists100000-bis.yml";
	string upFileName = "../../descriptors/words/ORB1000+ORB+1000.yml";
	string indexFileName = "../../descriptors/words/indexes.yml";
	string matsFileName = "../..//descriptors/words/mats.yml";

	//pre-compute: do this once

	//computeCenters(imagesDirectory, descriptorsDirectory, vocFileName, 100000);
	
	//bag(imagesDirectory, descriptorsDirectory, vocFileName, histsFileName);
	
	//upToBottom = bottomUp2(500, 2, vocFileName, upFileName, matsFileName, indexFileName);

	vector<Mat> mats;
	Mat upToBottom;
	
	FileStorage ifs(indexFileName, FileStorage::READ);
	ifs["index"] >> upToBottom;
	ifs.release();
	FileStorage mfs(matsFileName, FileStorage::READ);
	read(mfs["mats"], mats);
	mfs.release();


	//The histograms
	SparseMat hists;
	
	
	FileStorage fs(histsFileName, FileStorage::READ);
	fs["hists"] >> hists;
	fs.release();

	//BOW
	//IndexParams* indexParams = new LshIndexParams(6, 12, 1);
	//Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher(indexParams));
	//Ptr<DescriptorExtractor> descex(new ORB(1000));
	//Ptr<DescriptorMatcher> matcher(new BFMatcher(NORM_HAMMING));

	//Mat vocabulary;
	//FileStorage voc(vocFileName, FileStorage::READ);
	//voc["centers"] >> vocabulary;
	//voc.release();
	//vector<Mat> vocs;
	//vocs.push_back(vocabulary);
	/*matcher->add(vocs);
	matcher->train();*/
	//BOWImgDescriptorExtractor bow(descex, matcher);
	//bow.setVocabulary(vocabulary);
	
	vector<KeyPoint> kp;
	vector<vector<int>> hist;
	const int numberOfInput = 10;
	string inputs[numberOfInput] = {
		"../input/cc.jpg",
		"../input/ex2.jpg",
		"../input/woman1.jpg",
		"../input/woman2.jpg",
		"../input/liberte-glass.jpg",
		"../input/chartres-input.jpg",
		"../input/corot-pearl-input.jpg",
		"../input/meduse.jpg",
		"../input/lebrun-input.jpg",
		"../input/lebrun2-input.jpg"};

	Mat up;
	FileStorage u(upFileName, FileStorage::READ);
	u["centers"] >> up;
	u.release();
	cout << up.cols << " " << up.rows << " " << up.type();
	for(int k=0; k<numberOfInput; k++)
	{
		clock_t start = clock();

		cout << inputs[k] << endl;
		Mat input = imread(inputs[k], CV_LOAD_IMAGE_GRAYSCALE);
		if(input.empty())
			exit(-1);
		
		Mat descriptors;
		vector<KeyPoint> keypoints;
		ORB orb;
		orb.detect(input, keypoints);
		orb.compute(input, keypoints, descriptors);

		vector<DMatch> upMatches;
		BFMatcher(NORM_HAMMING).match(descriptors, up, upMatches);
		cout << ( clock() - start ) / (double) CLOCKS_PER_SEC << endl;
		
		vector<vector<int> > hist;
		for(int i=0; i<100000; i++)
			hist.push_back(vector<int>());
		
		for(int i=0; i<descriptors.rows; i++)
		{
			vector<DMatch> bottomMatch;
			BFMatcher(NORM_HAMMING).match(descriptors.row(i), mats[upMatches[i].trainIdx], bottomMatch);
			hist[upToBottom.at<INT32>(upMatches[i].trainIdx,bottomMatch[0].trainIdx)].push_back(i);
		}

		cout << ( clock() - start ) / (double) CLOCKS_PER_SEC << " ... second matching" << endl;

		Mat nhist = normalise(hist);
		vector<pair<int, float> > imageResponse;

		Mat answer = mul(hists, nhist);

		for(int i=0; i<answer.rows; i++)
		{
			imageResponse.push_back(pair<int, float>(i, answer.at<float>(i)));
		}

		std::sort(imageResponse.begin(), imageResponse.end(), irComparer2);
		cout << ( clock() - start ) / (double) CLOCKS_PER_SEC << endl;
		cout << imageResponse[0].first << " " << 
			imageResponse[1].first << " " <<
			imageResponse[2].first << " " <<
			imageResponse[3].first << " " <<
			imageResponse[4].first << endl;
		
		//vector<pair<int, float> > pss = paintingSearch(input, hists, bow, kp, hist);

		//for(unsigned i=0; i<5; i++)
		//	cout << (pss[i]).first << endl:

		cin.ignore();
	}
	return 0;
}