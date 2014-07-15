#include <GlassPainting_jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <string>
#include <vector>
#include <jni.h>
#include <android/log.h>


#define LOG_TAG "jni"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

using namespace std;
using namespace cv;

JNIEXPORT void JNICALL Java_fr_ens_di_willow_glasspainting_QueryActivity_detect
(JNIEnv*, jclass, jlong addrmRgba)
{
	Mat* mRgba = (Mat*)addrmRgba;
	vector<KeyPoint> keypoints;
	ORB orb;
	orb.detect(*mRgba, keypoints, Mat());
	Mat temp;
	Mat mRgb;
	cvtColor(*mRgba, mRgb, CV_BGRA2BGR);
	drawKeypoints(mRgb, keypoints, temp, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	temp.copyTo(*mRgba);
}

//query for first approach (one-to-one matching)
JNIEXPORT jint JNICALL Java_fr_ens_di_willow_glasspainting_QueryActivity_rank
(JNIEnv* env, jclass, jlong addrmRgba, jlongArray addrDescriptors)
{
	LOGD("beginning retrieval");
	jlong *descriptors = env->GetLongArrayElements(addrDescriptors,0);
	vector<Mat> descs;
	for(unsigned i=0; i<env->GetArrayLength(addrDescriptors); i++)
	{
		Mat* desc = (Mat*)descriptors[i];
		LOGD("loading desc");
		descs.push_back(*desc);
	}
	Mat* mRgba = (Mat*)addrmRgba;
	LOGD("retrieving descriptors");

	ORB orb;

	vector<KeyPoint> keypoints;
	orb.detect(*mRgba, keypoints, Mat());

	Mat mRgbaDescs;
	orb.compute(*mRgba, keypoints, mRgbaDescs);
	LOGD("descriptors computed for query image");

	float bestMatchScore = 0.;
	unsigned bestMatchIndex = -1;
	for(unsigned i=0; i<env->GetArrayLength(addrDescriptors); i++)
	{
		LOGD("comparing %d...", i);
		vector<vector<DMatch> > matchess;
		BFMatcher(NORM_HAMMING).knnMatch(mRgbaDescs, descs[i], matchess, 2);
		float goodMatch = 0.f;
		for(unsigned j=0; j<matchess.size(); j++)
		{
			if(matchess[j].size() == 1 || matchess[j][0].distance < 0.92*matchess[j][1].distance)
				goodMatch++;
		}
		if(goodMatch>bestMatchScore)
		{
			bestMatchIndex = i;
			bestMatchScore = goodMatch;
		}
	}
	LOGD("best match is #%d", bestMatchIndex);
	return jint(bestMatchIndex);
}

//loading the descriptors for first approach (one-to-one matching)
JNIEXPORT void JNICALL Java_fr_ens_di_willow_glasspainting_QueryActivity_load
(JNIEnv* env, jclass, jlongArray addrDescriptors)
{
	string prefix = "/sdcard/louvredescs/";
	string suffix = ".ORB+ORB.yml";
	string descs[10] = {"0013317N.BMP", "0013239N.BMP", "0039849N.BMP",
			"0079463N.BMP", "0079488N.BMP",  "0079397N.BMP", "0079402N.BMP",
			"0079483N.BMP", "0110856N.BMP", "0110932N.BMP"};
	jlong *descriptors = env->GetLongArrayElements(addrDescriptors,0);
	for(unsigned i=0; i<env->GetArrayLength(addrDescriptors); i++)
	{
		Mat* desc = (Mat*)descriptors[i];
		FileStorage descFS(prefix+descs[i]+suffix, FileStorage::READ);
		descFS["descriptors"] >> *desc;
		descFS.release();
		LOGD("loaded %d descriptors for image %d", desc->rows, i);
	}
	for(unsigned i=0; i<10; i++)
	{
		Mat* desc = (Mat*)descriptors[i];
		LOGD("checking: %d descriptors", desc->rows);
	}
	vector<Mat> mats;
	Mat upToBottom;
	FileStorage mfs("/sdcard/descriptors/words/mats.yml", FileStorage::READ);
	read(mfs["mats"], mats);
	mfs.release();
}
//loading vocabulary descriptors for second approach (hierachical bag of words)
JNIEXPORT jlong JNICALL Java_fr_ens_di_willow_glasspainting_QueryActivity_load2
(JNIEnv* env, jclass, jlong addrvoc, jlongArray addrMats, jlong addrIndex)
{
	LOGD("loading...");
	Mat* voc = (Mat*)addrvoc;
	string vocFileName = "/sdcard/descriptors/words/ORB1000+ORB+1000.yml";
	FileStorage vocfs(vocFileName, FileStorage::READ);
	vocfs["centers"] >> *voc;

	vocfs.release();

	Mat* index = (Mat*)addrIndex;
	string indexFileName = "/sdcard/descriptors/words/indexes.yml";
	FileStorage ifs(indexFileName, FileStorage::READ);
	ifs["index"] >> *index;
	ifs.release();

	jlong *pmats = env->GetLongArrayElements(addrMats,0);
	vector<Mat> mats;
	string matsFileName = "/sdcard/descriptors/words/mats.yml";
	FileStorage mfs(matsFileName, FileStorage::READ);
	read(mfs["mats"], mats);
	mfs.release();
	for(unsigned i=0; i<mats.size(); i++)
	{
		Mat* a = (Mat*)pmats[i];
		mats[i].copyTo(*a);
	}
	string histsFileName = "/sdcard/descriptors/words/samples-hists100000-bis.yml";
	SparseMat hists2;
	FileStorage fs(histsFileName, FileStorage::READ);
	LOGD("...");
	fs["hists"] >> hists2;
	fs.release();
	return (jlong) new SparseMat(hists2);
	LOGD("loading done");
}
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
using namespace cv::flann;
//ranking for first approach
vector<pair<int, float> > paintingSearch(const Mat& QI,const SparseMat& hists, BOWImgDescriptorExtractor bow,
                     vector<KeyPoint>& kp,vector<vector<int> >& hist)
{
	//SIFT detect;
	ORB detect;
    vector<pair<int, float> > imageResponse;
    detect(QI, Mat(), kp);


	Mat imgDescriptor;
    bow.compute(QI, kp, imgDescriptor, &hist);
    Mat nhist = normalise(hist);
    LOGD("ok");
    //Mat answer = hists*nhist;
    Mat answer = mul(hists, nhist);
    LOGD("mul");
    for(int i=0; i<answer.rows; i++)
    {
        imageResponse.push_back(pair<int, float>(i, answer.at<float>(i)));
    }
    LOGD("before sort");
    std::sort(imageResponse.begin(), imageResponse.end(), irComparer2);
    LOGD("after sort %d", imageResponse.size());
    pair<int, float> i = imageResponse[0];
    LOGD("that is ok");
    LOGD("and that %d",i.first);
    return imageResponse;
}
//ranking for second approach (hierarchical bag of words)
JNIEXPORT jint JNICALL Java_fr_ens_di_willow_glasspainting_QueryActivity_rank2
(JNIEnv* env, jclass, jlong addrmRgba, jlong addrVoc, jlongArray addrMats, jlong addrIndex, jlong addrHist)
{

	LOGD("rank 2 launched");

	//The histograms
	SparseMat hists;
	Mat vocabulary = *(Mat*)addrVoc;
	LOGD("retrieving voc : %d rows %d cols", vocabulary.rows, vocabulary.cols);

	hists =*(SparseMat*)addrHist;
	vector<Mat> mats;
	jlong *pmats = env->GetLongArrayElements(addrMats,0);
	for(unsigned i=0; i<env->GetArrayLength(addrMats); i++)
	{
		Mat* mat = (Mat*)(pmats[i]);
		mats.push_back(*mat);
	}
	Mat index = *(Mat*)addrIndex;


	LOGD("retieving hists");
	Mat mRgba = *(Mat*)addrmRgba;
	LOGD("let's go");
	Mat descriptors;
	vector<KeyPoint> keypoints;
	ORB orb;
	orb.detect(mRgba, keypoints);
	orb.compute(mRgba, keypoints, descriptors);

	LOGD("descriptors computed");
	vector<DMatch> upMatches;
	BFMatcher(NORM_HAMMING).match(descriptors, vocabulary, upMatches);
	LOGD("1st matching");

	vector<vector<int> > hist;
	for(int i=0; i<100000; i++)
		hist.push_back(vector<int>());

	for(int i=0; i<descriptors.rows; i++)
	{
		vector<DMatch> bottomMatch;
		BFMatcher(NORM_HAMMING).match(descriptors.row(i), mats[upMatches[i].trainIdx], bottomMatch);
		hist[index.at<int>(upMatches[i].trainIdx,bottomMatch[0].trainIdx)].push_back(i);
	}

	LOGD("second matching");

	Mat nhist = normalise(hist);
	LOGD("hist %d %d", nhist.cols, nhist.rows);
	vector<pair<int, float> > imageResponse;

	Mat answer = mul(hists, nhist);
	LOGD("multiplication %d %d", answer.rows, answer.cols);
	for(int i=0; i<answer.rows; i++)
	{
		imageResponse.push_back(pair<int, float>(i, answer.at<float>(i)));
	}

	std::sort(imageResponse.begin(), imageResponse.end(), irComparer2);

	int i = imageResponse[0].first;
	LOGD("result is %d",i);
	//This hack is to convert the answer to match it with the first rank function
	//TODO clean this mess up
	switch(i)
	{
	case 7:
		return 0;
	case 6:
		return 1;
	case 23:
		return 2;
	case 28:
		return 3;
	case 31:
		return 4;
	default:
		return 5;
	}
	return 5;
}
