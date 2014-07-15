GlassPainting
=============
GlassPainting, by Théophile Dalens

This source code is available to illustrate our work "Painting recognition from wearable cameras" (Théophile Dalens, Josef Sivic, Ivan Laptev, Marine Campedel; ENS/INRIA/CNRS UMR 8548 and Télécom ParisTECH). It is written by Théophile Dalens and Jia Li (WTF Public License 2.0), and contains code from OpenCV "face detection" sample (Willow Garage Inc., 3-clause BSD License). It is constituted of 3 parts:

1: EvaluateDescriptors. This is made to extract descriptors from images of a database, and eventually compare the accuracy of different descriptors.
2: Clustering. This is made to compute one or two layers of voabulary for a bag of binaryd descriptors approach
3: GlassPainting. This is the code for the Google Glass painting retrieval.

To run the Google Glass part:

1. Have installed OpenCV library on your computer, and (if need by your platform) the dirent.h library. Have installed also the Android development kit, the Glass development kit, the OpenCV Android SDK, Android NDK, and have OpenCV installed on the Glass. If you're able to run the OpenCV face detection sample on your Glass, then you should be just fine.

2. Get a database of paintings.

3. Run the computeFeatures() function of the EvaluateDescriptors project on the paintings.

4. (optionnal, if you want to have the bag of features retrieval.) Run the computeCenters(...), bag(...) and upToBottom(...) functions of the Clustering project. These functions are respectevely to build a first layer of the vocabulary, to represent each of your paintings by a histogram of words and to compute a second layer of the vocabulary.

5. Store what you have computed on your Glass, using e.g. the command "adb push..."

6. Connect your glass, compile and run GlassPainting. Swith the boolean "bag" to use the one-to-one approach (bag=false) or the bag of features approach (bag=true).

Any question or comment? Feel free to contact me, at theophile.dalens*at*gmail.com
