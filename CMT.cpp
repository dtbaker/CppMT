#include "CMT.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <math.h>

namespace cmt {

//http://stackoverflow.com/questions/36691050/opencv-3-list-of-available-featuredetectorcreate-and-descriptorextractorc
enum string_code{
    dBrisk, // detector + descriptor
    dOrb, // detector + descriptor
    dMser, // detector
    dFast, // detector
    dAgast, // detector
    dGTTT, // detector
    dSimpleBlog, // detector (fails on loosing pbject)
    dKaze, // detector + descriptor (not sure how to use)
    dAkaze, // detector + descriptor (not sure how to use)
    dFreak, // descriptor
    dStar, // detector
    dBrief, // descriptor
    dLucid, // descriptor
    dLatch, // descriptor
    dDaisy, // descriptor
    dMsd, // detector
    dSift, // detector + descriptor
    dSurf // detector + descriptor (HORRIBY SLOW ON PI)
};
string_code switch_string_input(std::string const& inString){
    if (inString == "BRISK") return dBrisk;
    if (inString == "ORB") return dOrb;
    if (inString == "MSER") return dMser;
    if (inString == "FAST") return dFast;
    if (inString == "AGAST") return dAgast;
    if (inString == "GFFT") return dGTTT;
    if (inString == "SimpleBlobDetector") return dSimpleBlog;
    if (inString == "FREAK") return dFreak;
    if (inString == "BRIEF") return dBrief;
    if (inString == "MSD") return dMsd;
    if (inString == "LUCID") return dLucid;
    if (inString == "LATCH") return dLatch;
    if (inString == "DAISY") return dDaisy;
    if (inString == "SIFT") return dSift;
    if (inString == "SURF") return dSurf;
}
void CMT::initialize(const Mat im_gray, const Rect rect, string tracker_name, int & threshold_value)
{
    initialized = false;
    name = tracker_name;
    FILE_LOG(logDEBUG) << "CMT::initialize() call";

    FILE_LOG(logDEBUG) << " threshold: " << threshold_value;

    //Remember initial size
    size_initial = rect.size();

    //Remember initial image
    im_prev = im_gray;

    //Compute center of rect
    Point2f center = Point2f(rect.x + rect.width / 2.0, rect.y + rect.height / 2.0);

    //Initialize rotated bounding box
    bb_rot = RotatedRect(center, size_initial, 0.0);

    //Initialize detector and descriptor
#if CV_MAJOR_VERSION > 2
    // see here for available feature detectors: http://stackoverflow.com/questions/36691050/opencv-3-list-of-available-featuredetectorcreate-and-descriptorextractorc
    switch(switch_string_input(str_detector)){
        case dBrisk:
            detector = cv::BRISK::create();
            break;
        case dOrb:
            detector = cv::ORB::create();
            break;
        case dMser:
            detector = cv::MSER::create();
            break;
        case dFast:
            detector = cv::FastFeatureDetector::create();
            break;
        case dAgast:
            detector = cv::AgastFeatureDetector::create();
            break;
        case dGTTT:
            detector = cv::GFTTDetector::create();
            break;
        case dMsd:
            detector = cv::xfeatures2d::MSDDetector::create();
            break;
        case dSimpleBlog:
            detector = cv::SimpleBlobDetector::create();
            break;
        case dSift:
            detector = cv::xfeatures2d::SIFT::create();
            break;
        case dSurf:
            detector = cv::xfeatures2d::SURF::create();
            break;
        default:
            detector = cv::FastFeatureDetector::create();
    }
    switch(switch_string_input(str_descriptor)){
        case dBrisk:
            descriptor = cv::BRISK::create();
            break;
        case dOrb:
            descriptor = cv::ORB::create();
            break;
        case dFreak:
            descriptor = cv::xfeatures2d::FREAK::create();
            break;
        case dLucid:
            descriptor = cv::xfeatures2d::LUCID::create(10, 10);
            break;
        case dDaisy:
            descriptor = cv::xfeatures2d::DAISY::create();
            break;
        case dLatch:
            descriptor = cv::xfeatures2d::LATCH::create();
            break;
        case dSift:
            descriptor = cv::xfeatures2d::SIFT::create();
            break;
        case dSurf:
            descriptor = cv::xfeatures2d::SURF::create();
            break;
        default:
            descriptor = cv::BRISK::create();
    }

#else
    detector = FeatureDetector::create(str_detector);
    descriptor = DescriptorExtractor::create(str_descriptor);
    FILE_LOG(logDEBUG) << "OpenCV 3: " << str_detector << "  Feature Detector.";
    FILE_LOG(logDEBUG) << "OpenCV 3: " << str_descriptor << " Descriptor Extractor.";
#endif


    //Get initial keypoints in whole image and compute their descriptors
    vector<KeyPoint> keypoints;
    detector->detect(im_gray, keypoints);

    FILE_LOG(logDEBUG) << keypoints.size() << " total keypoints.";

    //Divide keypoints into foreground and background keypoints according to selection
    vector<KeyPoint> keypoints_fg;
    vector<KeyPoint> keypoints_bg;

    for (size_t i = 0; i < keypoints.size(); i++)
    {
        KeyPoint k = keypoints[i];
        Point2f pt = k.pt;
//This is adding whatever is in the rect to be tracked by the system. So the most interesting points are this.
        if (pt.x > rect.x && pt.y > rect.y && pt.x < rect.br().x && pt.y < rect.br().y)
        {
            keypoints_fg.push_back(k);
        }
        else
        {
            keypoints_bg.push_back(k);
        }

    }

    //Create foreground classes
    vector<int> classes_fg;
    classes_fg.reserve(keypoints_fg.size());
    for (size_t i = 0; i < keypoints_fg.size(); i++)
    {
        classes_fg.push_back(i);
    }

    //Compute foreground/background features
    Mat descs_fg;
    Mat descs_bg;
    descriptor->compute(im_gray, keypoints_fg, descs_fg);
    descriptor->compute(im_gray, keypoints_bg, descs_bg);

    //Only now is the right time to convert keypoints to points, as compute() might remove some keypoints
    vector<Point2f> points_fg;
    vector<Point2f> points_bg;

    for (size_t i = 0; i < keypoints_fg.size(); i++)
    {
        points_fg.push_back(keypoints_fg[i].pt);
    }

    for (size_t i = 0; i < keypoints_bg.size(); i++)
    {
        points_bg.push_back(keypoints_bg[i].pt);
    }

    FILE_LOG(logDEBUG) << points_fg.size() << " computed foreground points.";
    FILE_LOG(logDEBUG) << points_bg.size() << " computed background points.";

    //Create normalized points
    vector<Point2f> points_normalized;
    for (size_t i = 0; i < points_fg.size(); i++)
    {
        points_normalized.push_back(points_fg[i] - center);
    }

    if(threshold_value == -1){
        // we pick a 70% value for threshold based on full point count
        threshold_value = floor(points_fg.size() * 0.7);
        if( threshold_value < 2) threshold_value = 0;
        FILE_LOG(logDEBUG) << threshold_value << " new threshold value based on point count of " << points_fg.size();
    }

    threshold_original = points_fg.size();
    threshold_maybe = floor(threshold_value * 0.4);
    threshold = threshold_value; // set public variable.

    //Initialize matcher
    matcher.initialize(points_normalized, descs_fg, classes_fg, descs_bg, center);

    //Initialize consensus
    consensus.initialize(points_normalized);

    //Create initial set of active keypoints
    for (size_t i = 0; i < keypoints_fg.size(); i++)
    {
        points_active.push_back(keypoints_fg[i].pt);
        classes_active = classes_fg;
    }

    //Now set the number of active points;
    num_initial_keypoints = points_active.size();

    //Now lets store the values of the images.
    //Now we are using it to detect faces in the system;
    //So let's store the inital face results and the inital face.

    //That way the face recognition can be done in another function.

    initialRect = rect;
    imArchive = im_gray(rect);
    pointsArchive.assign(points_fg.begin(), points_fg.end());
    classesArchive.assign(classes_fg.begin(), classes_fg.end());

    FILE_LOG(logDEBUG) << "CMT::initialize() return";
    initialized = true;
}

void CMT::set_name(string tracker_name)
{
    name = tracker_name;
    identified = true;
}

void CMT::processFrame(Mat im_gray, int threshold) {

//    FILE_LOG(logDEBUG) << "CMT::processFrame() call";

//    FILE_LOG(logDEBUG) << " threshold: " << threshold;

    //Track keypoints
    vector<Point2f> points_tracked;
    vector<unsigned char> status;
    opticalflow_results = tracker.track(im_prev, im_gray, points_active, points_tracked, status, threshold);

    if (!opticalflow_results)
    {

        FILE_LOG(logDEBUG) << " NO OPTICAL FLOW RESULTS! ";

        if (threshold != 0)
        {
            tracker_lost = true;
            //return;
        }
        else
        {
            //To evaluate the cmt results
            tracker_lost = false; 
        }
    }

    FILE_LOG(logDEBUG) << points_tracked.size() << " tracked points.";
    FILE_LOG(logDEBUG) << classes_active.size() << " class active size.";

    //keep only successful classes
    vector<int> classes_tracked;
    for (size_t i = 0; i < classes_active.size(); i++)
    {
        if (i < status.size() && status[i])
        {
            classes_tracked.push_back(classes_active[i]);
        }

    }


    FILE_LOG(logDEBUG) << classes_tracked.size() << " class tracked size.";

    //Detect keypoints, compute descriptors
    vector<KeyPoint> keypoints;
    detector->detect(im_gray, keypoints);

    FILE_LOG(logDEBUG) << keypoints.size() << " keypoints found.";

    Mat descriptors;
    descriptor->compute(im_gray, keypoints, descriptors);

    //Match keypoints globally
    vector<Point2f> points_matched_global;
    vector<int> classes_matched_global;
    matcher.matchGlobal(keypoints, descriptors, points_matched_global, classes_matched_global);

    FILE_LOG(logDEBUG) << points_matched_global.size() << " points matched globally.";

    //Fuse tracked and globally matched points
    vector<Point2f> points_fused;
    vector<int> classes_fused;
    fusion.preferFirst(points_tracked, classes_tracked, points_matched_global, classes_matched_global,
                       points_fused, classes_fused);

    FILE_LOG(logDEBUG) << points_fused.size() << " points fused.";

    //Estimate scale and rotation from the fused points
    float scale;
    float rotation;
    consensus.estimateScaleRotation(points_fused, classes_fused, scale, rotation);

    FILE_LOG(logDEBUG) << "scale " << scale << ", " << "rotation " << rotation;

    //Find inliers and the center of their votes
    Point2f center;
    vector<Point2f> points_inlier;
    vector<int> classes_inlier;
    consensus.findConsensus(points_fused, classes_fused, scale, rotation,
                            center, points_inlier, classes_inlier);

    FILE_LOG(logDEBUG) << points_inlier.size() << " inlier points.";
    FILE_LOG(logDEBUG) << "center " << center;

    //Match keypoints locally
    vector<Point2f> points_matched_local;
    vector<int> classes_matched_local;
    matcher.matchLocal(keypoints, descriptors, center, scale, rotation, points_matched_local, classes_matched_local);

    FILE_LOG(logDEBUG) << points_matched_local.size() << " points matched locally.";

    //Assing the active points in the space.

    //Clear active points
    points_active.clear();
    classes_active.clear();

    //Fuse locally matched points and inliers

    fusion.preferFirst(points_matched_local, classes_matched_local, points_inlier, classes_inlier, points_active, classes_active);
//    points_active = points_fused;
//    classes_active = classes_fused;
    num_active_keypoints = points_active.size();
    FILE_LOG(logDEBUG) << points_active.size() << " final fused points.";

    for (size_t i = 0; i < classesArchive.size(); i++)
    {
        FILE_LOG(logDEBUG) << " - ARCHIVE: " << classesArchive[i];
    }
    for (size_t i = 0; i < classes_active.size(); i++)
    {
        FILE_LOG(logDEBUG) << " - active: " << classes_active[i];
    }

    //TODO: Use theta to suppress result
    bb_rot = RotatedRect(center,  size_initial * scale, rotation / CV_PI * 180);

    //Remember current image
    im_prev = im_gray;


//    FILE_LOG(logDEBUG) << "CMT::processFrame() return";
}

} /* namespace CMT */
