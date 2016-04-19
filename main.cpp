// ./cmt --threshold -1 --detector SIFT --descriptor SURF


#include "CMT.h"
#include "gui.h"

#include "zhelpers.hpp"
#include <vector>
#include <string>
#include <algorithm>
#include <string.h>

#include <libmemcached/memcached.hpp>


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <unistd.h>
#include <sys/stat.h>

#include <raspicam/raspicam_cv.h>

#ifdef __GNUC__
#include <getopt.h>
#else
#include "getopt/getopt.h"
#endif

using cmt::CMT;
using cv::imread;
using cv::namedWindow;
using cv::destroyWindow;
using cv::startWindowThread;
using cv::Scalar;
using cv::VideoCapture;
using cv::waitKey;
using raspicam::RaspiCam_Cv;
using std::cerr;
using std::istream;
using std::ifstream;
using std::stringstream;
using std::ofstream;
using std::cout;
using std::min_element;
using std::max_element;
using std::endl;
using ::atof;

using namespace std;

// https://github.com/awslabs/aws-elasticache-cluster-client-libmemcached/blob/master/tests/cpp_example.cc


class MemCacheWrapper
{
public:

    memcache::Memcache *our_client = new memcache::Memcache("127.0.0.1:11211");

    static MemCacheWrapper &singleton()
    {
        static MemCacheWrapper instance;
        return instance;
    }

    void set(const std::string &key,
             const std::vector<char> &value)
    {
        time_t expiry= 0;
        uint32_t flags= 0;
        our_client->set(key, value, expiry, flags);
    }


    std::vector<char> get(const std::string &key)
    {
        std::vector<char> ret_value;
        our_client->get(key, ret_value);
        return ret_value;
    }

    void remove(const std::string &key)
    {
        our_client->remove(key);
    }


private:

    ~MemCacheWrapper()
    {
        delete our_client;
    }

    //MemCacheWrapper(const MemCacheWrapper&);
};


static string WIN_NAME = "CMT";
static string OUT_FILE_COL_HEADERS =
    "Frame,Timestamp (ms),Active points,"\
    "Bounding box centre X (px),Bounding box centre Y (px),"\
    "Bounding box width (px),Bounding box height (px),"\
    "Bounding box rotation (degrees),"\
    "Bounding box vertex 1 X (px),Bounding box vertex 1 Y (px),"\
    "Bounding box vertex 2 X (px),Bounding box vertex 2 Y (px),"\
    "Bounding box vertex 3 X (px),Bounding box vertex 3 Y (px),"\
    "Bounding box vertex 4 X (px),Bounding box vertex 4 Y (px)";

vector<float> getNextLineAndSplitIntoFloats(istream& str)
{
    vector<float>   result;
    string                line;
    getline(str,line);

    stringstream          lineStream(line);
    string                cell;
    while(getline(lineStream,cell,','))
    {
        result.push_back(atof(cell.c_str()));
    }
    return result;
}

int display(Mat im, CMT & cmt)
{
    Mat im_display;
    cvtColor(im, im_display, CV_GRAY2BGR);

    //Visualize the output
    for(size_t i = 0; i < cmt.points_active.size(); i++)
    {
        circle(im_display, cmt.points_active[i], 2, cmt.points_active.size() < cmt.threshold ? Scalar(0,0,255) : Scalar(0,255,0));
    }

    if(cmt.points_active.size() >= cmt.threshold_maybe){
        Point2f vertices[4];
        cmt.bb_rot.points(vertices);
        for (int i = 0; i < 4; i++)
        {
            line(im_display, vertices[i], vertices[(i+1)%4], cmt.points_active.size() >= cmt.threshold ? Scalar(0,255,0) : Scalar(0,0,255));
        }
    }

    imshow(WIN_NAME, im_display);

    return waitKey(5);
}

string write_rotated_rect(RotatedRect rect)
{
    Point2f verts[4];
    rect.points(verts);
    stringstream coords;

    coords << rect.center.x << "," << rect.center.y << ",";
    coords << rect.size.width << "," << rect.size.height << ",";
    coords << rect.angle << ",";

    for (int i = 0; i < 4; i++)
    {
        coords << verts[i].x << "," << verts[i].y;
        if (i != 3) coords << ",";
    }

    return coords.str();
}

inline bool file_exists_test (const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}

int main(int argc, char **argv)
{
    //Create a CMT object
    CMT cmt;

    //Initialization bounding box
    Rect rect;

    //Parse args
    int challenge_flag = 0;
    int loop_flag = 0;
    int verbose_flag = 0;
    int bbox_flag = 0;
    int skip_frames = 0;
    int skip_msecs = 0;
    int quiet_flag = 0;
    int output_flag = 0;
    int threshold_value = 0;
    string input_path;
    string output_path;

    const int detector_cmd = 1000;
    const int descriptor_cmd = 1001;
    const int bbox_cmd = 1002;
    const int frameimage_cmd = 10023;
    const int no_scale_cmd = 1003;
    const int with_rotation_cmd = 1004;
    const int skip_cmd = 1005;
    const int skip_msecs_cmd = 1006;
    const int output_file_cmd = 1007;
    const int threshold_cmd = 1008;

    struct option longopts[] =
    {
        //No-argument options
        {"challenge", no_argument, &challenge_flag, 1},
        {"loop", no_argument, &loop_flag, 1},
        {"verbose", no_argument, &verbose_flag, 1},
        {"no-scale", no_argument, 0, no_scale_cmd},
        {"with-rotation", no_argument, 0, with_rotation_cmd},
        {"quiet", no_argument, &quiet_flag, 1},
        //Argument options
        {"frameimage", required_argument, 0, frameimage_cmd},
        {"bbox", required_argument, 0, bbox_cmd},
        {"detector", required_argument, 0, detector_cmd},
        {"descriptor", required_argument, 0, descriptor_cmd},
        {"output-file", required_argument, 0, output_file_cmd},
        {"skip", required_argument, 0, skip_cmd},
        {"skip-msecs", required_argument, 0, skip_msecs_cmd},
        {"threshold", required_argument, 0, threshold_cmd},
        {0, 0, 0, 0}
    };

    int index = 0;
    int c;
    while((c = getopt_long(argc, argv, "v", longopts, &index)) != -1)
    {
        switch (c)
        {
            case 'v':
                verbose_flag = true;
                break;
            case bbox_cmd:
                {
                    //TODO: The following also accepts strings of the form %f,%f,%f,%fxyz...
                    string bbox_format = "%f,%f,%f,%f";
                    float x,y,w,h;
                    int ret = sscanf(optarg, bbox_format.c_str(), &x, &y, &w, &h);
                    if (ret != 4)
                    {
                        cerr << "bounding box must be given in format " << bbox_format << endl;
                        return 1;
                    }

                    bbox_flag = 1;
                    rect = Rect(x,y,w,h);
                }
                break;
            case frameimage_cmd:
                cmt.initial_frame_image = optarg;
                break;
            case detector_cmd:
                cmt.str_detector = optarg;
                break;
            case descriptor_cmd:
                cmt.str_descriptor = optarg;
                break;
            case output_file_cmd:
                output_path = optarg;
                output_flag = 1;
                break;
            case skip_cmd:
                {
                    int ret = sscanf(optarg, "%d", &skip_frames);
                    if (ret != 1)
                    {
                      skip_frames = 0;
                    }
                }
                break;
            case skip_msecs_cmd:
                {
                    int ret = sscanf(optarg, "%d", &skip_msecs);
                    if (ret != 1)
                    {
                      skip_msecs = 0;
                    }
                }
                break;
            case threshold_cmd:
                {
                    int ret = sscanf(optarg, "%d", &threshold_value);
                    if (ret != 1)
                    {
                      threshold_value = 0;
                    }
                }
                break;
            case no_scale_cmd:
                cmt.consensus.estimate_scale = false;
                break;
            case with_rotation_cmd:
                cmt.consensus.estimate_rotation = true;
                break;
            case '?':
                return 1;
        }

    }

    // Can only skip frames or milliseconds, not both.
    if (skip_frames > 0 && skip_msecs > 0)
    {
      cerr << "You can only skip frames, or milliseconds, not both." << endl;
      return 1;
    }

    //One argument remains
    if (optind == argc - 1)
    {
        input_path = argv[optind];
    }

    else if (optind < argc - 1)
    {
        cerr << "Only one argument is allowed." << endl;
        return 1;
    }

    //http://stackoverflow.com/questions/8263926/how-to-copy-stdstring-into-stdvectorchar
    string str = "12334434";
    vector<char> memcache_val(str.begin(), str.end());
    MemCacheWrapper::singleton().set("run_tracker", memcache_val);


    zmq::context_t context (1);
    zmq::socket_t socket (context, ZMQ_REQ);

    std::cout << "Connecting to robot server..." << std::endl;
    socket.connect ("tcp://localhost:5555");
    // send the message
    s_send (socket, "none");
    //  Get the reply.
    string reply = s_recv (socket);


    //Set up logging. Quiet takes preference over verbose.
    if (verbose_flag){
        FILELog::ReportingLevel() = verbose_flag ? logDEBUG : logINFO;
        Output2FILE::Stream() = stdout; //Log to stdout
    } else if (quiet_flag) {
        FILELog::ReportingLevel() = logERROR;
    } else {
        FILELog::ReportingLevel() = verbose_flag ? logDEBUG : logINFO;
        Output2FILE::Stream() = stdout; //Log to stdout
    }



    //Normal mode

    //Create window and allow preview if not in quiet mode
    bool show_preview;
    if (quiet_flag)
    {
        show_preview = false;
    }
    else
    {
    namedWindow(WIN_NAME);
        show_preview = true;
    }

    // VideoCapture cap;
    raspicam::RaspiCam_Cv Camera;
    Camera.set(CV_CAP_PROP_FRAME_WIDTH,  320 );
    Camera.set(CV_CAP_PROP_FRAME_HEIGHT, 240 );
    /*std::vector<char> input_width = MemCacheWrapper::singleton().get("input_width");
    std::string input_width_s(input_width.begin(),input_width.end());
    std::vector<char> input_height = MemCacheWrapper::singleton().get("input_height");
    std::string input_height_s(input_height.begin(),input_height.end());
    Camera.set(CV_CAP_PROP_FRAME_WIDTH,  atoi(input_width_s.c_str()) );
    Camera.set(CV_CAP_PROP_FRAME_HEIGHT, atoi(input_height_s.c_str()) );
    FILE_LOG(logDEBUG) << "Camera init size: " << atoi(input_width_s.c_str()) << " x " << atoi(input_height_s.c_str());*/
    Camera.set( CV_CAP_PROP_FORMAT, CV_8UC1 ); // CV_8UC3
    if (!Camera.open()) {cerr<<"Error opening the camera"<<endl;return -1;}

    sleep(4);
    //If no input was specified
//    if (input_path.length() == 0)
//    {
//        cap.open(0); //Open default camera device
//    }
//
//    //Else open the video specified by input_path
//    else
//    {
//        cap.open(input_path);
//
//        if (skip_frames > 0)
//        {
//          cap.set(CV_CAP_PROP_POS_FRAMES, skip_frames);
//        }
//
//        if (skip_msecs > 0)
//        {
//          cap.set(CV_CAP_PROP_POS_MSEC, skip_msecs);
//
//          // Now which frame are we on?
//          skip_frames = (int) cap.get(CV_CAP_PROP_POS_FRAMES);
//        }
//
//        show_preview = false;
//    }
//
//    //If it doesn't work, stop
//    if(!cap.isOpened())
//    {
//        cerr << "Unable to open video capture." << endl;
//        return -1;
//    }

    //Show preview until key is pressed
    while (show_preview)
    {

        cv::Mat preview;
        // cap >> preview;
        Camera.grab();
        Camera.retrieve ( preview );

        screenLog(preview, "Press a key to start selecting an object.");
        imshow(WIN_NAME, preview);

        char k = waitKey(10);
        if (k != -1) {
            show_preview = false;
        }
    }

    //Get initial image
    cv::Mat im0;
    //cap >> im0;
    if(cmt.initial_frame_image.length() > 0){
        cv::Mat image_read = cv::imread(cmt.initial_frame_image, 0);
        if(!image_read.data )                              // Check for invalid input
        {
            cout <<  "Could not open still image file" << std::endl ;
            return -1;
        }
        cv::Size resizesize(Camera.get(CV_CAP_PROP_FRAME_WIDTH),Camera.get(CV_CAP_PROP_FRAME_HEIGHT));
        cv::resize(image_read, im0, resizesize);
        cout << "Resized image from: " << image_read.size().width << " x " << image_read.size().height << std::endl;
        cout << "Resized image to: " << im0.size().width << " x " << im0.size().height << std::endl;
    }else{
        Camera.grab();
        Camera.retrieve ( im0 );
    }

    //If no bounding was specified, get it from user
    if (!bbox_flag)
    {
        if (quiet_flag)
        {
            startWindowThread(); //Cannot destroy window without this!
            namedWindow(WIN_NAME);
        rect = getRect(im0, WIN_NAME);
            destroyWindow(WIN_NAME);
        }
        else
        {
            rect = getRect(im0, WIN_NAME);
        }
    }

    FILE_LOG(logINFO) << "Using " << rect.x << "," << rect.y << "," << rect.width << "," << rect.height
        << " as initial bounding box.";

    //Convert im0 to grayscale
    Mat im0_gray;
    if (im0.channels() > 1) {
        cvtColor(im0, im0_gray, CV_BGR2GRAY);
    } else {
        im0_gray = im0;
    }

    //Initialize CMT
    string name = "foo";
    cmt.initialize(im0_gray, rect, name, threshold_value);

    int frame = skip_frames;

    //Open output file.
    ofstream output_file;

    if (output_flag)
    {
        //int msecs = (int) cap.get(CV_CAP_PROP_POS_MSEC);

        output_file.open(output_path.c_str());
        output_file << OUT_FILE_COL_HEADERS << endl;
        //output_file << frame << "," << msecs << ",";
        output_file << frame << ",";
        output_file << threshold_value << ",";
        output_file << cmt.points_active.size() << ",";
        output_file << write_rotated_rect(cmt.bb_rot) << endl;
    }

    //Main loop
    while (true)
    {

        frame++;

        cv::Mat im;

        //If loop flag is set, reuse initial image (for debugging purposes)
        if (loop_flag) im0.copyTo(im);
        else {
            // cap >> im; //Else use next image in stream
            Camera.grab();
            Camera.retrieve ( im );
        }

        if (im.empty()) break; //Exit at end of video stream

        Mat im_gray;
        if (im.channels() > 1) {
            cvtColor(im, im_gray, CV_BGR2GRAY);
        } else {
            im_gray = im;
        }

        //Let CMT process the frame
        cmt.processFrame(im_gray, threshold_value);

        // send the results to our robot server if they are good.
        // $bbox = "$left:$top|$right:$top|$right:$bottom|$left:$bottom|$confidence";
        if(cmt.num_active_keypoints > 0) {
            Point2f verts[4];
            cmt.bb_rot.points(verts);
            stringstream this_coords;
            // left:top
            this_coords << verts[1].x << ":" << verts[1].y << "|";
            // right:top
            this_coords << verts[2].x << ":" << verts[2].y << "|";
            // right:bottom
            this_coords << verts[3].x << ":" << verts[3].y << "|";
            // left:bottom
            this_coords << verts[0].x << ":" << verts[0].y << "|";
            // confidence
            this_coords << int(round((cmt.num_active_keypoints / cmt.threshold_original) * 100));
            FILE_LOG(logDEBUG) << "#" << frame << " sending: " << this_coords.str();



            // send the message
            s_send (socket, this_coords.str());
            //  Get the reply.
            string this_reply = s_recv (socket);
        }

        //Output.
        if (output_flag)
        {
            //int msecs = (int) cap.get(CV_CAP_PROP_POS_MSEC);
            //output_file << frame << "," << msecs << ",";
            output_file << frame << ",";
            output_file << cmt.points_active.size() << ",";
            output_file << write_rotated_rect(cmt.bb_rot) << endl;
        }
        else
        {
            //TODO: Provide meaningful output
            FILE_LOG(logDEBUG) << "";
            FILE_LOG(logINFO) << "#" << frame << " active: " << cmt.points_active.size();
        }

        if (!quiet_flag)
        {
            char key = display(im, cmt);
            if(key == 'q') break;
        }
        // check memcache that we're allowed to continue
        // this doesn't work :( no idea why.
        /*vector<char> continue_status = MemCacheWrapper::singleton().get("run_tracker");
        string foo(continue_status.begin(),continue_status.end());
        if(continue_status.empty() || continue_status != memcache_val){
            FILE_LOG(logDEBUG) << "# exit status was: " << foo << " = " << continue_status.size() << " orig: " << memcache_val.size();
            break;
        }*/

    }

    //Close output file.
    if (output_flag) output_file.close();

    Camera.release();


    return 0;
}
