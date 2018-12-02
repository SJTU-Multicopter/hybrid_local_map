#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/Twist.h>
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <cv_bridge/cv_bridge.h>

using namespace message_filters;
using namespace std;

#define MASK_DIST 6.4

float linear_v = 0.f;
float angular_v = 0.f;
float linear_v_abs_max = 1.0;
float angular_v_abs_max = 1.0;
const int image_width = 640;
const int image_height = 480;

void callback(const sensor_msgs::ImageConstPtr& depth, const sensor_msgs::ImageConstPtr& rgb)
{
	// Read from sensor msg
	cv_bridge::CvImagePtr rgb_ptr;
	try
	{
		rgb_ptr = cv_bridge::toCvCopy(rgb, sensor_msgs::image_encodings::BGR8);
	}
	catch (cv_bridge::Exception& e1)
	{
		ROS_ERROR("cv_bridge 1 exception: %s", e1.what());
		return;
	}

	cv_bridge::CvImagePtr depth_ptr;
	try
	{
		depth_ptr = cv_bridge::toCvCopy(depth);
	}
	catch (cv_bridge::Exception& e2)
	{
		ROS_ERROR("cv_bridge 2 exception: %s", e2.what());
		return;
	}

	cv::Mat rgb_img = rgb_ptr->image;
	cv::Mat depth_img = depth_ptr->image;

	// Merge by iteration
	int nr=depth_img.rows;
	int nc=depth_img.cols;

	int channel = rgb_img.channels();

	for(int i=0; i<nr ;i++)
	{
		const float* inDepth = depth_img.ptr<float>(i); // float
		uchar* inRgb = rgb_img.ptr<uchar>(i);

		for(int j=0; j<nc; j++)
		{
			// cout << inDepth[j] << " ";
			if(inDepth[j] != inDepth[j] || inDepth[j] > MASK_DIST)
			{
				inRgb[channel*j] = 0;
				inRgb[channel*j + 1] = 0;
				inRgb[channel*j + 2] = 0;
			}			
		}
	}

	// add panels
	int linear_l_x = 10;
	int linear_l_y = 40;
	int linear_r_x = 20;
	int linear_r_y = image_height - 40;

	int angular_l_x = 40;
	int angular_l_y = image_height - 20;
	int angular_r_x = image_width - 40;
	int angular_r_y = image_height - 10;

	int linear_range = linear_r_y - linear_l_y;
	int angular_range = angular_r_x - angular_l_x;

	int linear_v_bar_center_x = (linear_l_x + linear_r_x) / 2;
	int linear_v_bar_center_y = -(float)linear_v / (float)linear_v_abs_max * (float)linear_range / 2.0 + (linear_l_y + linear_r_y) / 2;
	int angular_v_bar_center_x = -(float)angular_v / (float)angular_v_abs_max * (float)angular_range / 2.0 + (angular_l_x + angular_r_x) / 2;
	int angular_v_bar_center_y = (angular_l_y + angular_r_y) / 2;

	int bar_size_half = 8;

	cv::Mat show_img;
	rgb_img.copyTo(show_img);

	cv::rectangle(show_img, cv::Point(linear_l_x, linear_l_y), cv::Point(linear_r_x, linear_r_y), cv::Scalar(200, 20, 0), 2, 1 ,0);
	cv::rectangle(show_img, cv::Point(angular_l_x, angular_l_y), cv::Point(angular_r_x, angular_r_y), cv::Scalar(200, 20, 0), 2, 1 ,0);

	cv::rectangle(show_img, cv::Point(linear_v_bar_center_x - bar_size_half, linear_v_bar_center_y - bar_size_half),
		cv::Point(linear_v_bar_center_x + bar_size_half, linear_v_bar_center_y + bar_size_half), cv::Scalar(0, 0, 255), -1, 1 ,0);
	cv::rectangle(show_img, cv::Point(angular_v_bar_center_x - bar_size_half, angular_v_bar_center_y - bar_size_half),
		cv::Point(angular_v_bar_center_x + bar_size_half, angular_v_bar_center_y + bar_size_half), cv::Scalar(0, 255, 0), -1, 1 ,0);

	cv::imshow("merged", show_img);
	cv::waitKey(5);
}


void callBackCmd(const geometry_msgs::Twist::ConstPtr& data)
{
    linear_v = data->linear.x;
    angular_v = data->angular.z;
}


int main(int argc, char** argv)
{
	ros::init(argc, argv, "lantern_with_pannel");
	ros::NodeHandle nh;

	message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "/camera/depth/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "/camera/rgb/image_raw", 1);

    ros::Subscriber Cmd_sub = nh.subscribe("/mobile_base/commands/velocity", 2, callBackCmd);

    typedef sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
    Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), depth_sub, rgb_sub);
    sync.registerCallback(boost::bind(&callback, _1, _2));

    ros::spin();

	return 0;
}

