#include <ros/ros.h> 
#include <std_msgs/Float64.h> 
#include <std_msgs/Float64MultiArray.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cmath>
#include <std_msgs/Float64MultiArray.h>
using namespace cv;

double init_yaw = 0.0;
double current_yaw = 0.0;
double suggested_x = 0.0;
double suggested_y = 0.0;
bool updated_1 = false;
bool updated_2 = false;
bool init_recored = false;

void yawCallback(const std_msgs::Float64& msg)
{
	if(!updated_1)
	{
		current_yaw = msg.data;
		updated_1 = true;
	}

	if(!init_recored)
	{
		init_yaw = msg.data;
		init_recored = true;
	}
	
} 

void vecCallback(const std_msgs::Float64MultiArray& msg)
{
	if(!updated_2)
	{
		suggested_x = msg.data[0];
		suggested_y = msg.data[1];
		updated_2 = true;
	}
	
}
 
int main(int argc, char **argv) 
{ 
	ros::init(argc,argv,"radar"); 
	ros::NodeHandle n; 

	ros::Subscriber yaw_sub= n.subscribe("/firefly/current_yaw",1,yawCallback); 
	ros::Subscriber vec_sub= n.subscribe("/firefly/suggested_vec",1,vecCallback); 

	namedWindow( "Compass", CV_WINDOW_AUTOSIZE );

	ros::Publisher s_vec_body_pub = n.advertise<std_msgs::Float64MultiArray>("/firefly/s_vec_body", 10);
	ros::Publisher s_vec_init_pub = n.advertise<std_msgs::Float64MultiArray>("/firefly/s_vec_init", 10);

	std_msgs::Float64MultiArray s_vec_body;
	s_vec_body.data.resize(2);

	std_msgs::Float64MultiArray s_vec_init;
	s_vec_init.data.resize(2);

    ros::Rate loop_rate(20);
    while(ros::ok())
    {
    	if(updated_1 && updated_2)
    	{
    		Mat img(300, 300, CV_8UC3, Scalar(0,0,0));
	    	Point p(150, 150);
	    	circle(img, p, 100, Scalar(0, 255, 0), 3);
	    	circle(img, p, 5, Scalar(0, 0, 255), 3);
	    	line(img, Point(150, 270), Point(150, 30), Scalar(255, 20, 0), 3);
	    	line(img, Point(140, 40), Point(150, 30), Scalar(255, 20, 0), 3);
	    	line(img, Point(160, 40), Point(150, 30), Scalar(255, 20, 0), 3);

	    	/* Convert to body coordinate */
	    	double suggested_body_x = suggested_x * cos(current_yaw) + suggested_y * sin(current_yaw); 
	    	double suggested_body_y = -suggested_x * sin(current_yaw) + suggested_y * cos(current_yaw);
	    	s_vec_body.data[0] = suggested_body_x;
	    	s_vec_body.data[1] = suggested_body_y;
	    	s_vec_body_pub.publish(s_vec_body);

	    	/* Convert to a initialized body coordinate (for local map) */
	    	double suggested_init_x = suggested_x * cos(init_yaw) + suggested_y * sin(init_yaw); 
	    	double suggested_init_y = -suggested_x * sin(init_yaw) + suggested_y * cos(init_yaw);
	    	s_vec_init.data[0] = suggested_init_x;
	    	s_vec_init.data[1] = suggested_init_y;
	    	s_vec_init_pub.publish(s_vec_init);

	    	
	    	/* Draw suggested direction */
	    	Point p_dsr( -suggested_body_y* 140 + 150 , -suggested_body_x * 140 + 150);
	    	line(img, p, p_dsr, Scalar(0, 0, 255), 4);	    	

	    	imshow("Compass", img);
	    	waitKey(5);

	    	updated_1 = false;
	    	updated_2 = false;
    	}

    	ros::spinOnce(); 
    	loop_rate.sleep();
    }

	return 0;
} 
