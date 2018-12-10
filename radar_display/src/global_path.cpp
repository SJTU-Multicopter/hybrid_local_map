#include <ros/ros.h> 
#include <std_msgs/Float64MultiArray.h> 
#include <std_msgs/Float64.h> 
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Point.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cmath>
#include <iostream>

using namespace cv;

/* Values about map and route */
const double points[12][2] = {{-15, -12.7}, {-23, -6.6}, {-16, -6.5}, {13, -6.8}, {-23, 1.3}, {-16.1, 1.3}, 
						{-4, 1.3}, {13, 1.3}, {23, 1.3}, {-3.8, 6.8}, {12.83, 6.7}, {23, 5}};

const double spawn_position[2] = {-16, -6.5};

const int point_num = 5;
const int route[point_num] ={2, 3, 7, 8, 11}; 

const double close_dist = 2.0;



/* Variables */
double position[3]={0.0, 0.0, 0.0};
double angle[3] = {0.0, 0.0, 0.0};

void odometryCallback(const nav_msgs::Odometry& msg)
{
	position[0] = msg.pose.pose.position.x + spawn_position[0];
	position[1] = msg.pose.pose.position.y + spawn_position[1];
	position[2] = msg.pose.pose.position.z;

	float q0 = msg.pose.pose.orientation.x;
	float q1 = msg.pose.pose.orientation.y;
	float q2 = msg.pose.pose.orientation.z;
	float q3 = msg.pose.pose.orientation.w;

	/* Pitch roll may be needed for MAVs */
	angle[2] = atan2(2*q3*q2 + 2*q0*q1, -2*q1*q1 - 2*q2*q2 + 1);  // Yaw

}


int main(int argc, char **argv) 
{ 
	ros::init(argc,argv,"global_path"); 
	ros::NodeHandle n; 

	ros::Subscriber yaw_sub= n.subscribe("/odom",1,odometryCallback); 

	namedWindow( "Compass", CV_WINDOW_AUTOSIZE );

	ros::Publisher target_pos_pub = n.advertise<geometry_msgs::Point>("/radar/target_point", 1);  // Gloabl coordinate, not robot odom coord
	ros::Publisher current_pos_pub = n.advertise<geometry_msgs::Point>("/radar/current_point", 1); // Gloabl coordinate, not robot odom coord
	ros::Publisher target_yaw_pub = n.advertise<std_msgs::Float64>("/radar/target_yaw", 1); // Gloabl coordinate, same with robot odom coord
	ros::Publisher current_yaw_pub = n.advertise<std_msgs::Float64>("/radar/current_yaw", 1);
	ros::Publisher delt_yaw_pub = n.advertise<std_msgs::Float64>("/radar/delt_yaw", 1);
    ros::Publisher direction_pub = n.advertise<std_msgs::Float64MultiArray>("/radar/direction", 1);

	std_msgs::Float64 target_yaw;
	std_msgs::Float64 current_yaw;
	std_msgs::Float64 delt_yaw;
    std_msgs::Float64MultiArray direction;

	geometry_msgs::Point target_point;
	geometry_msgs::Point current_point;

	int route_point_counter = 0;

	double target_x = points[route[route_point_counter]][0];
	double target_y = points[route[route_point_counter]][1];

    ros::Rate loop_rate(20);

    while(ros::ok())
    {
    	/* Close detection */
    	double dist_x = sqrt((target_x - position[0])*(target_x - position[0]) + (target_y - position[1])*(target_y - position[1]));
    	if(dist_x < close_dist) 
    	{
    		route_point_counter += 1;
    		if(route_point_counter >= point_num)
    		{
    			std::cout<< " You achieved the target!! Mission completed!!" << std::endl;
    			break;
    		}

    		target_x = points[route[route_point_counter]][0];
			target_y = points[route[route_point_counter]][1];
    	}

    	/* Calculate target yaw */
    	double delt_x = target_x - position[0];
    	double delt_y = target_y - position[1];

    	double yaw_t = atan2(delt_y, delt_x);

    	/* Calculate delt yaw */
    	double delt_yaw_value = 0.0;

        double delt_yaw_direct = yaw_t - angle[2];
        double delt_yaw_direct_abs = std::fabs(delt_yaw_direct);
        double sup_yaw_direct_abs = 2*M_PI - delt_yaw_direct_abs;

        if(delt_yaw_direct_abs < sup_yaw_direct_abs)
            delt_yaw_value = delt_yaw_direct;
        else
            delt_yaw_value = - sup_yaw_direct_abs * delt_yaw_direct / delt_yaw_direct_abs;

        /* Calculate diraction */
        direction.data.clear();
        if(delt_yaw_value > -M_PI/4.0 && delt_yaw_value < M_PI/4.0)  // Move forward
        {
            direction.data.push_back(1.0);
            direction.data.push_back(0.0);
            direction.data.push_back(0.0);
            direction.data.push_back(0.0);
        }
        else if(delt_yaw_value >= -3*M_PI/4.0 && delt_yaw_value <= -M_PI/4.0)  // Turn right
        {
            direction.data.push_back(0.0);
            direction.data.push_back(0.0);
            direction.data.push_back(0.0);
            direction.data.push_back(1.0);
        }
        else if(delt_yaw_value >= M_PI/4.0 && delt_yaw_value <= 3*M_PI/4.0) // Turn left
        {
            direction.data.push_back(0.0);
            direction.data.push_back(0.0);
            direction.data.push_back(1.0);
            direction.data.push_back(0.0);
        }
        else  // Move backward
        {
            direction.data.push_back(0.0);
            direction.data.push_back(1.0);
            direction.data.push_back(0.0);
            direction.data.push_back(0.0);
        }
        direction_pub.publish(direction);

    	/* Update and publish*/
    	target_point.x = target_x;
    	target_point.y = target_y;
    	target_point.z = 0.0;
    	target_pos_pub.publish(target_point);

    	current_point.x = position[0];
    	current_point.y = position[1];
    	current_point.z = 0.0;
    	current_pos_pub.publish(current_point);

    	target_yaw.data = yaw_t;
    	target_yaw_pub.publish(target_yaw);

    	current_yaw.data = angle[2];
    	current_yaw_pub.publish(current_yaw);

    	delt_yaw.data = delt_yaw_value;
    	delt_yaw_pub.publish(delt_yaw);


    	/* Convert to body coordinate */
    	double suggested_body_x = delt_x * cos(angle[2]) + delt_y * sin(angle[2]); 
    	double suggested_body_y = -delt_x * sin(angle[2]) + delt_y * cos(angle[2]);

    	/* Draw radar */
    	Mat img(300, 300, CV_8UC3, Scalar(0,0,0));
    	Point p(150, 150);
    	circle(img, p, 60, Scalar(0, 255, 0), 10);
    	circle(img, p, 5, Scalar(0, 0, 255), 3);
    	line(img, Point(150, 270), Point(150, 30), Scalar(255, 20, 0), 3);
    	line(img, Point(140, 40), Point(150, 30), Scalar(255, 20, 0), 3);
    	line(img, Point(160, 40), Point(150, 30), Scalar(255, 20, 0), 3);

    	Point p_dsr( -suggested_body_y* 140 + 150 , -suggested_body_x * 140 + 150);
    	line(img, p, p_dsr, Scalar(0, 0, 255), 4);

    	imshow("Compass", img);
    	waitKey(5);

    	ros::spinOnce(); 
    	loop_rate.sleep();
    }

	return 0;
} 

