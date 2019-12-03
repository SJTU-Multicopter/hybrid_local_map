//
// Created by cc on 2019/12/3.
//
#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <std_msgs/Float64MultiArray.h>
#include <math.h>

void rotateVector(cv::Point &center, cv::Point &start_point, float angle, cv::Point &end_point)
{
    cv::Point start_vector = start_point - center;
    cv::Point new_point_vector;
    new_point_vector.x = cos(angle)*start_vector.x + sin(angle)*start_vector.y;
    new_point_vector.y = -sin(angle)*start_vector.x + cos(angle)*start_vector.y;
    end_point = new_point_vector + center;
}

void costHeadObjectsCallback(const std_msgs::Float64MultiArray &msg)
{
    int num = msg.data.size();
    int rows = 480;
    int cols = 480;
    cv::Point center = cv::Point(cols/2, rows/2);
    cv::Mat background = cv::Mat::zeros(rows, cols, CV_8UC3);
    float angle_one_piece = 2*3.14159/num;

    /** Map color to 0, 255 **/
    double min_value = 1000000.f;
    double max_value = -1000000.f;
    for(auto & value_i : msg.data){
        if(value_i < min_value){
            min_value = value_i;
        }
        if(value_i > max_value){
            max_value = value_i;
        }
    }
    double delt_value = (max_value - min_value) / 250;

    cv::Point start_point = cv::Point(cols/2, rows/4*3);
    for(int i=0; i<num; i++)
    {
        float delt_angle_rad = angle_one_piece * i;
        cv::Point middle_point, left_point, right_point;
        rotateVector(center, start_point, delt_angle_rad,middle_point);
        rotateVector(center, middle_point, angle_one_piece/2.f,left_point);
        rotateVector(center, middle_point, -angle_one_piece/2.f,right_point);

        std::vector<cv::Point> contour;
        contour.push_back(center);
        contour.push_back(left_point);
        contour.push_back(right_point);

        std::vector<std::vector<cv::Point >> contours;
        contours.push_back(contour);

        int color = (int)(250 - (msg.data[i] - min_value)/delt_value) + 5;

        cv::polylines(background, contours, true, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
        cv::fillPoly(background, contours, cv::Scalar(0, 0, color));
    }

    cv::imshow("costHeadObjects", background);
    cv::waitKey(2);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "display_costs");
    ros::NodeHandle nh;
    ros::Subscriber objects_sub = nh.subscribe("/head_cost/cost_head_objects", 1, costHeadObjectsCallback);

    ros::spin();
}