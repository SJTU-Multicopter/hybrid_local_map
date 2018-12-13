//
// Created by lucasyu on 18-11-29.
//
#include <ros/ros.h>
#include <tf/transform_datatypes.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PointStamped.h>
#include <std_msgs/Float64.h>

#include <iostream>
#include <string>
#include <fstream>
#include <array>
#include <time.h>
#include <numeric>

using namespace std;
std::vector< std::vector<float> > data_pcl;
std::vector< std::vector<float> > data_uav;
std::vector< std::vector<float> > data_label;

ros::Time timestamp;
float vel_odom = 0.0f;
float angular_odom = 0.0f;
float position_odom_x = 0.0f;
float position_odom_y = 0.0f;
float position_odom_z = 0.0f;
float position_radar_x = 0.0f;
float position_radar_y = 0.0f;
float vel_smoother = 0.0f;
float angular_smoother = 0.0f;
float vel_teleop = 0.0f;
float angular_teleop = 0.0f;
float pos_target_x = 0.0f;
float pos_target_y = 0.0f;
float yaw_target = 0.0f;
float yaw_current = 0.0f;
float yaw_current_x = 0.0f;
float yaw_current_y = 0.0f;
float yaw_delt = 0.0f;
float yaw_forward = 0.0f;
float yaw_backward = 0.0f;
float yaw_leftward = 0.0f;
float yaw_rightward = 0.0f;


const int num_uavdata = 17;
const int num_label = 4;
const int img_width = 64;
//const int img_height = 64;
const int img_height_uplimit = 20;
const int img_height_downlimit = 4;
const int info_dim = 1;


ofstream outFile_uavdata;
ofstream outFile_labels;
ofstream outFile_pcd;
ofstream outFile_intensity_proportion;


void writeCsv(const std::vector<std::vector<float>> vec, const string& filename)
{
    // 写文件
    ofstream outFile;
    outFile.open(filename, ios::out); // 打开模式可省略
    for(int i_vec=0; i_vec<vec.size(); ++i_vec) {
        for (int j=0; j<vec[i_vec].size(); ++j)
        {
            char c[20];
            sprintf(c, "%f", vec[i_vec][j]);
            outFile << c << ",";
        }
        outFile << endl;
    }

    outFile.close();
}

void writeCsvOneLine(const std::vector<float> uav_data, const std::vector<float> labels)
{
    char c[20];
    for (int j=0; j<uav_data.size(); ++j)
    {
        sprintf(c, "%f", uav_data[j]);
        outFile_uavdata << c << ",";
    }

    outFile_uavdata << endl;

    for (int j=0; j<labels.size(); ++j)
    {
        sprintf(c, "%f", labels[j]);
        outFile_labels << c << ",";
    }
    outFile_labels << endl;
}

void CallbackPointCloud(const sensor_msgs::PointCloud2ConstPtr& cloud)
{
    static bool firstCome = true;
    static array<array<array<array<float, info_dim>, img_height_uplimit + img_height_downlimit>, img_width>, img_width>cloud_modified;
    static array<array<array<array<float, info_dim>, img_height_uplimit + img_height_downlimit>, img_width>, img_width>all_zeros;

    if(firstCome) {
        firstCome = false;
        for (auto & l3 : all_zeros) {
            for (auto & l2 : l3) {
                for (auto & l1 : l2) {
                    for (auto & f : l1) {
                        f = 0.0;  //unknown: 0
                    }
                }
            }
        }
    }

    cloud_modified = all_zeros;

    timestamp = cloud->header.stamp;
    float tmp_uav[num_uavdata] = {position_odom_x, position_odom_y, vel_odom, angular_odom,position_radar_x, 
                            position_radar_y, pos_target_x, pos_target_y, yaw_target,yaw_current, yaw_current_x, 
                            yaw_current_y, yaw_delt, yaw_forward, yaw_backward, yaw_leftward, yaw_rightward};
    float tmp_label[num_label] = {vel_smoother, angular_smoother, vel_teleop, angular_teleop};
    std::vector<float>data_label_tmp;
    std::vector<float>data_uav_tmp;
    data_uav_tmp.insert(data_uav_tmp.begin(), tmp_uav, tmp_uav+num_uavdata);
    data_label_tmp.insert(data_label_tmp.begin(), tmp_label, tmp_label+num_label);
    writeCsvOneLine(data_uav_tmp, data_label_tmp);
//    cout<<"reading odom cloud data..."<<endl;

    // convert cloud to pcl form
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::fromROSMsg(*cloud, *cloud_in);

    vector<int> intensity_count = {0,0,0,0,0,0,0,0};
    for (int i_pt=0; i_pt<cloud_in->points.size(); ++i_pt) {
        auto point = cloud_in->points[i_pt];
        
        float intensity;
        intensity = point.data_c[0]/7.f;

        int x_tmp;
        x_tmp = int((point.data[0] - position_odom_x) * 5 + 0.5f);
        int y_tmp;
        y_tmp = int((point.data[1] - position_odom_y) * 5 + 0.5f);
        int z_tmp;
        z_tmp = int((point.data[2] - position_odom_z) * 5 + 0.5f);

        if (abs(x_tmp) < img_width/2 && abs(y_tmp) < img_width/2 && z_tmp > -img_height_downlimit && z_tmp < img_height_uplimit)
        {
            int x_index = x_tmp + img_width / 2;
            int y_index = y_tmp + img_width / 2;
            int z_index = z_tmp + img_height_downlimit;
//            cout<<"indx and info: "<<x_index<<' '<<y_index<<' '<<z_index<<' '<< intensity << endl;
            cloud_modified[x_index][y_index][z_index][0] = intensity;

            // To count persentage of each semantic label
            int indensity_index;
            indensity_index = int(point.data_c[0] + 0.49f);
            intensity_count[indensity_index] += 1;
        }
    }

    // To count persentage of each semantic label
    char c_intensity[20];
    int intensity_sum = accumulate(intensity_count.begin(),intensity_count.end(),0);
    int rest_unkown = img_width * img_width * (img_height_uplimit + img_height_downlimit) - intensity_sum;

    for (int i_item=0; i_item<intensity_count.size();i_item++)
    {
        float item = intensity_count[i_item];

        float item_proportion = item / (img_width * img_width * (img_height_uplimit + img_height_downlimit));
        if (i_item == 1)     // unkown case
        {
            item_proportion = (item + rest_unkown) / (img_width * img_width * (img_height_uplimit + img_height_downlimit));
        }
        sprintf(c_intensity, "%f", item_proportion);
        outFile_intensity_proportion << c_intensity << ",";
    }
    outFile_intensity_proportion << endl;

    // Write to csv to save
    char c[20];
    for (int ii=0; ii<img_width; ++ii) {
        for (int jj = 0; jj < img_width; ++jj) {
            for (int kk = 0; kk < (img_height_uplimit+img_height_downlimit); ++kk) {
                sprintf(c, "%f", cloud_modified[ii][jj][kk][0]);
                outFile_pcd << c << ",";
            }
        }
    }

    outFile_pcd << endl;

    return;
}


void callBackOdom(const nav_msgs::OdometryConstPtr& odom)
{
    position_odom_x = odom->pose.pose.position.x;
    position_odom_y = odom->pose.pose.position.y;
    vel_odom = odom->twist.twist.linear.x / 0.8f;
    angular_odom = odom->twist.twist.angular.z / 0.8f;

}

void callBackRadar(const geometry_msgs::Point::ConstPtr& data)
{
    pos_target_x = data->x;
    pos_target_y = data->y;
}

void callBackTeleop(const geometry_msgs::Twist::ConstPtr& data)
{
    vel_teleop = data->linear.x / 0.8f;
    angular_teleop = data->angular.z / 0.8f;
}

void callBackCmdMobileBase(const geometry_msgs::Twist::ConstPtr& data)
{
    vel_smoother = data->linear.x / 0.8f;
    angular_smoother = data->angular.z / 0.8f;
}

void callBackTargetPos(const geometry_msgs::Point::ConstPtr& data)
{
    pos_target_x = data->x;
    pos_target_y = data->y;
}

void callBackTargetYaw(const std_msgs::Float64::ConstPtr& data)
{
    yaw_target = data->data / 3.15f;
}

void callBackCurrentYaw(const std_msgs::Float64::ConstPtr& data)
{
    yaw_current = data->data / 3.15f;
    yaw_current_x = cos(data->data);
    yaw_current_y = sin(data->data);
}

void callBackDeltYaw(const std_msgs::Float64::ConstPtr& data)
{
    yaw_delt = data->data;
    if (-3.15f/4.f < yaw_delt && yaw_delt < 3.15f/4.f){
        yaw_forward = 1.0f;
        yaw_backward = 0.0f;
        yaw_leftward = 0.0f;
        yaw_rightward = 0.0f;
    }
    else if ( 3.15f/4.f*3.f < yaw_delt || yaw_delt < -3.15f/4.f*3.f )
    {
        yaw_forward = 0.0f;
        yaw_backward = 1.0f;
        yaw_leftward = 0.0f;
        yaw_rightward = 0.0f;
    }
    else if ( 3.15f/4.f < yaw_delt && yaw_delt < 3.15f/4.f*3.f )
    {
        yaw_forward = 0.0f;
        yaw_backward = 0.0f;
        yaw_leftward = 1.0f;
        yaw_rightward = 0.0f;
    }
    else
    {
        yaw_forward = 0.0f;
        yaw_backward = 0.0f;
        yaw_leftward = 0.0f;
        yaw_rightward = 1.0f;
    }
    yaw_delt = data->data / 3.15f;

}


int main(int argc, char** argv)
{
    time_t t = time(0);
    char tmp[64];
    strftime( tmp, sizeof(tmp), "%Y_%m_%d_%X",localtime(&t) );
    char c_uav_data[100];
    sprintf(c_uav_data,"uav_data_%s.csv",tmp);
    cout<<"----- file uav data: "<< c_uav_data<<endl;
    char c_label_data[100];
    sprintf(c_label_data,"label_data_%s.csv",tmp);
    cout<<"----- file label data: "<< c_label_data<<endl;
    char c_pcl_data[100];
    sprintf(c_pcl_data,"pcl_data_%s.csv",tmp);
    cout<<"----- file pcl data: "<< c_pcl_data<<endl;
    char c_intensity_proportion[100];
    sprintf(c_intensity_proportion,"intensity_proportion_%s.csv",tmp);

    outFile_uavdata.open(c_uav_data, ios::out);
//    outFile_uavdata<<"position_odom_x"<<","<<"position_odom_y"<<","<<"vel_odom"<<","<<"angular_odom"
//           <<","<<"position_radar_x"<<","<<"position_radar_y"<<","<<"pos_target_x"
//           <<","<<"pos_target_y"<<","<<"yaw_target" <<","<<"yaw_current"<<","<<yaw_current_x<<","<<yaw_current_y<<","<<"yaw_delt"<<endl;

    outFile_labels.open(c_label_data, ios::out);
//    outFile_labels<<"vel_smoother"<<","<<"angular_smoother"<<","<<"vel_teleop"<<","<<"angular_teleop"<<endl;

    outFile_pcd.open(c_pcl_data, ios::out);
    outFile_intensity_proportion.open(c_intensity_proportion, ios::out);

    ros::init(argc, argv, "uavdataprocess");
    ros::NodeHandle nh;
    ros::Subscriber OdomCloud_sub = nh.subscribe("/ring_buffer/cloud_semantic", 2, CallbackPointCloud);
    ros::Subscriber Odom_sub = nh.subscribe("/odom", 2, callBackOdom);
    ros::Subscriber Radar_sub = nh.subscribe("/radar/current_point", 2, callBackRadar);
    ros::Subscriber Cmd_sub = nh.subscribe("/mobile_base/commands/velocity", 2, callBackCmdMobileBase);
    ros::Subscriber CmdSmoother_sub = nh.subscribe("/teleop_velocity_smoother/raw_cmd_vel", 2, callBackTeleop);
    ros::Subscriber TargetPos_sub = nh.subscribe("/radar/target_point", 2, callBackTargetPos);
    ros::Subscriber TargetYaw_sub = nh.subscribe("/radar/target_yaw", 2, callBackTargetYaw);
    ros::Subscriber CurrentYaw_sub = nh.subscribe("/radar/current_yaw", 2, callBackCurrentYaw);
    ros::Subscriber DeltYaw_sub = nh.subscribe("/radar/delt_yaw", 2, callBackDeltYaw);
    ros::spin();
    outFile_uavdata.close();
    outFile_labels.close();
    return 0;
}