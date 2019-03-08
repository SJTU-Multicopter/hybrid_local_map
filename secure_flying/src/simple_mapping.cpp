//
// Created by clarence on 19-3-5.
//
#include <ewok/ed_nor_ring_buffer.h>

#include <ros/ros.h>

#include <tf/transform_datatypes.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PointStamped.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float64MultiArray.h>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <iostream>
#include <string>
#include <fstream>


using namespace message_filters;
using namespace std;

// global declaration
ros::Time _last_time;
ros::Time _data_input_time;

bool initialized = false;
const double resolution = 0.1;

static const int POW = 7;
static const int N = (1 << POW);

ewok::EuclideanDistanceNormalRingBuffer<POW> rrb(resolution, 1.0);

ros::Publisher occ_marker_pub, free_marker_pub, dist_marker_pub, norm_marker_pub;
ros::Publisher cloud2_pub, cloud_fs_pub, cloud_semantic_pub, center_pub, traj_pub;

bool objects_updated = false;


// this callback use input cloud to update ring buffer, and update odometry of UAV
void odomCloudCallback(const nav_msgs::OdometryConstPtr& odom, const sensor_msgs::PointCloud2ConstPtr& cloud)
{
    ROS_INFO("Received Point Cloud!");
    _data_input_time = ros::Time::now();

    double elp = ros::Time::now().toSec() - _last_time.toSec();

    tf::Quaternion q1(odom->pose.pose.orientation.x, odom->pose.pose.orientation.y,
                      odom->pose.pose.orientation.z, odom->pose.pose.orientation.w);
    tf::Matrix3x3 m(q1);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    // ROS_INFO("Updating ringbuffer map");
    // update ring buffer
    // tranform from optical frame to uav frame
    Eigen::Matrix4f t_c_b = Eigen::Matrix4f::Zero();
    t_c_b(0, 2) = 1;
    t_c_b(1, 0) = -1;
    t_c_b(2, 1) = -1;
    t_c_b(3, 3) = 1;

    // transform from uav to world
    // get orientation and translation
    Eigen::Quaternionf q;
    q.w() = odom->pose.pose.orientation.w;
    q.x() = odom->pose.pose.orientation.x;
    q.y() = odom->pose.pose.orientation.y;
    q.z() = odom->pose.pose.orientation.z;

    // create transform matrix
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block(0, 0, 3, 3) = Eigen::Matrix3f(q);
    transform(0, 3) = odom->pose.pose.position.x;
    transform(1, 3) = odom->pose.pose.position.y;
    transform(2, 3) = odom->pose.pose.position.z;
    // std::cout << transform.matrix() << "\n\n";

    // convert cloud to pcl form
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::fromROSMsg(*cloud, *cloud_in);
    // transform to world frame
    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_2(new pcl::PointCloud<pcl::PointXYZRGB>());
    //pcl::transformPointCloud(*cloud_in, *cloud_2, transform);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_1(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_2(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::transformPointCloud(*cloud_in, *cloud_1, t_c_b);
    pcl::transformPointCloud(*cloud_1, *cloud_2, transform);

    // down-sample for all
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setInputCloud(cloud_2);
    float res = 0.1f;
    sor.setLeafSize(res, res, res);
    sor.filter(*cloud_filtered);

    // compute ewol pointcloud and origin
    Eigen::Vector3f origin = (transform * Eigen::Vector4f(0, 0, 0, 1)).head<3>();
    ewok::EuclideanDistanceNormalRingBuffer<POW>::PointCloud cloud_ew;
    std::vector<pcl::PointXYZRGB, Eigen::aligned_allocator<pcl::PointXYZRGB> > points =
            cloud_filtered->points; //  cloud_2->points;

    for(int i = 0; i < points.size(); ++i)
    {
        cloud_ew.push_back(Eigen::Vector4f(points.at(i).x, points.at(i).y, points.at(i).z, 0));
    }

    // initialize the ringbuffer map
    if(!initialized)
    {
        Eigen::Vector3i idx;
        rrb.getIdx(origin, idx);
        rrb.setOffset(idx);
        initialized = true;
    }
    else
    {
        // move buffer when its center is not the same as UAV
        while(true)
        {
            Eigen::Vector3i origin_idx, offset, diff;
            rrb.getIdx(origin, origin_idx);
            offset = rrb.getVolumeCenter();
            //std::cout << "origin :" << origin_idx << " center:" << offset << std::endl;
            diff = origin_idx - offset;
            if(diff.array().any())
                rrb.moveVolume(diff.head<3>());
            else
                break;
        }
    }

    // insert point cloud to ringbuffer
    rrb.insertPointCloud(cloud_ew, origin);

    rrb.updateDistance();
}

void timerCallback(const ros::TimerEvent& e)
{
    if(!initialized) return;

    /*Obstacle cloud*/
    pcl::PointCloud<pcl::PointXYZ> cloud;
    Eigen::Vector3d center;
    rrb.getBufferAsCloud(cloud, center);

    // convert to ROS message and publish
    sensor_msgs::PointCloud2 cloud2;
    pcl::toROSMsg(cloud, cloud2);

    // message publish should have the same time stamp
    cloud2.header.stamp = ros::Time::now();
    cloud2.header.frame_id = "world";
    cloud2_pub.publish(cloud2);

    //publish center
    geometry_msgs::PointStamped center_p;
    center_p.header = cloud2.header;
    center_p.point.x = center(0);
    center_p.point.y = center(1);
    center_p.point.z = center(2);
    center_pub.publish(center_p);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "simple_mapping");
    ros::NodeHandle nh;

    // ringbuffer cloud2
    cloud2_pub = nh.advertise<sensor_msgs::PointCloud2>("ring_buffer/cloud_ob", 1, true);
    center_pub = nh.advertise<geometry_msgs::PointStamped>("ring_buffer/center",1,true) ;

    message_filters::Subscriber<nav_msgs::Odometry> odom_sub(nh, "/firefly/ground_truth/odometry", 1);
    message_filters::Subscriber<sensor_msgs::PointCloud2> pcl_sub(nh, "/firefly/vi_sensor/camera_depth/depth/points", 1);

    typedef sync_policies::ApproximateTime<nav_msgs::Odometry, sensor_msgs::PointCloud2> MySyncPolicy;
    Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), odom_sub, pcl_sub);
    sync.registerCallback(boost::bind(&odomCloudCallback, _1, _2));


    // timer for publish ringbuffer as pointcloud
    ros::Timer timer1 = nh.createTimer(ros::Duration(0.2), timerCallback); // RATE

    ros::Duration(0.5).sleep();


    _last_time = ros::Time::now();
    std::cout << "Start mapping!" << std::endl;

    ros::spin();

    return 0;
}

