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
#include <cmath>
#include <Eigen/Eigen>
#include <algorithm>

using namespace message_filters;

// global declaration
ros::Time _data_input_time;

bool initialized = false;
const double resolution = 0.2;

static const int POW = 6;
static const int N = (1 << POW);

ewok::EuclideanDistanceNormalRingBuffer<POW> rrb(resolution, 1.0);

ros::Publisher occ_marker_pub, free_marker_pub, dist_marker_pub, norm_marker_pub;
ros::Publisher cloud2_pub, cloud_fs_pub, cloud_semantic_pub, center_pub, traj_pub;

bool objects_updated = false;


/****** Parameters for path planning ******/
const int ANGLE_H_NUM = 18;
const int ANGLE_V_NUM = 7;
Eigen::VectorXd Fov_half(2); //Fov parameters
Eigen::VectorXd Angle_h(ANGLE_H_NUM);  // initiate later in the main function
Eigen::VectorXd Angle_v(ANGLE_V_NUM); // initiate later in the main function
Eigen::MatrixXd F_cost;

Eigen::Vector3d p_goal;
Eigen::Vector3d p0; // Should be updated in a subscriber of the vehicle state
Eigen::Vector3d v0;
Eigen::Vector3d a0;
double yaw0 = 0.0;
double theta_h_last = 0.0;
double theta_v_last = 0.0;

struct  Path_Planning_Parameters
{
    double d_ref = 3.0;
    double k1_xy = 4; //% Goal directed coefficient
    double k1_z = 4; //% Goal directed coefficient
    double k2_xy = 2; //% Rotation coefficient
    double k2_z = 2; //% Rotation coefficient
    double k3 = M_PI*0.1; //% FOV coefficient
    double kk_h = 1; //% FOV horisontal cost coefficient
    double kk_v = 1; //% FOV vertical cost coefficient
    double v_max_ori = 5; //% m/s, just reference
    double v_scale_min = 0.4;
    double delt_t = 0.05; //%time interval between two control points
    int max_plan_num = 100;
}pp;

/**************************************************/


// this callback use input cloud to update ring buffer, and update odometry of UAV
void odomCloudCallback(const nav_msgs::OdometryConstPtr& odom, const sensor_msgs::PointCloud2ConstPtr& cloud)
{
    // ROS_INFO("Received Point Cloud!");
    _data_input_time = ros::Time::now();

    tf::Quaternion q1(odom->pose.pose.orientation.x, odom->pose.pose.orientation.y,
                      odom->pose.pose.orientation.z, odom->pose.pose.orientation.w);
    tf::Matrix3x3 m(q1);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

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

    double elp = ros::Time::now().toSec() - _data_input_time.toSec();
    std::cout << "Map update time = " << elp << " s" << std::endl;
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

void motion_primitives(Eigen::Vector3d p0, Eigen::Vector3d v0, Eigen::Vector3d a0, double yaw0, double theta_h,
                       double theta_v, Eigen::Vector3d goal, double d, double v_max, double delt_t,
                       Eigen::MatrixXd &p, Eigen::MatrixXd &v, Eigen::MatrixXd &a, Eigen::VectorXd &t)
{
    double delt_x = d*cos(theta_v)*cos(theta_h+yaw0);
    double delt_y = d*cos(theta_v)*sin(theta_h+yaw0);
    double delt_z = d*sin(theta_v);
    Eigen::Vector3d pf;
    pf << p0(0)+delt_x, p0(1)+delt_y, p0(2)+delt_z;

    Eigen::Vector3d l = goal - pf;
    Eigen::Vector3d vf = (v_max / l.norm()) * l;
    vf(2) = 0; // % Note: 0 maybe better, for the p curve wont go down to meet the vf

    Eigen::Vector3d af = Eigen::Vector3d::Zero();

    // % Choose the time as running in average velocity
    double decay_parameter = 0.5;
    double T = 0.2;
    if(vf(0)+v0(0) != 0)
    {
        double T1 = 2*delt_x/(vf(0)+v0(0)) * decay_parameter;
        T1 > T ? T = T1 : T = T;
    }

    if(vf(1)+v0(1) != 0)
    {
        double T2 = 2*delt_y/(vf(1)+v0(1)) * decay_parameter;
        T2 > T ? T = T2 : T = T;
    }

    if(vf(2)+v0(2) != 0)
    {
        double T3 = 2*delt_z/(vf(2)+v0(2)) * decay_parameter;
        T3 > T ? T = T3 : T = T;
    }

    int times = T / delt_t;
    p = Eigen::MatrixXd::Zero(times, 3);
    v = Eigen::MatrixXd::Zero(times, 3);
    a = Eigen::MatrixXd::Zero(times, 3);
    t = Eigen::VectorXd::Zero(times);

    // % calculate optimal jerk controls by Mark W. Miller
    for(int ii=0; ii<3; ii++)
    {
        double delt_a = af(ii) - a0(ii);
        double delt_v = vf(ii) - v0(ii) - a0(ii)*T;
        double delt_p = pf(ii) - p0(ii) - v0(ii)*T - 0.5*a0(ii)*T*T;

        //%  if vf is not free
        double alpha = delt_a*60/pow(T,3) - delt_v*360/pow(T,4) + delt_p*720/pow(T,5);
        double beta = -delt_a*24/pow(T,2) + delt_v*168/pow(T,3) - delt_p*360/pow(T,4);
        double gamma = delt_a*3/T - delt_v*24/pow(T,2) + delt_p*60/pow(T,3);

        for(int jj=0; jj<times; jj++)
        {
            double tt = (times + 1)*delt_t;
            t(jj) = tt;
            p(jj,ii) = alpha/120*pow(tt,5) + beta/24*pow(tt,4) + gamma/6*pow(tt,3) + a0(ii)/2*pow(tt,2) + v0(ii)*tt + p0(ii);
            v(jj,ii) = alpha/24*pow(tt,4) + beta/6*pow(tt,3) + gamma/2*pow(tt,2) + a0(ii)*tt + v0(ii);
            a(jj,ii) = alpha/6*pow(tt,3) + beta/2*pow(tt,2) + gamma*tt + a0(ii);
        }
    }


}

void trajectoryCallback(const ros::TimerEvent& e) {
    // TODO: Safety strategy for emergency stop should be added here

    /** Moition primitives **/
    Eigen::Vector3d delt_p = p_goal - p0;
    double phi_h = atan2(delt_p(1), delt_p(0)); //% horizental offset angle
    double phi_v = atan2(delt_p(2), sqrt(delt_p(0) * delt_p(0) + delt_p(1) * delt_p(1))); //% vertical offset angle

    // %calculate cost for sampled points
    Eigen::MatrixXd cost = Eigen::MatrixXd::Zero(ANGLE_H_NUM * ANGLE_V_NUM, 4);
    double theta_h = 0;
    double theta_v = 0;

    for(int i=0; i<ANGLE_H_NUM; i++)
    {
        for(int j=0; j<ANGLE_V_NUM; j++)
        {
            theta_h = Angle_h(i);
            theta_v = Angle_v(j);
            int m = i*ANGLE_V_NUM + j; //sequence number
            cost(m, 0) = pp.k1_xy*(yaw0+theta_h-phi_h)*(yaw0+theta_h-phi_h) + pp.k1_z*(theta_v-phi_v)*(theta_v-phi_v) +
                    pp.k2_xy*(theta_h-theta_h_last)*(theta_h-theta_h_last)+pp.k2_z*(theta_v-theta_v_last)*(theta_v-theta_v_last) + pp.k3*F_cost(i,j);
            cost(m, 1) = theta_h;
            cost(m, 2) = theta_v;
            cost(m, 3) = (1-F_cost(i,j)) * pp.d_ref;
        }
    }

    //% Rank by cost, small to large
    for(int m=0; m<ANGLE_H_NUM*ANGLE_V_NUM-1; m++)
    {
        for(int n=0; n<ANGLE_H_NUM*ANGLE_V_NUM-m-1; n++)
        {
            if(cost(n,0) > cost(n+1,0))
            {
                Eigen::Vector4d temp = cost.row(n+1);
                cost.row(n+1) = cost.row(n);
                cost.row(n) = temp;
            }
        }
    }

    //% max velocity is decreased concerning current velocity direction and goal
    //% direction
    double v_scale = std::max(delt_p.dot(v0)/v0.norm()/delt_p.norm(), pp.v_scale_min);
    double v_max = pp.v_max_ori * v_scale;

    for(int seq=0; seq<pp.max_plan_num; seq++)
    {
        Eigen::MatrixXd p;
        Eigen::MatrixXd v;
        Eigen::MatrixXd a;
        Eigen::VectorXd t;
        motion_primitives(p0, v0, a0, yaw0, cost(seq,1), cost(seq,2), p_goal, cost(seq,3), v_max, pp.delt_t, p, v, a, t);

        // TODO: Collision checking goes here
    }

    // TODO: Publish the control values

}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "local_planning");
    ros::NodeHandle nh;

    // Fov sample parameters
    Fov_half << 35, 20;
    Angle_h << -180, -150, -120, -90, -70, -50, -30, -20, -10, 0, 10, 20, 30, 50, 70, 90, 120, 150;
    Angle_v << -40, -20, -10, 0, 10, 20, 40;

    Angle_h = Angle_h * M_PI / 180.0;
    Angle_v = Angle_v * M_PI / 180.0;
    Fov_half = Fov_half * M_PI / 180.0;

    F_cost = Eigen::MatrixXd::Zero(ANGLE_H_NUM, ANGLE_V_NUM);
    for(int i = 0; i < ANGLE_H_NUM; i++)
    {
        for(int j = 0; j < ANGLE_V_NUM; j++)
        {
            if(fabs(Angle_h(i)) < Fov_half(0) && fabs(Angle_v(j)) < Fov_half(1)) {
                continue;
            } else
            {
                double delt_h_angle = std::min(fabs(Angle_h(i)-Fov_half(0)), fabs(Angle_h(i)+Fov_half(0)));
                double delt_v_angle = std::min(fabs(Angle_v(j)-Fov_half(1)), fabs(Angle_v(j)+Fov_half(1)));
                F_cost(i,j) = (pp.kk_h*delt_h_angle + pp.kk_v*delt_v_angle)/(270/180*M_PI); // % vertical max error + horizontal max error = 270¡ã
            }
        }
    }

    // ringbuffer cloud2
    cloud2_pub = nh.advertise<sensor_msgs::PointCloud2>("ring_buffer/cloud_ob", 1, true);
    center_pub = nh.advertise<geometry_msgs::PointStamped>("ring_buffer/center",1,true) ;

    message_filters::Subscriber<nav_msgs::Odometry> odom_sub(nh, "/firefly/ground_truth/odometry", 1);
    message_filters::Subscriber<sensor_msgs::PointCloud2> pcl_sub(nh, "/firefly/vi_sensor/camera_depth/depth/points", 1);

    typedef sync_policies::ApproximateTime<nav_msgs::Odometry, sensor_msgs::PointCloud2> MySyncPolicy;
    Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), odom_sub, pcl_sub);
    sync.registerCallback(boost::bind(&odomCloudCallback, _1, _2));

    // timer for publish ringbuffer as pointcloud
    ros::Timer timer1 = nh.createTimer(ros::Duration(0.5), timerCallback); // RATE 2 Hz to publish

    // timer for trajectory generation
    ros::Timer timer2 = nh.createTimer(ros::Duration(0.02), trajectoryCallback); // RATE 50 Hz to generate trajectory

    std::cout << "Start mapping!" << std::endl;

    ros::AsyncSpinner spinner(4); // Use 4 threads
    spinner.start();
    ros::waitForShutdown();

    return 0;
}

