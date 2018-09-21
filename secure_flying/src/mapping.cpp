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
//#include "object_msgs/ObjectsInBoxes.h"
#include "/home/ubuntu/catkin_ws/devel/include/darknet_ros_msgs/BoundingBoxes.h"
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
static const int IMGWIDTH = 640; // !!!! CHG NOTE
static const int IMGHEIGHT = 480;
static const bool save_pcds = false;
static const string save_path = "/home/clarence/workspace/PointCloud/Gazebo_PCD_Training_data/";

//ewok::EuclideanDistanceNormalRingBuffer<POW>::Ptr rrb;
ewok::EuclideanDistanceNormalRingBuffer<POW> rrb(resolution, 1.0);

ros::Publisher occ_marker_pub, free_marker_pub, dist_marker_pub, norm_marker_pub;
ros::Publisher cloud2_pub, cloud_fs_pub, cloud_semantic_pub, center_pub, traj_pub;

bool objects_updated = false;


unsigned char senantic_labels[IMGWIDTH][IMGHEIGHT];
bool label_mat_locked = false;
double direction_x = 1.0;
double direction_y = 0.0;
int control_label = 0;
int save_counter = 0;

//CHG
void directionCallback(const std_msgs::Float64MultiArray& direction)
{
    direction_x = direction.data[0];
    direction_y = direction.data[1];
}

//CHG
void keyboardCallback(const geometry_msgs::Twist& msg)
{
    if(msg.linear.x > 0.f)
    {
        control_label = 1; // Move forward
    }
    else if(msg.linear.z > 0.f)
    {
        control_label = 2; // Move upward
    }
    else if(msg.linear.z < -0.f)
    {
        control_label = 3; // Move downward
    }
    else if(msg.angular.z > 0.f)
    {
        control_label = 4;  // Turn left
    }
    else if(msg.angular.z < 0.f)
    {
        control_label = 5;  // Turn right
    }
    else
    {
        control_label = 0;  // Hover
    }
}

//CHG
void objectsCallback(const darknet_ros_msgs::BoundingBoxes& objects)
{
    if(objects_updated == false)
    {
        /*initialize labels*/
        for(int i = 0; i < IMGWIDTH; i++)
        {
            for(int j = 0; j < IMGHEIGHT; j++)
            {
                senantic_labels[i][j] = 0;
            }
        }

        /* Set labels of each roi. If there are overlaps, only take the last input label.*/
        for (int m = 0; m < objects.bounding_boxes.size(); m++)
        {
            unsigned char label;
            if(objects.bounding_boxes[m].Class == "person")
                label = 5;
            else if(objects.bounding_boxes[m].Class == "cat")
                label = 5;
            else if(objects.bounding_boxes[m].Class == "dog")
                label = 5;
            else if(objects.bounding_boxes[m].Class == "laptop")
                label = 4;
            else if(objects.bounding_boxes[m].Class == "bed")
                label = 4;
            else if(objects.bounding_boxes[m].Class == "tvmonitor")
                label = 4;
            else if(objects.bounding_boxes[m].Class == "chair")
                label = 4;
            else if(objects.bounding_boxes[m].Class == "diningtable")
                label = 4;
            else if(objects.bounding_boxes[m].Class == "sofa")
                label = 4;
            else
                label = 3;

            while(label_mat_locked)
            {
                ros::Duration(0.001).sleep(); // Sleep one second to wait for others using the label matirx
            }

            label_mat_locked = true;

            int range_x_min = objects.bounding_boxes[m].xmin;
            int range_x_max = objects.bounding_boxes[m].xmax;
            int range_y_min = objects.bounding_boxes[m].ymin;
            int range_y_max = objects.bounding_boxes[m].ymax;

            for(int i = range_x_min; i <= range_x_max; i++)
            {
                for(int j = range_y_min; j <= range_y_max; j++)
                {
                    senantic_labels[i][j] = label;
                }
            }
            label_mat_locked = false;

            //cout << "Label=" <<(int)label<< endl;
        }

        objects_updated = true;
    }
}

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

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_dynamic(new pcl::PointCloud<pcl::PointXYZRGB>()); //chg, for dynamic objects

    // add semantic label, chg

    while(label_mat_locked)
    {
        ros::Duration(0.00001).sleep(); // Sleep to wait for others using the label matrix
    }
    label_mat_locked = true;

    for(int i = 0; i < cloud_2->width; i++)
    {
        for(int j = 0; j < cloud_2->height; j++)
        {
            cloud_2->points[i + j * IMGWIDTH].rgb = senantic_labels[i][j];
            if(senantic_labels[i][j] == 5)  // Record dynamic obstacles: man 5
            {
                pcl::PointXYZRGB temp;
                temp.x = cloud_2->points[i + j * IMGWIDTH].x;
                temp.y = cloud_2->points[i + j * IMGWIDTH].y;
                temp.z = cloud_2->points[i + j * IMGWIDTH].z;
                temp.rgb = 0.f;
                cloud_dynamic->push_back(temp);
            }
        }
    }
    label_mat_locked = false;

    // down-sample for all
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setInputCloud(cloud_2);
    float res = 0.1f;
    sor.setLeafSize(res, res, res);
    sor.filter(*cloud_filtered);

    // down-sample for dynamic objects, chg
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_dynamic(new pcl::PointCloud<pcl::PointXYZRGB>());
    ewok::EuclideanDistanceNormalRingBuffer<POW>::PointCloud cloud_dyn;

    if(cloud_dynamic->width > 0)
    {
        pcl::VoxelGrid<pcl::PointXYZRGB> sor_d;
        sor_d.setInputCloud(cloud_dynamic);
        sor_d.setLeafSize(res, res, res);
        sor_d.filter(*cloud_filtered_dynamic);

        //chg
        std::vector<pcl::PointXYZRGB, Eigen::aligned_allocator<pcl::PointXYZRGB> > points_dyn =
                cloud_filtered_dynamic->points; //  cloud_2->points;
        for(int i = 0; i < points_dyn.size(); ++i)
        {
            cloud_dyn.push_back(Eigen::Vector4f(points_dyn.at(i).x, points_dyn.at(i).y, points_dyn.at(i).z, 0));
        }

    }


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
        //ROS_INFO_STREAM("Origin: " << origin.transpose() << " idx " << idx.transpose());
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
    double t1 = ros::Time::now().toSec();
    rrb.insertPointCloud(cloud_ew, origin);

    // insert dynamic points to ringbuffer
    if(cloud_dynamic->width > 0)
        rrb.insertPointCloudDynamic(cloud_dyn, origin);

    // Insert Semantic Info here, CHG
    rrb.insertPointCloudSemanticLabel(*cloud_2, objects_updated);
    rrb.updateDistance();

    double t2 = ros::Time::now().toSec();
    ROS_INFO("Updating ringbuffer time: %lf ms", 1000 * (t2 - t1));

    _last_time = ros::Time::now();

    // Check time, CHG
    double dt = ros::Time::now().toSec() - _data_input_time.toSec();
    ROS_INFO("Delayed time: %lf ms", 1000 * (dt));

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


    /* Free space cloud*/
    pcl::PointCloud<pcl::PointXYZ> free_cloud;
    Eigen::Vector3d center_fs;
    rrb.getBufferFSCloud(free_cloud, center_fs);

    // convert to ROS message and publish
    sensor_msgs::PointCloud2 cloud2_fs;
    pcl::toROSMsg(free_cloud, cloud2_fs);

    // message publish should have the same time stamp
    cloud2_fs.header.stamp = ros::Time::now();
    cloud2_fs.header.frame_id = "world";
    cloud_fs_pub.publish(cloud2_fs);


    /* Semantic cloud */
    pcl::PointCloud<pcl::PointXYZI> semantic_cloud;
    Eigen::Vector3d center_s;
    // rrb->getBufferSemanticCloud(semantic_cloud, center_s, direction_x, direction_y);
    rrb.getBufferObstacleSemanticCloud(semantic_cloud, center_s, direction_x, direction_y);

    // convert to ROS message and publish
    sensor_msgs::PointCloud2 cloud2_semantic;
    pcl::toROSMsg(semantic_cloud, cloud2_semantic);

    // message publish should have the same time stamp
    cloud2_semantic.header.stamp = ros::Time::now();
    cloud2_semantic.header.frame_id = "world";
    cloud_semantic_pub.publish(cloud2_semantic);
    

    //publish center
    geometry_msgs::PointStamped center_p;
    center_p.header = cloud2_fs.header;
    center_p.point.x = center_fs(0);
    center_p.point.y = center_fs(1);
    center_p.point.z = center_fs(2);
    center_pub.publish(center_p);

     /* Write */
    if(save_pcds)
    {       
        string pcd_name = save_path + to_string(save_counter) + ".pcd";
        string file_name = save_path + "labels.txt";

        pcl::PCDWriter writer;
        semantic_cloud.width = semantic_cloud.points.size ();
        semantic_cloud.height = 1;
        writer.write(pcd_name, semantic_cloud);

        ofstream label_of;

        label_of.open(file_name, std::ios::out | std::ios::app);  
        if (label_of.is_open())
        {
            string content = to_string(center_p.point.x) + "\t" + to_string(center_p.point.y) + "\t" + to_string(center_p.point.z)+ "\t" + to_string(control_label)+"\n";
            label_of << content;
            label_of.close();
        }

        save_counter ++;
    }

}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "sim_example");
    ros::NodeHandle nh;

    // ringbuffer cloud2
    cloud2_pub = nh.advertise<sensor_msgs::PointCloud2>("ring_buffer/cloud_ob", 1, true);
    cloud_fs_pub = nh.advertise<sensor_msgs::PointCloud2>("ring_buffer/cloud_fs", 1, true);
    cloud_semantic_pub = nh.advertise<sensor_msgs::PointCloud2>("ring_buffer/cloud_semantic", 1, true); //CHG
    center_pub = nh.advertise<geometry_msgs::PointStamped>("ring_buffer/center",1,true) ;

    // synchronized subscriber for pointcloud and odometry
    // message_filters::Subscriber<nav_msgs::Odometry> odom_sub(nh, "/zed/odom", 1);
    // message_filters::Subscriber<sensor_msgs::PointCloud2> pcl_sub(nh, "/zed/point_cloud/cloud_registered", 1);

    message_filters::Subscriber<nav_msgs::Odometry> odom_sub(nh, "/firefly/ground_truth/odometry", 1);
    message_filters::Subscriber<sensor_msgs::PointCloud2> pcl_sub(nh, "/firefly/vi_sensor/camera_depth/depth/points", 1);

    typedef sync_policies::ApproximateTime<nav_msgs::Odometry, sensor_msgs::PointCloud2> MySyncPolicy;
    Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), odom_sub, pcl_sub);
    sync.registerCallback(boost::bind(&odomCloudCallback, _1, _2));

    // subscribe RGB image detection result. Delay is ignored!!! CHG
    ros::Subscriber detection_sub = nh.subscribe("/darknet_ros/bounding_boxes", 2, objectsCallback);
    ros::Subscriber direction_sub =  nh.subscribe("/firefly/s_vec_init", 2, directionCallback);
    ros::Subscriber keyboard_sub =  nh.subscribe("/keyboard/twist", 2, keyboardCallback);

    

    // timer for publish ringbuffer as pointcloud
    ros::Timer timer1 = nh.createTimer(ros::Duration(0.2), timerCallback); // RATE

    ros::Duration(0.5).sleep();

    // setup ring buffer
//    rrb = ewok::EuclideanDistanceNormalRingBuffer<POW>::Ptr(
//        new ewok::EuclideanDistanceNormalRingBuffer<POW>(resolution, 1.0));

    _last_time = ros::Time::now();
    std::cout << "Start mapping!" << std::endl;

    ros::spin();

    return 0;
}
