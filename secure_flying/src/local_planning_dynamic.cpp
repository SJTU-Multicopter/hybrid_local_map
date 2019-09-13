// Coordinate follows mavros 

#include <ewok/ed_nor_ring_buffer.h>

#include <ros/ros.h>

#include <tf/transform_datatypes.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Point32.h>

#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Float64MultiArray.h>
#include <visualization_msgs/Marker.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <../include/Eigen/Eigen>
#include <algorithm>

#include <px4_autonomy/Position.h> // Publish control points to node 'px4_autonomy'

using namespace message_filters;

#define GRAVATY 9.8
#define PIx2 6.28318
#define PI 3.14159
#define PI_2 1.5708


/**** Parameters to tune, some initialization needs to be changed in main function ****/
const double resolution = 0.1;

static const int POW = 6;
static const int N = (1 << POW);

const float cal_duration = 0.05;

ewok::EuclideanDistanceNormalRingBuffer<POW> rrb(resolution, 1.0); //Distance truncation threshold

const int ANGLE_H_NUM = 17;
const int ANGLE_V_NUM = 7;

const int LOOKING_PIECES_SIZE = 18; // Should be even, Resolution: 20 degrees one piece. MID=8. Yaw=0. Larger: 9 - 17 :  0 to Pi;  Smaller: 7-0 : 0 to (nearly)-Pi;
const float CAMERA_H_FOV = 62;  //degrees
const float HEAD_BUFFER_HIT_INCREASE = 0.4;
const float HEAD_BUFFER_MISS_DECREASE_STANDARD_V = 0.05; // Miss add value when velocity is 1m/s
const float HEAD_BUFFER_MISS_DECREASE_MIN = 0.05;

struct  Path_Planning_Parameters
{
    double d_ref = 1.5;
    double k1_xy = 2; //% Goal directed coefficient
    double k1_z = 2; //% Goal directed coefficient
    double k2_xy = 4; //% Rotation coefficient
    double k2_z = 4; //% Rotation coefficient
    double v_max_ori = 1.0; //% m/s, just reference  5.0 originally
    double v_scale_min = 0.1;
    double delt_t = 0.05; //%time interval between two control points
    int max_plan_num = ANGLE_H_NUM * ANGLE_V_NUM;  // Previously it was 100, as 18*7 = 126 > 100
}pp;

struct  Head_Planning_Parameters
{
   double k_current_v = 0.7;
   double k_planned_dir = 0.3;
   double k_v_fluctuation = 0.2;
}hp;

/*** End of Parameters ***/

/** Basic global variables **/
ros::Publisher current_marker_pub;
ros::Publisher cloud2_pub, cloud_edf_pub, center_pub;

ros::Publisher traj_point_pub; // add on 9 Mar
ros::Publisher head_cmd_pub; // add on 9 Mar

double x_centre, y_centre, z_centre;

bool objects_updated = false;
bool imu_initilized = false;
bool state_locked = false;
bool state_updating = false;
bool in_safety_mode = true;

/****** Global variables for path planning ******/
ros::Time _data_input_time;
ros::Time _algorithm_time;

bool initialized = false;

Eigen::VectorXd Fov_half(2); //Fov parameters
Eigen::VectorXd Angle_h(ANGLE_H_NUM);  // initiate later in the main function, rad
Eigen::VectorXd Angle_v(ANGLE_V_NUM); // initiate later in the main function, rad

Eigen::Vector3d p_goal;
Eigen::Vector3d p0;
Eigen::Vector3d v0;
Eigen::Vector3d a0;
Eigen::Quaternionf quad(1.0, 0.0, 0.0, 0.0);
double yaw0;
double v_direction;
//Eigen::VectorXd quad(4); 

double yaw_init = 0.0;  //Keep zero in this edition, the sampled directions are from all sides
double theta_h_last = 0.0;
double theta_v_last = 0.0;

Eigen::Vector3d p_store;
double yaw_store = 0.0;


/*** Global variables for rotatin head ***/
Eigen::VectorXf _direction_update_buffer; // Range [0, 1]
float heading_resolution;
int valid_piece_num_one_side;
int mid_seq_num; 

double motor_yaw = 0.0;
double motor_yaw_rate = 0.0;

/** Declaration of functions**/

void sendMotorCommands(double yaw, double yaw_rate_abs);

/**************************************************/

double getHeadingYawFromSeq(int seq)
{
    if(seq >= LOOKING_PIECES_SIZE || seq < 0){
        ROS_ERROR("Seq for yaw buffer of the head out of range.");
        return 0.f;
    }
    else{
        return (seq - mid_seq_num) * heading_resolution;
    }
}

int getHeadingSeq(float direction)
{
    if(direction > PI){
        direction -= PIx2;
    }
    else if(direction < -PI){
        direction += PIx2;
    } 

    if(direction > 0){  // To sovle the truncation problem
        direction += heading_resolution / 2.f;
    }
    else{
        direction -= heading_resolution / 2.f;  
    }
    /// Caution!!!!! the rotating range of the camera can only be within (-PI-heading_resolution, PI+heading_resolution), chg
    int heading_seq =  (int)(direction / heading_resolution) + mid_seq_num;  

    if(heading_seq < 0) heading_seq = LOOKING_PIECES_SIZE - 1;

    return heading_seq;
}

void correctPieceSeq(int &seq)
{
    if(seq >= LOOKING_PIECES_SIZE){   // to form a ring
        seq = seq - LOOKING_PIECES_SIZE;
    }
    else if(seq < 0){
        seq = seq + LOOKING_PIECES_SIZE;
    }
}

void addHitOnePiece(int seq)
{
    //std::cout<<"hit seq = "<<seq<<std::endl;
    _direction_update_buffer(seq)+HEAD_BUFFER_HIT_INCREASE<=1.f ? _direction_update_buffer(seq)+=(float)HEAD_BUFFER_HIT_INCREASE : _direction_update_buffer(seq)=1.f;
}

void addMissOnePiece(int seq)
{
    float delt_miss = HEAD_BUFFER_MISS_DECREASE_MIN;
    float delt_miss_by_velocity = HEAD_BUFFER_MISS_DECREASE_STANDARD_V * std::max(fabs(v0(0)), fabs(v0(1)));
    if(delt_miss_by_velocity > delt_miss){
        delt_miss = delt_miss_by_velocity;
    }

    //std::cout<<"miss seq = "<<seq<<std::endl;
    _direction_update_buffer(seq)-delt_miss>=0.f ? _direction_update_buffer(seq)-=delt_miss : _direction_update_buffer(seq)=0.f;
}

void updateHeadBuffer(const int &heading_direction_seq)
{
    Eigen::VectorXi update_flag_buf = Eigen::VectorXi::Zero(LOOKING_PIECES_SIZE);

    //Add hit for the direction in FOV
    addHitOnePiece(heading_direction_seq);
    update_flag_buf(heading_direction_seq) = 1; 

    for(int i=1; i<=valid_piece_num_one_side; i++){
        int seq_this_1 = heading_direction_seq + i;
        int seq_this_2 = heading_direction_seq - i;
        
        correctPieceSeq(seq_this_1);
        correctPieceSeq(seq_this_2);

        addHitOnePiece(seq_this_1);
        addHitOnePiece(seq_this_2);

        update_flag_buf(seq_this_1) = 1;
        update_flag_buf(seq_this_2) = 1;
    }
    //Add miss for the rest
    for(int j=0; j<LOOKING_PIECES_SIZE; j++){
        if(update_flag_buf(j) != 1){
            addMissOnePiece(j);
        } 
    }
}


// this callback use input cloud to update ring buffer, and update odometry of UAV
void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud)
{
    ROS_INFO("Received Point Cloud!");
    _data_input_time = ros::Time::now();

    // Add the rotation of head
    Eigen::Quaternionf q1(0, 0, 0, 1);
    Eigen::Quaternionf axis = quad * q1 * quad.inverse();
    axis.w() = cos(motor_yaw/2.0);
    axis.x() = axis.x() * sin(motor_yaw/2.0);
    axis.y() = axis.y() * sin(motor_yaw/2.0);
    axis.z() = axis.z() * sin(motor_yaw/2.0);
    Eigen::Quaternionf quad_rotate = quad * axis;


    // create transform matrix
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block(0, 0, 3, 3) = Eigen::Matrix3f(quad_rotate);
    transform(0, 3) = p0(0);
    transform(1, 3) = p0(1);
    transform(2, 3) = p0(2);
    // std::cout << transform.matrix() << "\n\n";

    // convert cloud to pcl form
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*cloud, *cloud_in);

    // down-sample for all
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud_in);
    float res = 0.2f;
    sor.setLeafSize(res, res, res);
    sor.filter(*cloud_filtered);

    double elp_sample = ros::Time::now().toSec() - _data_input_time.toSec();
    //std::cout << "Map sample time = " << elp_sample << " s" << std::endl;

    // transform to world frame
    Eigen::Matrix4f t_c_b = Eigen::Matrix4f::Zero();
    t_c_b(0, 2) = 1;
    t_c_b(1, 0) = -1;
    t_c_b(2, 1) = -1;
    t_c_b(3, 3) = 1;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_1(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_2(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*cloud_filtered, *cloud_1, t_c_b);
    pcl::transformPointCloud(*cloud_1, *cloud_2, transform);

    double elp1 = ros::Time::now().toSec() - _data_input_time.toSec();
    //std::cout << "Map transfer time = " << elp1 << " s" << std::endl;

    // t_c_b is never needed when used in the real world
    //pcl::transformPointCloud(*cloud_in, *cloud_2, transform);

    // compute ewol pointcloud and origin
    Eigen::Vector3f origin = (transform * Eigen::Vector4f(0, 0, 0, 1)).head<3>(); //  position (x,y,z)
    ewok::EuclideanDistanceNormalRingBuffer<POW>::PointCloud cloud_ew;
    std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ> > points = cloud_2->points; //  cloud_2->points;

    x_centre = p0(0);
    y_centre = p0(1);
    z_centre = p0(2);

    for(int i = 0; i < points.size(); ++i)
    {
        double x_diff = fabs(x_centre - points.at(i).x);
        double y_diff = fabs(y_centre - points.at(i).y);
        double z_diff = fabs(z_centre - points.at(i).z);
        double noise_threshold = x_diff*x_diff + y_diff*y_diff + z_diff*z_diff;

        if (noise_threshold > 0.2)
            cloud_ew.push_back(Eigen::Vector4f(points.at(i).x, points.at(i).y, points.at(i).z, 0));
    }

    // initialize the ringbuffer map
    if(!initialized)
    {
        Eigen::Vector3i idx;
        rrb.getIdx(origin, idx);
        rrb.setOffset(idx);
        initialized = true;
        ROS_WARN("Initialized!!");
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

    double preprocess = ros::Time::now().toSec() - _data_input_time.toSec();
    //std::cout << "Map preprocess time = " << preprocess << " s" << std::endl;

    // insert point cloud to ringbuffer
    rrb.insertPointCloud(cloud_ew, origin);

    double insert_t = ros::Time::now().toSec() - _data_input_time.toSec();
    //std::cout << "Map insert time = " << insert_t << " s" << std::endl;

    // Calculate distance field consider newly imported points (dynamic points)
    rrb.updateDistanceDynamic(cloud_ew, origin);
    //rrb.updateDistance();

    //double elp = ros::Time::now().toSec() - _data_input_time.toSec();
    //std::cout << "Map update time = " << elp << " s" << std::endl;

    /* Update buffer for rolling head */
    int heading_direction_seq = getHeadingSeq(motor_yaw);  //current heading direction
    std::cout << "heading_direction_seq=" << heading_direction_seq << std::endl;
    updateHeadBuffer(heading_direction_seq);
    double head_up = ros::Time::now().toSec() - _data_input_time.toSec();
    std::cout << "Head buffer update time = " << head_up << " s" << std::endl;

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
    ROS_INFO("Cloud published!");

    //publish center
    geometry_msgs::PointStamped center_p;
    center_p.header = cloud2.header;
    center_p.point.x = center(0);
    center_p.point.y = center(1);
    center_p.point.z = center(2);
    center_pub.publish(center_p);

    /*EDF showing*/
     // Since the pointcloud which is too close to the drone is invalid, skip 1.0m

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_edf_field(new pcl::PointCloud<pcl::PointXYZRGB>());
    double trunc_distance = 1.0; // Truncation distance is 1.0
    double boundary = pow(2.0, POW) * resolution / 2; // Here the boundary is 6.4m

    for (double vis_x = resolution/2; vis_x < boundary; vis_x += resolution) {
        for (double vis_y = resolution/2; vis_y < boundary; vis_y += resolution) { //遍历整个空间填充distance颜色
            // Skip the truncation area
            if (vis_x <= trunc_distance && vis_y <= trunc_distance) {
                continue;
            }

            pcl::PointXYZRGB vis_point;
            int dir_x[4] = {1, 1, -1, -1};
            int dir_y[4] = {1, -1, 1, -1};

            for (int i = 0; i < 4; ++i) {
                vis_point.x = x_centre + vis_x * dir_x[i];
                vis_point.y = y_centre + vis_y * dir_y[i];
                vis_point.z = z_centre;
  
                Eigen::Vector3i point_RGB = rrb.get_rgb_edf_dynamic(vis_point.x,
                     vis_point.y, vis_point.z);

                vis_point.r = point_RGB(0);
                vis_point.g = point_RGB(1);
                vis_point.b = point_RGB(2);

                cloud_edf_field->points.push_back(vis_point);
            }
        }
    }
    // Publish edf as point cloud 
    sensor_msgs::PointCloud2 edf_ros_cloud;
    pcl::toROSMsg(*cloud_edf_field, edf_ros_cloud);
    edf_ros_cloud.header.stamp = ros::Time::now();
    edf_ros_cloud.header.frame_id = "world";
    cloud_edf_pub.publish(edf_ros_cloud);

}

/** This function is to generate state to state trajectory **/
void motion_primitives(Eigen::Vector3d p0, Eigen::Vector3d v0, Eigen::Vector3d a0, double yaw_current, double theta_h,
                       double theta_v, Eigen::Vector3d goal, double d, double v_max, double delt_t,
                       Eigen::MatrixXd &p, Eigen::MatrixXd &v, Eigen::MatrixXd &a, Eigen::VectorXd &t)
{
    double delt_x = d*cos(theta_v)*cos(theta_h+yaw_current);
    double delt_y = d*cos(theta_v)*sin(theta_h+yaw_current);
    double delt_z = d*sin(theta_v);

    Eigen::Vector3d pf;
    pf << p0(0)+delt_x, p0(1)+delt_y, p0(2)+delt_z;

    Eigen::Vector3d l = goal - pf;
    Eigen::Vector3d vf = (v_max / l.norm()) * l;
    vf(2) = 0; // % Note: 0 maybe better, for the p curve wont go down to meet the vf

    Eigen::Vector3d af = Eigen::Vector3d::Zero();

    // % Choose the time as running in average velocity
    // double decay_parameter = 0.5;
    // double T = 0.2;

    double j_limit = 4;
    double a_limit = 3;
    double v_limit = 3;

    double T1 = fabs(af(0)-a0(0))/j_limit > fabs(af(1)-a0(1))/j_limit ? fabs(af(0)-a0(0))/j_limit : fabs(af(1)-a0(1))/j_limit;
    T1 = T1 > fabs(af(2)-a0(2))/j_limit ? T1 : fabs(af(2)-a0(2))/j_limit;
    double T2 = fabs(vf(0)-v0(0))/a_limit > fabs(vf(1)-v0(1))/a_limit ? fabs(vf(0)-v0(0))/a_limit : fabs(vf(1)-v0(1))/a_limit;
    T2 = T2 > fabs(vf(2)-v0(2))/a_limit ? T2 : fabs(vf(2)-v0(2))/a_limit;
    double T3 = fabs(pf(0)-p0(0))/v_limit > fabs(pf(1)-p0(1))/v_limit ? fabs(pf(0)-p0(0))/v_limit : fabs(pf(1)-p0(1))/v_limit;
    T3 = T3 > fabs(pf(2)-p0(2))/v_limit ? T3 : fabs(pf(2)-p0(2))/v_limit;

    double T = T1 > T2 ? T1 : T2;
    T = T > T3 ? T : T3;
    T = T < 0.5 ? 0.5 : T;

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
            double tt = (jj + 1)*delt_t;
            t(jj) = tt;
            p(jj,ii) = alpha/120*pow(tt,5) + beta/24*pow(tt,4) + gamma/6*pow(tt,3) + a0(ii)/2*pow(tt,2) + v0(ii)*tt + p0(ii);
            v(jj,ii) = alpha/24*pow(tt,4) + beta/6*pow(tt,3) + gamma/2*pow(tt,2) + a0(ii)*tt + v0(ii);
            a(jj,ii) = alpha/6*pow(tt,3) + beta/2*pow(tt,2) + gamma*tt + a0(ii);
        }
    }
}

/* Publish markers to show the path in rviz */
void marker_publish(Eigen::MatrixXd &Points)
{
    visualization_msgs::Marker points, line_strip;
    points.header.frame_id = line_strip.header.frame_id = "world";
    points.header.stamp = line_strip.header.stamp = ros::Time::now();
    points.action = line_strip.action = visualization_msgs::Marker::ADD;
    points.ns = line_strip.ns = "points_and_lines";

    points.id = 0;
    line_strip.id = 1;

    points.type = visualization_msgs::Marker::POINTS;
    line_strip.type = visualization_msgs::Marker::LINE_STRIP;

    points.scale.x = 0.2;
    points.scale.y = 0.2;

    // Line width
    line_strip.scale.x = 0.1;

    // Points are green
    points.color.g = 1.0;
    points.color.a = 1.0;

    // Line strip is blue
    line_strip.color.b = 1.0;
    line_strip.color.a = 1.0;

    line_strip.lifetime = ros::Duration(0);

    int point_num = Points.rows();

    for(int i=0; i<point_num; i++)
    {
        geometry_msgs::Point p;
        p.x = Points(i, 0);
        p.y = Points(i, 1);
        p.z = Points(i, 2);

        // ROS_INFO("p.x %lf", p.x);

        points.points.push_back(p);
        line_strip.points.push_back(p);
    }

    current_marker_pub.publish(points);
    current_marker_pub.publish(line_strip);
}



/** This is the function to generate the collision-free path, which is trigered by a timer defined in main function. **/
void trajectoryCallback(const ros::TimerEvent& e) {
    /** Generate trajectory for uav first***/
    double theta_h_chosen;

    if(!state_updating) /// Need to rethink the necessity of this lock!!!! CHG
    {
        // To do: Update p_goal in accordance with keyboard/joy command
        state_locked = true;

        // TODO: Safety strategy for emergency stop should be added here

        // geometry_msgs::Point traj_pt;
        px4_autonomy::Position traj_pt;

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
                theta_h = Angle_h(i) + phi_h;  //start from goal position, chg, 2019.8.31
                theta_v = Angle_v(j) + phi_v;  //start from goal position, chg, 2019.8.31
                int m = i*ANGLE_V_NUM + j; //sequence number
                // Vectorial angle can never be larger than PI
                double goal_diff_h = fabs(theta_h-phi_h) > M_PI ? 2*M_PI - fabs(theta_h-phi_h) : theta_h-phi_h;
                double goal_diff_v = fabs(theta_v-phi_v) > M_PI ? 2*M_PI - fabs(theta_v-phi_v) : theta_v-phi_v;

                cost(m, 0) = pp.k1_xy*goal_diff_h*goal_diff_h + pp.k1_z*goal_diff_v*goal_diff_v +
                        pp.k2_xy*(theta_h-theta_h_last)*(theta_h-theta_h_last) + pp.k2_z*(theta_v-theta_v_last)*(theta_v-theta_v_last);
                cost(m, 1) = theta_h;
                cost(m, 2) = theta_v;
                cost(m, 3) = pp.d_ref;
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
        // If v0.norm() = 0, v_scale = nan
        v_scale = (v0.norm() == 0) ? pp.v_scale_min : v_scale;
        double v_max = pp.v_max_ori * v_scale;

        bool flag = false;

        theta_h_chosen = theta_h_last; //if the uav is in safe mode, this would be the last value so the head wont rotate
        for(int seq=0; seq<pp.max_plan_num; seq++)
        {
            Eigen::MatrixXd p;
            Eigen::MatrixXd v;
            Eigen::MatrixXd a;
            Eigen::VectorXd t;
            motion_primitives(p0, v0, a0, yaw_init, cost(seq,1), cost(seq,2), p_goal, cost(seq,3), v_max, pp.delt_t, p, v, a, t);

            // Get points number on the path
            const int Num = p.rows();
            Eigen::Vector3f *sim_traj = new Eigen::Vector3f[Num];

            for (int i = 0; i < Num; ++i) {
                sim_traj[i](0) = (float)p.row(i)(0);
                sim_traj[i](1) = (float)p.row(i)(1);
                sim_traj[i](2) = (float)p.row(i)(2);
            }

            // Obstacle threshold is 0.9 now
            flag = rrb.collision_checking(sim_traj, Num, 0.3); // collision_checking

            if(flag)
            {
                ROS_INFO("traj_safe");
                theta_h_chosen = cost(seq,1); 
                theta_h_last = cost(seq,1); // Update last theta
                theta_v_last = cost(seq,2);

                // Publish down sampled path points
                const int point_num_pub = 5;
                int interval_num = Num / point_num_pub;
                if (interval_num > 0)
                {
                    Eigen::MatrixXd show_points = Eigen::MatrixXd::Zero(point_num_pub+1, 3);
                    for(int pubi = 0; pubi < point_num_pub; pubi++)
                    {
                        show_points.row(pubi) = p.row(pubi*interval_num);
                    }
                    show_points.row(point_num_pub) = p.row(Num-1);
                    marker_publish(show_points);

                    // TODO: Publish control values of the UAV in obstacle avoidance mode here, chg

                }

                break;
            }
            else
            {
                // ROS_INFO("traj_unsafe");
            }
        }

        if(!flag){  // Safety mode
            ROS_WARN("No valid trajectory found! Trapped in safety mode!");
            in_safety_mode = true;

            //emergency stop?      4th July
            p_store = p0;
            yaw_store = yaw0;

            traj_pt.x = p_store(0);
            traj_pt.y = p_store(1);
            traj_pt.z = p_store(2);
            traj_pt.yaw = yaw_store;

            // TODO: Change the control values in safety mode here, chg

            for (int i = 0; i < 3; ++i) {
                traj_pt.header.stamp = ros::Time::now();
                traj_point_pub.publish(traj_pt);  
            }

            Eigen::MatrixXd show_points = Eigen::MatrixXd::Zero(6, 3);
            for(int pubi = 0; pubi < 6; pubi++)
            {
                show_points(pubi, 0) = p_store(0);
                show_points(pubi, 1) = p_store(1);
                show_points(pubi, 2) = p_store(2);
            }
            marker_publish(show_points);
        }

        double algo_time = ros::Time::now().toSec() - _algorithm_time.toSec();
        ROS_INFO("algorithm time is: %lf", algo_time);
        _algorithm_time = ros::Time::now();

        /// TODO: Publish the control values, chg

        state_locked = false;
    }

    /** Generate trajectory for rotating head **/
    static double last_head_yaw_plan = 0.0;
    static bool first_head_control_flag = true;

    if(first_head_control_flag){
        first_head_control_flag = false;
        sendMotorCommands(0.0, 0.5);
    }
    else{
        /// Rank by the cost given from current velocity direction, the sampled last direction and the last planned head direction
        Eigen::VectorXd cost_vector = Eigen::VectorXd::Zero(LOOKING_PIECES_SIZE);

        /// Current velocity direction(v_direction), planned velocity direction(theta_h_chosen), yaw of the head must be in the same coordinate!!
        double yaw_to_send, yaw_rate_to_send;

        double coefficient_current_v =  1.0 - _direction_update_buffer(getHeadingSeq(v_direction));
        double coefficient_planned_dir =  1.0 - _direction_update_buffer(getHeadingSeq(theta_h_chosen));
        ROS_INFO("coefficient_current_v=%lf, coefficient_planned_dir=%lf", coefficient_current_v, coefficient_planned_dir);

        double min_head_plan_cost = 10000000.0;
        for(int i=0; i<LOOKING_PIECES_SIZE; i++){
            double head_yaw_plan_temp = getHeadingYawFromSeq(i);
            double cost_temp = hp.k_current_v * (v_direction-head_yaw_plan_temp) * (v_direction-head_yaw_plan_temp) * coefficient_current_v
                               + hp.k_planned_dir * (theta_h_chosen-head_yaw_plan_temp) * (theta_h_chosen-head_yaw_plan_temp) * coefficient_planned_dir
                               + hp.k_v_fluctuation * (head_yaw_plan_temp - last_head_yaw_plan) * (head_yaw_plan_temp - last_head_yaw_plan);
            if(cost_temp < min_head_plan_cost)
            {
                min_head_plan_cost = cost_temp;
                yaw_to_send = head_yaw_plan_temp;
            }
        }

        yaw_rate_to_send = 0.9; // const speed for now
        sendMotorCommands(yaw_to_send, yaw_rate_to_send); //send to motor

        last_head_yaw_plan = yaw_to_send;
    }
}


void positionCallback(const geometry_msgs::PoseStamped& msg)
{
    if(!state_locked)
    {
        state_updating = true;

        /** Change from ENU to NWU, NEEDS CAREFUL CHECKING!!!!, chg**/
        p0(0) = msg.pose.position.y;
        p0(1) = -msg.pose.position.x;
        p0(2) = msg.pose.position.z;

        quad.x() = msg.pose.orientation.x;
        quad.y() = msg.pose.orientation.y;   
        quad.z() = msg.pose.orientation.z;
        quad.w() = msg.pose.orientation.w;

        Eigen::Quaternionf q1(0, 0, 0, 1);
        Eigen::Quaternionf axis = quad * q1 * quad.inverse();
        axis.w() = cos(-PI_2/2.0);
        axis.x() = axis.x() * sin(-PI_2/2.0);
        axis.y() = axis.y() * sin(-PI_2/2.0);
        axis.z() = axis.z() * sin(-PI_2/2.0);
        quad = quad * axis;
        /// Update yaw0 here, should be among [-PI, PI] 
        yaw0 = atan2(2*(quad.w()*quad.z()+quad.x()*quad.y()), 1-2*(quad.z()*quad.z()+quad.y()*quad.y()));// - PI_2;
        //if(yaw0 < -PI) yaw0 += PIx2;

        ROS_INFO("Current yaw = %f", yaw0);

        if (!in_safety_mode) {
            p_store = p0;
        }
        state_updating = false;
    }
}


void velocityCallback(const geometry_msgs::TwistStamped& msg)
{
    if(!state_locked)
    {
        state_updating = true;

        /** Change from ENU to NWU, NEEDS CAREFUL CHECKING!!!!, chg**/
        v0(0) = msg.twist.linear.y;
        v0(1) = -msg.twist.linear.x;
        v0(2) = msg.twist.linear.z;
        if(fabs(v0(0)) > 0.1 || fabs(v0(1)) > 0.1){  //add a dead zone
            v_direction = atan2(v0(1), v0(0));  
        }
        // else{
        //     v_direction = 0.0;
        // }

        ROS_INFO("v_direction(yaw) = %f, v0(0)=%f, v0(1)=%f", v_direction, v0(0), v0(1));

        state_updating = false;
    }
}

double init_head_yaw = 0.0;
void motorCallback(const geometry_msgs::Point32& msg)
{
    static bool init_time = true;

    if(init_time)
    {
        init_head_yaw = msg.x;
        init_time = false;
        ROS_INFO("Head Init Yaw in motor coordinate=%f", init_head_yaw);
    }
    else
    {
        motor_yaw = -msg.x + init_head_yaw; // + PI_2?? //start with zero, original z for motor is down. now turn to ENU coordinate. Head forward is PI/2 ???????????
        motor_yaw_rate = -msg.y;
    }
}

void sendMotorCommands(double yaw, double yaw_rate_abs) // Range[-Pi, Pi], [0, 1]
{
    static geometry_msgs::Point32 head_cmd;
    head_cmd.x = -yaw + init_head_yaw;  // + PI_2??  CHG
    head_cmd.y = yaw_rate_abs * 72;
    head_cmd_pub.publish(head_cmd);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "local_planning");
    ros::NodeHandle nh;

    // State parameters initiate
    p_goal << 20.0, 0.0, 1.5;  //x, y, z
    p0 << 0.0, 0.0, 0.0;
    v0 << 0.0, 0.0, 0.0;
    a0 << 0.0, 0.0, 0.0;
    yaw0 = 0.0;
    v_direction = 0.0;

    // Fov sample parameters
    Fov_half << 35, 20;
    // Horizontal angles larger than 90 degree are deleted
    //Angle_h << -90, -70, -50, -30, -20, -10, 0, 10, 20, 30, 50, 70, 90;
    Angle_h << -90, -75, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 75, 90;
    Angle_v << -40, -20, -10, 0, 10, 20, 40;

    Angle_h = Angle_h * M_PI / 180.0;
    Angle_v = Angle_v * M_PI / 180.0;
    Fov_half = Fov_half * M_PI / 180.0;

    _direction_update_buffer = Eigen::VectorXf::Zero(LOOKING_PIECES_SIZE); 
    heading_resolution =  2.f * PI / (float)LOOKING_PIECES_SIZE;
    mid_seq_num = (int)(LOOKING_PIECES_SIZE / 2 - 1); // start from 0, mid is 8 when LOOKING_PIECES_SIZE is 18

    int valid_piece_num = (int)((float)CAMERA_H_FOV / 180.f * PI / heading_resolution);
    if(valid_piece_num % 2 == 0)  valid_piece_num -= 1;
    if(valid_piece_num < 1){
        ROS_ERROR("No enough view field with the current camera set!");
        return 0;
    }
    valid_piece_num_one_side = (valid_piece_num - 1) / 2;

    ROS_INFO("Heading resolution = %f (rad), Fov = %f, valid_piece_num = %d", heading_resolution, CAMERA_H_FOV, valid_piece_num);


    // ringbuffer cloud2
    cloud2_pub = nh.advertise<sensor_msgs::PointCloud2>("/ring_buffer/cloud_ob", 1, true);
    cloud_edf_pub = nh.advertise<sensor_msgs::PointCloud2>("/ring_buffer/edf", 1, true);

    center_pub = nh.advertise<geometry_msgs::PointStamped>("/ring_buffer/center",1,true) ;
    current_marker_pub = nh.advertise<visualization_msgs::Marker>("/visualization_marker", 1);

    traj_point_pub = nh.advertise<px4_autonomy::Position>("/px4/cmd_pose", 5, true); // add on 9 Mar
    head_cmd_pub = nh.advertise<geometry_msgs::Point32>("/gimbal_commands", 2, true); 

    ros::Subscriber position_isolate_sub =  nh.subscribe("/mavros/local_position/pose", 1, positionCallback);
    ros::Subscriber velocity_isolate_sub = nh.subscribe("/mavros/local_position/velocity", 1, velocityCallback);

    ros::Subscriber motor_sub = nh.subscribe("/place_velocity_info", 1, motorCallback);
    ros::Subscriber cloud_sub = nh.subscribe("/camera/depth/color/points", 1, cloudCallback);

    // timer for publish ringbuffer as pointcloud
    ros::Timer timer1 = nh.createTimer(ros::Duration(0.2), timerCallback); // RATE 5 Hz to publish

    // timer for trajectory generation
    ros::Timer timer2 = nh.createTimer(ros::Duration(cal_duration), trajectoryCallback);

    std::cout << "Start mapping!" << std::endl;

    // ros::spin();
    ros::AsyncSpinner spinner(2); // Use 2 threads
    spinner.start();
    ros::waitForShutdown();

    return 0;
}

// Created by clarence on 19-8-29.
//

