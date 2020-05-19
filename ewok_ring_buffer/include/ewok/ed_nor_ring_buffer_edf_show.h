#ifndef EWOK_RING_BUFFER_INCLUDE_EWOK_ED_NOR_RING_BUFFER_H_
#define EWOK_RING_BUFFER_INCLUDE_EWOK_ED_NOR_RING_BUFFER_H_

#include <ewok/raycast_ring_buffer.h>

#include <deque>
#include <vector>

#include <pcl/features/normal_3d.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <iostream>

namespace ewok
{
template <int _POW, typename _Datatype = int16_t, typename _Scalar = float, typename _Flag = uint8_t>
class EuclideanDistanceNormalRingBuffer
{
  public:
    static const int _N = (1 << _POW);  // 2 to the power of POW

    // Other definitions
    typedef Eigen::Matrix<_Scalar, 4, 1> Vector4;
    typedef Eigen::Matrix<_Scalar, 3, 1> Vector3;
    typedef Eigen::Matrix<int, 3, 1> Vector3i;
    typedef std::vector<Vector4, Eigen::aligned_allocator<Vector4>> PointCloud;

    typedef std::shared_ptr<EuclideanDistanceNormalRingBuffer<_POW, _Datatype, _Scalar, _Flag>> Ptr;

    EuclideanDistanceNormalRingBuffer(const _Scalar &resolution, const _Scalar &truncation_distance)
      : resolution_(resolution)
      , truncation_distance_(truncation_distance)
      , occupancy_buffer_(resolution)
      , tmp_buffer1_(resolution)
      , tmp_buffer2_(resolution)
      , distance_buffer_(resolution, truncation_distance)
      , norm_buffer_x_(resolution, _Datatype(0))
      , norm_buffer_y_(resolution, _Datatype(0))
      , norm_buffer_z_(resolution, _Datatype(0))
      , semantic_label_(resolution, _Flag(0)) //CHG
      , label_possibility_(resolution, _Flag(0)) //CHG
    {
        distance_buffer_.setEmptyElement(std::numeric_limits<_Scalar>::max());
    }

    inline void getIdx(const Vector3 &point, Vector3i &idx) const { distance_buffer_.getIdx(point, idx); }

    inline void getPoint(const Vector3i &idx, Vector3 &point) const { distance_buffer_.getPoint(idx, point); }

    inline Vector3i getVolumeCenter() { return distance_buffer_.getVolumeCenter(); }

    void updateDistance() { compute_edt3d(); }

    void insertPointCloud(const PointCloud &cloud, const Vector3 &origin)
    {
        occupancy_buffer_.insertPointCloud(cloud, origin);
    }

    void insertPointCloudDynamic(const PointCloud &cloud, const Vector3 &origin)
    {
        std::cout<<"into dynamic processing!!"<<std::endl;
        occupancy_buffer_.insertPointCloudDynamic(cloud, origin);
    }


    void insertPointCloudSemanticLabel(const pcl::PointCloud<pcl::PointXYZRGB> &cloud_label, bool &objects_updated)
    {
        static int decay_step = 1;
        static int start_line = 5;
        static int inverse_threshold = 3;

        if(objects_updated)
        {
            std::vector<pcl::PointXYZRGB, Eigen::aligned_allocator<pcl::PointXYZRGB>> points = cloud_label.points;
       
            // add label and posibility
            for(int i = 0; i < int(points.size()); ++i)
            //for(int i = 0; i < cloud_label.points.size(); ++i)
            {
                Vector3 p;
                Vector3i idx; ///CHG
                p(0) = points.at(i).x;
                p(1) = points.at(i).y;
                p(2) = points.at(i).z;
                semantic_label_.getIdx(p, idx);

                if(semantic_label_.insideVolume(idx))
                {
                    // Update labels and possibility
                    float label_temp = points.at(i).rgb; //cloud_label.points[i].intensity;

                    if(semantic_label_.at(idx) == label_temp) // The same label comes, add possibility
                    {
                        if(label_possibility_.at(idx) < 250) label_possibility_.at(idx) += 3;
                    }
                    else  // A different label comes
                    {
                        if(label_temp == 3 && semantic_label_.at(idx) > 3) // 3: ordinary obstacle, reduce possibility
                        {
                            if(label_possibility_.at(idx) > inverse_threshold)
                                label_possibility_.at(idx) -= decay_step; // decay by time

                            if(label_possibility_.at(idx) <= inverse_threshold)
                            {
                                semantic_label_.at(idx) = 3; // 3: ordinary obstacle
                                label_possibility_.at(idx) = start_line;
                            }
                        }
                        else if(label_temp == 3 && semantic_label_.at(idx) < 3) // Original value is 0, free space is 2, set as ordinary obstacle: 3
                        {
                            semantic_label_.at(idx) = 3;
                            label_possibility_.at(idx) = start_line;
                        }
                        else if(label_temp == 0)
                        {
                            if(label_possibility_.at(idx) > decay_step)
                                label_possibility_.at(idx) -= decay_step; // decay by time
                        }
                        else 
                        {
                            semantic_label_.at(idx) = label_temp;  // New type(Not ordinary obstacle) comes. Replace
                            label_possibility_.at(idx) = start_line;
                            //std::cout << "label=" <<  label_temp << " ";
                        }
                    }
                }
            }
        }

        objects_updated = false;
    }

    // Add normal updating
    void insertPointCloudNormal(const pcl::PointCloud<pcl::PointNormal> &pn)
    {
        std::vector<pcl::PointNormal, Eigen::aligned_allocator<pcl::PointNormal>> points = pn.points;
        // add normal to belonging voxel
        for(int i = 0; i < int(points.size()); ++i)
        {
            Vector3 p;
            Vector3i idx; ///CHG
            p(0) = points.at(i).x;
            p(1) = points.at(i).y;
            p(2) = points.at(i).z;
            norm_buffer_x_.getIdx(p, idx);
            if(norm_buffer_x_.insideVolume(idx))
            {
                norm_buffer_x_.at(idx) += points.at(i).normal_x;
                norm_buffer_y_.at(idx) += points.at(i).normal_y;
                norm_buffer_z_.at(idx) += points.at(i).normal_z;
            }
        }

        // normalize all normal
        Vector3i off;
        norm_buffer_x_.getOffset(off);
        for(int x = 0; x < _N; x++)
        {
            for(int y = 0; y < _N; y++)
            {
                for(int z = 0; z < _N; z++)
                {
                    Vector3i coord(x, y, z);
                    coord += off;
                    if(occupancy_buffer_.isOccupied(coord))
                    {
                        float norm = sqrt(pow(norm_buffer_x_.at(coord), 2) + pow(norm_buffer_y_.at(coord), 2) +
                                          pow(norm_buffer_z_.at(coord), 2));
                        if(norm < 1e-3)
                            norm_buffer_x_.at(coord) = norm_buffer_y_.at(coord) = norm_buffer_z_.at(coord) = 0.0;
                        else
                        {
                            norm_buffer_x_.at(coord) /= norm;
                            norm_buffer_y_.at(coord) /= norm;
                            norm_buffer_z_.at(coord) /= norm;
                        }

                        // std::cout << "Normal at " << coord.transpose() << " is:" << norm_buffer_x_.at(coord) << ","
                        //           << norm_buffer_y_.at(coord) << "," << norm_buffer_z_.at(coord) << std::endl;
                    }
                }
            }
        }
    }

    // Add normal offset
    virtual void setOffset(const Vector3i &off)
    {
        occupancy_buffer_.setOffset(off);
        distance_buffer_.setOffset(off);
        norm_buffer_x_.setOffset(off);
        norm_buffer_y_.setOffset(off);
        norm_buffer_z_.setOffset(off);
        semantic_label_.setOffset(off);
        label_possibility_.setOffset(off);
    }

    // Add normal moveVolume
    virtual void moveVolume(const Vector3i &direction)
    {
        occupancy_buffer_.moveVolume(direction);
        distance_buffer_.moveVolume(direction);
        norm_buffer_x_.moveVolume(direction);
        norm_buffer_y_.moveVolume(direction);
        norm_buffer_z_.moveVolume(direction);
        semantic_label_.moveVolume(direction);
        label_possibility_.moveVolume(direction);
    }

    // get ringbuffer as pointcloud
    void getBufferAsCloud(pcl::PointCloud<pcl::PointXYZ> &cloud, Eigen::Vector3d &center)
    {
        // get center of ring buffer
        Vector3i c_idx = getVolumeCenter();
        Vector3 ct;
        getPoint(c_idx, ct);
        center(0) = ct(0);
        center(1) = ct(1);
        center(2) = ct(2);

        // convert ring buffer to point cloud
        Vector3i off;
        norm_buffer_x_.getOffset(off);
        for(int x = 0; x < _N; x++)
        {
            for(int y = 0; y < _N; y++)
            {
                for(int z = 0; z < _N; z++)
                {
                    // only occupied voxel is return
                    Vector3i coord(x, y, z);
                    coord += off;
                    if(occupancy_buffer_.isOccupied(coord))
                    {
                        Vector3 p;
                        getPoint(coord, p);
                        pcl::PointXYZ pclp;
                        pclp.x = p(0);
                        pclp.y = p(1);
                        pclp.z = p(2);
                        cloud.points.push_back(pclp);
                    }
                }
            }
        }
    }

    // get semantic ring buffer. intensity is samantic label
    void getBufferSemanticCloud(pcl::PointCloud<pcl::PointXYZI> &cloud, Eigen::Vector3d &center, double dir_x, double dir_y)
    {
        static int inverse_threshold = 3;

        // get center of ring buffer
        Vector3i c_idx = getVolumeCenter();
        Vector3 ct;
        getPoint(c_idx, ct);
        center(0) = ct(0);
        center(1) = ct(1);
        center(2) = ct(2);

        // convert ring buffer to point cloud
        Vector3i off;
        semantic_label_.getOffset(off);
        for(int x = 0; x < _N; x++)
        {
            for(int y = 0; y < _N; y++)
            {
                for(int z = 0; z < _N; z++)
                {
                    // only occupied voxel is return
                    Vector3i coord(x, y, z);
                    coord += off;
                    if(occupancy_buffer_.isOccupied(coord))
                    {
                        Vector3 p;
                        getPoint(coord, p);

                        pcl::PointXYZI pclp;
                        pclp.x = p(0);
                        pclp.y = p(1);
                        pclp.z = p(2);

                        if(semantic_label_.at(coord) > 3.f && label_possibility_.at(coord) > inverse_threshold)
                            pclp.intensity = semantic_label_.at(coord);
                        else
                            pclp.intensity = 3.f; //Ordernary obstacle

                        cloud.points.push_back(pclp);
                        //if(pclp.intensity > 4.f) std::cout << "person\t";
                    }
                    else if(occupancy_buffer_.isFree(coord)) //Free space semantic label should be eliminated
                    {
                        Vector3 p;
                        getPoint(coord, p);

                        pcl::PointXYZI pclp;
                        pclp.x = p(0);
                        pclp.y = p(1);
                        pclp.z = p(2);

                        // Consider x, y plain for fly direction
                        double temp_x = pclp.x - center(0);
                        double temp_y = pclp.y - center(1);

                        double temp_len = sqrt(temp_x * temp_x + temp_y * temp_y);
                        double x_norm = temp_x / temp_len;
                        double y_norm = temp_y / temp_len;

                        if(x_norm * dir_x + y_norm * dir_y > 0.8)
                        {
                            semantic_label_.at(coord) = 1; //Free space target
                            label_possibility_.at(coord) = 5;
                            pclp.intensity = 1;
                        }
                        else
                        {
                            semantic_label_.at(coord) = 2; //Free space
                            label_possibility_.at(coord) = 5;
                            pclp.intensity = 2;
                        }
                    
                        cloud.points.push_back(pclp);
                    }
                }
            }
        }
    }

    // get obstacle semantic ring buffer. intensity is samantic label
    void getBufferObstacleSemanticCloud(pcl::PointCloud<pcl::PointXYZI> &cloud, Eigen::Vector3d &center, double dir_x, double dir_y)
    {
        static int inverse_threshold = 3;

        // get center of ring buffer
        Vector3i c_idx = getVolumeCenter();
        Vector3 ct;
        getPoint(c_idx, ct);
        center(0) = ct(0);
        center(1) = ct(1);
        center(2) = ct(2);

        // convert ring buffer to point cloud
        Vector3i off;
        semantic_label_.getOffset(off);
        for(int x = 0; x < _N; x++)
        {
            for(int y = 0; y < _N; y++)
            {
                for(int z = 0; z < _N; z++)
                {
                    // only occupied voxel is return
                    Vector3i coord(x, y, z);
                    coord += off;
                    if(occupancy_buffer_.isOccupied(coord))
                    {
                        Vector3 p;
                        getPoint(coord, p);

                        pcl::PointXYZI pclp;
                        pclp.x = p(0);
                        pclp.y = p(1);
                        pclp.z = p(2);

                        if(semantic_label_.at(coord) > 3.f && label_possibility_.at(coord) > inverse_threshold)
                            pclp.intensity = semantic_label_.at(coord);
                        else
                            pclp.intensity = 3.f; //Ordernary obstacle

                        cloud.points.push_back(pclp);
                        //if(pclp.intensity > 4.f) std::cout << "person\t";
                    }
                }
            }
        }
    }

    // get ringbuffer free space pointcloud
    void getBufferFSCloud(pcl::PointCloud<pcl::PointXYZ> &cloud, Eigen::Vector3d &center)
    {
        // get center of ring buffer
        Vector3i c_idx = getVolumeCenter();
        Vector3 ct;
        getPoint(c_idx, ct);
        center(0) = ct(0);
        center(1) = ct(1);
        center(2) = ct(2);

        // convert ring buffer to point cloud
        Vector3i off;
        norm_buffer_x_.getOffset(off);
        for(int x = 0; x < _N; x++)
        {
            for(int y = 0; y < _N; y++)
            {
                for(int z = 0; z < _N; z++)
                {
                    // only free voxel is return
                    Vector3i coord(x, y, z);
                    coord += off;
                                       
                    if(occupancy_buffer_.isFree(coord))
                    {

                        Vector3 p;
                        getPoint(coord, p); 
                        pcl::PointXYZ pclp;

                        pclp.x = p(0);
                        pclp.y = p(1);
                        pclp.z = p(2);
                        cloud.points.push_back(pclp);
                    }
                   
                }
            }
        }
    }

    // Add normal visualizer
    void getMarkerNormal(visualization_msgs::Marker &m)
    {
        // use line list to represent normal
        m.type = visualization_msgs::Marker::LINE_LIST;
        m.header.frame_id = "world";
        m.header.stamp = ros::Time::now();
        m.ns = "normal";
        m.id = 0;
        m.action = visualization_msgs::Marker::MODIFY;
        m.scale.x = 0.03;

        m.color.r = m.color.g = 1.0;
        m.color.b = 0.0;
        m.color.a = 0.5;

        Vector3i off;
        norm_buffer_x_.getOffset(off);

        // add normal to line list
        for(int x = 0; x < _N; x++)
        {
            for(int y = 0; y < _N; y++)
            {
                for(int z = 0; z < _N; z++)
                {
                    // only visualize occupied points' normal
                    Vector3i coord(x, y, z);
                    coord += off;
                    if(!occupancy_buffer_.isOccupied(coord)) continue;

                    geometry_msgs::Point p;
                    // start point of normal, at voxel's center
                    p.x = coord(0) * resolution_;
                    p.y = coord(1) * resolution_;
                    p.z = coord(2) * resolution_;
                    m.points.push_back(p);

                    // end point of normal
                    p.x += 0.5 * norm_buffer_x_.at(coord);
                    p.y += 0.5 * norm_buffer_y_.at(coord);
                    p.z += 0.5 * norm_buffer_z_.at(coord);
                    m.points.push_back(p);
                }
            }
        }
    }

    void getMarkerFree(visualization_msgs::Marker &m) { occupancy_buffer_.getMarkerFree(m); }

    void getMarkerOccupied(visualization_msgs::Marker &m) { occupancy_buffer_.getMarkerOccupied(m); }

    void getMarkerUpdated(visualization_msgs::Marker &m) { occupancy_buffer_.getMarkerUpdated(m); }

    void getMarkerDistance(visualization_msgs::Marker &m, _Scalar distance)
    {
        distance_buffer_.getMarkerHelper(m, "ring_buffer_distance", 0, Vector4(0, 0, 1, 0.5),
                                         [=](const _Scalar &d) { return d <= distance; });
    }

    template <class Derived>
    _Scalar getDistanceWithGrad(const Eigen::MatrixBase<Derived> &point_const,
                                const Eigen::MatrixBase<Derived> &grad_const)
    {
        EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 3);
        Eigen::MatrixBase<Derived> &grad = const_cast<Eigen::MatrixBase<Derived> &>(grad_const);

        Vector3 point = point_const.template cast<_Scalar>();
        Vector3 point_m = point.array() - 0.5 * resolution_;

        Vector3i idx;
        distance_buffer_.getIdx(point_m, idx);

        Vector3 idx_point, diff;
        distance_buffer_.getPoint(idx, idx_point);

        diff = (point - idx_point) / resolution_;

        bool all_valid = true;
        _Scalar values[2][2][2];

        for(int x = 0; x < 2 && all_valid; x++)
        {
            for(int y = 0; y < 2 && all_valid; y++)
            {
                for(int z = 0; z < 2 && all_valid; z++)
                {
                    Vector3i current_idx = idx + Vector3i(x, y, z);

                    if(distance_buffer_.insideVolume(current_idx))
                    {
                        values[x][y][z] = distance_buffer_.at(current_idx);
                    }
                    else
                    {
                        all_valid = false;
                    }
                }
            }
        }

        if(all_valid)
        {
            // Trilinear interpolation
            _Scalar v00 = (1 - diff[0]) * values[0][0][0] + diff[0] * values[1][0][0];
            _Scalar v01 = (1 - diff[0]) * values[0][0][1] + diff[0] * values[1][0][1];
            _Scalar v10 = (1 - diff[0]) * values[0][1][0] + diff[0] * values[1][1][0];
            _Scalar v11 = (1 - diff[0]) * values[0][1][1] + diff[0] * values[1][1][1];

            _Scalar v0 = (1 - diff[1]) * v00 + diff[1] * v10;
            _Scalar v1 = (1 - diff[1]) * v01 + diff[1] * v11;

            _Scalar v = (1 - diff[2]) * v0 + diff[2] * v1;

            grad[2] = (v1 - v0) / resolution_;
            grad[1] = ((1 - diff[2]) * (v10 - v00) + diff[2] * (v11 - v01)) / resolution_;

            grad[0] = (1 - diff[2]) * (1 - diff[1]) * (values[1][0][0] - values[0][0][0]);
            grad[0] += (1 - diff[2]) * diff[1] * (values[1][1][0] - values[0][1][0]);
            grad[0] += diff[2] * (1 - diff[1]) * (values[1][0][1] - values[0][0][1]);
            grad[0] += diff[2] * diff[1] * (values[1][1][1] - values[0][1][1]);

            grad[0] /= resolution_;

            return v;
        }
        else
        {
            return truncation_distance_;
        }
    }

    bool collision_checking(Eigen::Vector3f *traj_points, int Num, _Scalar threshold) {

        bool all_safe = true;

        for (int i = 0; i < Num; ++i) {
            Vector3 traj_point = traj_points[i].template cast<_Scalar>();
            Vector3i traj_point_idx;

            distance_buffer_.getIdx(traj_point, traj_point_idx);

            // std::cout<<"point_ori=("<< traj_point(0) <<","<< traj_point(1)<<","<<traj_point(2) << ")"<<std::endl;
            // std::cout<<"point_idx=("<< traj_point_idx(0) <<","<< traj_point_idx(1)<<","<<traj_point_idx(2) << ")"<<std::endl;
            // std::cout << "dis:" << distance_buffer_.at(traj_point_idx) << std::endl;

            if (distance_buffer_.insideVolume(traj_point_idx)) {  //if inside
                if (distance_buffer_.at(traj_point_idx) < threshold) {
                    all_safe = false;
                    break;
                }
                
            } else {
                all_safe = false;
                break;
            }
        }
        // std::cout<<"************************"<<std::endl;
        return all_safe;
    }

    Vector3i getRGBFromDistance(float x, float y, float z) {
        Vector3 cloud_point;
        Vector3i cloud_point_idx;
        cloud_point(0) = (_Scalar)x;
        cloud_point(1) = (_Scalar)y;
        cloud_point(2) = (_Scalar)z;

        distance_buffer_.getIdx(cloud_point, cloud_point_idx);
        float distance = (float)distance_buffer_.at(cloud_point_idx);

        int value = floor(distance * 240); // Mapping 0~1.0 to 0~240 degree
        value = value > 240 ? 240 : value;

        // 240 degrees are divided into 4 sections, 0 is Red-dominant, 1 and 2 are Green-dominant,
        // 3 is Blue-dominant. The dorminant color is 255 and another color is always 0 while the
        // remaining color increases or decreases between 0~255
        int section = value / 60;
        float float_key = (value % 60) / (float)60 * 255;
        int key = floor(float_key);
        int nkey = 255 - key;

        Vector3i point_RGB;

        switch(section) {
            case 0: // G increase
                point_RGB(0) = 255;
                point_RGB(1) = key;
                point_RGB(2) = 0;
                break;
            case 1: // R decrease
                point_RGB(0) = nkey;
                point_RGB(1) = 255;
                point_RGB(2) = 0;
                break;
            case 2: // B increase
                point_RGB(0) = 0;
                point_RGB(1) = 255;
                point_RGB(2) = key;
                break;
            case 3: // G decrease
                point_RGB(0) = 0;
                point_RGB(1) = nkey;
                point_RGB(2) = 255;
                break;
            case 4:
                point_RGB(0) = 0;
                point_RGB(1) = 0;
                point_RGB(2) = 255;
                break;
            default: // White
            	point_RGB(0) = 255;
                point_RGB(1) = 255;
                point_RGB(2) = 255;
        }

        return point_RGB;
    }

    // Compute entropy
    // The value of distance transform is a 0.0 to 1.0 range, which is divided into 10 sections
    // Compute the probability that the value lies in each section
    _Scalar compute_entropy2d(float origin_x, float origin_y, float origin_z) {
        int frequency[10][10] = {0};

        Vector3 entro_point;
        Vector3i entro_point_idx;
        Vector3 adjac_entro_point;
        Vector3i adjac_entro_point_idx;

        // Compute area is a polygon in front of the drone to assure that all the elements
        // are in FOV
        for (int i = 5; i <= 30; ++i) {
            if (i <= 10) {
                for (int j = -5 - (i-5); j <= 5 + (i-5); ++j) {
                    _Scalar x = (_Scalar)origin_x + j * resolution_;
                    _Scalar y = (_Scalar)origin_y + i * resolution_;
                    _Scalar z = (_Scalar)origin_z;

                    entro_point(0) = x;
                    entro_point(1) = y;
                    entro_point(2) = z;
                    distance_buffer_.getIdx(entro_point, entro_point_idx);
                    _Scalar distance = distance_buffer_.at(entro_point_idx);

                    // 0-1.0 -> 0-10.0
                    distance *= 10.0;
                    distance = distance >= 10.0 ? 9.0 : distance;
                    int center_dis = floor(distance);

                    int dir_x[8] = {1, 1, 0, -1, -1, -1, 0, 1};
                    int dir_y[8] = {0, 1, 1, 1, 0, -1, -1, -1};
                    _Scalar adjac_distance = 0;
                    int adjac_dis;
                    _Scalar adjac_x;
                    _Scalar adjac_y;
                    _Scalar adjac_z;

                    for (int k = 0; k < 8; ++k) {
                        adjac_x = x + dir_x[k] * resolution_;
                        adjac_y = y + dir_y[k] * resolution_;
                        adjac_z = z;

                        adjac_entro_point(0) = adjac_x;
                        adjac_entro_point(1) = adjac_y;
                        adjac_entro_point(2) = adjac_z;
                        distance_buffer_.getIdx(adjac_entro_point, adjac_entro_point_idx);
                        adjac_distance += distance_buffer_.at(adjac_entro_point_idx);
                    }

                    // Average value of the distance transform of the 8 adjacent elements
                    adjac_distance = adjac_distance / 8.0 * 10.0;
                    adjac_distance = adjac_distance > 10.0 ? 9.0 : adjac_distance;
                    adjac_dis = floor(adjac_distance);

                    ++frequency[center_dis][adjac_dis];
                }
            } else {
                for (int j = -10; j <= 10; ++j) {
                    _Scalar x = (_Scalar)origin_x + j * resolution_;
                    _Scalar y = (_Scalar)origin_y + i * resolution_;
                    _Scalar z = (_Scalar)origin_z;

                    entro_point(0) = x;
                    entro_point(1) = y;
                    entro_point(2) = z;
                    distance_buffer_.getIdx(entro_point, entro_point_idx);
                    _Scalar distance = distance_buffer_.at(entro_point_idx);

                    // 0-0.5 -> 0-10
                    distance *= 10.0;
                    distance = distance >= 10.0 ? 9.0 : distance;
                    int center_dis = floor(distance);

                    int dir_x[8] = {1, 1, 0, -1, -1, -1, 0, 1};
                    int dir_y[8] = {0, 1, 1, 1, 0, -1, -1, -1};
                    _Scalar adjac_distance = 0;
                    int adjac_dis;
                    _Scalar adjac_x;
                    _Scalar adjac_y;
                    _Scalar adjac_z;

                    for (int k = 0; k < 8; ++k) {
                        adjac_x = x + dir_x[k] * resolution_;
                        adjac_y = y + dir_y[k] * resolution_;
                        adjac_z = z;

                        adjac_entro_point(0) = adjac_x;
                        adjac_entro_point(1) = adjac_y;
                        adjac_entro_point(2) = adjac_z;
                        distance_buffer_.getIdx(adjac_entro_point, adjac_entro_point_idx);
                        adjac_distance += distance_buffer_.at(adjac_entro_point_idx);
                    }

                    // Average of adjacent distance
                    adjac_distance = adjac_distance / 8.0 * 10.0;
                    adjac_distance = adjac_distance > 10.0 ? 9.0 : adjac_distance;
                    adjac_dis = floor(adjac_distance);

                    ++frequency[center_dis][adjac_dis];
                }
            }
        }

        _Scalar entropy = 0;
        _Scalar freq;
        for (int i = 0; i < 10; ++i) {
            for (int j = 0; j < 10; ++j) {
                if (frequency[i][j] == 0) {
                    continue;
                } else {
                    freq = frequency[i][j]/516.0;
                    entropy -= freq*log(freq);
                }
            }
        }

        return entropy;
    }

    _Scalar compute_entropy1d(float origin_x, float origin_y, float origin_z) {
        int frequency[10] = {0};

        Vector3 entro_point;
        Vector3i entro_point_idx;

        for (int i = 5; i <= 30; ++i) {
            if (i <= 10) {
                for (int j = -5 - (i-5); j <= 5 + (i-5); ++j) {
                    _Scalar x = (_Scalar)origin_x + j * resolution_;
                    _Scalar y = (_Scalar)origin_y + i * resolution_;
                    _Scalar z = (_Scalar)origin_z;

                    entro_point(0) = x;
                    entro_point(1) = y;
                    entro_point(2) = z;
                    distance_buffer_.getIdx(entro_point, entro_point_idx);
                    _Scalar distance = distance_buffer_.at(entro_point_idx);

                    // 0-1.0 -> 0-10
                    distance *= 10.0;
                    distance = distance >= 10.0 ? 9.0 : distance;
                    int center_dis = floor(distance);

                    ++frequency[center_dis];
                }
            } else {
                for (int j = -10; j <= 10; ++j) {
                    _Scalar x = (_Scalar)origin_x + j * resolution_;
                    _Scalar y = (_Scalar)origin_y + i * resolution_;
                    _Scalar z = (_Scalar)origin_z;

                    entro_point(0) = x;
                    entro_point(1) = y;
                    entro_point(2) = z;
                    distance_buffer_.getIdx(entro_point, entro_point_idx);
                    _Scalar distance = distance_buffer_.at(entro_point_idx);

                    // 0-1.0 -> 0-10
                    distance *= 10.0;
                    distance = distance >= 10.0 ? 9.0 : distance;
                    int center_dis = floor(distance);

                    ++frequency[center_dis];
                }
            }
        }

        _Scalar entropy = 0;
        _Scalar freq;
        for (int i = 0; i < 10; ++i) {
            if (frequency[i] == 0) {
                continue;
            } else {
                freq = frequency[i]/516.0;
                entropy -= freq*log(freq);
            }
        }

        return entropy;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  protected:
    void compute_edt3d()
    {
        Vector3i offset;
        distance_buffer_.getOffset(offset); //off_set in distance buffer, integer, sequence of distance ring buffer

        Vector3i min_vec, max_vec; // To store the max and min index of the points whose distance value is required
        occupancy_buffer_.getUpdatedMinMax(min_vec, max_vec); // min_vec = off_set in occupancy buffer + (N-1), max_vec = off_set in occupancy buffer

        min_vec -= offset;  // min_vec = off_set in occupancy buffer - off_set in distance buffer + (N-1)
        max_vec -= offset;  // max_vec = off_set in occupancy buffer - off_set in distance buffer

        min_vec.array() -= truncation_distance_ / resolution_; // 1.0 / 0.2 = 5, truncation_distance_ is set by user, 1.0 in local_planning.cpp
        max_vec.array() += truncation_distance_ / resolution_; // Shrink the computing area

        min_vec.array() = min_vec.array().max(Vector3i(0, 0, 0).array());  // Set a limit, in case the calculated min_vec is too large
        max_vec.array() = max_vec.array().min(Vector3i(_N - 1, _N - 1, _N - 1).array());

        // ROS_INFO_STREAM("min_vec: " << min_vec.transpose() << " max_vec: " << max_vec.transpose());

        for(int x = min_vec[0]; x <= max_vec[0]; x++)  //x, y plane search
        {
            for(int y = min_vec[1]; y <= max_vec[1]; y++)
            {
                fill_edt(
                    [&](int z) {
                        return occupancy_buffer_.isOccupied(offset + Vector3i(x, y, z)) ?
                                   0 :
                                   std::numeric_limits<_Scalar>::max(); // _Scalar: float
                    },
                    [&](int z, _Scalar val) { tmp_buffer1_.at(Vector3i(x, y, z)) = val; }, min_vec[2], max_vec[2]);
            }
        }

        for(int x = min_vec[0]; x <= max_vec[0]; x++)  // x, z plane search
        {
            for(int z = min_vec[2]; z <= max_vec[2]; z++)
            {
                fill_edt([&](int y) { return tmp_buffer1_.at(Vector3i(x, y, z)); },
                         [&](int y, _Scalar val) { tmp_buffer2_.at(Vector3i(x, y, z)) = val; }, min_vec[1], max_vec[1]);
            }
        }

        for(int y = min_vec[1]; y <= max_vec[1]; y++) // y, z plane search
        {
            for(int z = min_vec[2]; z <= max_vec[2]; z++)
            {
                fill_edt([&](int x) { return tmp_buffer2_.at(Vector3i(x, y, z)); },
                         [&](int x, _Scalar val) {
                             distance_buffer_.at(offset + Vector3i(x, y, z)) =
                                 std::min(resolution_ * std::sqrt(val), truncation_distance_);  // Set final offset
                         },
                         min_vec[0], max_vec[0]);
            }
        }

        occupancy_buffer_.clearUpdatedMinMax();
    }

    template <typename F_get_val, typename F_set_val>
    void fill_edt(F_get_val f_get_val, F_set_val f_set_val, int start = 0, int end = _N - 1) // CHG, distance field calculation
    {
        int v[_N];
        _Scalar z[_N + 1];

        int k = start;
        v[start] = start;
        z[start] = -std::numeric_limits<_Scalar>::max();
        z[start + 1] = std::numeric_limits<_Scalar>::max();

        for(int q = start + 1; q <= end; q++)
        {
            k++;
            _Scalar s;

            do
            {
                k--;
                s = ((f_get_val(q) + q * q) - (f_get_val(v[k]) + v[k] * v[k])) / (2 * q - 2 * v[k]);
                // ROS_INFO_STREAM("k: " << k << " s: " <<  s  << " z[k] " << z[k] << " v[k] " << v[k]);

            } while(s <= z[k]);

            k++;
            v[k] = q;
            z[k] = s;
            z[k + 1] = std::numeric_limits<_Scalar>::max();
        }

        k = start;

        for(int q = start; q <= end; q++)
        {
            while(z[k + 1] < q) k++;
            _Scalar val = (q - v[k]) * (q - v[k]) + f_get_val(v[k]);
            //      if(val < std::numeric_limits<_Scalar>::max())
            //  ROS_INFO_STREAM("val: " << val << " q: " << q << " v[k] " << v[k]);
            // if(val > truncation_distance_*truncation_distance_) val = std::numeric_limits<_Scalar>::max();
            f_set_val(q, val);
        }
    }

    _Scalar resolution_;
    _Scalar truncation_distance_;

    RaycastRingBuffer<_POW, _Datatype, _Scalar, _Flag> occupancy_buffer_;

    RingBufferBase<_POW, _Scalar, _Scalar> distance_buffer_;  // CHG

    RingBufferBase<_POW, _Scalar, _Scalar> tmp_buffer1_, tmp_buffer2_;

    // Normal buffer of x, y, z
    RingBufferBase<_POW, _Scalar, _Scalar> norm_buffer_x_;  // CHG
    RingBufferBase<_POW, _Scalar, _Scalar> norm_buffer_y_;
    RingBufferBase<_POW, _Scalar, _Scalar> norm_buffer_z_;

    // Buffer for semantic label and posibility
    RingBufferBase<_POW, _Scalar, _Scalar> semantic_label_; //0-255
    RingBufferBase<_POW, _Scalar, _Scalar> label_possibility_; //0-255

};
}

#endif  // EWOK_RING_BUFFER_INCLUDE_EWOK_ED_RING_BUFFER_H_
