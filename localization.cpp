#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/RegionOfInterest.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <geometry_msgs/TransformStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <std_srvs/Empty.h>
#include <vector>
#include <deque>
#include <cmath>
#include <iostream>

// Global variables
nav_msgs::OccupancyGrid mapMsg;
cv::Mat mapCropped;
cv::Mat mapTemp;
sensor_msgs::RegionOfInterest mapRoiInfo;
std::vector<cv::Point2f> scanPoints;
ros::ServiceClient clearCostmapsClient;
std::string baseFrame;
std::string odomFrame;
std::string laserFrame;
std::string laserTopic;

float lidarX = 0.0f;
float lidarY = 0.0f;
float lidarYaw = 0.0f;
const float degToRad = M_PI / 180.0f;
int clearCountdown = -1;
int scanCount = 0;

std::deque<std::tuple<float, float, float>> dataQueue;
const size_t maxQueueSize = 10;

void initialPoseCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg) {
    // Extract pose from message
    double mapX = msg->pose.pose.position.x;
    double mapY = msg->pose.pose.position.y;
    tf2::Quaternion quaternion;
    tf2::fromMsg(msg->pose.pose.orientation, quaternion);

    // Convert quaternion to yaw
    tf2::Matrix3x3 matrix(quaternion);
    double roll, pitch, yaw;
    matrix.getRPY(roll, pitch, yaw);

    // Validate map data
    if (mapMsg.info.resolution <= 0) {
        ROS_ERROR("Invalid map resolution");
        return;
    }

    // Convert map coordinates (meters) to cropped grid coordinates (pixels)
    lidarX = (mapX - mapMsg.info.origin.position.x) / mapMsg.info.resolution - mapRoiInfo.x_offset;
    lidarY = (mapY - mapMsg.info.origin.position.y) / mapMsg.info.resolution - mapRoiInfo.y_offset;
    lidarYaw = -yaw;

    // Trigger costmap clearing after 30 iterations
    clearCountdown = 30;
    ROS_INFO("Initial pose set successfully");
}

void cropMap() {
    // Extract map metadata
    nav_msgs::MapMetaData info = mapMsg.info;

    // Initialize bounding box for occupied cells
    unsigned int xMin = info.width / 2, xMax = xMin;
    unsigned int yMin = info.height / 2, yMax = yMin;
    bool firstPoint = true;

    // Convert map to OpenCV matrix
    cv::Mat mapRaw(info.height, info.width, CV_8UC1, cv::Scalar(128));
    for (unsigned int y = 0; y < info.height; ++y) {
        for (unsigned int x = 0; x < info.width; ++x) {
            unsigned int index = y * info.width + x;
            mapRaw.at<uchar>(y, x) = static_cast<uchar>(mapMsg.data[index]);

            // Update bounding box for occupied cells
            if (mapMsg.data[index] == 100) {
                if (firstPoint) {
                    xMin = xMax = x;
                    yMin = yMax = y;
                    firstPoint = false;
                } else {
                    xMin = std::min(xMin, x);
                    xMax = std::max(xMax, x);
                    yMin = std::min(yMin, y);
                    yMax = std::max(yMax, y);
                }
            }
        }
    }

    // Calculate ROI center and size
    unsigned int centerX = (xMin + xMax) / 2;
    unsigned int centerY = (yMin + yMax) / 2;
    unsigned int halfWidth = (xMax - xMin) / 2 + 50;
    unsigned int halfHeight = (yMax - yMin) / 2 + 50;
    unsigned int originX = centerX > halfWidth ? centerX - halfWidth : 0;
    unsigned int originY = centerY > halfHeight ? centerY - halfHeight : 0;
    unsigned int width = halfWidth * 2;
    unsigned int height = halfHeight * 2;

    // Ensure ROI is within map bounds
    originX = std::max(0u, originX);
    width = std::min(info.width - originX, width);
    originY = std::max(0u, originY);
    height = std::min(info.height - originY, height);

    // Crop map
    cv::Rect roi(originX, originY, width, height);
    mapCropped = mapRaw(roi).clone();

    // Store ROI info
    mapRoiInfo.x_offset = originX;
    mapRoiInfo.y_offset = originY;
    mapRoiInfo.width = width;
    mapRoiInfo.height = height;

    // Set default initial pose
    geometry_msgs::PoseWithCovarianceStamped initPose;
    initPose.pose.pose.position.x = -3.0;
    initPose.pose.pose.position.y = 1.0;
    initPose.pose.pose.orientation.z = 0.0;
    initPose.pose.pose.orientation.w = 1.0;
    ROS_INFO("Calling initialPoseCallback with default pose");
    initialPoseCallback(geometry_msgs::PoseWithCovarianceStamped::ConstPtr(
        new geometry_msgs::PoseWithCovarianceStamped(initPose)));
}

cv::Mat createGradientMask(int size) {
    cv::Mat mask(size, size, CV_8UC1);
    int center = size / 2;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            double distance = std::hypot(x - center, y - center);
            mask.at<uchar>(y, x) = cv::saturate_cast<uchar>(255 * std::max(0.0, 1.0 - distance / center));
        }
    }
    return mask;
}

void processMap() {
    if (mapCropped.empty()) {
        ROS_WARN("Cropped map is empty");
        return;
    }

    // Initialize temporary map
    mapTemp = cv::Mat::zeros(mapCropped.size(), CV_8UC1);
    cv::Mat gradientMask = createGradientMask(101);

    // Apply gradient mask to occupied cells
    for (int y = 0; y < mapCropped.rows; ++y) {
        for (int x = 0; x < mapCropped.cols; ++x) {
            if (mapCropped.at<uchar>(y, x) == 100) {
                int left = std::max(0, x - 50);
                int top = std::max(0, y - 50);
                int right = std::min(mapCropped.cols - 1, x + 50);
                int bottom = std::min(mapCropped.rows - 1, y + 50);

                cv::Rect roi(left, top, right - left + 1, bottom - top + 1);
                cv::Mat region = mapTemp(roi);

                int maskLeft = 50 - (x - left);
                int maskTop = 50 - (y - top);
                cv::Rect maskRoi(maskLeft, maskTop, roi.width, roi.height);
                cv::Mat mask = gradientMask(maskRoi);

                cv::max(region, mask, region);
            }
        }
    }
}

void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
    mapMsg = *msg;
    cropMap();
    processMap();
}

bool checkConvergence(float x, float y, float yaw) {
    if (x == 0 && y == 0 && yaw == 0) {
        dataQueue.clear();
        return true;
    }

    dataQueue.push_back(std::make_tuple(x, y, yaw));
    if (dataQueue.size() > maxQueueSize) {
        dataQueue.pop_front();
    }

    if (dataQueue.size() == maxQueueSize) {
        auto& first = dataQueue.front();
        auto& last = dataQueue.back();
        float dx = std::abs(std::get<0>(last) - std::get<0>(first));
        float dy = std::abs(std::get<1>(last) - std::get<1>(first));
        float dyaw = std::abs(std::get<2>(last) - std::get<2>(first));

        if (dx < 5 && dy < 5 && dyaw < 5 * degToRad) {
            dataQueue.clear();
            return true;
        }
    }
    return false;
}

bool check(float x, float y, float yaw); // Hàm kiểm tra hội tụ (giả định đã được định nghĩa)
void scanCallback(const sensor_msgs::LaserScan::ConstPtr& msg) {
    scanPoints.clear();

    // Convert laser scan data to pixel coordinates
    double angle = msg->angle_min;
    for (size_t i = 0; i < msg->ranges.size(); ++i) {
        if (msg->ranges[i] >= msg->range_min && msg->ranges[i] <= msg->range_max) {
            float x = msg->ranges[i] * std::cos(angle) / mapMsg.info.resolution;
            float y = -msg->ranges[i] * std::sin(angle) / mapMsg.info.resolution;
            scanPoints.push_back(cv::Point2f(x, y));
        }
        angle += msg->angle_increment;
    }

    if (scanCount == 0) {
        ++scanCount;
    }

    while (ros::ok()) {
        if (mapCropped.empty()) {
            ROS_WARN("Cropped map is empty, skipping scan processing");
            break;
        }

        std::vector<cv::Point2f> transformPoints, clockwisePoints, counterPoints;
        int maxSum = 0;
        float bestDx = 0, bestDy = 0, bestDyaw = 0;

        for (const auto& point : scanPoints) {
            float rotatedX = point.x * std::cos(lidarYaw) - point.y * std::sin(lidarYaw);
            float rotatedY = point.x * std::sin(lidarYaw) + point.y * std::cos(lidarYaw);
            transformPoints.push_back(cv::Point2f(rotatedX + lidarX, lidarY - rotatedY));

            float clockwiseYaw = lidarYaw + degToRad;
            rotatedX = point.x * std::cos(clockwiseYaw) - point.y * std::sin(clockwiseYaw);
            rotatedY = point.x * std::sin(clockwiseYaw) + point.y * std::cos(clockwiseYaw);
            clockwisePoints.push_back(cv::Point2f(rotatedX + lidarX, lidarY - rotatedY));

            float counterYaw = lidarYaw - degToRad;
            rotatedX = point.x * std::cos(counterYaw) - point.y * std::sin(counterYaw);
            rotatedY = point.x * std::sin(counterYaw) + point.y * std::cos(counterYaw);
            counterPoints.push_back(cv::Point2f(rotatedX + lidarX, lidarY - rotatedY));
        }

        std::vector<cv::Point2f> offsets = {{0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        std::vector<std::vector<cv::Point2f>> pointSets = {transformPoints, clockwisePoints, counterPoints};
        std::vector<float> yawOffsets = {0, degToRad, -degToRad};

        for (size_t i = 0; i < offsets.size(); ++i) {
            for (size_t j = 0; j < pointSets.size(); ++j) {
                int sum = 0;
                for (const auto& point : pointSets[j]) {
                    float px = point.x + offsets[i].x;
                    float py = point.y + offsets[i].y;
                    if (px >= 0 && px < mapTemp.cols && py >= 0 && py < mapTemp.rows) {
                        sum += mapTemp.at<uchar>(static_cast<int>(py), static_cast<int>(px));
                    }
                }
                if (sum > maxSum) {
                    maxSum = sum;
                    bestDx = offsets[i].x;
                    bestDy = offsets[i].y;
                    bestDyaw = yawOffsets[j];
                }
            }
        }

        // Update pose
        lidarX += bestDx;
        lidarY += bestDy;
        lidarYaw += bestDyaw;

        // Check for convergence
        if (checkConvergence(lidarX, lidarY, lidarYaw)) {
            ROS_INFO("Pose converged");
            break;
        }
    }

    // Handle costmap clearing
    if (clearCountdown > -1) {
        --clearCountdown;
        if (clearCountdown == 0) {
            std_srvs::Empty srv;
            if (clearCostmapsClient.call(srv)) {
                ROS_INFO("Cleared costmaps");
            } else {
                ROS_WARN("Failed to clear costmaps");
            }
        }
    }
}

void publishPoseTf() {
    if (scanCount == 0 || mapCropped.empty() || mapMsg.data.empty()) {
        return;
    }

    static tf2_ros::Buffer tfBuffer;
    static tf2_ros::TransformListener tfListener(tfBuffer);
    static tf2_ros::TransformBroadcaster broadcaster;

    // Convert pixel coordinates to meters
    double xMeters = (lidarX + mapRoiInfo.x_offset) * mapMsg.info.resolution + mapMsg.info.origin.position.x;
    double yMeters = (lidarY + mapRoiInfo.y_offset) * mapMsg.info.resolution + mapMsg.info.origin.position.y;
    double yawRos = -lidarYaw;

    // Create quaternion
    tf2::Quaternion quaternion;
    quaternion.setRPY(0, 0, yawRos);

    // Get odom-to-base transform
    geometry_msgs::TransformStamped odomToBase;
    try {
        odomToBase = tfBuffer.lookupTransform(odomFrame, laserFrame, ros::Time(0));
    } catch (tf2::TransformException& ex) {
        ROS_WARN("Transform lookup failed: %s", ex.what());
        return;
    }

    // Compute map-to-odom transform
    tf2::Transform mapToBase;
    mapToBase.setOrigin(tf2::Vector3(xMeters, yMeters, 0));
    mapToBase.setRotation(quaternion);

    tf2::Transform odomToBaseTf2;
    tf2::fromMsg(odomToBase.transform, odomToBaseTf2);
    tf2::Transform mapToOdom = mapToBase * odomToBaseTf2.inverse();

    // Publish transform
    geometry_msgs::TransformStamped mapToOdomMsg;
    mapToOdomMsg.header.stamp = ros::Time::now();
    mapToOdomMsg.header.frame_id = "map";
    mapToOdomMsg.child_frame_id = odomFrame;
    mapToOdomMsg.transform = tf2::toMsg(mapToOdom);

    broadcaster.sendTransform(mapToOdomMsg);
}

int main(int argc, char** argv) {
    setlocale(LC_ALL, "");
    ros::init(argc, argv, "lidarLocalization");
    ros::NodeHandle privateNh("~");

    // Load parameters
    privateNh.param<std::string>("baseFrame", baseFrame, "base_link");
    privateNh.param<std::string>("odomFrame", odomFrame, "odom");
    privateNh.param<std::string>("laserFrame", laserFrame, "armbot_laser_scanner");
    privateNh.param<std::string>("laserTopic", laserTopic, "/armbot_laser_scanner/laser/scan");

    // Initialize subscribers and service client
    ros::NodeHandle nh;
    ros::Subscriber mapSub = nh.subscribe("map", 1, mapCallback);
    ros::Subscriber scanSub = nh.subscribe(laserTopic, 1, scanCallback);
    ros::Subscriber initialPoseSub = nh.subscribe("initialpose", 1, initialPoseCallback);
    clearCostmapsClient = nh.serviceClient<std_srvs::Empty>("move_base/clear_costmaps");

    // Main loop
    ros::Rate rate(30);
    while (ros::ok()) {
        publishPoseTf();
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
