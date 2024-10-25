#ifndef RUNE_DETECTOR_RUNE_DETECTOR_HPP_
#define RUNE_DETECTOR_RUNE_DETECTOR_HPP_

// std
#include <algorithm>
#include <array>
#include <string>
#include <vector>
// ros2
#include <image_transport/image_transport.hpp>
#include <image_transport/publisher.hpp>
#include <image_transport/subscriber_filter.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
// 3rd party
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
// project
#include "rm_interfaces/msg/rune_target.hpp"
#include "rm_interfaces/msg/serial_receive_data.hpp"
#include "rm_interfaces/srv/set_mode.hpp"
#include "rm_utils/common.hpp"
#include "infer_pool/thread_pool.hpp"
#include "rune_detector/rune_detector.hpp"

namespace rune
{
    class RuneDetectorNode : public rclcpp::Node
    {
    public:
        RuneDetectorNode(const rclcpp::NodeOptions &options);
        ~RuneDetectorNode();

    private:
        std::vector<std::shared_ptr<RuneDetector>> initDetector();

        void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr img_msg);

        void createDebugPublishers();
        void destroyDebugPublishers();

        void inferResultCallback(std::vector<RuneObject> &objs,
                                 int64_t timestamp_nanosec,
                                 const cv::Mat &src_img);

        void setModeCallback(const std::shared_ptr<rm_interfaces::srv::SetMode::Request> request,
                             std::shared_ptr<rm_interfaces::srv::SetMode::Response> response);
        // Dynamic Parameter
        rcl_interfaces::msg::SetParametersResult onSetParameters(
            std::vector<rclcpp::Parameter> parameters);
        rclcpp::Node::OnSetParametersCallbackHandle::SharedPtr on_set_parameters_callback_handle_;

        // Image subscription
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;

        // Target publisher
        std::string frame_id_;
        rclcpp::Publisher<rm_interfaces::msg::RuneTarget>::SharedPtr rune_pub_;

        // Enable/Disable Rune Detector
        rclcpp::Service<rm_interfaces::srv::SetMode>::SharedPtr set_rune_mode_srv_;

        // Rune params
        fyt::EnemyColor detect_color_;
        bool is_rune_;
        bool is_big_rune_;

        // For R tag detection
        bool detect_r_tag_;

        // Debug infomation
        bool debug_;
        image_transport::Publisher result_img_pub_;

        //thread pool configure
        int thread_num_;
        unsigned long long id_;
        std::mutex idMtx, queueMtx;
        std::unique_ptr<ipool::ThreadPool> pool;
        std::queue<std::future<bool>> futs;
        std::vector<std::shared_ptr<RuneDetector>> models;

        bool put(cv::Mat &src,int64_t timestamp_nanosec);
        bool get();
        int getModelId();

        //externed detect function
        int binary_thresh_;
        std::tuple<cv::Point2f, cv::Mat> detectRTag(const cv::Mat &img,int binary_thresh, const cv::Point2f &prior);
    };
} // namespace rune

#endif