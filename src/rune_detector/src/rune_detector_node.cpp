// ros2
#include <cv_bridge/cv_bridge.h>
#include <rmw/qos_profiles.h>

#include <rclcpp/qos.hpp>
// std
#include <algorithm>
#include <array>
#include <filesystem>
#include <numeric>
#include <vector>
// third party
#include <opencv2/imgproc.hpp>
// project
#include "rm_utils/assert.hpp"
#include "rm_utils/common.hpp"
#include "rm_utils/logger/log.hpp"
#include "rm_utils/url_resolver.hpp"
#include "rune_detector/types.hpp"
#include "rune_detector/rune_detector_node.hpp"

namespace rune
{

    RuneDetectorNode::RuneDetectorNode(const rclcpp::NodeOptions &options)
        : Node("rune_detector", options), is_rune_(false)
    {
        FYT_REGISTER_LOGGER("rune_detector", "~/fyt2024-log", INFO);
        FYT_INFO("rune_detector", "Starting RuneDetectorNode!");

        frame_id_ = declare_parameter("frame_id", "camera_optical_frame");
        detect_r_tag_ = declare_parameter("detect_r_tag", true);
        binary_thresh_ = declare_parameter("min_lightness", 100);
        thread_num_ = declare_parameter("thread_num", 3);

        // Detector
        models = initDetector();
        // Rune Publisher
        rune_pub_ = this->create_publisher<rm_interfaces::msg::RuneTarget>("rune_detector/rune_target",
                                                                           rclcpp::SensorDataQoS());

        // Debug Publishers
        this->debug_ = declare_parameter("debug", true);
        if (this->debug_)
        {
            createDebugPublishers();
        }
        auto qos = rclcpp::SensorDataQoS();
        qos.keep_last(1);
        img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "image_raw", qos, std::bind(&RuneDetectorNode::imageCallback, this, std::placeholders::_1));
        set_rune_mode_srv_ = this->create_service<rm_interfaces::srv::SetMode>(
            "rune_detector/set_mode",
            std::bind(
                &RuneDetectorNode::setModeCallback, this, std::placeholders::_1, std::placeholders::_2));
    }

    RuneDetectorNode::~RuneDetectorNode()
    {
        while (!futs.empty())
        {
            futs.pop();
        }
    }

    std::vector<std::shared_ptr<RuneDetector>> RuneDetectorNode::initDetector()
    {
        std::string model_path =
            this->declare_parameter("detector.model", "/home/dfg/backup/RM2025_VISION/muti-thread-ros-2/src/rune_detector/model/yolox_rune_3.6m.onnx");
        std::string device_type = this->declare_parameter("detector.device_type", "CPU");
        bool auto_init = this->declare_parameter("detector.auto_init", false);
        FYT_ASSERT(!model_path.empty());
        FYT_INFO("rune_detector", "model : {}, device : {}", model_path, device_type);

        rcl_interfaces::msg::ParameterDescriptor param_desc;
        param_desc.integer_range.resize(1);
        param_desc.description = "0-RED, 1-BLUE";
        param_desc.integer_range[0].from_value = 0;
        param_desc.integer_range[0].to_value = 1;
        detect_color_ = static_cast<fyt::EnemyColor>(declare_parameter("detect_color", 0, param_desc));

        float conf_threshold = this->declare_parameter("detector.confidence_threshold", 0.50);
        int top_k = this->declare_parameter("detector.top_k", 128);
        float nms_threshold = this->declare_parameter("detector.nms_threshold", 0.3);

        namespace fs = std::filesystem;
        // fs::path resolved_path = fyt::utils::URLResolver::getResolvedPath(model_path);
        // FYT_ASSERT_MSG(fs::exists(resolved_path), resolved_path.string() + " Not Found");

        // Set dynamic parameter callback
        rcl_interfaces::msg::SetParametersResult onSetParameters(
            std::vector<rclcpp::Parameter> parameters);
        on_set_parameters_callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&RuneDetectorNode::onSetParameters, this, std::placeholders::_1));

        // Create detectors in thread pool
        std::vector<std::shared_ptr<RuneDetector>> models;

        try
        {
            this->pool = std::make_unique<ipool::ThreadPool>(this->thread_num_);

            for (int i = 0; i < this->thread_num_; i++)
            {
                auto model = std::make_shared<RuneDetector>(model_path, device_type, conf_threshold, top_k, nms_threshold, auto_init);
                if (!auto_init)
                    model->init();
                model->setCallback(std::bind(&RuneDetectorNode::inferResultCallback,
                                             this,
                                             std::placeholders::_1,
                                             std::placeholders::_2,
                                             std::placeholders::_3));
                models.emplace_back(model);
            }
        }
        catch (const std::bad_alloc &e)
        {
            std::cout << "Out of memory: " << e.what() << std::endl;
        }

        return models;
    }
    void RuneDetectorNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr img_msg)
    {
        if (is_rune_ == false)
        {
            return;
        }

        // Limits request size
        while (futs.size() > static_cast<size_t>(thread_num_))
        {
            if (!this->get())
            {
                FYT_WARN("rune_detector", "detector_pool_result is empty");
                break;
            }
        }

        auto timestamp = rclcpp::Time(img_msg->header.stamp);
        frame_id_ = img_msg->header.frame_id;
        auto img = cv_bridge::toCvCopy(img_msg, "rgb8")->image;

        // Push image to detector
        this->put(img, timestamp.nanoseconds());
    }

    void RuneDetectorNode::createDebugPublishers()
    {
        result_img_pub_ = image_transport::create_publisher(this, "rune_detector/result_img");
    }

    void RuneDetectorNode::destroyDebugPublishers()
    {
        result_img_pub_.shutdown();
    }

    void RuneDetectorNode::inferResultCallback(std::vector<RuneObject> &objs, int64_t timestamp_nanosec, const cv::Mat &src_img)
    {
        auto timestamp = rclcpp::Time(timestamp_nanosec);
        // Used to draw debug info
        cv::Mat debug_img;
        if (debug_)
        {
            debug_img = src_img.clone();
        }

        rm_interfaces::msg::RuneTarget rune_msg;
        rune_msg.header.frame_id = frame_id_;
        rune_msg.header.stamp = timestamp;
        rune_msg.is_big_rune = is_big_rune_;

        // Erase all object that not match the color
        objs.erase(
            std::remove_if(objs.begin(),
                           objs.end(),
                           [c = detect_color_](const auto &obj) -> bool
                           { return obj.color != c; }),
            objs.end());

        if (!objs.empty())
        {
            // Sort by probability
            std::sort(objs.begin(), objs.end(), [](const RuneObject &a, const RuneObject &b)
                      { return a.prob > b.prob; });

            cv::Point2f r_tag;
            cv::Mat binary_roi = cv::Mat::zeros(1, 1, CV_8UC3);
            if (detect_r_tag_)
            {
                // Detect R tag using traditional method
                std::tie(r_tag, binary_roi) =
                    detectRTag(src_img, binary_thresh_, objs.at(0).pts.r_center);
            }
            else
            {
                // Use the average center of all objects as the center of the R tag
                r_tag = std::accumulate(objs.begin(),
                                        objs.end(),
                                        cv::Point2f(0, 0),
                                        [n = static_cast<float>(objs.size())](cv::Point2f p, auto &o)
                                        {
                                            return p + o.pts.r_center / n;
                                        });
            }
            // Assign the center of the R tag to all objects
            std::for_each(objs.begin(), objs.end(), [r = r_tag](RuneObject &obj)
                          { obj.pts.r_center = r; });

            // Draw binary roi
            if (debug_ && !debug_img.empty())
            {
                cv::Rect roi =
                    cv::Rect(debug_img.cols - binary_roi.cols, 0, binary_roi.cols, binary_roi.rows);
                binary_roi.copyTo(debug_img(roi));
                cv::rectangle(debug_img, roi, cv::Scalar(150, 150, 150), 2);
            }

            // The final target is the inactivated rune with the highest probability
            auto result_it =
                std::find_if(objs.begin(), objs.end(), [c = detect_color_](const auto &obj) -> bool
                             { return obj.type == RuneType::INACTIVATED && obj.color == c; });

            if (result_it != objs.end())
            {
                // FYT_DEBUG("rune_detector", "Detected!");
                rune_msg.is_lost = false;
                rune_msg.pts[0].x = result_it->pts.r_center.x;
                rune_msg.pts[0].y = result_it->pts.r_center.y;
                rune_msg.pts[1].x = result_it->pts.bottom_left.x;
                rune_msg.pts[1].y = result_it->pts.bottom_left.y;
                rune_msg.pts[2].x = result_it->pts.top_left.x;
                rune_msg.pts[2].y = result_it->pts.top_left.y;
                rune_msg.pts[3].x = result_it->pts.top_right.x;
                rune_msg.pts[3].y = result_it->pts.top_right.y;
                rune_msg.pts[4].x = result_it->pts.bottom_right.x;
                rune_msg.pts[4].y = result_it->pts.bottom_right.y;
            }
            else
            {
                // All runes are activated
                rune_msg.is_lost = true;
            }
        }
        else
        {
            // All runes are not the target color
            rune_msg.is_lost = true;
        }

        rune_pub_->publish(std::move(rune_msg));

        if (debug_)
        {
            if (debug_img.empty())
            {
                // Avoid debug_mode change in processing
                return;
            }

            // Draw detection result
            for (auto &obj : objs)
            {
                auto pts = obj.pts.toVector2f();
                cv::Point2f aim_point = std::accumulate(pts.begin() + 1, pts.end(), cv::Point2f(0, 0)) / 4;

                cv::Scalar line_color =
                    obj.type == RuneType::INACTIVATED ? cv::Scalar(50, 255, 50) : cv::Scalar(255, 50, 255);
                cv::putText(debug_img,
                            fmt::format("{:.2f}", obj.prob),
                            cv::Point2i(pts[1]),
                            cv::FONT_HERSHEY_SIMPLEX,
                            0.8,
                            line_color,
                            2);
                cv::polylines(debug_img, obj.pts.toVector2i(), true, line_color, 2);
                cv::circle(debug_img, aim_point, 5, line_color, -1);

                std::string rune_type = obj.type == RuneType::INACTIVATED ? "_HIT" : "_OK";
                std::string rune_color = enemyColorToString(obj.color);
                cv::putText(debug_img,
                            rune_color + rune_type,
                            cv::Point2i(pts[2]),
                            cv::FONT_HERSHEY_SIMPLEX,
                            0.8,
                            line_color,
                            2);
            }

            auto end = this->get_clock()->now();
            auto duration = end.seconds() - timestamp.seconds();
            std::string letency = fmt::format("Latency: {:.3f}ms", duration * 1000);
            cv::putText(debug_img,
                        letency,
                        cv::Point2i(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.8,
                        cv::Scalar(0, 255, 255),
                        2);
            result_img_pub_.publish(cv_bridge::CvImage(rune_msg.header, "rgb8", debug_img).toImageMsg());
        }
    }

    void RuneDetectorNode::setModeCallback(const std::shared_ptr<rm_interfaces::srv::SetMode::Request> request, std::shared_ptr<rm_interfaces::srv::SetMode::Response> response)
    {
        response->success = true;

        fyt::VisionMode mode = static_cast<fyt::VisionMode>(request->mode);
        std::string mode_name = visionModeToString(mode);
        if (mode_name == "UNKNOWN")
        {
            FYT_ERROR("rune_detector", "Invalid mode: {}", request->mode);
            return;
        }

        auto createImageSub = [this]()
        {
            if (img_sub_ == nullptr)
            {
                img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
                    "image_raw",
                    rclcpp::SensorDataQoS(),
                    std::bind(&RuneDetectorNode::imageCallback, this, std::placeholders::_1));
            }
        };

        detect_color_ = static_cast<fyt::EnemyColor>(this->get_parameter("detect_color").as_int());
        FYT_WARN("rune_detector", "color: {}", (this->get_parameter("detect_color").as_int()));

        switch (mode)
        {
        case fyt::VisionMode::SMALL_RUNE:
        {
            is_rune_ = true;
            is_big_rune_ = false;
            createImageSub();
            break;
        }
        case fyt::VisionMode::BIG_RUNE:
        {
            is_rune_ = true;
            is_big_rune_ = true;
            createImageSub();
            break;
        }
        default:
        {
            is_rune_ = false;
            is_big_rune_ = false;
            img_sub_.reset();
            break;
        }
        }

        FYT_WARN("rune_detector", "Set Rune Mode: {}", visionModeToString(mode));
    }

    rcl_interfaces::msg::SetParametersResult RuneDetectorNode::onSetParameters(std::vector<rclcpp::Parameter> parameters)
    {
        rcl_interfaces::msg::SetParametersResult result;
        for (const auto &param : parameters)
        {
            if (param.get_name() == "binary_thresh")
            {
                binary_thresh_ = param.as_int();
            }
            else if (param.get_name() == "detect_color")
            {
                detect_color_ = static_cast<fyt::EnemyColor>(param.as_int());
            }
        }
        result.successful = true;
        return result;
    }

    std::tuple<cv::Point2f, cv::Mat> RuneDetectorNode::detectRTag(const cv::Mat &img, int binary_thresh,const cv::Point2f &prior)
    {

        if (prior.x < 0 || prior.x > img.cols || prior.y < 0 || prior.y > img.rows)
        {
            return {prior, cv::Mat::zeros(cv::Size(200, 200), CV_8UC3)};
        }

        // Create ROI
        const int roi_x = (prior.x - 100) > 0 ? (prior.x - 100) : 0;
        const int roi_y = (prior.y - 100) > 0 ? (prior.y - 100) : 0;
        const cv::Point roi_tl = cv::Point(roi_x, roi_y);
        const int roi_w = (roi_tl.x + 200) > img.cols ? (img.cols - roi_tl.x) : 200;
        const int roi_h = (roi_tl.y + 200) > img.rows ? (img.rows - roi_tl.y) : 200;
        const cv::Rect roi = cv::Rect(roi_tl, cv::Size(roi_w, roi_h));
        const cv::Point2f prior_in_roi = prior - cv::Point2f(roi_tl);

        cv::Mat img_roi = img(roi);

        // Gray -> Binary -> Dilate
        cv::Mat gray_img;
        cv::cvtColor(img_roi, gray_img, cv::COLOR_BGR2GRAY);
        cv::Mat binary_img;
        cv::threshold(gray_img, binary_img, 0, binary_thresh_, cv::THRESH_BINARY | cv::THRESH_OTSU);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::dilate(binary_img, binary_img, kernel);

        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        auto it = std::find_if(contours.begin(),
                               contours.end(),
                               [p = prior_in_roi](const std::vector<cv::Point> &contour) -> bool
                               {
                                   return cv::pointPolygonTest(contour, p, false) >= 0;
                               });

        cv::cvtColor(binary_img, binary_img, cv::COLOR_GRAY2BGR);
        if (it == contours.end())
        {
            return {prior, binary_img};
        }
        else
        {
            cv::drawContours(binary_img, contours, it - contours.begin(), cv::Scalar(0, 255, 0), 2);
            cv::Point2f center =
                std::accumulate(it->begin(),
                                it->end(),
                                cv::Point2f(0, 0),
                                [n = static_cast<float>(it->size())](cv::Point2f a, auto b)
                                {
                                    return a + cv::Point2f(b.x, b.y) / n;
                                });
            center += cv::Point2f(roi_tl);
            return {center, binary_img};
        }
    }


    bool RuneDetectorNode::put(cv::Mat &src, int64_t timestamp_nanosec)
    {
        std::lock_guard<std::mutex> lock(queueMtx);
        futs.push(pool->submit(&RuneDetector::infer, models[this->getModelId()], src, timestamp_nanosec));
        return true;
    }

    bool RuneDetectorNode::get()
    {
        std::lock_guard<std::mutex> lock(queueMtx);
        if (futs.empty() == true)
            return false;
        bool success = futs.front().get();
        futs.pop();
        return success;
    }

    int RuneDetectorNode::getModelId()
    {
        std::lock_guard<std::mutex> lock(idMtx);
        int modelId = id_ % thread_num_;
        id_++;
        return modelId;
    }

} // namespace rune

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rune::RuneDetectorNode)
