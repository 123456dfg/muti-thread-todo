#ifndef _RUNEMODEL_H_
#define _RUNEMODEL_H_

// std
#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>
#include <numeric>
#include <unordered_map>
// third party
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <opencv2/imgproc.hpp>
#include "fmt/format.h"
// project
#include "types.hpp"

namespace rune
{
    struct GridAndStride
    {
        int grid0;
        int grid1;
        int stride;
    };

    class RuneDetector
    {
    public:
        using CallbackType = std::function<void(std::vector<RuneObject> &, int64_t, const cv::Mat &)>;

    private:
        std::string model_path_;
        std::string device_name_;
        float conf_threshold_;
        int top_k_;
        float nms_threshold_;
        std::mutex mtx;
        std::unique_ptr<ov::Core> ov_core_;
        std::unique_ptr<ov::CompiledModel> compiled_model_;
        std::vector<int> strides_;
        std::vector<GridAndStride> grid_strides_;
        CallbackType infer_callback_;

        bool process(const cv::Mat resized_img, Eigen::Matrix3f transform_matrix, int64_t timestamp_nanosec, cv::Mat &src_img);

    public:
        void init();
        void setCallback(CallbackType callback);
        explicit RuneDetector(const std::filesystem::path &model_path,
                              const std::string &device_name,
                              float conf_threshold = 0.25,
                              int top_k = 128,
                              float nms_threshold = 0.3,
                              bool auto_init = false);

        // static std::tuple<cv::Point2f, cv::Mat> detectRTag(const cv::Mat &img, const cv::Point2f &prior);

        bool infer(cv::Mat &src, int64_t timestamp_nanosec);
    };

} // namespace rune

#endif // _RUNEMODEL_H_