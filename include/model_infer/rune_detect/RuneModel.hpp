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
    static constexpr int INPUT_W = 480;   // Width of input
    static constexpr int INPUT_H = 480;   // Height of input
    static constexpr int NUM_CLASSES = 2; // Number of classes
    static constexpr int NUM_COLORS = 2;  // Number of color
    static constexpr int NUM_POINTS = 5;
    static constexpr int NUM_POINTS_2 = 2 * NUM_POINTS;
    static constexpr float MERGE_CONF_ERROR = 0.15;
    static constexpr float MERGE_MIN_IOU = 0.9;
    static std::unordered_map<int, EnemyColor> DNN_COLOR_TO_ENEMY_COLOR = {{0, EnemyColor::BLUE},
                                                                           {1, EnemyColor::RED}};
    struct GridAndStride
    {
        int grid0;
        int grid1;
        int stride;
    };

    static cv::Mat letterbox(const cv::Mat &img,
                             Eigen::Matrix3f &transform_matrix,
                             std::vector<int> new_shape = {INPUT_W, INPUT_H})
    {
        // Get current image shape [height, width]

        int img_h = img.rows;
        int img_w = img.cols;

        // Compute scale ratio(new / old) and target resized shape
        float scale = std::min(new_shape[1] * 1.0 / img_h, new_shape[0] * 1.0 / img_w);
        int resize_h = static_cast<int>(round(img_h * scale));
        int resize_w = static_cast<int>(round(img_w * scale));

        // Compute padding
        int pad_h = new_shape[1] - resize_h;
        int pad_w = new_shape[0] - resize_w;

        // Resize and pad image while meeting stride-multiple constraints
        cv::Mat resized_img;
        cv::resize(img, resized_img, cv::Size(resize_w, resize_h));

        // divide padding into 2 sides
        float half_h = pad_h * 1.0 / 2;
        float half_w = pad_w * 1.0 / 2;

        // Compute padding boarder
        int top = static_cast<int>(round(half_h - 0.1));
        int bottom = static_cast<int>(round(half_h + 0.1));
        int left = static_cast<int>(round(half_w - 0.1));
        int right = static_cast<int>(round(half_w + 0.1));

        /* clang-format off */
        /* *INDENT-OFF* */

        // Compute point transform_matrix
        transform_matrix << 1.0 / scale, 0, -half_w / scale,
                            0, 1.0 / scale, -half_h / scale,
                            0, 0, 1;

        /* *INDENT-ON* */
        /* clang-format on */

        // Add border
        cv::copyMakeBorder(resized_img,
                           resized_img,
                           top,
                           bottom,
                           left,
                           right,
                           cv::BORDER_CONSTANT,
                           cv::Scalar(114, 114, 114));

        return resized_img;
    }

    static void generateGridsAndStride(const int target_w,
                                       const int target_h,
                                       std::vector<int> &strides,
                                       std::vector<GridAndStride> &grid_strides)
    {
        for (auto stride : strides)
        {
            int num_grid_w = target_w / stride;
            int num_grid_h = target_h / stride;

            for (int g1 = 0; g1 < num_grid_h; g1++)
            {
                for (int g0 = 0; g0 < num_grid_w; g0++)
                {
                    grid_strides.emplace_back(GridAndStride{g0, g1, stride});
                }
            }
        }
    }

    // Decode output tensor
    static void generateProposals(std::vector<RuneObject> &output_objs,
                                  std::vector<float> &scores,
                                  std::vector<cv::Rect> &rects,
                                  const cv::Mat &output_buffer,
                                  const Eigen::Matrix<float, 3, 3> &transform_matrix,
                                  float conf_threshold,
                                  std::vector<GridAndStride> grid_strides)
    {
        const int num_anchors = grid_strides.size();

        for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
        {
            float confidence = output_buffer.at<float>(anchor_idx, NUM_POINTS_2);
            if (confidence < conf_threshold)
            {
                continue;
            }

            const int grid0 = grid_strides[anchor_idx].grid0;
            const int grid1 = grid_strides[anchor_idx].grid1;
            const int stride = grid_strides[anchor_idx].stride;

            double color_score, class_score;
            cv::Point color_id, class_id;
            cv::Mat color_scores =
                output_buffer.row(anchor_idx).colRange(NUM_POINTS_2 + 1, NUM_POINTS_2 + 1 + NUM_COLORS);
            cv::Mat num_scores =
                output_buffer.row(anchor_idx)
                    .colRange(NUM_POINTS_2 + 1 + NUM_COLORS, NUM_POINTS_2 + 1 + NUM_COLORS + NUM_CLASSES);
            // Argmax
            cv::minMaxLoc(color_scores, NULL, &color_score, NULL, &color_id);
            cv::minMaxLoc(num_scores, NULL, &class_score, NULL, &class_id);

            float x_1 = (output_buffer.at<float>(anchor_idx, 0) + grid0) * stride;
            float y_1 = (output_buffer.at<float>(anchor_idx, 1) + grid1) * stride;
            float x_2 = (output_buffer.at<float>(anchor_idx, 2) + grid0) * stride;
            float y_2 = (output_buffer.at<float>(anchor_idx, 3) + grid1) * stride;
            float x_3 = (output_buffer.at<float>(anchor_idx, 4) + grid0) * stride;
            float y_3 = (output_buffer.at<float>(anchor_idx, 5) + grid1) * stride;
            float x_4 = (output_buffer.at<float>(anchor_idx, 6) + grid0) * stride;
            float y_4 = (output_buffer.at<float>(anchor_idx, 7) + grid1) * stride;
            float x_5 = (output_buffer.at<float>(anchor_idx, 8) + grid0) * stride;
            float y_5 = (output_buffer.at<float>(anchor_idx, 9) + grid1) * stride;

            Eigen::Matrix<float, 3, 5> apex_norm;
            Eigen::Matrix<float, 3, 5> apex_dst;

            /* clang-format off */
            /* *INDENT-OFF* */
            apex_norm << x_1, x_2, x_3, x_4, x_5,
                        y_1, y_2, y_3, y_4, y_5,
                        1,   1,   1,   1,   1;
            /* *INDENT-ON* */
            /* clang-format on */

            apex_dst = transform_matrix * apex_norm;

            RuneObject obj;

            obj.pts.r_center = cv::Point2f(apex_dst(0, 0), apex_dst(1, 0));
            obj.pts.bottom_left = cv::Point2f(apex_dst(0, 1), apex_dst(1, 1));
            obj.pts.top_left = cv::Point2f(apex_dst(0, 2), apex_dst(1, 2));
            obj.pts.top_right = cv::Point2f(apex_dst(0, 3), apex_dst(1, 3));
            obj.pts.bottom_right = cv::Point2f(apex_dst(0, 4), apex_dst(1, 4));

            auto rect = cv::boundingRect(obj.pts.toVector2f());

            obj.box = rect;
            obj.color = DNN_COLOR_TO_ENEMY_COLOR[color_id.x];
            obj.type = static_cast<RuneType>(class_id.x);
            obj.prob = confidence;

            rects.push_back(rect);
            scores.push_back(confidence);
            output_objs.push_back(std::move(obj));
        }
    }

    // Calculate intersection area between Object a and Object b.
    static inline float intersectionArea(const RuneObject &a, const RuneObject &b)
    {
        cv::Rect_<float> inter = a.box & b.box;
        return inter.area();
    }

    static void nmsMergeSortedBboxes(std::vector<RuneObject> &faceobjects,
                                     std::vector<int> &indices,
                                     float nms_threshold)
    {
        indices.clear();

        const int n = faceobjects.size();

        std::vector<float> areas(n);
        for (int i = 0; i < n; i++)
        {
            areas[i] = faceobjects[i].box.area();
        }

        for (int i = 0; i < n; i++)
        {
            RuneObject &a = faceobjects[i];

            int keep = 1;
            for (size_t j = 0; j < indices.size(); j++)
            {
                RuneObject &b = faceobjects[indices[j]];

                // intersection over union
                float inter_area = intersectionArea(a, b);
                float union_area = areas[i] + areas[indices[j]] - inter_area;
                float iou = inter_area / union_area;
                if (iou > nms_threshold || isnan(iou))
                {
                    keep = 0;
                    // Stored for Merge
                    if (a.type == b.type && a.color == b.color && iou > MERGE_MIN_IOU &&
                        abs(a.prob - b.prob) < MERGE_CONF_ERROR)
                    {
                        a.pts.children.push_back(b.pts);
                    }
                    // cout<<b.pts_x.size()<<endl;
                }
            }

            if (keep)
            {
                indices.push_back(i);
            }
        }
    }

    class RuneDetector
    {
    private:
        std::string model_path_;
        std::string device_name_;
        std::vector<RuneObject> objs_;
        std::mutex mtx;
        std::unique_ptr<ov::Core> ov_core_;
        std::unique_ptr<ov::CompiledModel> compiled_model_;
        float conf_threshold_;
        int top_k_;
        float nms_threshold_;
        std::vector<int> strides_;
        std::vector<GridAndStride> grid_strides_;

        bool process(const cv::Mat resized_img,
                     Eigen::Matrix3f transform_matrix);
        void init();

    public:
        explicit RuneDetector(const std::filesystem::path &model_path,
                              const std::string &device_name,
                              float conf_threshold = 0.25,
                              int top_k = 128,
                              float nms_threshold = 0.3,
                              bool auto_init = false);

        std::tuple<cv::Point2f, cv::Mat> detectRTag(const cv::Mat &img,
                                                    int binary_thresh,
                                                    const cv::Point2f &prior);

        cv::Mat infer(cv::Mat src);
        void drawResult(cv::Mat &src);
    };

    std::tuple<cv::Point2f, cv::Mat> RuneDetector::detectRTag(const cv::Mat &img, int binary_thresh, const cv::Point2f &prior)
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
        cv::threshold(gray_img, binary_img, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
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

    void RuneDetector::drawResult(cv::Mat &src_img)
    {
        cv::Mat debug_img;
        if (debug_)
        {
            debug_img = src_img.clone();
        }

        // Erase all object that not match the color
        objs_.erase(
            std::remove_if(objs_.begin(),
                           objs_.end(),
                           [c = static_cast<EnemyColor>(detect_color_)](const auto &obj) -> bool
                           { return obj.color != c; }),
            objs_.end());

        if (!objs_.empty())
        {
            // Sort by probability
            std::sort(objs_.begin(), objs_.end(), [](const RuneObject &a, const RuneObject &b)
                      { return a.prob > b.prob; });

            cv::Point2f r_tag;
            cv::Mat binary_roi = cv::Mat::zeros(1, 1, CV_8UC3);
            if (detect_r_tag_)
            {
                // Detect R tag using traditional method to correct error
                std::tie(r_tag, binary_roi) =
                    detectRTag(src_img, binary_thresh_, objs_.at(0).pts.r_center);
            }
            else
            {
                // Use the average center of all objects as the center of the R tag
                r_tag = std::accumulate(objs_.begin(),
                                        objs_.end(),
                                        cv::Point2f(0, 0),
                                        [n = static_cast<float>(objs_.size())](cv::Point2f p, auto &o)
                                        {
                                            return p + o.pts.r_center / n;
                                        });
            }
            // Assign the center of the R tag to all objects
            std::for_each(objs_.begin(), objs_.end(), [r = r_tag](RuneObject &objs_)
                          { objs_.pts.r_center = r; });

            // Draw binary roi
            if (debug_ && !debug_img.empty())
            {
                cv::Rect roi =
                    cv::Rect(debug_img.cols - binary_roi.cols, 0, binary_roi.cols, binary_roi.rows);
                binary_roi.copyTo(debug_img(roi));
                cv::rectangle(debug_img, roi, cv::Scalar(150, 150, 150), 2);
            }
            auto result_it =
                std::find_if(objs_.begin(), objs_.end(), [c = static_cast<EnemyColor>(detect_color_)](const auto &obj) -> bool
                             { return obj.type == RuneType::INACTIVATED && obj.color == c; });
        }

        if (debug_)
        {
            if (debug_img.empty())
            {
                // Avoid debug_mode change in processing
                return;
            }

            // Draw detection result
            for (auto &obj : objs_)
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
        }

        //     auto end = std::chrono::steady_clock::now();
        //     auto duration = std::chrono::duration<double, std::milli>(end - timestamp).count();

        //     std::string latency = fmt::format(" Latency: {:.3f}ms\n", duration);

        //     cv::putText(debug_img,
        //                 latency,
        //                 cv::Point2i(10, 30),
        //                 cv::FONT_HERSHEY_SIMPLEX,
        //                 0.8,
        //                 cv::Scalar(0, 255, 255),
        //                 2);
        // }
        src_img = debug_img.clone();
        return;
    }

    bool RuneDetector::process(const cv::Mat resized_img, Eigen::Matrix3f transform_matrix)
    {
        // BGR->RGB, u8(0-255)->f32(0.0-1.0), HWC->NCHW
        // note: TUP's model no need to normalize

        cv::Mat blob =
            cv::dnn::blobFromImage(resized_img, 1., cv::Size(INPUT_W, INPUT_H), cv::Scalar(0, 0, 0), true);

        // Feed blob into input
        auto input_port = compiled_model_->input();

        ov::Tensor input_tensor(input_port.get_element_type(),
                                ov::Shape(std::vector<size_t>{1, 3, INPUT_W, INPUT_H}),
                                blob.ptr(0));

        // Start inference
        auto infer_timestamp = std::chrono::steady_clock::now();
        auto infer_request = compiled_model_->create_infer_request();
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();

        auto output = infer_request.get_output_tensor();

        // Process output data
        auto output_shape = output.get_shape();
        // 3549 x 21 Matrix
        cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, output.data());

        // Parsed variable
        std::vector<RuneObject> objs_tmp, objs_result;
        std::vector<cv::Rect> rects;
        std::vector<float> scores;
        std::vector<int> indices;

        // Parse YOLO output
        generateProposals(objs_tmp,
                          scores,
                          rects,
                          output_buffer,
                          transform_matrix,
                          this->conf_threshold_,
                          this->grid_strides_);

        // TopK
        std::sort(objs_tmp.begin(), objs_tmp.end(), [](const RuneObject &a, const RuneObject &b)
                  { return a.prob > b.prob; });
        if (objs_tmp.size() > static_cast<size_t>(this->top_k_))
        {
            objs_tmp.resize(this->top_k_);
        }

        nmsMergeSortedBboxes(objs_tmp, indices, this->nms_threshold_);

        for (size_t i = 0; i < indices.size(); i++)
        {
            objs_result.push_back(std::move(objs_tmp[indices[i]]));

            if (objs_result[i].pts.children.size() > 0)
            {
                const float N = static_cast<float>(objs_result[i].pts.children.size() + 1);
                FeaturePoints pts_final = std::accumulate(
                    objs_result[i].pts.children.begin(), objs_result[i].pts.children.end(), objs_result[i].pts);
                objs_result[i].pts = pts_final / N;
            }
        }

        objs_ = objs_result;

        return true;
    }

    RuneDetector::RuneDetector(const std::filesystem::path &model_path,
                               const std::string &device_name,
                               float conf_threshold,
                               int top_k,
                               float nms_threshold,
                               bool auto_init)
        : model_path_(model_path), device_name_(device_name), conf_threshold_(conf_threshold), top_k_(top_k), nms_threshold_(nms_threshold)
    {
        if (auto_init)
        {
            init();
        }
    }
    inline cv::Mat RuneDetector::infer(cv::Mat src)
    {
        std::lock_guard<std::mutex> lock(mtx);
        Eigen::Matrix3f transform_matrix; // transform matrix from resized image to source image.
        cv::Mat resized_img = letterbox(src, transform_matrix);
        if (process(resized_img, transform_matrix))
        {
            cv::Mat result_img;
            drawResult(result_img);
            return result_img;
        }
        return src;
    }

    void RuneDetector::init()
    {
        if (ov_core_ == nullptr)
        {
            ov_core_ = std::make_unique<ov::Core>();
        }

        auto model = ov_core_->read_model(model_path_);

        // Set infer type
        ov::preprocess::PrePostProcessor ppp(model);
        // Set input output precision
        auto elem_type = device_name_ == "GPU" ? ov::element::f16 : ov::element::f32;
        auto perf_mode = device_name_ == "CPU"
                             ? ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)
                             : ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT);
        ppp.input().tensor().set_element_type(elem_type);
        ppp.output().tensor().set_element_type(elem_type);

        // Compile model
        compiled_model_ =
            std::make_unique<ov::CompiledModel>(ov_core_->compile_model(model, device_name_, perf_mode));

        strides_ = {8, 16, 32};
        generateGridsAndStride(INPUT_W, INPUT_H, strides_, grid_strides_);
    }

} // namespace rune

#endif // _RUNEMODEL_H_