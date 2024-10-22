#include <iostream>
#include <memory>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "thread_pool/InferPool.hpp"
#include "model_infer/auto_driver/YoloXModel.hpp"
#include "model_infer/rune_detect/RuneModel.hpp"

int main()
{
    int thread_num = 2;
    constexpr bool use_video = true;
    constexpr bool use_img = false;

    const std::string matadata_path = "/home/dfg/backup/RM2025_VISION/muti-thread-todo/resouce/metadata.yaml";
    const std::string img_path = "/home/dfg/driver_pkg_demo/000_1_0012.png";
    const std::string device = "CPU";

#ifdef AUTO_DRIVER
    constexpr float conf = 0.6;
    const std::string video_path = "/home/dfg/backup/RM2025_VISION/muti-thread-todo/resouce/traffic_sign1.mp4";
    ipool::InferPool<ipool::YoloXModel, cv::Mat, cv::Mat> testPool(model_path, matadata_path, device, conf, thread_num);
#endif
#ifdef RUNE_DRIVER
    const std::string video_path = "/home/dfg/backup/RM2025_VISION/muti-thread-todo/resouce/file.mp4";
    constexpr int topk = 128;
    constexpr float conf = 0.6;
    constexpr float nms_threshold = 0.1;
    ipool::InferPool<rune::RuneDetector, cv::Mat, cv::Mat> testPool(model_path, device, conf, topk, nms_threshold, thread_num);
#endif

    if (!testPool.init())
    {
        printf("iPool init fail!\n");
        return -1;
    }

    cv::VideoCapture capture;
    if (!use_video)
        capture.open(0);
    else
        capture.open(video_path);
    cv::namedWindow("Camera FPS", cv::WINDOW_AUTOSIZE);

    int frames = 0;
    auto beforeTime = std::chrono::steady_clock::now();
    if (use_img)
    {
        cv::Mat img = cv::imread(img_path);
        while (true)
        {

            if (!testPool.put(img))
                break;
            cv::Mat res = img.clone();
            if (frames >= thread_num && !testPool.get(res))
                break;
            if (cv::waitKey(1) == 'q') // 延时1毫秒,按q键退出/Press q to exit
                break;
            frames++;
            if (frames % 120 == 0)
            {
                auto currentTime = std::chrono::steady_clock::now();
                printf("120帧内平均帧率:\t %f fps/s\n", 120.0 / std::chrono::duration<double, std::milli>(currentTime - beforeTime).count() * 1000.0);
                beforeTime = currentTime;
            }
        }
    }
    else
        while (capture.isOpened())
        {
            cv::Mat img;
            if (capture.read(img) == false)
                break;
            if (!testPool.put(img))
                break;
            cv::Mat res = img.clone();
            if (frames >= thread_num && !testPool.get(res))
                break;
            //cv::imshow("Camera FPS", res);
            if (cv::waitKey(1) == 'q') // 延时1毫秒,按q键退出/Press q to exit
                break;
            frames++;
            if (frames % 120 == 0)
            {
                auto currentTime = std::chrono::steady_clock::now();
                printf("120帧内平均帧率:\t %f fps/s\n", 120.0 / std::chrono::duration<double, std::milli>(currentTime - beforeTime).count() * 1000.0);
                beforeTime = currentTime;
            }
        }
}