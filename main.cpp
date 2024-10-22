#include <iostream>
#include <memory>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "thread_pool/InferPool.hpp"
#include "model_infer/YoloXModel.hpp"

int main()
{
    int thread_num = 3;
    constexpr bool use_video = true;
    const std::string model_path = "/home/dfg/auto_driver_demo/src/resource/models/traffic_sign_model/best.onnx";
    const std::string video_path = "/home/dfg/backup/RM2025_VISION/muti-thread-todo/resouce/traffic_sign1.mp4";
    const std::string matadata_path = "/home/dfg/backup/RM2025_VISION/muti-thread-todo/resouce/metadata.yaml";
    const std::string device = "CPU";
    constexpr float conf = 0.6;
    ipool::InferPool<ipool::YoloXModel, cv::Mat, cv::Mat> testPool(model_path, matadata_path, device, conf, thread_num);

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
    auto beforeTime=std::chrono::steady_clock::now();
    while (capture.isOpened())
    {
        cv::Mat img;
        if (capture.read(img) == false)
            break;
        if (!testPool.put(img))
            break;
        if (frames >= thread_num && !testPool.get(img))
            break;
        cv::imshow("Camera FPS", img);
        if (cv::waitKey(1) == 'q') // 延时1毫秒,按q键退出/Press q to exit
            break;
        frames++;
        if (frames % 120 == 0)
        {
            auto currentTime = std::chrono::steady_clock::now();
            printf("120帧内平均帧率:\t %f fps/s\n", 120.0 / std::chrono::duration<double,std::milli>(currentTime-beforeTime).count()*1000.0);
            beforeTime = currentTime;
        }

    }
}