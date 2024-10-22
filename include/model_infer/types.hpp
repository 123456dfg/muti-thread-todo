#ifndef _TYPES_H_
#define _TYPES_H_

#include <opencv2/core.hpp>
namespace YoloModel
{
    struct TrafficSign
    {
        int class_id;
        float confidence;
        cv::Rect box;
    };

    struct TransformStruct
    {
        TransformStruct() {}
        TransformStruct(float scale,
                        float hh,
                        float hw) : scale_(scale), hh_(hh), hw_(hw) {}
        float scale_;
        float hh_;
        float hw_;
    };
} // namespace YoloModel

#endif // _TYPES_H_