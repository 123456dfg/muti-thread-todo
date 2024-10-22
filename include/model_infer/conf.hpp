#ifndef _CONF_H_
#define _CONF_H_

#include <string>
#include <iostream>

#define RUNE_DRIVER // 模型选择

#ifdef RUNE_DRIVER

#define detect_r_tag_ false
#define binary_thresh_ 100
#define debug_ false
const int detect_color_ = 1; // 0-RED 1-BLUE
const std::string model_path = "/home/dfg/backup/RM2025_VISION/rune_model_test/yolox_rune_3.6m.xml";

#endif

#ifdef AUTO_DRIVER

const std::string model_path = "/home/dfg/backup/RM2025_VISION/muti-thread-todo/resouce/best.onnx";

#endif

#endif // _CONF_H_
