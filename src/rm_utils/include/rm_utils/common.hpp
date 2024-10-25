// Created by Chengfu Zou
// Copyright (C) FYT Vision Group. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef RM_UTILS_COMMON_HPP_
#define RM_UTILS_COMMON_HPP_

#include <string>

namespace fyt {

enum class EnemyColor {
  RED = 0,
  BLUE = 1,
  WHITE = 2,
};

inline std::string enemyColorToString(EnemyColor color) {
  switch (color) {
    case EnemyColor::RED:
      return "RED";
    case EnemyColor::BLUE:
      return "BLUE";
    case EnemyColor::WHITE:
      return "WHITE";
    default:
      return "UNKNOWN";
  }
}

enum VisionMode {
  AUTO_AIM = 0,
  SMALL_RUNE,
  BIG_RUNE,
};

inline std::string visionModeToString(VisionMode mode) {
  switch (mode) {
    // case VisionMode::AIM:
    //   return "AIM";
    case VisionMode::AUTO_AIM:
      return "AUTO_AIM";
    case VisionMode::SMALL_RUNE:
      return "SMALL_RUNE";
    case VisionMode::BIG_RUNE:
      return "BIG_RUNE";
    default:
      return "UNKNOWN";
  }
}

}  // namespace fyt
#endif
