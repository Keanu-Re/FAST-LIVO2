/**
 * @brief Frame类 - 视觉里程计中的单帧图像处理类
 * 
 * 该类用于处理和存储单帧图像数据,包含以下主要功能:
 * - 图像金字塔构建
 * - 特征点管理
 * - 相机模型关联
 * 
 * @param cam 相机模型指针
 * @param img 输入的灰度图像
 * 
 * frame_utils命名空间:
 * @namespace frame_utils 提供Frame类相关的工具函数
 * @function createImgPyramid 创建图像金字塔
 * - 输入参数:
 *   @param img_level_0 原始图像(金字塔第0层)
 *   @param n_levels 金字塔层数
 *   @param pyr 输出的图像金字塔
 */
/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include <boost/bind/bind.hpp>
#include "feature.h"
#include "frame.h"
#include "visual_point.h"
#include <stdexcept>
#include <vikit/math_utils.h>
#include <vikit/performance_monitor.h>
#include <vikit/vision.h>

int Frame::frame_counter_ = 0;

Frame::Frame(vk::AbstractCamera *cam, const cv::Mat &img)
    : id_(frame_counter_++), 
      cam_(cam)
{
  initFrame(img);
}

Frame::~Frame()
{
  std::for_each(fts_.begin(), fts_.end(), [&](Feature *i) { delete i; });
}

void Frame::initFrame(const cv::Mat &img)
{
  if (img.empty()) { throw std::runtime_error("Frame: provided image is empty"); }

  if (img.cols != cam_->width() || img.rows != cam_->height())
  {
    throw std::runtime_error("Frame: provided image has not the same size as the camera model");
  }

  if (img.type() != CV_8UC1) { throw std::runtime_error("Frame: provided image is not grayscale"); }

  img_ = img;
}

/// Utility functions for the Frame class
namespace frame_utils
{

void createImgPyramid(const cv::Mat &img_level_0, int n_levels, ImgPyr &pyr)
{
  pyr.resize(n_levels);
  pyr[0] = img_level_0;
  for (int i = 1; i < n_levels; ++i)
  {
    pyr[i] = cv::Mat(pyr[i - 1].rows / 2, pyr[i - 1].cols / 2, CV_8U);
    vk::halfSample(pyr[i - 1], pyr[i]);
  }
}

} // namespace frame_utils
