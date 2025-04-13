/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef LIVO_POINT_H_
#define LIVO_POINT_H_

#include <boost/noncopyable.hpp>
#include "common_lib.h"
#include "frame.h"

class Feature;

/// A visual map point on the surface of the scene.
/**
 * @brief 视觉特征点类
 * 
 * 该类表示三维空间中的视觉特征点,包含点的位置、法向量、观测信息等属性。
 * 用于视觉SLAM中的特征点管理和跟踪。
 * 
 * 主要属性:
 * - 3D位置(pos_)
 * - 表面法向量(normal_)及其协方差信息(normal_information_) 
 * - 观测到该点的特征(obs_)
 * - 点的协方差矩阵(covariance_)
 * - 收敛状态标志(is_converged_)
 * - 法向量初始化标志(is_normal_initialized_)
 * - 参考patch相关信息(has_ref_patch_, ref_patch)
 * 
 * 该类继承自boost::noncopyable,禁止拷贝构造和赋值操作。
 * 使用EIGEN_MAKE_ALIGNED_OPERATOR_NEW保证内存对齐。
 */
class VisualPoint : boost::noncopyable
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Vector3d pos_;                //!< 3d pos of the point in the world coordinate frame.
  Vector3d normal_;             //!< Surface normal at point.
  Matrix3d normal_information_; //!< Inverse covariance matrix of normal estimation.
  Vector3d previous_normal_;    //!< Last updated normal vector.
  list<Feature *> obs_;         //!< Reference patches which observe the point.
  Eigen::Matrix3d covariance_;  //!< Covariance of the point.
  bool is_converged_;           //!< True if the point is converged.
  bool is_normal_initialized_;  //!< True if the normal is initialized.
  bool has_ref_patch_;          //!< True if the point has a reference patch.
  Feature *ref_patch;           //!< Reference patch of the point.

  VisualPoint(const Vector3d &pos);
  ~VisualPoint();
  void findMinScoreFeature(const Vector3d &framepos, Feature *&ftr) const;
  void deleteNonRefPatchFeatures();
  void deleteFeatureRef(Feature *ftr);
  void addFrameRef(Feature *ftr);
  bool getCloseViewObs(const Vector3d &pos, Feature *&obs, const Vector2d &cur_px) const;
};

#endif // LIVO_POINT_H_
