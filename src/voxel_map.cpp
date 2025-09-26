/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "voxel_map.h"

void calcBodyCov(Eigen::Vector3d &pb, const float range_inc, const float degree_inc, Eigen::Matrix3d &cov)
{
  if (pb[2] == 0) pb[2] = 0.0001;
  float range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
  float range_var = range_inc * range_inc;
  Eigen::Matrix2d direction_var;
  direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0, pow(sin(DEG2RAD(degree_inc)), 2);
  Eigen::Vector3d direction(pb);
  direction.normalize();
  Eigen::Matrix3d direction_hat;
  direction_hat << 0, -direction(2), direction(1), direction(2), 0, -direction(0), -direction(1), direction(0), 0;
  Eigen::Vector3d base_vector1(1, 1, -(direction(0) + direction(1)) / direction(2));
  base_vector1.normalize();
  Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
  base_vector2.normalize();
  Eigen::Matrix<double, 3, 2> N;
  N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1), base_vector1(2), base_vector2(2);
  Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
  cov = direction * range_var * direction.transpose() + A * direction_var * A.transpose();
}

void loadVoxelConfig(ros::NodeHandle &nh, VoxelMapConfig &voxel_config)
{
  nh.param<bool>("publish/pub_plane_en", voxel_config.is_pub_plane_map_, false);
  
  nh.param<int>("lio/max_layer", voxel_config.max_layer_, 1);
  nh.param<double>("lio/voxel_size", voxel_config.max_voxel_size_, 0.5);
  nh.param<double>("lio/min_eigen_value", voxel_config.planner_threshold_, 0.01);
  nh.param<double>("lio/sigma_num", voxel_config.sigma_num_, 3);
  nh.param<double>("lio/beam_err", voxel_config.beam_err_, 0.02);
  nh.param<double>("lio/dept_err", voxel_config.dept_err_, 0.05);
  nh.param<vector<int>>("lio/layer_init_num", voxel_config.layer_init_num_, vector<int>{5,5,5,5,5});
  nh.param<int>("lio/max_points_num", voxel_config.max_points_num_, 50);
  nh.param<int>("lio/max_iterations", voxel_config.max_iterations_, 5);

  nh.param<bool>("local_map/map_sliding_en", voxel_config.map_sliding_en, false);
  nh.param<int>("local_map/half_map_size", voxel_config.half_map_size, 100);
  nh.param<double>("local_map/sliding_thresh", voxel_config.sliding_thresh, 8);
}

void VoxelOctoTree::init_plane(const std::vector<pointWithVar> &points, VoxelPlane *plane)
{
  plane->plane_var_ = Eigen::Matrix<double, 6, 6>::Zero();
  plane->covariance_ = Eigen::Matrix3d::Zero();
  plane->center_ = Eigen::Vector3d::Zero();
  plane->normal_ = Eigen::Vector3d::Zero();
  plane->points_size_ = points.size();
  plane->radius_ = 0;
  for (auto pv : points)
  {
    plane->covariance_ += pv.point_w * pv.point_w.transpose();
    plane->center_ += pv.point_w;
  }
  plane->center_ = plane->center_ / plane->points_size_;
  plane->covariance_ = plane->covariance_ / plane->points_size_ - plane->center_ * plane->center_.transpose();
  Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance_);
  Eigen::Matrix3cd evecs = es.eigenvectors();
  Eigen::Vector3cd evals = es.eigenvalues();
  Eigen::Vector3d evalsReal;
  evalsReal = evals.real();
  Eigen::Matrix3f::Index evalsMin, evalsMax;
  evalsReal.rowwise().sum().minCoeff(&evalsMin);
  evalsReal.rowwise().sum().maxCoeff(&evalsMax);
  int evalsMid = 3 - evalsMin - evalsMax;
  Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
  Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
  Eigen::Vector3d evecMax = evecs.real().col(evalsMax);
  Eigen::Matrix3d J_Q;
  J_Q << 1.0 / plane->points_size_, 0, 0, 0, 1.0 / plane->points_size_, 0, 0, 0, 1.0 / plane->points_size_;
  // && evalsReal(evalsMid) > 0.05
  //&& evalsReal(evalsMid) > 0.01
  if (evalsReal(evalsMin) < planer_threshold_)
  {
    for (int i = 0; i < points.size(); i++)
    {
      Eigen::Matrix<double, 6, 3> J;
      Eigen::Matrix3d F;
      for (int m = 0; m < 3; m++)
      {
        if (m != (int)evalsMin)
        {
          Eigen::Matrix<double, 1, 3> F_m =
              (points[i].point_w - plane->center_).transpose() / ((plane->points_size_) * (evalsReal[evalsMin] - evalsReal[m])) *
              (evecs.real().col(m) * evecs.real().col(evalsMin).transpose() + evecs.real().col(evalsMin) * evecs.real().col(m).transpose());
          F.row(m) = F_m;
        }
        else
        {
          Eigen::Matrix<double, 1, 3> F_m;
          F_m << 0, 0, 0;
          F.row(m) = F_m;
        }
      }
      J.block<3, 3>(0, 0) = evecs.real() * F;
      J.block<3, 3>(3, 0) = J_Q;
      plane->plane_var_ += J * points[i].var * J.transpose();
    }

    plane->normal_ << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
    plane->y_normal_ << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid), evecs.real()(2, evalsMid);
    plane->x_normal_ << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax), evecs.real()(2, evalsMax);
    plane->min_eigen_value_ = evalsReal(evalsMin);
    plane->mid_eigen_value_ = evalsReal(evalsMid);
    plane->max_eigen_value_ = evalsReal(evalsMax);
    plane->radius_ = sqrt(evalsReal(evalsMax));
    plane->d_ = -(plane->normal_(0) * plane->center_(0) + plane->normal_(1) * plane->center_(1) + plane->normal_(2) * plane->center_(2));
    plane->is_plane_ = true;
    plane->is_update_ = true;
    if (!plane->is_init_)
    {
      plane->id_ = voxel_plane_id;
      voxel_plane_id++;
      plane->is_init_ = true;
    }
  }
  else
  {
    plane->is_update_ = true;
    plane->is_plane_ = false;
  }
}

void VoxelOctoTree::init_octo_tree()
{
  if (temp_points_.size() > points_size_threshold_)
  {
    init_plane(temp_points_, plane_ptr_);
    if (plane_ptr_->is_plane_ == true)
    {
      octo_state_ = 0;
      // new added
      if (temp_points_.size() > max_points_num_)
      {
        update_enable_ = false;
        std::vector<pointWithVar>().swap(temp_points_);
        new_points_ = 0;
      }
    }
    else
    {
      octo_state_ = 1;
      cut_octo_tree();
    }
    init_octo_ = true;
    new_points_ = 0;
  }
}

void VoxelOctoTree::cut_octo_tree()
{
  if (layer_ >= max_layer_)
  {
    octo_state_ = 0;
    return;
  }
  for (size_t i = 0; i < temp_points_.size(); i++)
  {
    int xyz[3] = {0, 0, 0};
    if (temp_points_[i].point_w[0] > voxel_center_[0]) { xyz[0] = 1; }
    if (temp_points_[i].point_w[1] > voxel_center_[1]) { xyz[1] = 1; }
    if (temp_points_[i].point_w[2] > voxel_center_[2]) { xyz[2] = 1; }
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    if (leaves_[leafnum] == nullptr)
    {
      leaves_[leafnum] = new VoxelOctoTree(max_layer_, layer_ + 1, layer_init_num_[layer_ + 1], max_points_num_, planer_threshold_);
      leaves_[leafnum]->layer_init_num_ = layer_init_num_;
      leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
      leaves_[leafnum]->quater_length_ = quater_length_ / 2;
    }
    leaves_[leafnum]->temp_points_.push_back(temp_points_[i]);
    leaves_[leafnum]->new_points_++;
  }
  for (uint i = 0; i < 8; i++)
  {
    if (leaves_[i] != nullptr)
    {
      if (leaves_[i]->temp_points_.size() > leaves_[i]->points_size_threshold_)
      {
        init_plane(leaves_[i]->temp_points_, leaves_[i]->plane_ptr_);
        if (leaves_[i]->plane_ptr_->is_plane_)
        {
          leaves_[i]->octo_state_ = 0;
          // new added
          if (leaves_[i]->temp_points_.size() > leaves_[i]->max_points_num_)
          {
            leaves_[i]->update_enable_ = false;
            std::vector<pointWithVar>().swap(leaves_[i]->temp_points_);
            new_points_ = 0;
          }
        }
        else
        {
          leaves_[i]->octo_state_ = 1;
          leaves_[i]->cut_octo_tree();
        }
        leaves_[i]->init_octo_ = true;
        leaves_[i]->new_points_ = 0;
      }
    }
  }
}

void VoxelOctoTree::UpdateOctoTree(const pointWithVar &pv)
{
  if (!init_octo_)
  {
    new_points_++;
    temp_points_.push_back(pv);
    if (temp_points_.size() > points_size_threshold_) { init_octo_tree(); }
  }
  else
  {
    if (plane_ptr_->is_plane_)
    {
      if (update_enable_)
      {
        new_points_++;
        temp_points_.push_back(pv);
        if (new_points_ > update_size_threshold_)
        {
          init_plane(temp_points_, plane_ptr_);
          new_points_ = 0;
        }
        if (temp_points_.size() >= max_points_num_)
        {
          update_enable_ = false;
          std::vector<pointWithVar>().swap(temp_points_);
          new_points_ = 0;
        }
      }
    }
    else
    {
      if (layer_ < max_layer_)
      {
        int xyz[3] = {0, 0, 0};
        if (pv.point_w[0] > voxel_center_[0]) { xyz[0] = 1; }
        if (pv.point_w[1] > voxel_center_[1]) { xyz[1] = 1; }
        if (pv.point_w[2] > voxel_center_[2]) { xyz[2] = 1; }
        int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
        if (leaves_[leafnum] != nullptr) { leaves_[leafnum]->UpdateOctoTree(pv); }
        else
        {
          leaves_[leafnum] = new VoxelOctoTree(max_layer_, layer_ + 1, layer_init_num_[layer_ + 1], max_points_num_, planer_threshold_);
          leaves_[leafnum]->layer_init_num_ = layer_init_num_;
          leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
          leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
          leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
          leaves_[leafnum]->quater_length_ = quater_length_ / 2;
          leaves_[leafnum]->UpdateOctoTree(pv);
        }
      }
      else
      {
        if (update_enable_)
        {
          new_points_++;
          temp_points_.push_back(pv);
          if (new_points_ > update_size_threshold_)
          {
            init_plane(temp_points_, plane_ptr_);
            new_points_ = 0;
          }
          if (temp_points_.size() > max_points_num_)
          {
            update_enable_ = false;
            std::vector<pointWithVar>().swap(temp_points_);
            new_points_ = 0;
          }
        }
      }
    }
  }
}

VoxelOctoTree *VoxelOctoTree::find_correspond(Eigen::Vector3d pw)
{
  if (!init_octo_ || plane_ptr_->is_plane_ || (layer_ >= max_layer_)) return this;

  int xyz[3] = {0, 0, 0};
  xyz[0] = pw[0] > voxel_center_[0] ? 1 : 0;
  xyz[1] = pw[1] > voxel_center_[1] ? 1 : 0;
  xyz[2] = pw[2] > voxel_center_[2] ? 1 : 0;
  int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];

  // printf("leafnum: %d. \n", leafnum);

  return (leaves_[leafnum] != nullptr) ? leaves_[leafnum]->find_correspond(pw) : this;
}

VoxelOctoTree *VoxelOctoTree::Insert(const pointWithVar &pv)
{
  if ((!init_octo_) || (init_octo_ && plane_ptr_->is_plane_) || (init_octo_ && (!plane_ptr_->is_plane_) && (layer_ >= max_layer_)))
  {
    new_points_++;
    temp_points_.push_back(pv);
    return this;
  }

  if (init_octo_ && (!plane_ptr_->is_plane_) && (layer_ < max_layer_))
  {
    int xyz[3] = {0, 0, 0};
    xyz[0] = pv.point_w[0] > voxel_center_[0] ? 1 : 0;
    xyz[1] = pv.point_w[1] > voxel_center_[1] ? 1 : 0;
    xyz[2] = pv.point_w[2] > voxel_center_[2] ? 1 : 0;
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    if (leaves_[leafnum] != nullptr) { return leaves_[leafnum]->Insert(pv); }
    else
    {
      leaves_[leafnum] = new VoxelOctoTree(max_layer_, layer_ + 1, layer_init_num_[layer_ + 1], max_points_num_, planer_threshold_);
      leaves_[leafnum]->layer_init_num_ = layer_init_num_;
      leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
      leaves_[leafnum]->quater_length_ = quater_length_ / 2;
      return leaves_[leafnum]->Insert(pv);
    }
  }
  return nullptr;
}

/**
 * @brief 状态估计函数，基于扩展卡尔曼滤波(EKF)实现激光雷达惯性里程计(LIO)的状态更新
 * @param state_propagat 通过IMU传播得到的状态预测值
 * 
 * 功能流程：
 * 1. 初始化数据结构和变量
 * 2. 计算点云协方差矩阵
 * 3. 构建残差项
 * 4. 迭代EKF更新状态
 * 5. 收敛判断和协方差更新
 */
void VoxelMapManager::StateEstimation(StatesGroup &state_propagat) {
  // ============= 1. 初始化数据结构 =============
  cross_mat_list_.clear();  // 清空反对称矩阵列表
  cross_mat_list_.reserve(feats_down_size_);  // 预分配内存
  body_cov_list_.clear();   // 清空体坐标系协方差列表
  body_cov_list_.reserve(feats_down_size_);  // 预分配内存

  // ============= 2. 计算点云协方差 =============
  for (size_t i = 0; i < feats_down_body_->size(); i++) {
    // 获取当前点坐标
    V3D point_this(feats_down_body_->points[i].x, 
                  feats_down_body_->points[i].y, 
                  feats_down_body_->points[i].z);
    
    // 处理z=0的特殊情况
    if (point_this[2] == 0) { point_this[2] = 0.001; }
    
    // 计算基础协方差矩阵
    M3D var;
    calcBodyCov(point_this, config_setting_.dept_err_, config_setting_.beam_err_, var);
    body_cov_list_.push_back(var);  // 存储体坐标系协方差
    
    // 转换到世界坐标系并计算反对称矩阵
    point_this = extR_ * point_this + extT_;
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(point_this);
    cross_mat_list_.push_back(point_crossmat);  // 存储反对称矩阵
  }

  // ============= 3. 准备点云数据 =============
  vector<pointWithVar>().swap(pv_list_);  // 清空并释放内存
  pv_list_.resize(feats_down_size_);     // 重新分配大小

  // ============= 4. 初始化EKF变量 =============
  int rematch_num = 0;  // 重匹配计数器
  MD(DIM_STATE, DIM_STATE) G, H_T_H, I_STATE;
  G.setZero();      // 增益矩阵
  H_T_H.setZero();  // H转置*H矩阵
  I_STATE.setIdentity();  // 单位矩阵

  // ============= 5. 迭代EKF更新 =============
  bool flg_EKF_inited, flg_EKF_converged, EKF_stop_flg = 0;
  for (int iterCount = 0; iterCount < config_setting_.max_iterations_; iterCount++) {
    double total_residual = 0.0;  // 总残差
    
    // 5.1 将点云转换到世界坐标系
    pcl::PointCloud<pcl::PointXYZI>::Ptr world_lidar(new pcl::PointCloud<pcl::PointXYZI>);
    TransformLidar(state_.rot_end, state_.pos_end, feats_down_body_, world_lidar);
    
    // 5.2 计算每个点的世界坐标系协方差
    M3D rot_var = state_.cov.block<3, 3>(0, 0);  // 旋转协方差
    M3D t_var = state_.cov.block<3, 3>(3, 3);   // 平移协方差
    for (size_t i = 0; i < feats_down_body_->size(); i++) {
      pointWithVar &pv = pv_list_[i];
      // 存储体坐标系和世界坐标系点坐标
      pv.point_b << feats_down_body_->points[i].x, feats_down_body_->points[i].y, feats_down_body_->points[i].z;
      pv.point_w << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
      
      // 计算综合协方差（考虑旋转和平移不确定性）
      M3D cov = body_cov_list_[i];
      M3D point_crossmat = cross_mat_list_[i];
      cov = state_.rot_end * cov * state_.rot_end.transpose() + 
            (-point_crossmat) * rot_var * (-point_crossmat).transpose() + 
            t_var;
      pv.var = cov;  // 存储最终协方差
      pv.body_var = body_cov_list_[i];  // 存储体坐标系协方差
    }
    
    // 5.3 构建残差列表（并行计算）
    ptpl_list_.clear();
    BuildResidualListOMP(pv_list_, ptpl_list_);
    
    // 5.4 计算总残差
    for (int i = 0; i < ptpl_list_.size(); i++) {
      total_residual += fabs(ptpl_list_[i].dis_to_plane_);
    }
    effct_feat_num_ = ptpl_list_.size();  // 有效特征点数量
    
    // 打印调试信息
    cout << "[ LIO ] Raw feature num: " << feats_undistort_->size() 
         << ", downsampled feature num:" << feats_down_size_ 
         << " effective feature num: " << effct_feat_num_ 
         << " average residual: " << total_residual / effct_feat_num_ << endl;

    // 5.5 计算测量雅可比矩阵H和测量协方差
    MatrixXd Hsub(effct_feat_num_, 6);  // 雅可比矩阵
    MatrixXd Hsub_T_R_inv(6, effct_feat_num_);  // H转置*R逆
    VectorXd R_inv(effct_feat_num_);    // 残差协方差逆
    VectorXd meas_vec(effct_feat_num_); // 测量向量
    meas_vec.setZero();
    
    for (int i = 0; i < effct_feat_num_; i++) {
      auto &ptpl = ptpl_list_[i];
      // 计算世界坐标系点坐标
      V3D point_this(ptpl.point_b_);
      point_this = extR_ * point_this + extT_;
      
      // 计算反对称矩阵
      M3D point_crossmat;
      point_crossmat << SKEW_SYM_MATRX(point_this);
      
      // 计算法向量相关项
      V3D point_world = state_propagat.rot_end * point_this + state_propagat.pos_end;
      Eigen::Matrix<double, 1, 6> J_nq;
      J_nq.block<1, 3>(0, 0) = point_world - ptpl_list_[i].center_;
      J_nq.block<1, 3>(0, 3) = -ptpl_list_[i].normal_;
      
      // 计算点协方差
      M3D var = state_propagat.rot_end * extR_ * ptpl_list_[i].body_cov_ * (state_propagat.rot_end * extR_).transpose();
      
      // 计算残差协方差逆
      double sigma_l = J_nq * ptpl_list_[i].plane_var_ * J_nq.transpose();
      R_inv(i) = 1.0 / (0.001 + sigma_l + ptpl_list_[i].normal_.transpose() * var * ptpl_list_[i].normal_);
      
      // 计算雅可比矩阵H
      V3D A(point_crossmat * state_.rot_end.transpose() * ptpl_list_[i].normal_);
      Hsub.row(i) << VEC_FROM_ARRAY(A), ptpl_list_[i].normal_[0], ptpl_list_[i].normal_[1], ptpl_list_[i].normal_[2];
      Hsub_T_R_inv.col(i) << A[0] * R_inv(i), A[1] * R_inv(i), A[2] * R_inv(i), 
                            ptpl_list_[i].normal_[0] * R_inv(i),
                            ptpl_list_[i].normal_[1] * R_inv(i),
                            ptpl_list_[i].normal_[2] * R_inv(i);
      meas_vec(i) = -ptpl_list_[i].dis_to_plane_;  // 测量残差
    }

    // 5.6 EKF更新
    EKF_stop_flg = false;
    flg_EKF_converged = false;
    MatrixXd K(DIM_STATE, effct_feat_num_);  // 卡尔曼增益
    auto &&HTz = Hsub_T_R_inv * meas_vec;    // H转置*R逆*残差
    H_T_H.block<6, 6>(0, 0) = Hsub_T_R_inv * Hsub;  // 信息矩阵
    
    // 计算卡尔曼增益
    MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H.block<DIM_STATE, DIM_STATE>(0, 0) + 
                                      state_.cov.block<DIM_STATE, DIM_STATE>(0, 0).inverse()).inverse();
    G.block<DIM_STATE, 6>(0, 0) = K_1.block<DIM_STATE, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
    
    // 计算状态更新量
    auto vec = state_propagat - state_;
    VD(DIM_STATE) solution = K_1.block<DIM_STATE, 6>(0, 0) * HTz + 
                            vec.block<DIM_STATE, 1>(0, 0) - 
                            G.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0);
    
    // 更新状态
    state_ += solution;
    
    // 检查收敛条件
    auto rot_add = solution.block<3, 1>(0, 0);
    auto t_add = solution.block<3, 1>(3, 0);
    if ((rot_add.norm() * 57.3 < 0.01) && (t_add.norm() * 100 < 0.015)) { 
      flg_EKF_converged = true; 
    }
    
    // 计算当前欧拉角
    V3D euler_cur = state_.rot_end.eulerAngles(2, 1, 0);

    // ============= 6. 重匹配判断 =============
    if (flg_EKF_converged || ((rematch_num == 0) && (iterCount == (config_setting_.max_iterations_ - 2)))) { 
      rematch_num++; 
    }

    // ============= 7. 收敛判断和协方差更新 =============
    if (!EKF_stop_flg && (rematch_num >= 2 || (iterCount == config_setting_.max_iterations_ - 1))) {
      // 更新协方差矩阵
      state_.cov.block<DIM_STATE, DIM_STATE>(0, 0) =
          (I_STATE.block<DIM_STATE, DIM_STATE>(0, 0) - 
           G.block<DIM_STATE, DIM_STATE>(0, 0)) * state_.cov.block<DIM_STATE, DIM_STATE>(0, 0);
      
      // 更新位置和四元数
      position_last_ = state_.pos_end;
      geoQuat_ = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));
      
      EKF_stop_flg = true;  // 标记EKF完成
    }
    
    if (EKF_stop_flg) break;  // 如果收敛则退出迭代
  }

  // double t2 = omp_get_wtime();
  // scan_count++;
  // ekf_time = t2 - t0 - build_residual_time;

  // ave_build_residual_time = ave_build_residual_time * (scan_count - 1) / scan_count + build_residual_time / scan_count;
  // ave_ekf_time = ave_ekf_time * (scan_count - 1) / scan_count + ekf_time / scan_count;

  // cout << "[ Mapping ] ekf_time: " << ekf_time << "s, build_residual_time: " << build_residual_time << "s" << endl;
  // cout << "[ Mapping ] ave_ekf_time: " << ave_ekf_time << "s, ave_build_residual_time: " << ave_build_residual_time << "s" << endl;
}

void VoxelMapManager::TransformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud,
                                     pcl::PointCloud<pcl::PointXYZI>::Ptr &trans_cloud)
{
  pcl::PointCloud<pcl::PointXYZI>().swap(*trans_cloud);
  trans_cloud->reserve(input_cloud->size());
  for (size_t i = 0; i < input_cloud->size(); i++)
  {
    pcl::PointXYZINormal p_c = input_cloud->points[i];
    Eigen::Vector3d p(p_c.x, p_c.y, p_c.z);
    p = (rot * (extR_ * p + extT_) + t);
    pcl::PointXYZI pi;
    pi.x = p(0);
    pi.y = p(1);
    pi.z = p(2);
    pi.intensity = p_c.intensity;
    trans_cloud->points.push_back(pi);
  }
}

/**
 * @brief 构建体素地图的核心方法
 * 功能：将输入的点云数据分配到体素中，并根据配置动态管理体素结构（分割或合并）
 */
void VoxelMapManager::BuildVoxelMap() {
  // ============= 1. 参数初始化 =============
  float voxel_size = config_setting_.max_voxel_size_;        // 体素边长
  float planer_threshold = config_setting_.planner_threshold_; // 平面拟合阈值
  int max_layer = config_setting_.max_layer_;                // 八叉树最大深度
  int max_points_num = config_setting_.max_points_num_;      // 单个体素最大点数
  std::vector<int> layer_init_num = config_setting_.layer_init_num_; // 各层初始化配置

  // ============= 2. 点云预处理 =============
  std::vector<pointWithVar> input_points; // 存储带协方差的点云
  for (size_t i = 0; i < feats_down_world_->size(); i++) {
    pointWithVar pv;
    // 世界坐标系下的点坐标
    pv.point_w << feats_down_world_->points[i].x, 
                  feats_down_world_->points[i].y, 
                  feats_down_world_->points[i].z;
    
    // 体坐标系下的点坐标（用于协方差计算）
    V3D point_this(feats_down_body_->points[i].x, 
                   feats_down_body_->points[i].y, 
                   feats_down_body_->points[i].z);
    
    // 计算基础协方差（考虑传感器误差）
    M3D var;
    calcBodyCov(point_this, config_setting_.dept_err_, config_setting_.beam_err_, var);
    
    // 计算反对称矩阵
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(point_this);
    
    // 综合协方差（叠加位姿不确定性）
    var = (state_.rot_end * extR_) * var * (state_.rot_end * extR_).transpose() +
          (-point_crossmat) * state_.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose() + 
          state_.cov.block<3, 3>(3, 3);
    
    pv.var = var; // 存储最终协方差
    input_points.push_back(pv); // 加入处理队列
  }

  // ============= 3. 体素化映射 =============
  uint plsize = input_points.size();
  for (uint i = 0; i < plsize; i++) {
    const pointWithVar p_v = input_points[i];
    float loc_xyz[3]; // 计算点所在的体素索引
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_v.point_w[j] / voxel_size;
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; } // 处理负坐标
    }
    
    // 生成体素位置键
    VOXEL_LOCATION position((int64_t)loc_xyz[0], 
                           (int64_t)loc_xyz[1], 
                           (int64_t)loc_xyz[2]);
    
    // 查找体素是否存在
    auto iter = voxel_map_.find(position);
    if (iter != voxel_map_.end()) {
      // 体素已存在：追加点云数据
      voxel_map_[position]->temp_points_.push_back(p_v);
      voxel_map_[position]->new_points_++; // 更新点数统计
    } else {
      // 体素不存在：创建新体素节点
      VoxelOctoTree *octo_tree = new VoxelOctoTree(
          max_layer, 0, layer_init_num[0], max_points_num, planer_threshold);
      
      // 初始化体素属性
      voxel_map_[position] = octo_tree;
      voxel_map_[position]->quater_length_ = voxel_size / 4; // 体素半径
      voxel_map_[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size; // 中心坐标X
      voxel_map_[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size; // 中心坐标Y
      voxel_map_[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size; // 中心坐标Z
      voxel_map_[position]->temp_points_.push_back(p_v); // 存储点云
      voxel_map_[position]->new_points_++; // 更新点数统计
      voxel_map_[position]->layer_init_num_ = layer_init_num; // 初始化层级配置
    }
  }

  // ============= 4. 初始化八叉树结构 =============
  for (auto iter = voxel_map_.begin(); iter != voxel_map_.end(); ++iter) {
    iter->second->init_octo_tree(); // 对每个体素触发八叉树初始化
  }
}


V3F VoxelMapManager::RGBFromVoxel(const V3D &input_point)
{
  int64_t loc_xyz[3];
  for (int j = 0; j < 3; j++)
  {
    loc_xyz[j] = floor(input_point[j] / config_setting_.max_voxel_size_);
  }

  VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
  int64_t ind = loc_xyz[0] + loc_xyz[1] + loc_xyz[2];
  uint k((ind + 100000) % 3);
  V3F RGB((k == 0) * 255.0, (k == 1) * 255.0, (k == 2) * 255.0);
  // cout<<"RGB: "<<RGB.transpose()<<endl;
  return RGB;
}

/**
 * @brief 更新体素地图的核心方法
 * @param input_points 输入的点云数据（带协方差信息）
 * 
 * 功能流程：
 * 1. 从配置中读取体素化参数
 * 2. 遍历所有输入点云
 * 3. 计算每个点所属的体素位置
 * 4. 更新或创建对应的体素八叉树节点
 */
void VoxelMapManager::UpdateVoxelMap(const std::vector<pointWithVar> &input_points) {
  // ============= 1. 参数初始化 =============
  float voxel_size = config_setting_.max_voxel_size_;         // 体素边长（单位：米）
  float planer_threshold = config_setting_.planner_threshold_; // 平面拟合阈值
  int max_layer = config_setting_.max_layer_;                 // 八叉树最大深度
  int max_points_num = config_setting_.max_points_num_;       // 单个体素最大点数
  std::vector<int> layer_init_num = config_setting_.layer_init_num_; // 各层初始化配置

  uint plsize = input_points.size(); // 输入点云数量

  // ============= 2. 遍历所有点云 =============
  for (uint i = 0; i < plsize; i++) {
    const pointWithVar p_v = input_points[i]; // 当前点（带协方差）

    // ============= 3. 计算体素位置 =============
    float loc_xyz[3]; // 存储归一化后的体素坐标
    for (int j = 0; j < 3; j++) {
      // 归一化到体素网格（除以体素边长）
      loc_xyz[j] = p_v.point_w[j] / voxel_size;
      
      // 处理负坐标（确保向下取整）
      if (loc_xyz[j] < 0) { 
        loc_xyz[j] -= 1.0; 
      }
    }

    // 生成体素位置键（64位整数坐标）
    VOXEL_LOCATION position(
      (int64_t)loc_xyz[0], 
      (int64_t)loc_xyz[1], 
      (int64_t)loc_xyz[2]
    );

    // ============= 4. 查找或创建体素节点 =============
    auto iter = voxel_map_.find(position); // 查找体素是否已存在
    if (iter != voxel_map_.end()) {
      // 情况1：体素已存在 -> 更新八叉树
      voxel_map_[position]->UpdateOctoTree(p_v);
    } else {
      // 情况2：体素不存在 -> 创建新八叉树节点
      VoxelOctoTree *octo_tree = new VoxelOctoTree(
        max_layer,            // 最大深度
        0,                    // 当前层（根节点为0）
        layer_init_num[0],    // 初始点数阈值
        max_points_num,       // 最大点数
        planer_threshold      // 平面拟合阈值
      );

      // 初始化体素属性
      voxel_map_[position] = octo_tree; // 加入体素地图
      voxel_map_[position]->layer_init_num_ = layer_init_num; // 设置层级配置
      voxel_map_[position]->quater_length_ = voxel_size / 4; // 体素半径（1/4边长）

      // 计算体素中心坐标（世界坐标系）
      voxel_map_[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
      voxel_map_[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
      voxel_map_[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;

      // 更新八叉树（插入当前点）
      voxel_map_[position]->UpdateOctoTree(p_v);
    }
  }
}


/**
 * @brief 并行构建点云到平面残差列表（用于点云配准优化）
 * @param pv_list 输入点云列表（带协方差信息）
 * @param ptpl_list 输出残差列表（点对平面的匹配关系）
 * 
 * 功能流程：
 * 1. 初始化并行计算环境和数据结构
 * 2. 并行计算每个点到最近体素平面的残差
 * 3. 收集有效残差并输出结果
 */
void VoxelMapManager::BuildResidualListOMP(std::vector<pointWithVar> &pv_list, 
                                          std::vector<PointToPlane> &ptpl_list) 
{
    // ============= 1. 参数初始化 =============
    int max_layer = config_setting_.max_layer_;        // 八叉树最大深度
    double voxel_size = config_setting_.max_voxel_size_; // 体素边长
    double sigma_num = config_setting_.sigma_num_;     // 残差过滤阈值系数
    std::mutex mylock;                                 // 线程锁（保护共享数据）

    // ============= 2. 初始化输出数据结构 =============
    ptpl_list.clear();  // 清空输出残差列表
    std::vector<PointToPlane> all_ptpl_list(pv_list.size()); // 临时存储所有残差
    std::vector<bool> useful_ptpl(pv_list.size(), false);    // 标记有效残差
    std::vector<size_t> index(pv_list.size());               // 点云索引

    // 初始化索引数组（0,1,2,...）
    for (size_t i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    // ============= 3. 并行计算残差（OpenMP加速） =============
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);  // 设置线程数
        #pragma omp parallel for           // 开启并行for循环
    #endif
    for (int i = 0; i < index.size(); i++) {
        pointWithVar &pv = pv_list[i];     // 获取当前点（带协方差）

        // 3.1 计算当前点所在的体素位置（归一化坐标）
        float loc_xyz[3];
        for (int j = 0; j < 3; j++) {
            loc_xyz[j] = pv.point_w[j] / voxel_size;
            if (loc_xyz[j] < 0) { 
                loc_xyz[j] -= 1.0;  // 处理负坐标（确保向下取整）
            }
        }

        // 3.2 生成体素位置键（64位整数坐标）
        VOXEL_LOCATION position(
            (int64_t)loc_xyz[0], 
            (int64_t)loc_xyz[1], 
            (int64_t)loc_xyz[2]
        );

        // 3.3 查找当前体素是否存在
        auto iter = voxel_map_.find(position);
        if (iter != voxel_map_.end()) {
            VoxelOctoTree *current_octo = iter->second;  // 获取体素八叉树

            // 3.4 计算当前点到体素平面的残差
            PointToPlane single_ptpl;
            bool is_sucess = false;
            double prob = 0;
            build_single_residual(pv, current_octo, 0, is_sucess, prob, single_ptpl);

            // 3.5 如果当前体素匹配失败，尝试邻近体素
            if (!is_sucess) {
                VOXEL_LOCATION near_position = position;
                // 检查X方向邻近体素
                if (loc_xyz[0] > (current_octo->voxel_center_[0] + current_octo->quater_length_)) { 
                    near_position.x += 1; 
                } else if (loc_xyz[0] < (current_octo->voxel_center_[0] - current_octo->quater_length_)) { 
                    near_position.x -= 1; 
                }
                // 检查Y方向邻近体素
                if (loc_xyz[1] > (current_octo->voxel_center_[1] + current_octo->quater_length_)) { 
                    near_position.y += 1; 
                } else if (loc_xyz[1] < (current_octo->voxel_center_[1] - current_octo->quater_length_)) { 
                    near_position.y -= 1; 
                }
                // 检查Z方向邻近体素
                if (loc_xyz[2] > (current_octo->voxel_center_[2] + current_octo->quater_length_)) { 
                    near_position.z += 1; 
                } else if (loc_xyz[2] < (current_octo->voxel_center_[2] - current_octo->quater_length_)) { 
                    near_position.z -= 1; 
                }

                // 尝试匹配邻近体素
                auto iter_near = voxel_map_.find(near_position);
                if (iter_near != voxel_map_.end()) {
                    build_single_residual(pv, iter_near->second, 0, is_sucess, prob, single_ptpl);
                }
            }

            // 3.6 记录有效残差（线程安全操作）
            if (is_sucess) {
                mylock.lock();
                useful_ptpl[i] = true;
                all_ptpl_list[i] = single_ptpl;
                mylock.unlock();
            } else {
                mylock.lock();
                useful_ptpl[i] = false;
                mylock.unlock();
            }
        }
    }

    // ============= 4. 收集有效残差 =============
    for (size_t i = 0; i < useful_ptpl.size(); i++) {
        if (useful_ptpl[i]) {
            ptpl_list.push_back(all_ptpl_list[i]);  // 仅保留有效残差
        }
    }
}


/**
 * @brief 计算单个点到体素平面的残差（递归实现）
 * @param pv 输入点（包含坐标、协方差等信息）
 * @param current_octo 当前体素八叉树节点
 * @param current_layer 当前八叉树层级
 * @param is_sucess 输出标志，表示是否成功匹配到平面
 * @param prob 输出概率值，表示匹配质量
 * @param single_ptpl 输出点对平面的匹配关系（残差信息）
 * 
 * 功能流程：
 * 1. 检查当前体素是否为平面
 * 2. 计算点到平面的距离和投影距离
 * 3. 判断点是否在有效范围内
 * 4. 计算残差和概率
 * 5. 递归处理子节点（如果不是平面且未达到最大深度）
 */
void VoxelMapManager::build_single_residual(pointWithVar &pv, const VoxelOctoTree *current_octo, 
                                           const int current_layer, bool &is_sucess,
                                           double &prob, PointToPlane &single_ptpl) {
    // ============= 1. 参数初始化 =============
    int max_layer = config_setting_.max_layer_;   // 八叉树最大深度
    double sigma_num = config_setting_.sigma_num_; // 残差过滤阈值系数
    double radius_k = 3;                          // 有效范围系数
    Eigen::Vector3d p_w = pv.point_w;             // 点的世界坐标

    // ============= 2. 检查当前体素是否为平面 =============
    if (current_octo->plane_ptr_->is_plane_) {
        VoxelPlane &plane = *current_octo->plane_ptr_; // 获取平面信息

        // 2.1 计算点到平面中心的向量
        Eigen::Vector3d p_world_to_center = p_w - plane.center_;

        // 2.2 计算点到平面的垂直距离
        float dis_to_plane = fabs(plane.normal_(0) * p_w(0) + 
                                 plane.normal_(1) * p_w(1) + 
                                 plane.normal_(2) * p_w(2) + plane.d_);

        // 2.3 计算点到平面中心的欧氏距离平方
        float dis_to_center = (plane.center_(0) - p_w(0)) * (plane.center_(0) - p_w(0)) + 
                             (plane.center_(1) - p_w(1)) * (plane.center_(1) - p_w(1)) +
                             (plane.center_(2) - p_w(2)) * (plane.center_(2) - p_w(2));

        // 2.4 计算点到平面的投影距离（平面内距离）
        float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_plane);

        // ============= 3. 判断点是否在有效范围内 =============
        if (range_dis <= radius_k * plane.radius_) {
            // 3.1 计算雅可比矩阵和残差方差
            Eigen::Matrix<double, 1, 6> J_nq;
            J_nq.block<1, 3>(0, 0) = p_w - plane.center_; // 位置部分
            J_nq.block<1, 3>(0, 3) = -plane.normal_;      // 法向量部分
            double sigma_l = J_nq * plane.plane_var_ * J_nq.transpose(); // 平面方差传递
            sigma_l += plane.normal_.transpose() * pv.var * plane.normal_; // 点方差传递

            // 3.2 判断残差是否在阈值范围内
            if (dis_to_plane < sigma_num * sqrt(sigma_l)) {
                is_sucess = true; // 标记匹配成功
                double this_prob = 1.0 / (sqrt(sigma_l)) * exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);

                // 3.3 更新最优匹配结果
                if (this_prob > prob) {
                    prob = this_prob;
                    pv.normal = plane.normal_; // 更新点的法向量
                    // 填充点对平面匹配信息
                    single_ptpl.body_cov_ = pv.body_var;
                    single_ptpl.point_b_ = pv.point_b;
                    single_ptpl.point_w_ = pv.point_w;
                    single_ptpl.plane_var_ = plane.plane_var_;
                    single_ptpl.normal_ = plane.normal_;
                    single_ptpl.center_ = plane.center_;
                    single_ptpl.d_ = plane.d_;
                    single_ptpl.layer_ = current_layer;
                    single_ptpl.dis_to_plane_ = plane.normal_(0) * p_w(0) + 
                                               plane.normal_(1) * p_w(1) + 
                                               plane.normal_(2) * p_w(2) + plane.d_;
                }
                return; // 匹配成功，直接返回
            } else {
                // is_sucess = false; // 残差超出阈值，匹配失败
                return;
            }
        } else {
            // is_sucess = false; // 点不在有效范围内，匹配失败
            return;
        }
    } 
    // ============= 4. 递归处理子节点（非平面且未达到最大深度） =============
    else {
        if (current_layer < max_layer) {
            for (size_t leafnum = 0; leafnum < 8; leafnum++) {
                if (current_octo->leaves_[leafnum] != nullptr) {
                    // 递归处理子节点
                    VoxelOctoTree *leaf_octo = current_octo->leaves_[leafnum];
                    build_single_residual(pv, leaf_octo, current_layer + 1, is_sucess, prob, single_ptpl);
                }
            }
            return;
        } else { 
            return; // 达到最大深度，直接返回
        }
    }
}

void VoxelMapManager::pubVoxelMap()
{
  double max_trace = 0.25;
  double pow_num = 0.2;
  ros::Rate loop(500);
  float use_alpha = 0.8;
  visualization_msgs::MarkerArray voxel_plane;
  voxel_plane.markers.reserve(1000000);
  std::vector<VoxelPlane> pub_plane_list;
  for (auto iter = voxel_map_.begin(); iter != voxel_map_.end(); iter++)
  {
    GetUpdatePlane(iter->second, config_setting_.max_layer_, pub_plane_list);
  }
  for (size_t i = 0; i < pub_plane_list.size(); i++)
  {
    V3D plane_cov = pub_plane_list[i].plane_var_.block<3, 3>(0, 0).diagonal();
    double trace = plane_cov.sum();
    if (trace >= max_trace) { trace = max_trace; }
    trace = trace * (1.0 / max_trace);
    trace = pow(trace, pow_num);
    uint8_t r, g, b;
    mapJet(trace, 0, 1, r, g, b);
    Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
    double alpha;
    if (pub_plane_list[i].is_plane_) { alpha = use_alpha; }
    else { alpha = 0; }
    pubSinglePlane(voxel_plane, "plane", pub_plane_list[i], alpha, plane_rgb);
  }
  voxel_map_pub_.publish(voxel_plane);
  loop.sleep();
}

void VoxelMapManager::GetUpdatePlane(const VoxelOctoTree *current_octo, const int pub_max_voxel_layer, std::vector<VoxelPlane> &plane_list)
{
  if (current_octo->layer_ > pub_max_voxel_layer) { return; }
  if (current_octo->plane_ptr_->is_update_) { plane_list.push_back(*current_octo->plane_ptr_); }
  if (current_octo->layer_ < current_octo->max_layer_)
  {
    if (!current_octo->plane_ptr_->is_plane_)
    {
      for (size_t i = 0; i < 8; i++)
      {
        if (current_octo->leaves_[i] != nullptr) { GetUpdatePlane(current_octo->leaves_[i], pub_max_voxel_layer, plane_list); }
      }
    }
  }
  return;
}

void VoxelMapManager::pubSinglePlane(visualization_msgs::MarkerArray &plane_pub, const std::string plane_ns, const VoxelPlane &single_plane,
                                     const float alpha, const Eigen::Vector3d rgb)
{
  visualization_msgs::Marker plane;
  plane.header.frame_id = "camera_init";
  plane.header.stamp = ros::Time();
  plane.ns = plane_ns;
  plane.id = single_plane.id_;
  plane.type = visualization_msgs::Marker::CYLINDER;
  plane.action = visualization_msgs::Marker::ADD;
  plane.pose.position.x = single_plane.center_[0];
  plane.pose.position.y = single_plane.center_[1];
  plane.pose.position.z = single_plane.center_[2];
  geometry_msgs::Quaternion q;
  CalcVectQuation(single_plane.x_normal_, single_plane.y_normal_, single_plane.normal_, q);
  plane.pose.orientation = q;
  plane.scale.x = 3 * sqrt(single_plane.max_eigen_value_);
  plane.scale.y = 3 * sqrt(single_plane.mid_eigen_value_);
  plane.scale.z = 2 * sqrt(single_plane.min_eigen_value_);
  plane.color.a = alpha;
  plane.color.r = rgb(0);
  plane.color.g = rgb(1);
  plane.color.b = rgb(2);
  plane.lifetime = ros::Duration();
  plane_pub.markers.push_back(plane);
}

void VoxelMapManager::CalcVectQuation(const Eigen::Vector3d &x_vec, const Eigen::Vector3d &y_vec, const Eigen::Vector3d &z_vec,
                                      geometry_msgs::Quaternion &q)
{
  Eigen::Matrix3d rot;
  rot << x_vec(0), x_vec(1), x_vec(2), y_vec(0), y_vec(1), y_vec(2), z_vec(0), z_vec(1), z_vec(2);
  Eigen::Matrix3d rotation = rot.transpose();
  Eigen::Quaterniond eq(rotation);
  q.w = eq.w();
  q.x = eq.x();
  q.y = eq.y();
  q.z = eq.z();
}

void VoxelMapManager::mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b)
{
  r = 255;
  g = 255;
  b = 255;

  if (v < vmin) { v = vmin; }

  if (v > vmax) { v = vmax; }

  double dr, dg, db;

  if (v < 0.1242)
  {
    db = 0.504 + ((1. - 0.504) / 0.1242) * v;
    dg = dr = 0.;
  }
  else if (v < 0.3747)
  {
    db = 1.;
    dr = 0.;
    dg = (v - 0.1242) * (1. / (0.3747 - 0.1242));
  }
  else if (v < 0.6253)
  {
    db = (0.6253 - v) * (1. / (0.6253 - 0.3747));
    dg = 1.;
    dr = (v - 0.3747) * (1. / (0.6253 - 0.3747));
  }
  else if (v < 0.8758)
  {
    db = 0.;
    dr = 1.;
    dg = (0.8758 - v) * (1. / (0.8758 - 0.6253));
  }
  else
  {
    db = 0.;
    dg = 0.;
    dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758));
  }

  r = (uint8_t)(255 * dr);
  g = (uint8_t)(255 * dg);
  b = (uint8_t)(255 * db);
}

void VoxelMapManager::mapSliding()
{
  if((position_last_ - last_slide_position).norm() < config_setting_.sliding_thresh)
  {
    std::cout<<RED<<"[DEBUG]: Last sliding length "<<(position_last_ - last_slide_position).norm()<<RESET<<"\n";
    return;
  }

  //get global id now
  last_slide_position = position_last_;
  double t_sliding_start = omp_get_wtime();
  float loc_xyz[3];
  for (int j = 0; j < 3; j++)
  {
    loc_xyz[j] = position_last_[j] / config_setting_.max_voxel_size_;
    if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
  }
  // VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);//discrete global
  clearMemOutOfMap((int64_t)loc_xyz[0] + config_setting_.half_map_size, (int64_t)loc_xyz[0] - config_setting_.half_map_size,
                    (int64_t)loc_xyz[1] + config_setting_.half_map_size, (int64_t)loc_xyz[1] - config_setting_.half_map_size,
                    (int64_t)loc_xyz[2] + config_setting_.half_map_size, (int64_t)loc_xyz[2] - config_setting_.half_map_size);
  double t_sliding_end = omp_get_wtime();
  std::cout<<RED<<"[DEBUG]: Map sliding using "<<t_sliding_end - t_sliding_start<<" secs"<<RESET<<"\n";
  return;
}

void VoxelMapManager::clearMemOutOfMap(const int& x_max,const int& x_min,const int& y_max,const int& y_min,const int& z_max,const int& z_min )
{
  int delete_voxel_cout = 0;
  // double delete_time = 0;
  // double last_delete_time = 0;
  for (auto it = voxel_map_.begin(); it != voxel_map_.end(); )
  {
    const VOXEL_LOCATION& loc = it->first;
    bool should_remove = loc.x > x_max || loc.x < x_min || loc.y > y_max || loc.y < y_min || loc.z > z_max || loc.z < z_min;
    if (should_remove){
      // last_delete_time = omp_get_wtime();
      delete it->second;
      it = voxel_map_.erase(it);
      // delete_time += omp_get_wtime() - last_delete_time;
      delete_voxel_cout++;
    } else {
      ++it;
    }
  }
  std::cout<<RED<<"[DEBUG]: Delete "<<delete_voxel_cout<<" root voxels"<<RESET<<"\n";
  // std::cout<<RED<<"[DEBUG]: Delete "<<delete_voxel_cout<<" voxels using "<<delete_time<<" s"<<RESET<<"\n";
}