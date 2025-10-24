// Copyright 2020 LMNT, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

#include "device_ptr.h"
#include "haste.h"

using haste::v0::gru::ForwardPass;
using haste::v0::gru::BackwardPass;
using std::string;

using Tensor1 = Eigen::Tensor<float, 1>;
using Tensor2 = Eigen::Tensor<float, 2>;
using Tensor3 = Eigen::Tensor<float, 3>;

constexpr int BATCH_SIZE = 64; // 批大小
constexpr int SEQUENCE_LEN = 1000; // 序列长度(T), 每个样本有T个时间步
constexpr int HIDDEN_DIMS = 512; // 隐藏层维度(H), h_t的维度
constexpr int INPUT_DIMS = 512; // 输入维度(I), x_t的维度

static cublasHandle_t g_blas_handle;

class ScopeTimer { // 测量时间类
  public:
    ScopeTimer(const string& msg) : msg_(msg) {
      cudaEventCreate(&start_);
      cudaEventCreate(&stop_);
      cudaDeviceSynchronize();
      cudaEventRecord(start_);
    }

    ~ScopeTimer() {
      float elapsed_ms;
      cudaEventRecord(stop_);
      cudaEventSynchronize(stop_);
      cudaEventElapsedTime(&elapsed_ms, start_, stop_);
      printf("%s %fms\n", msg_.c_str(), elapsed_ms);
      cudaEventDestroy(start_);
      cudaEventDestroy(stop_);
    }

  private:
    string msg_;
    cudaEvent_t start_, stop_;
};

void GruInference(
    const Tensor2& W,
    const Tensor2& R,
    const Tensor1& bx,
    const Tensor1& br,
    const Tensor3& x) {
  const int time_steps = x.dimension(2);
  const int batch_size = x.dimension(1);
  const int input_size = x.dimension(0);
  const int hidden_size = R.dimension(1);

  // Copy weights over to GPU.
  device_ptr<Tensor2> W_dev(W);
  device_ptr<Tensor2> R_dev(R);
  device_ptr<Tensor1> bx_dev(bx);
  device_ptr<Tensor1> br_dev(br);
  device_ptr<Tensor3> x_dev(x);

  device_ptr<Tensor2> h_dev((time_steps + 1) * batch_size * hidden_size);
  device_ptr<Tensor3> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3);
  device_ptr<Tensor2> tmp_Rh_dev(batch_size * hidden_size * 3);

  h_dev.zero();

  ScopeTimer t("Inference:");

  ForwardPass<float> forward = ForwardPass<float>(
      false,  // training
      batch_size,
      input_size,
      hidden_size,
      g_blas_handle);

  forward.Run(
      time_steps,
      W_dev.data,
      R_dev.data,
      bx_dev.data,
      br_dev.data,
      x_dev.data,
      h_dev.data,
      nullptr,
      tmp_Wx_dev.data,
      tmp_Rh_dev.data,
      0.0f,
      nullptr);
}

void GruTrain(
    const Tensor2& W,
    const Tensor2& R,
    const Tensor1& bx,
    const Tensor1& br,
    const Tensor3& x,
    const Tensor3& dh_new) {
  const int time_steps = x.dimension(2);
  const int batch_size = x.dimension(1);
  const int input_size = x.dimension(0);
  const int hidden_size = R.dimension(1);

  // Copy weights over to GPU.
  device_ptr<Tensor2> W_dev(W);
  device_ptr<Tensor2> R_dev(R);
  device_ptr<Tensor1> bx_dev(bx);
  device_ptr<Tensor1> br_dev(br);
  device_ptr<Tensor3> x_dev(x);
  device_ptr<Tensor3> dh_new_dev(dh_new);

  device_ptr<Tensor2> h_dev((time_steps + 1) * batch_size * hidden_size);
  device_ptr<Tensor3> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3);
  device_ptr<Tensor2> tmp_Rh_dev(batch_size * hidden_size * 3);
  device_ptr<Tensor3> v_dev(time_steps * batch_size * hidden_size * 4);

  h_dev.zero();

  {
    ScopeTimer t("Train forward:");
    ForwardPass<float> forward = ForwardPass<float>(
        true,  // training
        batch_size,
        input_size,
        hidden_size,
        g_blas_handle);

    forward.Run(
        time_steps,
        W_dev.data,
        R_dev.data,
        bx_dev.data,
        br_dev.data,
        x_dev.data,
        h_dev.data,
        v_dev.data,
        tmp_Wx_dev.data,
        tmp_Rh_dev.data,
        0.0f,
        nullptr);
  }

  device_ptr<Tensor3> dx_dev(time_steps * batch_size * input_size);
  device_ptr<Tensor2> dW_dev(input_size * hidden_size * 3);
  device_ptr<Tensor2> dR_dev(hidden_size * hidden_size * 3);
  device_ptr<Tensor1> dbx_dev(hidden_size * 3);
  device_ptr<Tensor1> dbr_dev(hidden_size * 3);
  device_ptr<Tensor2> dh_dev(batch_size * hidden_size);
  device_ptr<Tensor3> dp_dev(time_steps * batch_size * hidden_size * 3);
  device_ptr<Tensor3> dq_dev(time_steps * batch_size * hidden_size * 3);

  {
    ScopeTimer t("Train backward:");
    BackwardPass<float> backward(
        batch_size,
        input_size,
        hidden_size,
        g_blas_handle);

    backward.Run(
        time_steps,
        W_dev.data,
        R_dev.data,
        bx_dev.data,
        br_dev.data,
        x_dev.data,
        h_dev.data,
        v_dev.data,
        dh_new_dev.data,
        dx_dev.data,
        dW_dev.data,
        dR_dev.data,
        dbx_dev.data,
        dbr_dev.data,
        dh_dev.data,
        dp_dev.data,
        dq_dev.data,
        nullptr);
  }
}

// t: 时间步索引
// xt: 第t时间步的输入数据(如音频中的第t个采样点). 维度: I * 1 (列向量, I为输入特征维度)
// ht: 第t时间步的隐藏状态(GRU的输出, 包含当前时间步的"记忆"). 维度: H * 1(列向量, H为隐藏层维度)
// ht-1: 第t-1时间步的隐藏状态(上一时刻的"记忆", 作为当前时刻的输入). 维度: H * 1

// 更新门(zt)公式
// zt: 更新门的输出(取值0~1): 控制"保留上一时刻隐藏状态ht-1"的比例(值越接近1保留越多). 维度: H * 1
// σ(⋅): sigmoid激活函数, 将值映射到0~1区间. (相当于归一化)
// Wz: 更新门的"输入层->隐藏层"权重矩阵(控制输如xt对更新门的影响). 维度: H * I
// Rz: 更新门的"隐藏层->隐藏层"权重矩阵(控制上一时刻隐藏状态ht-1对更新门的影响). 维度: H * H
// bz: 更新门的偏置项(挑战更新门的基线输出). 维度: H * 1
// Wz * xt: 输入xt与权重Wz的矩阵乘法(线性变化, 提取输入对更新门的特征). 维度: H * 1
// Rz * ht-1: 上一时刻隐藏状态ht-1与权重Rz的矩阵乘法(提取历史记忆对更新门的特征). 维度: H * 1

// 重置门(rt)公式
// rt: 重置门输出(取值0~1): 控制"遗忘上一时刻隐藏状态ht-1"的比例(值越接近0, 遗忘越多). 维度: H * 1
// Wr: 重置门的"输入层->隐藏层"权重矩阵(控制输入xt对重置门的影响). 维度: H * 1
// Rr: 重置门的 “隐藏层→隐藏层” 权重矩阵(控制上一时刻影藏状态ht-1对重置门的影响). 维度: H * H
// br: 重置门的偏置项(挑战重置门的基线输出). 维度: H * 1
// Wr * xt: 输入xt与权重Wr的矩阵乘法(根据重置门的 “功能需求”，对输入数据进行针对性提取). 维度: H * 1
// Rr * ht-1: 上一时刻隐藏状态ht-1与权重Rr的矩阵乘法(提取历史记忆对重置门的特征). 维度: H * 1
//

// 候选隐藏状态(~ht)公式
// ~ht: 候选隐藏状态(取值-1~1): 基于当前输入xt和"筛选后的历史记忆"计算的新信息. 维度: H * 1
// tanh(⋅): 双曲正切集合函数. 将值映射到-1~1区间
// Wh: 候选状态的 “输入层→隐藏层” 权重矩阵（控制输入xt对候选状态的影响）. 维度: H × I
// Rh: 候选状态的 “隐藏层→隐藏层” 权重矩阵（控制筛选后的历史记忆对候选状态的影响）. 维度: H * H
// bh: 候选状态的偏置项(调整候选状态的基线输出). 维度: H * 1
// ⊙: hadamard积. 两个同维度向量对应位置元素相乘
// rt ⊙ ht-1: 重置门对历史记忆的筛选: rt值越小, ht-1中被保留的信息越少. 维度: H * 1
// Wh * xt: 输入xt与权重Wh的矩阵乘法(线性变换，提取输入对候选状态的特征). 维度: H * 1
// Rh(rt ⊙ ht-1): 筛选后的历史记忆与权重Rh的矩阵乘法(提取筛选后记忆对候选状态的特征). 维度: H * 1

// 最终隐藏状态(ht)公式
// ht: 第t时间步的最终隐藏状态: 平衡了"新信息(~ht)"和"旧信息(ht-1)". 维度: H * 1
// 1 - zt: 与更新门互补的值: 控制保留新信息~ht的比例. 维度: H * 1
// (1 - zt) ⊙ ~ht: 新信息的保留部分: 1 - zt越大, 保留的新信息越多.
// zt ⊙ ht-1: 旧信息的保留部分: zt越大, 保留的旧信息越多. 维度: H * 1

int main() {
  srand(time(0));

  cublasCreate(&g_blas_handle);

  // Weights.
  Tensor2 W(HIDDEN_DIMS * 3, INPUT_DIMS); // 对应W_z/W_r/W_h的合并
  Tensor2 R(HIDDEN_DIMS * 3, HIDDEN_DIMS); // 对应R_z/R_r/R_h的合并
  Tensor1 bx(HIDDEN_DIMS * 3); // 对应b_z/b_r/b_h的合并. bx 负责给 “输入 x_t 到门控的线性变换” 加偏置
  Tensor1 br(HIDDEN_DIMS * 3); // br: 3H(部分实现中偏置分输出\隐藏层. br 负责给 “隐藏状态 h_{t-1} 到门控的线性变换” 加偏置

  // Input.
  Tensor3 x(INPUT_DIMS, BATCH_SIZE, SEQUENCE_LEN);

  // Gradients from upstream layers.
  Tensor3 dh(HIDDEN_DIMS, BATCH_SIZE, SEQUENCE_LEN + 1);

  W.setRandom();
  R.setRandom();
  bx.setRandom();
  br.setRandom();
  x.setRandom();
  dh.setRandom();

  GruInference(W, R, bx, br, x);
  GruTrain(W, R, bx, br, x, dh);

  cublasDestroy(g_blas_handle);

  return 0;
}
