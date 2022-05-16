/*
 * Copyright Codeplay Software Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "core/providers/sycl/sycl_kernel.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/sycl/sycl_fwd.h"

#include <CL/sycl.hpp>
#include "sycldnn/backend/sycl_blas_backend.h"

#include "sycldnn/conv2d/launch.h"
#include "sycldnn/conv2d/selector/default_selector.h"
#include "sycldnn/conv2d/workspace_size.h"
#include "sycldnn/binaryop/launch.h"
#include "sycldnn/binaryop/operators.h"
#include "sycldnn/transpose/launch.h"
#include "sycldnn/helpers/padding.h"
#include "sycldnn/status.h"

#include <gsl/gsl>

namespace snn = sycldnn;
using Backend = snn::backend::SyclBLASBackend;

namespace onnxruntime {
namespace sycl {

template <typename T>
struct SycldnnConvState {
  using DeviceMem = Backend::internal_pointer_type<T>;

  // keep track of prev x/w dims for eventual state update (per layer)
  gsl::span<const int64_t> prev_x_dims;
  gsl::span<const int64_t> prev_w_dims;

  // X & Y
  DeviceMem x_data;
  DeviceMem w_data;

  // Bias
  DeviceMem b_data;
  snn::binaryop::BinaryParams bias_params;

  // Output
  Tensor* Y = nullptr;
  DeviceMem y_data;

  // Shapes
  int64_t N;
  int64_t C;
  int64_t H_in;
  int64_t W_in;
  int64_t M;
  int64_t R;
  int64_t S;
  int64_t H_out;
  int64_t W_out;

  std::vector<int64_t> y_dims;

  // SNN conv params
  snn::conv2d::Conv2DParams snn_conv_params;

  // Workspace
  static inline DeviceMem workspace = DeviceMem();
  static inline size_t max_workspace_size = 0;
  size_t workspace_size;

  std::unique_ptr<snn::conv2d::Selector> selector;

#ifndef USE_SYCL_NHWC
  // Only used with NCHW layout
  DeviceMem input;
  DeviceMem weights;
  DeviceMem output;
  std::vector<int> input_sizes;
  std::vector<int> output_sizes;
  std::vector<int> weight_sizes;
  const std::vector<int> weight_permutations = {2, 3, 1, 0};
#endif
};

template <typename T>
class Conv final : public SyclKernel {
 public:
  Conv(const OpKernelInfo& info) : SyclKernel(info), conv_attrs_(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;

  ~Conv() override {
    Backend backend_{*Queue()};
    // Deallocating workspace at destruction time (since allocated once per layer)
#ifndef USE_SYCL_NHWC
    //Deallocating all the memory elements used
    backend_.template deallocate(state_.input);
    backend_.template deallocate(state_.weights);
    backend_.template deallocate(state_.output);
#endif
    if (SycldnnConvState<T>::max_workspace_size > 0) {
      // Deallocate once (the only largest workspace left)
      backend_.template deallocate(SycldnnConvState<T>::workspace);
      SycldnnConvState<T>::max_workspace_size = 0;
    }
  }

 protected:
  Status UpdateState(OpKernelContext* context, Backend& backend_) const;
  ConvAttributes conv_attrs_;
  mutable SycldnnConvState<T> state_;
};

}  // namespace sycl
}  // namespace onnxruntime
