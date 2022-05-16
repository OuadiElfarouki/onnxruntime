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

#include "core/providers/sycl/reduction/reduce.h"
#include "core/providers/common.h"

#include <CL/sycl.hpp>

#include "sycldnn/backend/sycl_blas_backend.h"
#include "sycldnn/transpose/launch.h"
#include "sycldnn/status.h"

namespace snn = sycldnn;
using Backend = snn::backend::SyclBLASBackend;

namespace onnxruntime {
namespace sycl {

// Registering Kernels
#define REGISTER_VERSIONED_REDUCE_KERNELS_TYPED(T, op, start, end) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                         \
      op,                                                          \
      kOnnxDomain,                                                 \
      start,                                                       \
      end,                                                         \
      T,                                                           \
      kSyclExecutionProvider,                                      \
      KernelDefBuilder()                                           \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),  \
      op<T>);

#define REGISTER_REDUCE_KERNELS_TYPED(T, op, start)               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      op,                                                         \
      kOnnxDomain,                                                \
      start,                                                      \
      T,                                                          \
      kSyclExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      op<T>);

template <typename T>
Status ReduceMean<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();
  size_t x_dims = x_shape.NumDimensions();

  std::vector<int> input_shape(x_dims);

  std::vector<int64_t> y_shape{1,x_shape[3]};

  Tensor* Y = context->Output(0, y_shape);

  const cl::sycl::buffer<T, 1> X_buffer = *X->template Ptr<cl::sycl::buffer<T, 1>>();
  cl::sycl::buffer<T, 1> Y_buffer = *Y->template MutablePtr<cl::sycl::buffer<T, 1>>();

  // SYCL DNN Backend
  Backend backend{*Queue()};

  using DeviceMem = Backend::internal_pointer_type<T>;

  //Creating Device Pointers to Buffers
  auto x_data = DeviceMem(X_buffer, static_cast<size_t>(X->ByteOffset() / sizeof(T)));
  auto y_data = DeviceMem(Y_buffer, static_cast<size_t>(Y->ByteOffset() / sizeof(T)));

  int preserve_dims, reduce_dims;
  preserve_dims = x_shape[3];
  reduce_dims = x_shape[1]*x_shape[2];

  auto executor = backend.get_executor();

  if (axes_.size() > 0) {
    blas::extension::_reduction<blas::MeanOperator, T>(
        executor, x_data, preserve_dims, y_data, preserve_dims, reduce_dims,
        blas::reduction_dim_t::outer);
  } else {
    preserve_dims = 1;
    for (size_t i = 0; i < x_dims; i++) {
      reduce_dims *= input_shape[i];
    }
    blas::extension::_reduction<blas::MeanOperator, T>(
        executor, x_data, reduce_dims, y_data, reduce_dims, preserve_dims,
        blas::reduction_dim_t::inner);
  }

  return Status::OK();
}

REGISTER_VERSIONED_REDUCE_KERNELS_TYPED(float, ReduceMean, 1, 10)
REGISTER_VERSIONED_REDUCE_KERNELS_TYPED(float, ReduceMean, 11, 12)
REGISTER_REDUCE_KERNELS_TYPED(float, ReduceMean, 13)

}  // namespace sycl
}  // namespace onnxruntime
