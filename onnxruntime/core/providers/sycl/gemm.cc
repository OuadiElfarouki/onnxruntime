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

#include "core/providers/sycl/gemm.h"
#include "core/providers/cpu/math/gemm_helper.h"

#include <CL/sycl.hpp>

#include "sycldnn/backend/sycl_blas_backend.h"
#include "sycldnn/binaryop/launch.h"
#include "sycldnn/binaryop/operators.h"
#include "sycldnn/binaryop/params.h"

#include "sycldnn/status.h"

namespace snn = sycldnn;
using Backend = snn::backend::SyclBLASBackend;

namespace onnxruntime {
namespace sycl {

// Registering VERSIONNED TYPED Kernels
#define REGISTER_VERSIONED_GEMM_KERNEL_TYPED(T, start, end)       \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Gemm,                                                       \
      kOnnxDomain,                                                \
      start,                                                      \
      end,                                                        \
      T,                                                          \
      kSyclExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Gemm<T>);

#define REGISTER_GEMM_KERNEL_TYPED(T, start)                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Gemm,                                                       \
      kOnnxDomain,                                                \
      start,                                                      \
      T,                                                          \
      kSyclExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Gemm<T>);

template <typename T>
Status Gemm<T>::ComputeInternal(OpKernelContext* context) const {
  // INPUT
  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* W = context->Input<Tensor>(1);
  const Tensor* B = context->Input<Tensor>(2);

  // Gemm helper for dimensions verification / computation
  GemmHelper helper(X->Shape(), trans_A_, W->Shape(), trans_B_, B != nullptr ? B->Shape() : TensorShape({}));
  if (!helper.State().IsOK())
    return helper.State();

  // Extracting dimensions
  int M = static_cast<int>(helper.M());
  int N = static_cast<int>(helper.N());
  int K = static_cast<int>(helper.K());

  // OUTPUT
  Tensor* Y = context->Output(0, {M, N});

  // if input is empty tensor, return as nothing need to be calculated and we've set the shape for the output
  if (M == 0 || N == 0)
    return Status::OK();

  // SYCL BUFFERS
  const cl::sycl::buffer<T, 1> X_buffer = *X->template Ptr<cl::sycl::buffer<T, 1>>();
  const cl::sycl::buffer<T, 1> W_buffer = *W->template Ptr<cl::sycl::buffer<T, 1>>();
  cl::sycl::buffer<T, 1> Y_buffer = *Y->template MutablePtr<cl::sycl::buffer<T, 1>>();

  // SYCL DNN Backend
  auto queue = *Queue();
  Backend backend{queue};

  using DeviceMem = Backend::internal_pointer_type<T>;

  // Creating Device Pointers to Buffers
  auto x_data = DeviceMem(X_buffer, static_cast<size_t>(X->ByteOffset() / sizeof(T)));
  auto w_data = DeviceMem(W_buffer, static_cast<size_t>(W->ByteOffset() / sizeof(T)));
  auto y_data = DeviceMem(Y_buffer, static_cast<size_t>(Y->ByteOffset() / sizeof(T)));


  backend.template matmul<false, false, T, int>(x_data, w_data, y_data, 0.f, M, K, N);  

  sycldnn::binaryop::BinaryParams params{M*N, N};

  cl::sycl::buffer<T, 1>* B_buffer = const_cast<cl::sycl::buffer<T, 1>*>(B->template Ptr<cl::sycl::buffer<T, 1>>());
  const TensorShape& b_shape = B->Shape();
  auto b_data = DeviceMem(*B_buffer, static_cast<size_t>(B->ByteOffset() / sizeof(T)));

  using ConstMem = Backend::pointer_type<T const>;

  sycldnn::binaryop::launch<T, sycldnn::binaryop::Add>(ConstMem{y_data}, b_data, y_data, params, backend); 

  return Status::OK();
}

// REGISTER KERNEL
REGISTER_VERSIONED_GEMM_KERNEL_TYPED(float, 7, 8)
REGISTER_VERSIONED_GEMM_KERNEL_TYPED(float, 9, 10)
REGISTER_VERSIONED_GEMM_KERNEL_TYPED(float, 11, 12)
REGISTER_GEMM_KERNEL_TYPED(float, 13)

}  // namespace sycl
}  // namespace onnxruntime
