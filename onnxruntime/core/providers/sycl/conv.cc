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

#include "core/providers/sycl/conv.h"

namespace onnxruntime {
namespace sycl {

// Registering Kernel
#define REGISTER_VERSIONED_CONV_KERNEL_TYPED(T, start, end)       \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Conv,                                                       \
      kOnnxDomain,                                                \
      start,                                                      \
      end,                                                        \
      T,                                                          \
      kSyclExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Conv<T>);

#define REGISTER_CONV_KERNEL_TYPED(T, start)                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Conv,                                                       \
      kOnnxDomain,                                                \
      start,                                                      \
      T,                                                          \
      kSyclExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Conv<T>);

template <typename T>
Status Conv<T>::UpdateState(OpKernelContext* context, Backend& backend_) const {
  // Set X & W
  const auto* X = context->Input<Tensor>(0);
  const auto* W = context->Input<Tensor>(1);

  const TensorShape& x_shape = X->Shape();
  gsl::span<const int64_t> x_dims = x_shape.GetDims();
  size_t x_num_dims = x_shape.NumDimensions();

  const TensorShape& w_shape = W->Shape();
  gsl::span<const int64_t> w_dims = w_shape.GetDims();
  size_t w_num_dims = w_shape.NumDimensions();

  // X & Y SYCL BUFFERS
  const cl::sycl::buffer<T, 1> X_buffer = *X->template Ptr<cl::sycl::buffer<T, 1>>();
  const cl::sycl::buffer<T, 1> W_buffer = *W->template Ptr<cl::sycl::buffer<T, 1>>();

  using DeviceMem = Backend::internal_pointer_type<T>;

  // Creating Device Pointers to Buffers
  state_.x_data = DeviceMem(X_buffer, static_cast<size_t>(X->ByteOffset() / sizeof(T)));
  state_.w_data = DeviceMem(W_buffer, static_cast<size_t>(W->ByteOffset() / sizeof(T)));

  // Set B
  if (context->InputCount() >= 3) {
    const Tensor* B = context->Input<Tensor>(2);
    const cl::sycl::buffer<T, 1> B_buffer = *B->template Ptr<cl::sycl::buffer<T, 1>>();
    state_.b_data = DeviceMem(B_buffer, static_cast<size_t>(B->ByteOffset() / sizeof(T)));
  }

  // Check if previous x & w dims are the same to avoid re-calculating shapes and workspace size
  bool x_dims_changed = (state_.prev_x_dims != x_dims);
  bool w_dims_changed = (state_.prev_w_dims != w_dims);

  if (x_dims_changed || w_dims_changed) {
    // input/weights dims check-update
    state_.prev_x_dims = x_dims_changed ? x_dims : state_.prev_x_dims;
    state_.prev_w_dims = w_dims_changed ? w_dims : state_.prev_w_dims;

    state_.N = x_shape[0];

#ifndef USE_SYCL_NHWC
    ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X, W));
    state_.C = x_num_dims > 1 ? x_shape[1] : 1;
    state_.H_in = x_num_dims > 2 ? x_shape[2] : 1;
    state_.W_in = x_num_dims > 3 ? x_shape[3] : 1;

    state_.M = w_shape[0];
    int64_t C_w = w_num_dims > 1 ? w_shape[1] : 1;
    state_.R = w_num_dims > 2 ? w_shape[2] : 1;
    state_.S = w_num_dims > 3 ? w_shape[3] : 1;
#else
    // TODO: Implement ValidateInputs for SYCL EP for the NHWC layout
    state_.C = x_num_dims > 3 ? x_shape[3] : 1;
    state_.H_in = x_num_dims > 1 ? x_shape[1] : 1;
    state_.W_in = x_num_dims > 2 ? x_shape[2] : 1;

    state_.R = w_shape[0];
    state_.S = w_num_dims > 1 ? w_shape[1] : 1;
    int64_t C_w = w_num_dims > 2 ? w_shape[2] : 1;
    state_.M = w_num_dims > 3 ? w_shape[3] : 1;
#endif
    if (state_.C != C_w) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Invalid Channel Dimensions");
    }

    std::vector<int64_t> kernel_shape = {state_.R, state_.S};
#ifndef USE_SYCL_NHWC
    ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(w_shape, kernel_shape));
#endif

    std::vector<int64_t> pads(conv_attrs_.pads);
    if (pads.size() < 2 * kernel_shape.size()) {
      pads.resize(kernel_shape.size() * 2, 0);
    }
    std::vector<int64_t> dilations(conv_attrs_.dilations);
    if (dilations.size() < kernel_shape.size()) {
      dilations.resize(kernel_shape.size(), 1);
    }
    std::vector<int64_t> strides(conv_attrs_.strides);
    if (strides.size() < kernel_shape.size()) {
      strides.resize(kernel_shape.size(), 1);
    }

    // Setting output Y
    std::vector<int64_t> Y_dims({state_.N});
    std::vector<int64_t> input_shape({state_.H_in});
#ifndef USE_SYCL_NHWC
    Y_dims.push_back(state_.M);
    if (x_num_dims > 3) input_shape.push_back(state_.W_in);
    ORT_RETURN_IF_ERROR(conv_attrs_.InferOutputShape((TensorShape)input_shape, kernel_shape, strides, dilations, pads, Y_dims));
#else
    if (x_num_dims > 2) input_shape.push_back(state_.W_in);
    ORT_RETURN_IF_ERROR(conv_attrs_.InferOutputShape((TensorShape)input_shape, kernel_shape, strides, dilations, pads, Y_dims));
    Y_dims.push_back(state_.M);
#endif

    state_.y_dims = Y_dims;
    state_.Y = context->Output(0, state_.y_dims);
    size_t y_num_dims = state_.Y->Shape().NumDimensions();

    // Bail out early if one of the Y dimensions is zero.
    if (state_.Y->Shape().Size() == 0) {
      return Status::OK();
    }

    cl::sycl::buffer<T, 1> Y_buffer = *(state_.Y->template MutablePtr<cl::sycl::buffer<T, 1>>());
    state_.y_data = DeviceMem(Y_buffer, static_cast<size_t>(state_.Y->ByteOffset() / sizeof(T)));

#ifndef USE_SYCL_NHWC
    state_.H_out = y_num_dims > 2 ? state_.Y->Shape()[2] : 1;
    state_.W_out = y_num_dims > 3 ? state_.Y->Shape()[3] : 1;
#else
    state_.H_out = y_num_dims > 1 ? state_.Y->Shape()[1] : 1;
    state_.W_out = y_num_dims > 2 ? state_.Y->Shape()[2] : 1;
#endif

    //Setting Bias parameters
    if (context->InputCount() >= 3) {
      state_.bias_params.lhs_items = static_cast<int>(state_.H_out * state_.W_out * state_.N * state_.M);
      state_.bias_params.rhs_items = static_cast<int>(state_.M);
    }

    // Setting Conv parameters
    state_.snn_conv_params.channels = static_cast<int>(state_.C);
    state_.snn_conv_params.features = static_cast<int>(state_.M);
    state_.snn_conv_params.batch = static_cast<int>(state_.N);
    state_.snn_conv_params.in_rows = static_cast<int>(state_.H_in);
    state_.snn_conv_params.in_cols = static_cast<int>(state_.W_in);
    state_.snn_conv_params.window_rows = static_cast<int>(state_.R);
    state_.snn_conv_params.window_cols = static_cast<int>(state_.S);
    state_.snn_conv_params.out_rows = static_cast<int>(state_.H_out);
    state_.snn_conv_params.out_cols = static_cast<int>(state_.W_out);

    state_.snn_conv_params.stride_rows = static_cast<int>(strides[0]);
    state_.snn_conv_params.stride_cols = static_cast<int>(strides[strides.size() - 1]);

    state_.snn_conv_params.pad_rows = static_cast<int>(pads[0]);
    state_.snn_conv_params.pad_cols = static_cast<int>(pads[pads.size() - 1]);

    // Conv selector instance
    state_.selector = snn::conv2d::get_default_selector(Queue()->get_device());

    //Querying the required workspace size
    state_.workspace_size = snn::conv2d::query_workspace_size<
                                snn::conv2d::conv_type::Forward>(state_.snn_conv_params, *(state_.selector))
                                .recommended_size;

    // Allocating workspace if required
    if (state_.workspace_size > SycldnnConvState<T>::max_workspace_size) {
      backend_.template deallocate(SycldnnConvState<T>::workspace);
      SycldnnConvState<T>::workspace = backend_.template allocate<T>(state_.workspace_size);
      SycldnnConvState<T>::max_workspace_size = state_.workspace_size;
    }

    // Extra state variables to be set for transpose to adequate data layout (NCHW case)
#ifndef USE_SYCL_NHWC
    state_.input_sizes = {(int)state_.N, (int)state_.C, (int)state_.H_in, (int)state_.W_in};
    state_.weight_sizes = {(int)state_.M, (int)state_.C, (int)state_.R, (int)state_.S};

    // Allocating Intermediate Memory to perform computations in NHWC format through
    // SYCL-DNN
    state_.input = backend_.template allocate<T>(static_cast<size_t>(state_.N * state_.C * state_.H_in * state_.W_in));
    state_.weights = backend_.template allocate<T>(static_cast<size_t>(state_.M * state_.C * state_.R * state_.S));
#endif

  } else {
    //set Y
    state_.Y = context->Output(0, state_.y_dims);
    if (state_.Y->Shape().Size() == 0) {
      return Status::OK();
    }
    cl::sycl::buffer<T, 1> Y_buffer = *(state_.Y->template MutablePtr<cl::sycl::buffer<T, 1>>());
    state_.y_data = DeviceMem(Y_buffer, static_cast<size_t>(state_.Y->ByteOffset() / sizeof(T)));
  }
  return Status::OK();
}

template <typename T>
Status Conv<T>::ComputeInternal(OpKernelContext* context) const {
  Backend backend_{*Queue()};

  ORT_RETURN_IF_ERROR(UpdateState(context, backend_));

#ifndef USE_SYCL_NHWC

  // Performing input conversion from NCHW to NHWC for feature map
  snn::transpose::convert_nchw_to_nhwc<T, Backend>(state_.x_data, state_.input, state_.input_sizes, backend_);

  // Performing conversion from MCHW to HWCM for weights
  snn::transpose::launch<T, Backend>(state_.w_data, state_.weights, state_.weight_sizes, state_.weight_permutations, backend_);

  state_.output = backend_.template allocate<T>(static_cast<size_t>(state_.N * state_.M * state_.H_out * state_.W_out));

  //Launching Conv kernel
  snn::conv2d::launch<T, snn::conv2d::conv_type::Forward>(
      state_.input, state_.weights, state_.output, state_.snn_conv_params, *(state_.selector), backend_, SycldnnConvState<T>::workspace, state_.workspace_size);

#else
  //Launching Conv kernel
  snn::conv2d::launch<T, snn::conv2d::conv_type::Forward>(
      state_.x_data, state_.w_data, state_.y_data, state_.snn_conv_params, *(state_.selector), backend_, SycldnnConvState<T>::workspace, state_.workspace_size);

#endif

  //Check if Bias Addition is required
  if (context->InputCount() >= 3) {
#ifndef USE_SYCL_NHWC
    // Launching Bias addition kernel
    snn::binaryop::launch<T, snn::binaryop::Add>(state_.output, state_.b_data, state_.output, state_.bias_params, backend_);

    // Converting back NHWC -> NCHW
    snn::transpose::convert_nhwc_to_nchw<T, Backend>(state_.output, state_.y_data, state_.output_sizes, backend_);
#else
    // Launching Bias addition kernel
    snn::binaryop::launch<T, snn::binaryop::Add>(state_.y_data, state_.b_data, state_.y_data, state_.bias_params, backend_);
#endif
  }

  return Status::OK();
}

// REGISTER KERNEL
REGISTER_VERSIONED_CONV_KERNEL_TYPED(float, 1, 10)
REGISTER_CONV_KERNEL_TYPED(float, 11)

}  // namespace sycl
}  // namespace onnxruntime
