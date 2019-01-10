// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/slice.h"
#include "core/providers/cpu/tensor/utils.h"
using namespace ::onnxruntime::common;
using namespace std;

namespace onnxruntime {

#define ADD_TYPED_SLICE_OP(data_type, indice_type)                                      \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                       \
      Slice,                                                                            \
      1,                                                                                \
      data_type,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()), \
      Slice<data_type, indice_type, false>);

ADD_TYPED_SLICE_OP(uint8_t,  int64_t);
ADD_TYPED_SLICE_OP(uint16_t, int64_t);
ADD_TYPED_SLICE_OP(uint32_t, int64_t);
ADD_TYPED_SLICE_OP(uint64_t, int64_t);
ADD_TYPED_SLICE_OP(int8_t,   int64_t);
ADD_TYPED_SLICE_OP(int16_t,  int64_t);
ADD_TYPED_SLICE_OP(int32_t,  int64_t);
ADD_TYPED_SLICE_OP(int64_t,  int64_t);
ADD_TYPED_SLICE_OP(float,    int64_t);
ADD_TYPED_SLICE_OP(double,   int64_t);
ADD_TYPED_SLICE_OP(MLFloat16,int64_t);
ADD_TYPED_SLICE_OP(bool,     int64_t);
ADD_TYPED_SLICE_OP(string,   int64_t);

#define ADD_TYPED_DYNAMIC_SLICE_OP(data_type, indice_type)                                   \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                            \
      DynamicSlice,                                                                          \
      1,                                                                                     \
      data_type##_##indice_type,                                                             \
      KernelDefBuilder().TypeConstraint("T",    DataTypeImpl::GetTensorType<data_type>())    \
                        .TypeConstraint("Tind", DataTypeImpl::GetTensorType<indice_type>()), \
      Slice<data_type, indice_type, true>);

ADD_TYPED_DYNAMIC_SLICE_OP(uint8_t,  int32_t);
ADD_TYPED_DYNAMIC_SLICE_OP(uint16_t, int32_t);
ADD_TYPED_DYNAMIC_SLICE_OP(uint32_t, int32_t);
ADD_TYPED_DYNAMIC_SLICE_OP(uint64_t, int32_t);
ADD_TYPED_DYNAMIC_SLICE_OP(int8_t,   int32_t);
ADD_TYPED_DYNAMIC_SLICE_OP(int16_t,  int32_t);
ADD_TYPED_DYNAMIC_SLICE_OP(int32_t,  int32_t);
ADD_TYPED_DYNAMIC_SLICE_OP(int64_t,  int32_t);
ADD_TYPED_DYNAMIC_SLICE_OP(float,    int32_t);
ADD_TYPED_DYNAMIC_SLICE_OP(double,   int32_t);
ADD_TYPED_DYNAMIC_SLICE_OP(MLFloat16,int32_t);
ADD_TYPED_DYNAMIC_SLICE_OP(bool,     int32_t);
ADD_TYPED_DYNAMIC_SLICE_OP(string,   int32_t);

ADD_TYPED_DYNAMIC_SLICE_OP(uint8_t,  int64_t);
ADD_TYPED_DYNAMIC_SLICE_OP(uint16_t, int64_t);
ADD_TYPED_DYNAMIC_SLICE_OP(uint32_t, int64_t);
ADD_TYPED_DYNAMIC_SLICE_OP(uint64_t, int64_t);
ADD_TYPED_DYNAMIC_SLICE_OP(int8_t,   int64_t);
ADD_TYPED_DYNAMIC_SLICE_OP(int16_t,  int64_t);
ADD_TYPED_DYNAMIC_SLICE_OP(int32_t,  int64_t);
ADD_TYPED_DYNAMIC_SLICE_OP(int64_t,  int64_t);
ADD_TYPED_DYNAMIC_SLICE_OP(float,    int64_t);
ADD_TYPED_DYNAMIC_SLICE_OP(double,   int64_t);
ADD_TYPED_DYNAMIC_SLICE_OP(MLFloat16,int64_t);
ADD_TYPED_DYNAMIC_SLICE_OP(bool,     int64_t);
ADD_TYPED_DYNAMIC_SLICE_OP(string,   int64_t);

namespace {
// std::clamp doesn't exist until C++17 so create a local version
template <typename T>
const T& clamp(const T& v, const T& lo, const T& hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}
}  // namespace

Status SliceBase::PrepareForCompute(const std::vector<int64_t>& raw_starts,
                                    const std::vector<int64_t>& raw_ends, 
                                    const std::vector<int64_t>& raw_axes,
                                    const size_t                dimension_count,
                                    const std::vector<int64_t>& input_dimensions,
                                    std::vector<int64_t>&       starts,
                                    std::vector<int64_t>&       output_dims) const {
  // Initialize axes to the provided axes attribute or to the default sequence
  std::vector<int64_t> axes(raw_axes);
  if (axes.size() == 0) {
    //axes are omitted, they are set to[0, ..., ndim - 1]
    axes.resize(starts.size());
    std::iota(axes.begin(), axes.end(), 0);
  }

  // Iterate through the provided axes and override the start/end ranges
  for (size_t axesIndex = 0; axesIndex < axes.size(); axesIndex++) {
    auto axis = axes[axesIndex] < 0 ? axes[axesIndex] + static_cast<int64_t>(dimension_count) : axes[axesIndex];
    if (axis >= static_cast<int64_t>(dimension_count) || axis < 0)
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'axes' has an axis outside of the tensor dimension count");
    auto start = raw_starts[axesIndex];
    if (start < 0)
      start += input_dimensions[axis];
    starts[axis] = clamp(start, int64_t{0}, input_dimensions[axis]);

    auto end = raw_ends[axesIndex];
    if (end < 0)
      end += input_dimensions[axis];
    output_dims[axis] = clamp(end, int64_t{0}, input_dimensions[axis]) - starts[axis];
    if (output_dims[axis] < 0)
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'starts' and 'ends' values resulted in a negative dimension");
  }

  return Status::OK();
}

template <typename Tind>
void SliceBase::FillVectorsFromInput(const OpKernelContext* context,
                                     std::vector<int64_t>&  input_starts,
                                     std::vector<int64_t>&  input_ends,
                                     std::vector<int64_t>&  input_axes) const {
  auto stat_tensor = context->Input<Tensor>(1);
  auto ends_tensor = context->Input<Tensor>(2);
  auto axes_tensor = context->Input<Tensor>(3);

  ORT_ENFORCE (nullptr != stat_tensor && stat_tensor->Shape().NumDimensions() == 1,    "Starts must be a 1-D array"    );
  ORT_ENFORCE (nullptr != ends_tensor && ends_tensor->Shape().NumDimensions() == 1,    "ends must be a 1-D array"      );
  ORT_ENFORCE (stat_tensor->Shape() == ends_tensor->Shape(),                           "Starts and ends shape mismatch");
  ORT_ENFORCE (nullptr == axes_tensor || stat_tensor->Shape() == axes_tensor->Shape(), "Starts and axes shape mismatch");

  auto size = stat_tensor->Shape().Size();
  input_starts.resize(size);
  std::copy(stat_tensor->Data<Tind>(), stat_tensor->Data<Tind>() + size, input_starts.begin());
  input_ends.resize(size);
  std::copy(ends_tensor->Data<Tind>(), ends_tensor->Data<Tind>() + size, input_ends.begin());
  if (nullptr != axes_tensor) {
    input_axes.resize(size);
    std::copy(axes_tensor->Data<Tind>(), axes_tensor->Data<Tind>() + size, input_axes.begin());
  }
}

template <typename T, typename Tind, bool dynamic>
Status Slice<T, Tind, dynamic>::Compute(OpKernelContext* ctx) const {
  const Tensor* input_tensor_ptr = ctx->Input<Tensor>(0);
  ORT_ENFORCE(input_tensor_ptr != nullptr);
  auto& input_tensor = *input_tensor_ptr;
  auto& input_dimensions = input_tensor.Shape().GetDims();

  // Initialize the starts & ends to the actual tensor shape
  const size_t dimension_count = input_dimensions.size();
  std::vector<int64_t> starts(dimension_count, 0);
  std::vector<int64_t> output_dims(input_dimensions);

  if (dynamic) {
    std::vector<int64_t> input_starts, input_ends, input_axes;
    FillVectorsFromInput<Tind>(ctx, input_starts, input_ends, input_axes);
    ORT_RETURN_IF_ERROR(PrepareForCompute(input_starts, input_ends, input_axes,
                        dimension_count, input_dimensions, starts, output_dims));
  } else {
    ORT_RETURN_IF_ERROR(PrepareForCompute(attr_starts_, attr_ends_, attr_axes_,
                        dimension_count, input_dimensions, starts, output_dims));
  }

  TensorShape output_shape(output_dims);
  auto& output_tensor = *ctx->Output(0, output_shape);
  auto* output = output_tensor.template MutableData<T>();
  const auto* output_end = output + output_shape.Size();

  SliceIterator<T> input_iterator(input_tensor, starts, output_dims);
  while (output != output_end)
    *output++ = *input_iterator++;

  return Status::OK();
}

}  // namespace onnxruntime
