// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * \param device_id cuda device id, starts from zero.
 * \param out Call ONNXRuntimeReleaseObject() method when you no longer need to use it.
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeCreateCUDAExecutionProviderFactory, int device_id, _Out_ ONNXRuntimeProviderFactoryInterface*** out);

#ifdef __cplusplus
}
#endif
