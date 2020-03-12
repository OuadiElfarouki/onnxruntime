// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include <atomic>
#include <cstdint>
#include <mutex>
#include <memory>
#include <condition_variable>

namespace onnxruntime {
namespace contrib {

class OrtEventPool final {
 public:
  static OrtEventPool& GetInstance() {
    static OrtEventPool instance_;
    return instance_;
  }
  void SignalEvent(int64_t id);
  void ResetEvent(int64_t id);
  bool QueryEvent(int64_t id) const;
  void WaitEvent(int64_t id) const;

  size_t GetPoolSize() const {
    return MaxNumItems;
  }

 private:
  OrtEventPool() = default;
  ~OrtEventPool() = default;
  OrtEventPool(const OrtEventPool&) = delete;
  OrtEventPool& operator=(const OrtEventPool&) = delete;

  struct Item {
    std::atomic<bool> signaled;
    mutable std::mutex mutex;
    mutable std::condition_variable cv;

    Item() {
      signaled.store(false);
    }
  };
  enum {
    MaxNumItems = 4096
  };
  Item pool_[MaxNumItems];
};

}  // namespace contrib
}  // namespace onnxruntime
