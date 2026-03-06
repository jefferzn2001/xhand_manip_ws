//
// Created by qiayuanl on 8/29/24.
//

#pragma once

#include <legged_model/LeggedModel.h>

#include <memory>
#include <vector>

namespace legged {
class GmObserver {
 public:
  using SharedPtr = std::shared_ptr<GmObserver>;

  explicit GmObserver(const std::shared_ptr<LeggedModel>& leggedModel, scalar_t cutoffFrequency = 10);

  void setCutOffFrequency(scalar_t cutoffFrequency) { cutoffFrequency_ = cutoffFrequency; }

  virtual void update(scalar_t period);

  std::vector<vector_t> getContactWrenches();

  vector_t getBaseWrench() const { return fExt_.tail<6>(); }

 protected:
  LeggedModel::SharedPtr leggedModel_;

  scalar_t cutoffFrequency_;

  vector_t lowPassLast_;
  vector_t tauExt_, fExt_;
};
}  // namespace legged
