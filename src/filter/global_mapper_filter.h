#pragma once

#include "glomap/controllers/track_establishment.h"
#include "glomap/controllers/track_retriangulation.h"
#include "glomap/estimators/bundle_adjustment.h"
#include "glomap/estimators/global_positioning.h"
#include "glomap/estimators/global_rotation_averaging.h"
#include "glomap/estimators/relpose_estimation.h"
#include "glomap/estimators/view_graph_calibration.h"
#include "glomap/types.h"
#include "glomap/controllers/global_mapper.h"

#include <colmap/scene/database.h>

namespace glomap {

class GlobalMapperFilter {
 public:
  GlobalMapperFilter(const GlobalMapperOptions& options) : options_(options) {}

  bool Solve(const colmap::Database& database,
             const std::string& image_path,
             ViewGraph& view_graph,
             std::unordered_map<camera_t, Camera>& cameras,
             std::unordered_map<image_t, Image>& images,
             std::unordered_map<track_t, Track>& tracks);

 private:
  const GlobalMapperOptions options_;
};

}  // namespace glomap
