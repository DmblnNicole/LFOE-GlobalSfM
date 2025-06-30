#include "global_mapper_filter.h"
#include "glomap/controllers/global_mapper.h"
#include "glomap/controllers/option_manager.h"
#include "glomap/io/colmap_io.h"
#include "glomap/types.h"

#include <colmap/util/file.h>
#include <colmap/util/misc.h>
#include <colmap/util/timer.h>

// Code adapted from https://github.com/colmap/glomap

namespace glomap {

// -------------------------------------
// Mappers starting from COLMAP database
// -------------------------------------
int RunMapper(int argc, char** argv) {
  std::string database_path;
  std::string output_path;

  std::string image_path = "";
  std::string constraint_type = "ONLY_POINTS";
  std::string output_format = "bin";

  OptionManager options;
  options.AddRequiredOption("database_path", &database_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("image_path", &image_path);
  options.AddDefaultOption("constraint_type",
                           &constraint_type,
                           "{ONLY_POINTS, ONLY_CAMERAS, "
                           "POINTS_AND_CAMERAS_BALANCED, POINTS_AND_CAMERAS}");
  options.AddDefaultOption("output_format", &output_format, "{bin, txt}");
  options.AddGlobalMapperFullOptions();

  options.Parse(argc, argv);

  if (!colmap::ExistsFile(database_path)) {
    LOG(ERROR) << "`database_path` is not a file";
    return EXIT_FAILURE;
  }

  if (constraint_type == "ONLY_POINTS") {
    options.mapper->opt_gp.constraint_type =
        GlobalPositionerOptions::ONLY_POINTS;
  } else if (constraint_type == "ONLY_CAMERAS") {
    options.mapper->opt_gp.constraint_type =
        GlobalPositionerOptions::ONLY_CAMERAS;
  } else if (constraint_type == "POINTS_AND_CAMERAS_BALANCED") {
    options.mapper->opt_gp.constraint_type =
        GlobalPositionerOptions::POINTS_AND_CAMERAS_BALANCED;
  } else if (constraint_type == "POINTS_AND_CAMERAS") {
    options.mapper->opt_gp.constraint_type =
        GlobalPositionerOptions::POINTS_AND_CAMERAS;
  } else {
    LOG(ERROR) << "Invalid constriant type";
    return EXIT_FAILURE;
  }

  // Check whether output_format is valid
  if (output_format != "bin" && output_format != "txt") {
    LOG(ERROR) << "Invalid output format";
    return EXIT_FAILURE;
  }

  // Load the database
  ViewGraph view_graph;
  std::unordered_map<camera_t, Camera> cameras;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<track_t, Track> tracks;

  const colmap::Database database(database_path);
  ConvertDatabaseToGlomap(database, view_graph, cameras, images);

  if (view_graph.image_pairs.empty()) {
    LOG(ERROR) << "Can't continue without image pairs";
    return EXIT_FAILURE;
  }

  GlobalMapperFilter global_mapper_filter(*options.mapper);

  // Main solver
  LOG(INFO) << "Loaded database";
  colmap::Timer run_timer;
  run_timer.Start();
  // With outlier filter
  global_mapper_filter.Solve(database, image_path, view_graph, cameras, images, tracks);
  run_timer.Pause();

  LOG(INFO) << "Reconstruction done in " << run_timer.ElapsedSeconds()
            << " seconds";

  WriteGlomapReconstruction(
      output_path, cameras, images, tracks, output_format, image_path);
  LOG(INFO) << "Export to COLMAP reconstruction done";

  return EXIT_SUCCESS;
}

}  // namespace glomap
