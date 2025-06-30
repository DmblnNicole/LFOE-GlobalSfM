#include "global_mapper_filter.h"
#include "run_mapper.h"
#include <colmap/util/logging.h>
#include <iostream>
#include <functional>
#include <vector>

namespace {

using command_func_t = std::function<int(int, char**)>;

int ShowHelp(const std::vector<std::pair<std::string, command_func_t>>& commands) {
  std::cout << "GLOMAP with translation outlier filter\n\n";
  std::cout << "Usage:\n";
  std::cout << "  glomap_filter mapper --database_path ... --output_path ...\n";
  std::cout << "\nAvailable commands:\n";
  for (const auto& cmd : commands) {
    std::cout << "  " << cmd.first << '\n';
  }
  std::cout << std::endl;
  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
  colmap::InitializeGlog(argv);
  FLAGS_alsologtostderr = true;

  std::vector<std::pair<std::string, command_func_t>> commands;
  commands.emplace_back("mapper", &glomap::RunMapper);

  if (argc == 1) return ShowHelp(commands);

  std::string command = argv[1];
  if (command == "help" || command == "-h" || command == "--help") {
    return ShowHelp(commands);
  }

  for (const auto& cmd : commands) {
    if (cmd.first == command) {
      int command_argc = argc - 1;
      char** command_argv = &argv[1];
      command_argv[0] = argv[0];
      return cmd.second(command_argc, command_argv);
    }
  }

  std::cerr << "Unknown command: " << command << std::endl;
  return EXIT_FAILURE;
}
