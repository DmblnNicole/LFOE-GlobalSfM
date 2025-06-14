#pragma once
#include <pybind11/embed.h>
#include <string>
#include <iostream>

namespace py = pybind11;

class PythonInference {
public:
    PythonInference() {
        py::initialize_interpreter();
        py::module sys = py::module::import("sys");
        const std::string rel_path = "../python";
        sys.attr("path").attr("insert")(0, rel_path);
    }

    ~PythonInference() {
        py::finalize_interpreter();
    }

    std::vector<std::pair<int, int>> run_inference(const std::string& path_images,
                                                const std::string& path_clusters,
                                                const std::string& path_viewgraph,
                                                const std::string& path_rotations,
                                                const std::string& path_model) {
        py::module inference = py::module::import("inference");
        py::object result = inference.attr("main_inference")(path_images, path_clusters, path_viewgraph, path_rotations, path_model);
        std::vector<std::pair<int, int>> edges_to_delete;
        for (auto item : result) {
            auto tuple = item.cast<std::tuple<int, int>>();
            edges_to_delete.emplace_back(std::get<0>(tuple), std::get<1>(tuple));
        }
        return edges_to_delete;
    }
};