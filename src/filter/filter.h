#include "glomap/scene/view_graph.h"
#include "glomap/scene/image.h"
#include "glomap/scene/image_pair.h"

#include <colmap/math/graph_cut.h>
#include <yaml-cpp/yaml.h>

namespace glomap {

int ClusterViewGraph(const std::vector<std::pair<int, int>>& edges, 
                    const std::vector<int>& weights, 
                    const std::string& path_clusters, 
                    const int init_num_clusters, 
                    int max_edges);

int CalculateNumEdgesInLinegraph(const std::vector<std::pair<int, int>>& edges);

void SaveViewGraph(ViewGraph& view_graph,
                        std::unordered_map<image_t, Image>& images,
                        const std::string& file_path);

void SaveRotations(std::unordered_map<image_t, Image>& images, const std::string& file_path);

int RemoveOutlierEdges(ViewGraph& view_graph, const std::vector<ImagePair>& edges_to_delete);

}