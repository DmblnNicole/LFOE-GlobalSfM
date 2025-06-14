#include "filter.h"
#include "glomap/math/rigid3d.h"

#include <fstream>
#include <iostream>
#include <iomanip> 

namespace glomap {

// Pre-processing
int CalculateNumEdgesInLinegraph(const std::vector<std::pair<int, int>>& edges) {
    // Calculate degrees of nodes
    std::unordered_map<int, int> degrees;
    for (const auto& edge : edges) {
        degrees[edge.first]++;
        degrees[edge.second]++;
    }
    // Calculate number of edges in the line graph
    int num_edges = 0;
    for (const auto& kv : degrees) {
        num_edges += (kv.second * kv.second);
    }
    return (num_edges / 2) - edges.size();
}

int ClusterViewGraph(const std::vector<std::pair<int, int>>& edges, 
                    const std::vector<int>& weights, 
                    const std::string& path_clusters, 
                    const int init_num_clusters, 
                    int max_edges) {
    int num_clusters = init_num_clusters;
    bool exceed_max_edges;
    std::unordered_map<int, int> cut_labels;
    do {
        exceed_max_edges = false;
        cut_labels = colmap::ComputeNormalizedMinGraphCut(edges, weights, num_clusters);
        if (cut_labels.empty()) {
            LOG(ERROR) << "ComputeNormalizedMinGraphCut failed" << std::endl;
            return 1;
        }
        // Mapping from node to cluster label
        std::unordered_map<int, int> node_to_cluster;
        for (const auto& label : cut_labels) {
            node_to_cluster[label.first] = label.second; // For example: node 13 is in cluster 0
        }
        std::unordered_map<int, std::vector<std::pair<int, int>>> cluster_edges;
        for (const auto& edge : edges) {
            // Check cluster IDs for both nodes of the edge
            int cluster_id1 = node_to_cluster.find(edge.first) != node_to_cluster.end() ? node_to_cluster[edge.first] : -1;
            int cluster_id2 = node_to_cluster.find(edge.second) != node_to_cluster.end() ? node_to_cluster[edge.second] : -1;
            // If the first node belongs to a cluster, add the edge to that cluster
            if(cluster_id1 != -1) {
                cluster_edges[cluster_id1].push_back(edge);
            }
            // If the second node belongs to a different cluster, add the edge to that cluster as well
            if(cluster_id2 != -1 && cluster_id1 != cluster_id2) {
                cluster_edges[cluster_id2].push_back(edge);
            }
        }
        for (const auto& kv : cluster_edges) {
            int num_edges_in_lg = CalculateNumEdgesInLinegraph(kv.second);
            if (num_edges_in_lg > max_edges) {
                exceed_max_edges = true;
                num_clusters++; 
                break;
            }
        }
    } while (exceed_max_edges);

    std::ofstream output_file(path_clusters);
    if (output_file.is_open()) {
        for (const auto& pair : cut_labels) {
            output_file << pair.first << " " << pair.second << std::endl;
        }
        output_file.close();
        LOG(INFO) << "Viewgraph clustered into " << num_clusters << " subgraphs" << std::endl;
    } else {
        LOG(ERROR) << "Unable to open the output file" << std::endl;
        return 1;
    }
    return 0;
}

void SaveViewGraph(ViewGraph& view_graph,
                   std::unordered_map<image_t, Image>& images,
                   const std::string& file_path) {
    std::ofstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        LOG(ERROR) << "Failed to open file for saving view graph: " << file_path;
        return;
    }

    for (const auto& [key, edge] : view_graph.image_pairs) {
        if (!edge.is_valid) continue;

        int src_id = static_cast<int>(edge.image_id1);
        int dst_id = static_cast<int>(edge.image_id2);

        const std::string& src_name = images[edge.image_id1].file_name;
        const std::string& dst_name = images[edge.image_id2].file_name;

        uint32_t src_len = static_cast<uint32_t>(src_name.size());
        uint32_t dst_len = static_cast<uint32_t>(dst_name.size());

        file.write(reinterpret_cast<const char*>(&src_id), sizeof(int));
        file.write(reinterpret_cast<const char*>(&dst_id), sizeof(int));
        file.write(reinterpret_cast<const char*>(&src_len), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&dst_len), sizeof(uint32_t));
        file.write(src_name.c_str(), src_len);
        file.write(dst_name.c_str(), dst_len);

        Eigen::Matrix3d rot_mat = edge.cam2_from_cam1.rotation.toRotationMatrix();
        Eigen::Vector3d trans = edge.cam2_from_cam1.translation;

        file.write(reinterpret_cast<const char*>(rot_mat.data()), sizeof(double) * 9);
        file.write(reinterpret_cast<const char*>(trans.data()), sizeof(double) * 3);
    }

    file.close();
    LOG(ERROR) << "Viewgraph saved to: " << file_path;
}

void SaveRotations(std::unordered_map<image_t, Image>& images,
                   const std::string& file_path) {
    std::ofstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        LOG(ERROR) << "Failed to open file for saving rotations: " << file_path;
        return;
    }

    for (const auto& [image_id, image] : images) {
        int id = static_cast<int>(image_id);
        Eigen::Matrix3d rot_mat = image.cam_from_world.rotation.toRotationMatrix();

        file.write(reinterpret_cast<const char*>(&id), sizeof(int));
        file.write(reinterpret_cast<const char*>(rot_mat.data()), sizeof(double) * 9);
    }

    file.close();
    LOG(ERROR) << "Rotations saved to: " << file_path;
}

int RemoveOutlierEdges(ViewGraph& view_graph, const std::vector<ImagePair>& outlier_edges) {
    int num_deleted = 0;
    for (const auto& image_pair : outlier_edges) {
        const image_pair_t pair_id = ImagePair::ImagePairToPairId(image_pair.image_id1, image_pair.image_id2);
        ImagePair& image_pair_view_graph = view_graph.image_pairs.at(pair_id);
        image_pair_view_graph.is_valid = false;
        num_deleted++;
    }
    return num_deleted;
}

} // namespace glomap