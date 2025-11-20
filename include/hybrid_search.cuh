#pragma once
#include <string>
#include <vector>

// API for Index Construction
void build_index_impl(
    const std::string& dense_data_path,
    const std::string& sparse_data_path,
    const std::string& bm25_data_path,
    const std::string& output_graph_path
);

// API for Query/Search
void search_index_impl(
    const std::string& dense_data_path,
    const std::string& dense_query_path,
    const std::string& sparse_data_path,
    const std::string& sparse_query_path,
    const std::string& bm25_data_path,
    const std::string& bm25_query_path,
    const std::string& keyword_id_path,
    const std::string& knowledge_path,
    const std::string& entity2doc_path,
    const std::string& query_entity_path,
    const std::string& graph_path,
    const std::string& ground_truth_path,
    int top_k,
    int cands,
    float sparse_weight,
    float bm25_weight,
    float dense_weight,
    float kg_weight
);