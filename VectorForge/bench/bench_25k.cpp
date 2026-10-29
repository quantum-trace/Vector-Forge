#include "vectorforge/collection.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kDim = 64;
constexpr size_t kCount = 25000;
constexpr size_t kQueries = 200;
constexpr size_t kK = 10;

using Vector = std::vector<float>;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t i = static_cast<size_t>(
        std::ceil(p * static_cast<double>(values.size())) - 1.0);
    return values[std::min(i, values.size() - 1)];
}

}  // namespace

int main() {
    try {
        std::mt19937 rng(20261029);
        std::normal_distribution<float> distribution(0.0f, 1.0f);

        std::vector<Vector> data(kCount, Vector(kDim));
        std::vector<Vector> queries(kQueries, Vector(kDim));

        for (auto& v : data)
            for (auto& x : v) x = distribution(rng);

        for (auto& q : queries)
            for (auto& x : q) x = distribution(rng);

        vf::CollectionConfig cfg;
        cfg.dim = kDim;
        cfg.metric = vf::Metric::L2;
        cfg.hnsw.M = 16;
        cfg.hnsw.ef_construction = 100;
        cfg.hnsw.seed = 42;

        vf::Collection collection(cfg);

        const auto build_start = Clock::now();
        for (size_t i = 0; i < data.size(); ++i) {
            collection.upsert(
                static_cast<uint64_t>(i),
                data[i].data(),
                data[i].size());
        }
        const double build_ms = elapsed_ms(build_start, Clock::now());

        std::vector<std::vector<uint64_t>> ground_truth;
        ground_truth.reserve(kQueries);

        std::vector<double> exact_latencies;
        uint64_t exact_distances = 0;

        vf::SearchParams exact;
        exact.k = kK;
        exact.ef = kK;
        exact.exact = true;

        for (const auto& q : queries) {
            vf::SearchStats stats{};
            const auto start = Clock::now();
            const auto hits = collection.search(q.data(), q.size(), exact, &stats);
            exact_latencies.push_back(elapsed_ms(start, Clock::now()));
            exact_distances += stats.dist_computations;

            std::vector<uint64_t> ids;
            ids.reserve(hits.size());
            for (const auto& hit : hits) ids.push_back(hit.id);
            ground_truth.push_back(std::move(ids));
        }

        const double exact_p50 = percentile(exact_latencies, 0.50);
        const double exact_p95 = percentile(exact_latencies, 0.95);

        std::cout << "VectorForge HNSW vs exact-search benchmark\n"
                  << "seed=20261029 dataset=" << kCount
                  << " queries=" << kQueries
                  << " dim=" << kDim
                  << " k=" << kK
                  << " metric=L2 M=" << cfg.hnsw.M
                  << " ef_construction=" << cfg.hnsw.ef_construction << '\n'
                  << "build_ms=" << std::fixed << std::setprecision(3)
                  << build_ms << '\n'
                  << "exact_p50_ms=" << exact_p50
                  << " exact_p95_ms=" << exact_p95
                  << " exact_avg_distance_computations="
                  << static_cast<double>(exact_distances) / kQueries << '\n'
                  << "\nef,recall_at_10,p50_ms,p95_ms,avg_distance_computations,speedup\n";

        for (const size_t ef : {16, 32, 64, 128}) {
            vf::SearchParams params;
            params.k = kK;
            params.ef = ef;

            std::vector<double> latencies;
            uint64_t distance_computations = 0;
            size_t correct = 0;
            size_t relevant = 0;

            for (size_t qi = 0; qi < queries.size(); ++qi) {
                vf::SearchStats stats{};
                const auto& q = queries[qi];

                const auto start = Clock::now();
                const auto hits = collection.search(
                    q.data(), q.size(), params, &stats);
                latencies.push_back(elapsed_ms(start, Clock::now()));
                distance_computations += stats.dist_computations;

                std::unordered_set<uint64_t> truth(
                    ground_truth[qi].begin(), ground_truth[qi].end());

                for (const auto& hit : hits) {
                    if (truth.count(hit.id)) ++correct;
                }
                relevant += ground_truth[qi].size();
            }

            const double p50 = percentile(latencies, 0.50);
            const double p95 = percentile(latencies, 0.95);
            const double recall = relevant
                ? static_cast<double>(correct) / relevant : 0.0;
            const double avg_dist =
                static_cast<double>(distance_computations) / kQueries;
            const double speedup =
                p50 > 0.0 ? exact_p50 / p50 : 0.0;

            std::cout << ef << ','
                      << recall << ','
                      << p50 << ','
                      << p95 << ','
                      << avg_dist << ','
                      << speedup << '\n';
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << '\n';
        return 1;
    }
}
