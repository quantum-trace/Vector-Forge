#include "vectorforge/collection.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    try {
        vf::CollectionConfig cfg;
        cfg.dim = 3;
        cfg.metric = vf::Metric::L2;

        vf::Collection collection(cfg);

        const std::vector<float> a{1.0f, 0.0f, 0.0f};
        const std::vector<float> b{0.0f, 1.0f, 0.0f};
        const std::vector<float> c{0.0f, 0.0f, 1.0f};

        collection.upsert(101, a.data(), a.size(), {{"category", "a"}});
        collection.upsert(202, b.data(), b.size(), {{"category", "b"}});
        collection.upsert(303, c.data(), c.size(), {{"category", "c"}});

        require(collection.size() == 3, "size should be 3 after inserts");
        require(collection.contains(101), "inserted ID 101 should exist");

        vf::SearchParams exact;
        exact.k = 1;
        exact.exact = true;

        auto exact_hits = collection.search(
            a.data(), a.size(), exact);

        require(!exact_hits.empty(), "exact search returned no results");
        require(exact_hits.front().id == 101,
                "exact search should return the nearest vector");
        require(std::abs(exact_hits.front().distance) < 1e-6f,
                "exact match should have zero squared L2 distance");

        vf::SearchParams approximate;
        approximate.k = 1;
        approximate.ef = 16;

        auto hnsw_hits = collection.search(
            a.data(), a.size(), approximate);

        require(!hnsw_hits.empty(), "HNSW search returned no results");
        require(hnsw_hits.front().id == 101,
                "HNSW should find the exact vector in this small dataset");

        const std::vector<float> updated{5.0f, 5.0f, 5.0f};
        collection.upsert(101, updated.data(), updated.size());

        require(collection.size() == 3, "upsert should not increase live size");
        require(collection.contains(101), "updated ID should still exist");

        exact_hits = collection.search(a.data(), a.size(), exact);
        require(exact_hits.front().id != 101,
                "updated vector should no longer be nearest to the old query");

        require(collection.erase(202), "deleting ID 202 should succeed");
        require(!collection.erase(202), "deleting ID 202 twice should return false");
        require(!collection.contains(202), "deleted ID 202 should not exist");
        require(collection.size() == 2, "size should be 2 after deletion");

        exact_hits = collection.search(b.data(), b.size(), exact);
        for (const auto& hit : exact_hits) {
            require(hit.id != 202, "exact search returned a deleted ID");
        }

        std::string reason;
        require(collection.check_invariants(&reason),
                "collection invariants failed: " + reason);

        collection.compact();

        require(collection.size() == 2, "compaction changed live size");
        require(collection.check_invariants(&reason),
                "invariants failed after compaction: " + reason);

        std::cout << "PASS: collection insert/search/update/delete/compact/invariants\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
