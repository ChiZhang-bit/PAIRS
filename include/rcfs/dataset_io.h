#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "rcfs/predicate.h"
#include "rcfs/types.h"

namespace rcfs {

enum class VectorFileFormat : uint8_t {
  kDenseFloatBinary = 0,
  kFvecs = 1,
};

struct IdMatrix {
  uint32_t rows = 0;
  uint32_t cols = 0;
  std::vector<NodeId> ids;
};

struct ResultMatrix {
  uint32_t rows = 0;
  uint32_t k = 0;
  std::vector<NodeId> ids;
  std::vector<float> distances;
};

struct AttributeTable {
  uint64_t rows = 0;
  std::vector<NumericColumn> numeric_columns;
  std::vector<CategoricalColumn> categorical_columns;
};

struct DatasetFiles {
  std::string vectors;
  std::string base_attributes;
  std::string color_mapping;  // v1 alias
  VectorFileFormat vector_format = VectorFileFormat::kFvecs;
  std::string tag_mapping;  // Optional v2/v3 dictionary; v4 stores it in base_attributes.
};

struct QueryFiles {
  std::string vectors;
  std::string predicates;
  VectorFileFormat vector_format = VectorFileFormat::kFvecs;
};

struct QueryWorkload {
  DenseFloatMatrix vectors;
  std::vector<PredicateRecord> predicates;
};

VectorFileFormat parse_vector_file_format(const std::string& name);
const char* vector_file_format_name(VectorFileFormat format);

// Dense binary: uint32 rows, uint32 cols, float32 payload.
DenseFloatMatrix read_dense_float_bin(const std::string& path);

// Standard fvecs: repeated int32 dimension followed by float32 coordinates.
DenseFloatMatrix read_fvecs(const std::string& path);
DenseFloatMatrix read_vectors(const std::string& path,
                              VectorFileFormat format);

// Standard ivecs: repeated int32 K followed by int32 IDs.  -1 is converted to
// kInvalidNodeId; other negative IDs are rejected.
IdMatrix read_ivecs(const std::string& path);

// RCFS result binary: uint32 nq, uint32 K, all IDs, then all distances.
ResultMatrix read_rcfs_result(const std::string& path);
void write_rcfs_result(const std::string& path, const ResultMatrix& result);

// Read all fields from URFATTR v2/v3/v4, or legacy v1 range1/range2 and bool/color.
// v4 appends uint32 JSON byte length and dictionary bytes after the column payload.
// v4 rejects external mappings. An empty mapping path uses sibling tag_mapping.json
// for v2/v3 and default colors for v1. This version is independent of index V4.
AttributeTable read_sift_gist_base_attributes(
    const std::string& path, const std::string& color_mapping_path = {});

Dataset read_sift_gist_dataset(const DatasetFiles& files);

// Query IDs are checked while reading predicates and must align with vector
// rows.  max_ast_depth only protects parsing; DNF size is bounded separately
// by compile_bounded_dnf.
QueryWorkload read_query_workload(const QueryFiles& files,
                                  const Schema& schema,
                                  uint32_t max_ast_depth = 64);

}  // namespace rcfs
