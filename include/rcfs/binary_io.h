#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <vector>

namespace rcfs {

// Allocation limits for length-prefixed data. The element limit accommodates
// a SIFT1M vector matrix (128 million floats).
constexpr uint64_t kDefaultMaxVectorElements = uint64_t{1} << 30;
constexpr uint64_t kDefaultMaxStringBytes = uint64_t{1} << 26;
constexpr uint64_t kDefaultMaxStringCount = uint64_t{1} << 20;

inline std::string binary_io_error(const std::string& operation,
                                   const std::string& context) {
  return context.empty() ? operation : operation + ": " + context;
}

inline bool host_is_little_endian() noexcept {
  const uint16_t value = 1;
  unsigned char first_byte = 0;
  std::memcpy(&first_byte, &value, sizeof(first_byte));
  return first_byte == 1;
}

inline void write_exact(std::ostream& output, const void* data, uint64_t bytes,
                        const std::string& context = {}) {
  if (bytes == 0) return;
  if (data == nullptr) {
    throw std::invalid_argument(
        binary_io_error("Cannot write from a null buffer", context));
  }

  const auto* cursor = static_cast<const char*>(data);
  const uint64_t chunk_limit = static_cast<uint64_t>(
      std::numeric_limits<std::streamsize>::max());
  uint64_t remaining = bytes;
  while (remaining != 0) {
    const uint64_t chunk = std::min(remaining, chunk_limit);
    output.write(cursor, static_cast<std::streamsize>(chunk));
    if (!output) {
      throw std::runtime_error(binary_io_error("Binary write failed", context));
    }
    cursor += static_cast<size_t>(chunk);
    remaining -= chunk;
  }
}

inline void read_exact(std::istream& input, void* data, uint64_t bytes,
                       const std::string& context = {}) {
  if (bytes == 0) return;
  if (data == nullptr) {
    throw std::invalid_argument(
        binary_io_error("Cannot read into a null buffer", context));
  }

  auto* cursor = static_cast<char*>(data);
  const uint64_t chunk_limit = static_cast<uint64_t>(
      std::numeric_limits<std::streamsize>::max());
  uint64_t remaining = bytes;
  while (remaining != 0) {
    const uint64_t chunk = std::min(remaining, chunk_limit);
    input.read(cursor, static_cast<std::streamsize>(chunk));
    if (input.gcount() != static_cast<std::streamsize>(chunk)) {
      throw std::runtime_error(
          binary_io_error("Unexpected end of binary input", context));
    }
    cursor += static_cast<size_t>(chunk);
    remaining -= chunk;
  }
}

inline uint64_t remaining_bytes(std::istream& input,
                                const std::string& context = {}) {
  const std::istream::pos_type current = input.tellg();
  if (current == std::istream::pos_type(-1)) {
    throw std::runtime_error(
        binary_io_error("Cannot determine the current stream position", context));
  }

  input.seekg(0, std::ios::end);
  const std::istream::pos_type end = input.tellg();
  if (end == std::istream::pos_type(-1)) {
    input.clear();
    input.seekg(current);
    throw std::runtime_error(
        binary_io_error("Cannot determine the input size", context));
  }
  input.seekg(current);
  if (!input) {
    throw std::runtime_error(
        binary_io_error("Cannot restore the input position", context));
  }
  if (end < current) {
    throw std::runtime_error(
        binary_io_error("Input position lies beyond the end of the stream",
                        context));
  }

  const auto difference = end - current;
  if (difference < 0 ||
      static_cast<uintmax_t>(difference) >
          std::numeric_limits<uint64_t>::max()) {
    throw std::runtime_error(
        binary_io_error("Input size cannot be represented", context));
  }
  return static_cast<uint64_t>(difference);
}

inline void require_remaining(std::istream& input, uint64_t bytes,
                              const std::string& context = {}) {
  const uint64_t available = remaining_bytes(input, context);
  if (bytes > available) {
    throw std::runtime_error(binary_io_error(
        "Binary input is truncated (need " + std::to_string(bytes) +
            " bytes, have " + std::to_string(available) + ")",
        context));
  }
}

inline void require_eof(std::istream& input,
                        const std::string& context = {}) {
  const uint64_t trailing = remaining_bytes(input, context);
  if (trailing != 0) {
    throw std::runtime_error(binary_io_error(
        "Binary input contains " + std::to_string(trailing) +
            " trailing bytes",
        context));
  }
}

inline uint64_t file_size_bytes(const std::filesystem::path& path) {
  std::error_code error;
  const uintmax_t bytes = std::filesystem::file_size(path, error);
  if (error) {
    throw std::runtime_error("Cannot determine file size for '" +
                             path.string() + "': " + error.message());
  }
  if (bytes > std::numeric_limits<uint64_t>::max()) {
    throw std::runtime_error("File is too large to represent: " +
                             path.string());
  }
  return static_cast<uint64_t>(bytes);
}

template <typename T>
constexpr bool is_binary_scalar_v =
    std::is_arithmetic<T>::value || std::is_enum<T>::value;

template <typename T>
inline void write_pod(std::ostream& output, const T& value,
                      const std::string& context = {}) {
  static_assert(is_binary_scalar_v<T>,
                "write_pod supports arithmetic and enum scalars only");

  if constexpr (std::is_enum<T>::value) {
    using Underlying = typename std::underlying_type<T>::type;
    write_pod(output, static_cast<Underlying>(value), context);
  } else if constexpr (std::is_same<T, bool>::value) {
    write_pod<uint8_t>(output, value ? uint8_t{1} : uint8_t{0}, context);
  } else {
    static_assert(!std::is_floating_point<T>::value ||
                      std::numeric_limits<T>::is_iec559,
                  "Binary floating-point I/O requires IEC 559 representation");
    std::array<unsigned char, sizeof(T)> bytes{};
    std::memcpy(bytes.data(), &value, sizeof(T));
    if (!host_is_little_endian() && sizeof(T) > 1) {
      std::reverse(bytes.begin(), bytes.end());
    }
    write_exact(output, bytes.data(), bytes.size(), context);
  }
}

template <typename T>
inline T read_pod(std::istream& input, const std::string& context = {}) {
  static_assert(is_binary_scalar_v<T>,
                "read_pod supports arithmetic and enum scalars only");

  if constexpr (std::is_enum<T>::value) {
    using Underlying = typename std::underlying_type<T>::type;
    return static_cast<T>(read_pod<Underlying>(input, context));
  } else if constexpr (std::is_same<T, bool>::value) {
    const uint8_t encoded = read_pod<uint8_t>(input, context);
    if (encoded > 1) {
      throw std::runtime_error(
          binary_io_error("Invalid encoded bool value", context));
    }
    return encoded != 0;
  } else {
    static_assert(!std::is_floating_point<T>::value ||
                      std::numeric_limits<T>::is_iec559,
                  "Binary floating-point I/O requires IEC 559 representation");
    std::array<unsigned char, sizeof(T)> bytes{};
    read_exact(input, bytes.data(), bytes.size(), context);
    if (!host_is_little_endian() && sizeof(T) > 1) {
      std::reverse(bytes.begin(), bytes.end());
    }
    T value{};
    std::memcpy(&value, bytes.data(), sizeof(T));
    return value;
  }
}

inline size_t checked_element_count(uint64_t count, uint64_t maximum,
                                    size_t container_max,
                                    const std::string& context = {}) {
  if (count > maximum) {
    throw std::runtime_error(binary_io_error(
        "Element count " + std::to_string(count) +
            " exceeds configured limit " + std::to_string(maximum),
        context));
  }
  if (count > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
      count > static_cast<uint64_t>(container_max)) {
    throw std::runtime_error(
        binary_io_error("Element count cannot be allocated", context));
  }
  return static_cast<size_t>(count);
}

template <typename T>
inline uint64_t checked_byte_count(uint64_t count,
                                   const std::string& context = {}) {
  if (count > std::numeric_limits<uint64_t>::max() / sizeof(T)) {
    throw std::runtime_error(
        binary_io_error("Element byte count overflows uint64", context));
  }
  return count * sizeof(T);
}

template <typename T>
inline void write_array(std::ostream& output, const T* values, size_t count,
                        const std::string& context = {}) {
  static_assert(is_binary_scalar_v<T> && !std::is_same<T, bool>::value,
                "write_array supports non-bool arithmetic/enum values only");
  if (count == 0) return;
  if (values == nullptr) {
    throw std::invalid_argument(
        binary_io_error("Cannot write a null array", context));
  }
  const uint64_t count64 = static_cast<uint64_t>(count);
  const uint64_t bytes = checked_byte_count<T>(count64, context);
  if (host_is_little_endian() && !std::is_enum<T>::value) {
    write_exact(output, values, bytes, context);
    return;
  }
  for (size_t i = 0; i < count; ++i) write_pod(output, values[i], context);
}

template <typename T>
inline void write_array(std::ostream& output, const std::vector<T>& values,
                        const std::string& context = {}) {
  write_array(output, values.data(), values.size(), context);
}

template <typename T>
inline std::vector<T> read_array(
    std::istream& input, uint64_t count,
    uint64_t maximum = kDefaultMaxVectorElements,
    const std::string& context = {}) {
  static_assert(is_binary_scalar_v<T> && !std::is_same<T, bool>::value,
                "read_array supports non-bool arithmetic/enum values only");
  const size_t size = checked_element_count(
      count, maximum, std::vector<T>().max_size(), context);
  const uint64_t bytes = checked_byte_count<T>(count, context);
  require_remaining(input, bytes, context);

  std::vector<T> values(size);
  if (size == 0) return values;
  if (host_is_little_endian() && !std::is_enum<T>::value) {
    read_exact(input, values.data(), bytes, context);
    return values;
  }
  for (size_t i = 0; i < size; ++i) values[i] = read_pod<T>(input, context);
  return values;
}

template <typename T>
inline void write_vector(std::ostream& output, const std::vector<T>& values,
                         const std::string& context = {}) {
  write_pod<uint64_t>(output, static_cast<uint64_t>(values.size()), context);
  write_array(output, values, context);
}

template <typename T>
inline std::vector<T> read_vector(
    std::istream& input, uint64_t maximum = kDefaultMaxVectorElements,
    const std::string& context = {}) {
  const uint64_t count = read_pod<uint64_t>(input, context);
  return read_array<T>(input, count, maximum, context);
}

inline void write_string(std::ostream& output, const std::string& value,
                         const std::string& context = {}) {
  write_pod<uint64_t>(output, static_cast<uint64_t>(value.size()), context);
  write_exact(output, value.data(), static_cast<uint64_t>(value.size()), context);
}

inline std::string read_string(std::istream& input,
                               uint64_t maximum = kDefaultMaxStringBytes,
                               const std::string& context = {}) {
  const uint64_t count = read_pod<uint64_t>(input, context);
  const size_t size = checked_element_count(
      count, maximum, std::string().max_size(), context);
  require_remaining(input, count, context);
  std::string value(size, '\0');
  read_exact(input, value.data(), count, context);
  return value;
}

inline void write_string_vector(std::ostream& output,
                                const std::vector<std::string>& values,
                                const std::string& context = {}) {
  write_pod<uint64_t>(output, static_cast<uint64_t>(values.size()), context);
  for (const std::string& value : values) write_string(output, value, context);
}

inline std::vector<std::string> read_string_vector(
    std::istream& input, uint64_t maximum_count = kDefaultMaxStringCount,
    uint64_t maximum_string_bytes = kDefaultMaxStringBytes,
    const std::string& context = {}) {
  const uint64_t count = read_pod<uint64_t>(input, context);
  const size_t size = checked_element_count(
      count, maximum_count, std::vector<std::string>().max_size(), context);
  std::vector<std::string> values;
  values.reserve(size);
  for (size_t i = 0; i < size; ++i) {
    values.push_back(read_string(input, maximum_string_bytes, context));
  }
  return values;
}

inline std::ifstream open_binary_input(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    throw std::runtime_error("Cannot open binary input: " + path.string());
  }
  return input;
}

inline std::ofstream open_binary_output(const std::filesystem::path& path) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output.is_open()) {
    throw std::runtime_error("Cannot open binary output: " + path.string());
  }
  return output;
}

}  // namespace rcfs
