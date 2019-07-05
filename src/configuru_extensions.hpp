#pragma once

#include <visit_struct/visit_struct.hpp>

#include <configuru.hpp>
#include <emath/matrix.hpp>

namespace emath {

template<typename T>
configuru::Config serialize(const emath::Matrix<T>& matrix)
{
	auto rows = configuru::Config::array();
	for (size_t y = 0; y < matrix.height(); ++y) {
		auto row = configuru::Config::array();
		for (size_t x = 0; x < matrix.width(); ++x) {
			row.push_back(configuru::serialize(matrix(x, y)));
		}
		rows.push_back(row);
	}
	return rows;
}

template<typename T>
void deserialize(emath::Matrix<T>* matrix, const configuru::Config& config, const configuru::ConversionError& on_error)
{
	if (!config.is_array()) {
		if (on_error) { on_error(config.where() + "Expected array of arrays"); }
		return;
	}
	const int height = config.array_size();
	if (height == 0) { *matrix = {}; return; }
	if (!config[0].is_array()) {
		if (on_error) { on_error(config.where() + "Expected array of arrays"); }
		return;
	}
	const int width = config[0].array_size();

	matrix->resize(width, height);

	for (size_t y = 0; y < height; ++y) {
		const auto& row = config[y];
		if (!row.is_array() && row.array_size() != width) {
			if (on_error) { on_error(config.where() + "Expected array of length " + std::to_string(width)); }
			return;
		}
		for (size_t x = 0; x < width; ++x) {
			configuru::deserialize(matrix->pointer_to(x, y), row[x], on_error);
		}
	}
}

} // namespace emath

template<typename T>
void load(T* to, const char* path)
{
	auto log_errors = [&](const auto& error) {
		LOG_F(ERROR, "While loading '%s': %s", path, error.c_str());
	};
	try {
		const auto cfg = configuru::parse_file(path, configuru::CFG);
		configuru::deserialize(to, cfg, log_errors);
	} catch (std::exception& e) {
		LOG_F(ERROR, "Failed to read '%s': %s", path, e.what());
	}
}

template<typename T>
void save(const T& from, const char* path)
{
	auto log_errors = [&](const auto& error) {
		LOG_F(ERROR, "While saving '%s': %s", path, error.c_str());
	};
	try {
		const auto cfg = configuru::serialize(from);
		configuru::dump_file(path, cfg, configuru::CFG);
	} catch (std::exception& e) {
		LOG_F(ERROR, "Failed to write to '%s': %s", path, e.what());
	}
}
