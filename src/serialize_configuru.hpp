#include <type_traits>
#include <vector>

#include <configuru.hpp>
#include <loguru.hpp>
#include <visit_struct/visit_struct.hpp>

// ----------------------------------------------------------------------------

template <typename Container>
struct is_container : std::false_type { };

// template <typename... Ts> struct is_container<std::list<Ts...> > : std::true_type { };
template <typename... Ts> struct is_container<std::vector<Ts...> > : std::true_type { };

// ----------------------------------------------------------------------------

template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value, configuru::Config>::type
to_config(const T& some_struct);

template<typename T, size_t N>
configuru::Config to_config(T (&some_array)[N]);

template<typename T>
typename std::enable_if<is_container<T>::value, configuru::Config>::type
to_config(const T& some_struct);

template<typename T>
typename std::enable_if<visit_struct::traits::is_visitable<T>::value, configuru::Config>::type
to_config(const T& some_struct);

// ----------------------------------------------------------------------------

template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value, configuru::Config>::type
to_config(const T& some_value)
{
	return some_value;
}

template<typename T, size_t N>
configuru::Config to_config(T (&some_array)[N])
{
	auto config = configuru::Config::array();
	for (size_t i = 0; i < N; ++i) {
		config.push_back(to_config(some_array[i]));
	}
	return config;
}

template<typename T>
typename std::enable_if<is_container<T>::value, configuru::Config>::type
to_config(const T& some_array)
{
	auto config = configuru::Config::array();
	for (const auto& value : some_array) {
		config.push_back(to_config(value));
	}
	return config;
}

template<typename T>
typename std::enable_if<visit_struct::traits::is_visitable<T>::value, configuru::Config>::type
to_config(const T& some_struct)
{
	auto config = configuru::Config::object();
	visit_struct::apply_visitor([&config](const char* name, const auto& value) {
		config[name] = to_config(value);
	}, some_struct);
	return config;
}

// ----------------------------------------------------------------------------

template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value>::type
from_config(T* value, const configuru::Config& config);

template<typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value>::type
from_config(T (*some_array)[N], const configuru::Config& config);

template<typename T>
typename std::enable_if<is_container<T>::value>::type
from_config(T* value, const configuru::Config& config);

template<typename T>
typename std::enable_if<visit_struct::traits::is_visitable<T>::value>::type
from_config(T* value, const configuru::Config& config);

// ----------------------------------------------------------------------------

template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value>::type
from_config(T* value, const configuru::Config& config)
{
	*value = configuru::as<T>(config);
}

template<typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value>::type
from_config(T (*some_array)[N], const configuru::Config& config)
{
	CHECK_EQ_F(config.array_size(), N);
	for (size_t i = 0; i < N; ++i) {
		from_config(&(*some_array)[i], config[i]);
	}
}

template<typename T>
typename std::enable_if<is_container<T>::value>::type
from_config(T* container, const configuru::Config& config)
{
	if (!config.is_array()) {
		LOG_F(WARNING, "Failed to deserialize container");
		return;
	}
	container->clear();
	for (const auto& value : config.as_array()) {
		container->push_back({});
		from_config(&container->back(), value);
	}
}

template<typename T>
typename std::enable_if<visit_struct::traits::is_visitable<T>::value>::type
from_config(T* some_struct, const configuru::Config& config)
{
	if (!config.is_object()) {
		LOG_F(WARNING, "Failed to deserialize container");
		return;
	}
	visit_struct::apply_visitor([&config](const char* name, auto& value) {
		if (config.has_key(name)) {
			from_config(&value, config[name]);
		}
	}, *some_struct);
}
