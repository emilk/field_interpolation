#pragma once

#include <emath/fwd.hpp>

#define IM_VEC2_CLASS_EXTRA                                              \
	ImVec2(emath::Vec2f v) : x(v.x), y(v.y) {}                           \
	operator emath::Vec2f() const { return emath::Vec2f(x, y); }

#define IM_VEC4_CLASS_EXTRA                                              \
	ImVec4(emath::Vec4f v) : x(v.x), y(v.y), z(v.z), w(v.w) {}           \
	operator emath::Vec4f() const { return emath::Vec4f(x, y, z, w); }

#include <emilib/imgui_helpers.hpp>
