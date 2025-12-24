#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(cpp_export, m) {
	m.attr("value") = 10;
	m.attr("room_number") = py::cast("MOS 0204");
}