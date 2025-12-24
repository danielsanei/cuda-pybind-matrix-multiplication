#include <string>
#include <pybind11/pybind11.h>

class myclass {
	public:
		myclass(const std::string &name) : m_name(name) {}
		const std::string &get_name() const { return m_name; }
		void set_name(const std::string &name) { m_name = name; }
	private:
		std::string m_name;
	
};

namespace py = pybind11;

PYBIND11_MODULE(cpp_class, m) {
	py::class_<myclass>(m, "Myclass")
		.def(py::init<const std::string &>())
		.def("getname", &myclass::get_name)
		.def("setname", &myclass::set_name);
}