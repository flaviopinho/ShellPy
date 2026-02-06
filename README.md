# ShellPy

**ShellPy** is a Python library for shell analysis using the Ritz method and associated numerical techniques.

## 🚀 Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/flaviopinho/ShellPy.git
cd ShellPy
pip install -e .
```

This will also install all dependencies listed in `requirements.txt`.

## 👢 Dependencies

ShellPy requires the following Python packages:

- `numpy~=1.26.4`
- `scipy~=1.12.0`
- `sympy~=1.12`
- `mpmath~=1.3.0`
- `multipledispatch~=1.0.0`
- `pandas~=2.2.3`
- `sparse~=0.15.1`
- `dill~=0.3.8`
- `matplotlib~=3.8.3`
- `pyvista~=0.46.5`

## 📚 Usage

Here is a simple example of using ShellPy:

```python
import shellpy
from shellpy.numeric_integration import gauss_quadrature

# Example function
result = gauss_quadrature(lambda x: x**2, a=0, b=1, n=4)
print("Result:", result)
```

You can replace this with real functions from your modules (`materiais`, `expansions`, etc.).

## 🧪 Tests

Tests are located in `shellpy/tests`.
Run tests using `pytest`:

```bash
pytest shellpy/tests
```

## ⚙️ Project Structure

```
ShellPy/
├── shellpy/
│   ├── __init__.py
│   ├── cache_decorator.py
│   ├── displacement_covariant_derivative.py
│   ├── displacement_expansion.py
│   ├── mid_surface_domain.py
│   ├── midsurface_geometry.py
│   ├── multiindex.py
│   ├── shell.py
│   ├── tensor_derivatives.py
│   ├── thickness.py
│   ├── expansions/
│   ├── fosd_theory/
│   ├── fosd_theory2/
│   ├── koiter_shell_theory/
│   ├── materials/
│   ├── numeric_integration/
│   ├── shell_loads/
│   └── tests/
├── fem_models/
├── linear_normal_modes/
├── linear_static_analysis/
├── nonlinear_static_analysis/
├── README.md
├── LICENSE
├── setup.py
├── pyproject.toml
└── requirements.txt
```

## 📝 License

This project is licensed under the MIT License — see the `LICENSE` file for details.

## ✨ Contributing

Feel free to open issues or pull requests. Please, include tests for new features.
