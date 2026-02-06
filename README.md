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
- `pytest~=9.0.2`

## 📚 Usage

Here is a simple example of using ShellPy:

```python
from shellpy.numeric_integration.numeric_integration import simple_integral, double_integral

# Example function
result = simple_integral(lambda x: x**2, (0, 1), 4)
print("Result:", result)
```

You can replace this with real functions from your modules (`materiais`, `expansions`, etc.).

## 🧪 Tests

Tests are located in `shellpy/tests`.
Run tests using `pytest`:

```bash
pytest -v
```

## ⚙️ Project Structure

```
ShellPy/
├───continuationpy
│   └───predator_pray
├───exemples
│   ├───linear_normal_modes
│   ├───linear_static_analysis
│   ├───nonlinear_static_analysis
│   └───paper_results
│       └───fem_models
├───shellpy
│   ├───expansions
│   ├───fsdt5
│   ├───fsdt6
│   ├───fsdt7_eas
│   ├───fsdt_tensor
│   ├───materials
│   ├───numeric_integration
│   ├───sanders_koiter
│   └───shell_loads
└───tests
├── .gitignore
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
