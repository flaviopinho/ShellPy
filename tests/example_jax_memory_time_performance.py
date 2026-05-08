import time
import sympy as sym
import jax
import jax.numpy as jnp
import psutil
import os

from shellpy.sanders_koiter import strain_energy_internal_force_and_tangent_matrix_jax

# Configuração de Precisão
jax.config.update("jax_enable_x64", True)

# Imports da ShellPy (ajuste conforme sua estrutura)
from shellpy.expansions.eigen_function_expansion import EigenFunctionExpansion
from shellpy import RectangularMidSurfaceDomain, Shell, ConstantThickness, MidSurfaceGeometry, xi1_, xi2_
from shellpy.materials.isotropic_homogeneous_linear_elastic_material import IsotropicHomogeneousLinearElasticMaterial
from shellpy.koiter_tensor import koiter_strain_energy_large_rotations
from shellpy.tensor_derivatives import tensor_derivative
from shellpy.shell_loads.shell_conservative_load import PressureLoad
from shellpy.numeric_integration.gauss_integral import gauss_weights_simple_integral
from shellpy.sanders_koiter._compute_constant_shell_matrices import compute_constant_shell_matrices


def get_process_memory():
    """Retorna a memória RAM atual usada pelo processo em MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def format_bytes(size_in_bytes):
    """Formata bytes para MB ou GB para leitura humana."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.2f} TB"


def run_performance_memory_benchmark():
    print("=" * 85)
    print(f"{'SHELLPY BENCHMARK: PERFORMANCE & MEMORY':^85}")
    print("=" * 85)

    # --- Setup ---
    R, a, b, h = 0.1, 0.1, 0.1, 0.0001
    E, nu, density = 1, 0.3, 2
    edges = RectangularMidSurfaceDomain(0, a, 0, b)

    # TESTE DE ESCALABILIDADE: Aumente 'modes' para ver a explosão do Tensorial
    modes = 6  # Se subir para 15 (~1000 DOFs), o Tensorial provavelmente vai travar seu PC
    expansion_size = {"u1": (modes, modes), "u2": (modes, modes), "u3": (modes, modes)}
    bc = {"xi1": ("S", "S"), "xi2": ("S", "S")}
    displacement_field = EigenFunctionExpansion(expansion_size, edges, {"u1": bc, "u2": bc, "u3": bc})
    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2)])
    shell = Shell(MidSurfaceGeometry(R_), ConstantThickness(h), edges,
                  IsotropicHomogeneousLinearElasticMaterial(E, nu, density), displacement_field, PressureLoad(1))

    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()
    print(f">> Model: {n_dof} DOFs | Integration: 10x10x5\n")

    results = {}

    # -------------------------------------------------------------------------
    # 1. MATRICIAL (NUMPY)
    # -------------------------------------------------------------------------
    mem_start = get_process_memory()
    t0 = time.perf_counter()
    Wxy1, C0, C1, C2, eps0_lin, eps1_lin, eps0_nl, eps1_nl = compute_constant_shell_matrices(
        shell, 10, 10, 5, gauss_weights_simple_integral
    )
    t_off = time.perf_counter() - t0
    # Medimos o tamanho dos tensores quadráticos que são o vilão aqui
    size_arrays = eps0_nl.nbytes + eps1_nl.nbytes + eps0_lin.nbytes
    mem_peak = get_process_memory() - mem_start
    results['SK Matrix'] = {'off': t_off, 'mem_arrays': size_arrays, 'mem_peak': mem_peak}

    # -------------------------------------------------------------------------
    # 2. TENSORIAL KOITER
    # -------------------------------------------------------------------------
    # CUIDADO: U4 cresce com N^4. Para 1000 DOFs, isso ocuparia Terabytes.
    mem_start = get_process_memory()
    t0 = time.perf_counter()
    try:
        U2, U3, U4 = koiter_strain_energy_large_rotations(shell, 10, 10, 5, gauss_weights_simple_integral)
        div = E * h ** 2
        F_t = (tensor_derivative(U2, 0) * h / div, tensor_derivative(U3, 0) * h ** 2 / div,
               tensor_derivative(U4, 0) * h ** 3 / div)
        J_t = (tensor_derivative(F_t[0], 1), tensor_derivative(F_t[1], 1), tensor_derivative(F_t[2], 1))
        t_off = time.perf_counter() - t0
        size_arrays = U4.nbytes + J_t[2].nbytes  # U4 e o maior tensor da Jacobiana
        mem_peak = get_process_memory() - mem_start
        results['Tensorial'] = {'off': t_off, 'mem_arrays': size_arrays, 'mem_peak': mem_peak}
    except MemoryError:
        results['Tensorial'] = {'off': float('nan'), 'mem_arrays': 0, 'mem_peak': 0, 'status': 'CRASHED'}

    # -------------------------------------------------------------------------
    # 3. JAX MATRIX-FREE
    # -------------------------------------------------------------------------
    mem_start = get_process_memory()
    t0 = time.perf_counter()
    get_F_and_J_jax = strain_energy_internal_force_and_tangent_matrix_jax(
        shell, 10, 10, 5, gauss_weights_simple_integral
    )
    # Warm-up (Compilação XLA consome memória temporária)
    _, _, _ = get_F_and_J_jax(jnp.zeros(n_dof))
    t_off = time.perf_counter() - t0
    # No Matrix-free, não guardamos eps_nl, apenas as bases Phi
    # (Como as bases estão dentro da closure, estimamos o peso do que foi passado ao JAX)
    mem_peak = get_process_memory() - mem_start
    results['JAX Matrix-Free'] = {'off': t_off, 'mem_arrays': 0, 'mem_peak': mem_peak}

    # --- PRINT RESULTS ---
    print("-" * 85)
    print(f"{'MÉTODO':<20} | {'OFFLINE (s)':<12} | {'EST. STORAGE':<15} | {'PEAK RAM (MB)':<15}")
    print("-" * 85)
    for method, data in results.items():
        status = data.get('status', '')
        if status == 'CRASHED':
            print(f"{method:<20} | {'OUT OF MEMORY':^45}")
        else:
            storage = format_bytes(data['mem_arrays'])
            print(f"{method:<20} | {data['off']:<12.4f} | {storage:<15} | {data['mem_peak']:<15.2f}")
    print("-" * 85)
    print("* Est. Storage: Tamanho dos tensores principais guardados na RAM.")
    print("* Peak RAM: Incremento real de memória durante a inicialização.")


if __name__ == "__main__":
    run_performance_memory_benchmark()