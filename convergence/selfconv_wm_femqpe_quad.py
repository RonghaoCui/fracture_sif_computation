"""
Self-convergence analysis for Westergaard's problem in the mixed mode solved 
by quadratic FEM with quarter-point elements, using overlap integration with 
high-degree quadrature rules.

This script computes the matrix-domain relative L2 error
    ||u_H^{T6} - u_h^{T6}|| / ||u_h^{T6}||
where the finest mesh in cell_sizes is used as the reference solution.
"""
from __future__ import annotations

import math
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import porepy as pp
from matplotlib.ticker import LogFormatterMathtext, LogLocator, ScalarFormatter

from fems import FemWestergaardMixedMode2d
from selfconv import (
    relative_error_p2_fem_qpe,
)


save_dir = "figures/convergence/selfconv/femqpe"
os.makedirs(save_dir, exist_ok=True)

save_csv = "data/convergence/selfconv/femqpe/wm"
os.makedirs(save_csv, exist_ok=True)


def make_model_params(h: float) -> dict:
    """Create parameters for the FEM benchmark model."""
    a = 0.30
    poi = 0.25
    shear_modulus = 1.0
    lam = 2.0 * shear_modulus * poi / (1.0 - 2.0 * poi)

    return {
        "prepare_simulation": True,
        "material_constants": {
            "solid": pp.SolidConstants(
                shear_modulus=shear_modulus,
                lame_lambda=lam,
            )
        },
        "a": a,
        "domain_size": 1.0,
        "p0": 0.0,
        "sigma_inf": 1e-4,    # far_field normal stress
        "tau_inf": 1e-4,      # far_field shear stress
        "poi": poi,
        "meshing_arguments": {"cell_size": h},
        "grid_type": "simplex",
        "theta_rad": math.radians(90.0 - 0),
        "num_bem_segments": 1000,
        "error_exclusion_zone_fracture_tips": 0.0,
    }


def run_fem_benchmark_model(
    h: float,
    use_qpe: bool,
) -> object:
    """Build and solve a FEM benchmark model."""
    params = make_model_params(h=h)
    model = FemWestergaardMixedMode2d(params, use_qpe=use_qpe)
    model.model_setup_and_sol()
    return model


if __name__ == "__main__":

    # cell_sizes = [0.1, 0.05, 0.025, 0.0124, 0.00625, 0.0050]
    cell_sizes = [0.1, 0.05, 0.025, 0.0124, 0.00625]

    use_qpe = True

    h_ref = cell_sizes[-1]
    print(f"Running reference FEM model h = {h_ref}")
    fine_model = run_fem_benchmark_model(
        h=h_ref,
        use_qpe=use_qpe,
    )

    results = []
    matrix_errors = []
    fracture_errors = []
    sqrt_dofs = []

    for h in cell_sizes[:-1]:
        print(f"\nRunning coarse FEM model h = {h}")
        coarse_model = run_fem_benchmark_model(
            h=h,
            use_qpe=use_qpe,
        )

        err_matrix, err_fracture = relative_error_p2_fem_qpe(
            coarse_model=coarse_model,
            fine_model=fine_model,
            num_quad=7,
            use_qpe=use_qpe,
            tol=1e-8,
        )

        # FEM displacement DOFs: ux and uy for every T6 node.
        matrix_dof = int(2 * coarse_model.nodes_p6.shape[0])
        sqrt_dof = float(np.sqrt(matrix_dof))

        matrix_errors.append(err_matrix)
        fracture_errors.append(err_fracture)
        sqrt_dofs.append(sqrt_dof)

        print(f"Matrix relative L2 error: {err_matrix:.6e}")
        print(f"Fracture relative L2 error: {err_fracture:.6e}")
        print(f"Matrix DOF: {matrix_dof}")

        results.append(
            {
                "h": h,
                "matrix_dof": matrix_dof,
                "matrix_error_quad": err_matrix,
                "fracture_error_quad": err_fracture,
            }
        )

    matrix_errors = np.asarray(matrix_errors)
    fracture_errors = np.asarray(fracture_errors)
    sqrt_dofs = np.asarray(sqrt_dofs)

    df = pd.DataFrame(results)
    csv_path = os.path.join(save_csv, "selfconv_femqpe_wm.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

    # Reference lines. Adjust initial_value_line if needed for your figure.
    num_dofs_line = np.array([10, 30, 50, 70, 100, 300, 500, 1000, 3000])
    initial_value_line = 1.0e-2
    first_order_line = initial_value_line * (num_dofs_line[0] / num_dofs_line)
    second_order_line = initial_value_line * (num_dofs_line[0] / num_dofs_line) ** 2.0

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1.0,
            "lines.markersize": 2.0,
            "mathtext.fontset": "dejavusans",
        }
    )

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    ax.loglog(
        sqrt_dofs,
        matrix_errors,
        "o-",
        color="tab:blue",
        label="QFEM matrix (quad)",
    )

    ax.loglog(
        sqrt_dofs,
        fracture_errors,
        "s-",
        color="tab:orange",
        label="QFEM fracture (quad)",
    )

    # Reference slopes.
    ax.loglog(
        num_dofs_line,
        first_order_line,
        "--",
        color="black",
        label="1st order",
    )
    ax.loglog(
        num_dofs_line,
        second_order_line,
        "--",
        color="gray",
        label="2nd order",
    )

    ax.set_xlabel(r"$\sqrt{\mathrm{DOF}}$")
    ax.set_ylabel("Relative error")

    ax.set_xlim(10, 1000)
    ax.set_ylim(0.00001, 1.0)

    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))

    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "self_convergence_wm_femqpe_quad.png"), 
        dpi=300, 
        bbox_inches="tight"
    )
    plt.close(fig)

