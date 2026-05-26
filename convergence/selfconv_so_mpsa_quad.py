"""
Self-convergence analysis for Sneddon's problem in the mixed-dimensional grids. 

"""

from __future__ import annotations

import porepy as pp
import numpy as np

from tests.functional.setups.manu_sneddon_2d import ManuSneddonModel2d

# import helpers.mdamr_local  # adds external/mdamr-main/src to sys.path
# from mdamr.non_matching.transfer_grid import TransferGrid

from selfconv import (
    relative_error_p1_mpsa, 
)

import pandas as pd

import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import os
save_dir = "figures/convergence/selfconv/mpsa"
os.makedirs(save_dir, exist_ok=True)

save_csv = "data/convergence/selfconv/mpsa/so"
os.makedirs(save_csv, exist_ok=True)


# Basic model
def make_sneddon_params(h: float) -> dict:
    a = 0.3
    poi = 0.25
    shear_modulus = 1.0
    lam = 2 * shear_modulus * poi / (1 - 2 * poi)

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
        "p0": 1e-4,
        "poi": poi,
        "meshing_arguments": {"cell_size": h},
        "grid_type": "simplex",
        "theta_rad": math.radians(90 - 0),
        "num_bem_segments": 1000,
        "error_exclusion_zone_fracture_tips": 0.1,
    }


def run_sneddon_model(h: float) -> object:
    params = make_sneddon_params(h)
    model = ManuSneddonModel2d(params=params)
    pp.run_time_dependent_model(model)
    return model

# Main convergence analysis
if __name__ == "__main__":
    # cell_sizes = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    cell_sizes = [0.1, 0.05, 0.025]

    # Use the finest grid as the reference solution.
    h_ref = cell_sizes[-1]
    print(f"Running reference model h = {h_ref}")
    fine_model = run_sneddon_model(h_ref)

    results = []
    matrix_errors = []
    fracture_errors = []
    interface_errors = []
    matrix_errors2 = []
    fracture_errors2 = []
    interface_errors2 = []
    sqrt_dofs = []

    for h in cell_sizes[:-1]:
        print(f"\nRunning coarse model h = {h}")
        coarse_model = run_sneddon_model(h)

        err_matrix, err_fracture, err_interface, err_matrix2, err_fracture2, err_interface2 = (
            relative_error_p1_mpsa(
                coarse_model,
                fine_model,
                averaging_weight="volume",
                tol=1e-8,
            )
        )

        sd_matrix = coarse_model.mdg.subdomains(dim=2)[0]
        sd_frac = coarse_model.mdg.subdomains(dim=1)[0]
        mg = coarse_model.mdg.interfaces()[0]

        # DOFs for subdomains and interfaces.
        matrix_dof = coarse_model.nd * sd_matrix.num_cells
        fracture_dof = coarse_model.nd * sd_frac.num_cells
        interface_dof = coarse_model.nd * mg.num_cells

        # 2D matrix
        dof_md = matrix_dof

        # 2D matrix + 1D fracture
        # dof_md = (
        #     coarse_model.nd * sd_matrix.num_cells 
        #     + coarse_model.nd * sd_frac.num_cells
        # )

        # 2D matrix + 1D interfaces
        # dof_md = (
        #     coarse_model.nd * sd_matrix.num_cells
        #     + coarse_model.nd * mg.num_cells
        # )

        sqrt_dof = np.sqrt(dof_md)

        matrix_errors.append(err_matrix)
        fracture_errors.append(err_fracture)
        interface_errors.append(err_interface)
        matrix_errors2.append(err_matrix2)
        fracture_errors2.append(err_fracture2)
        interface_errors2.append(err_interface2)
        sqrt_dofs.append(sqrt_dof)

        print(f"Matrix relative L2 error: {err_matrix:.6e}")
        print(f"Fracture relative L2 error: {err_fracture:.6e}")
        print(f"Interface relative L2 error: {err_interface:.6e}")

        # Save results for each model in a list of dictionaries. 
        results.append({
            "h": h,
            "matrix_dof": matrix_dof,
            "fracture_dof": fracture_dof,
            "interface_dof": interface_dof,
            "matrix_error_quad": err_matrix,
            "fracture_error_quad": err_fracture,
            "interface_error_quad": err_interface,
            "matrix_error_exact": err_matrix2,
            "fracture_error_exact": err_fracture2,
            "interface_error_exact": err_interface2,

        })

    matrix_errors = np.asarray(matrix_errors)
    fracture_errors = np.asarray(fracture_errors)
    interface_errors = np.asarray(interface_errors)
    matrix_errors2 = np.asarray(matrix_errors2)
    fracture_errors2 = np.asarray(fracture_errors2)
    interface_errors2 = np.asarray(interface_errors2)
    sqrt_dofs = np.asarray(sqrt_dofs)

    df = pd.DataFrame(results)
    csv_path = os.path.join(save_csv, "selfconv_mpsa_so_test.csv")
    df.to_csv(csv_path, index=False)

    # Reference line
    num_dofs_line = np.array([10, 30, 50, 70, 90, 100, 300, 500, 1000])
    initial_value_line = 1.1e-2
    first_order_line = initial_value_line * (num_dofs_line[0] / num_dofs_line)
    second_order_line = initial_value_line  * (num_dofs_line[0] / num_dofs_line) ** 2.0

    mpl.rcParams.update({
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
    })

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))            # Scientific form

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.loglog(
        sqrt_dofs, 
        matrix_errors, 
        "o-", color='tab:blue',
        label="MPSA matrix (quad)"
    )

    ax.loglog(
        sqrt_dofs,
        fracture_errors,
        "s-", color='tab:orange',
        label="MPSA fracture (quad)",
    )

    ax.loglog(
        sqrt_dofs,
        interface_errors,
        "^-",color='tab:pink',
        label="MPSA interface (quad)",
    )

    ax.loglog(
        sqrt_dofs, 
        matrix_errors2, 
        "o--", color='tab:blue',
        label="MPSA matrix (exact)",
    )

    ax.loglog(
        sqrt_dofs,
        fracture_errors2,
        "s--", color='tab:orange',
        label="MPSA fracture (exact)",
    )

    ax.loglog(
        sqrt_dofs,
        interface_errors2,
        "^--",color='tab:pink',
        label="MPSA interface (exact)",
    )

    # Reference slopes.
    ax.loglog(
        num_dofs_line, first_order_line, 
        "--", color='black', 
        label="1st order"
    )
    ax.loglog(
        num_dofs_line, second_order_line, 
        "--", color='gray', 
        label="2nd order"
    )
    ax.set_xlabel(r"$\sqrt{\mathrm{DOF}}$")
    ax.set_ylabel("Relative error")

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)

    ax.set_xlim(10, 300)
    ax.set_ylim(0.001, 1.0)
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
        os.path.join(save_dir, "self_convergence_so_mpsa_test.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
