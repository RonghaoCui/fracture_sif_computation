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
    matrix_overlap_l2_components_p1,
    fracture_overlap_l2_components_p1,
    interface_overlap_l2_components_p1,
    matrix_overlap_l2_components_p1_quad,
    fracture_overlap_l2_components_p1_quad,
    interface_overlap_l2_components_p1_quad, 
)

import pandas as pd

import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import os
save_dir = "figures/convergence/selfconv"
os.makedirs(save_dir, exist_ok=True)

save_csv = "data/convergence/selfconv/mpsa/so"
os.makedirs(save_csv, exist_ok=True)

# a = 0.3
# poi = 0.25
# shear_modulus = 1.0
# lam = 2 * shear_modulus * poi / (1 - 2 * poi)

# params = {
#     "prepare_simulation": True,
#     "material_constants": {"solid": pp.SolidConstants(shear_modulus=shear_modulus, lame_lambda=lam)},
#     "a": a,
#     "domain_size": 1.0,
#     "p0": 1e-4,
#     "poi": poi,
#     "meshing_arguments": {"cell_size": 0.1},
#     "grid_type": "simplex",
#     "theta_rad": math.radians(90 - 0),
#     "num_bem_segments": 1000,
#     "error_exclusion_zone_fracture_tips": 0.1,
# }

# h = 0.1
# params["meshing_arguments"] = {"cell_size": h}
# model = ManuSneddonModel2d(params=params)


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


# Extract matrix grid and cell-centered MPSA displacement
# def matrix_grid(model: ManuSneddonModel2d) -> pp.Grid:
#     return model.mdg.subdomains(dim=2)[0]

def extract_matrix_displacement_cc(
        model: object
    ) -> np.ndarray:
    """Compute cell-centered displacement on the 2D matrix grid.

    Parameters:
        model: A solved PorePy model.

    Returns: 
        cell-centered displacement (shape=(2, num_cells)).
    """
    sd = model.mdg.subdomains(dim=2)[0]
    u = model.displacement([sd]).value(model.equation_system)
    return u.reshape((model.nd, sd.num_cells), order="F")


# Displacement jump on 1D fracture. 
# def fracture_grid(model: ManuSneddonModel2d) -> pp.Grid:
#     """Return the 1D fracture grid."""
#     return model.mdg.subdomains(dim=1)[0]

def extract_fracture_displacement_jump_cc(
        model: object
    ) -> np.ndarray:
    """Compute cell-centered displacement jump on the 1D fracture cells.

    Parameters:
        model: A solved PorePy model.

    Returns:
        cell-centered displacement (shape=(2, num_cells)).
    """
    sd = model.mdg.subdomains(dim=1)[0]
    u_jump = model.displacement_jump([sd]).value(
        model.equation_system
    )
    return u_jump.reshape((model.nd, sd.num_cells), order="F")

# Displacement on 1D interfaces
# def mortar_grid(model: ManuSneddonModel2d) -> pp.MortarGrid:
#     return model.mdg.interfaces()[0]

def extract_interface_displacement_cc(
    model: object,
) -> np.ndarray:
    """Compute cell-centered displacement on the matrix-fracture 
    mortar/interface cells.

    Parameters:
        model: A solved PorePy model.

    Returns:
        cell-centered displacement (shape=(2, num_cells)).

    Notes: 
        For one fracture with two sides, num_mortar_cells = 2 * num_fracture_cells.
    """
    md = model.mdg.interfaces()[0]
    u_intf = model.interface_displacement([md]).value(
        model.equation_system
    )
    return u_intf.reshape((model.nd, md.num_cells), order="F")


def relative_l2_error_mixed_dimensional_overlap_between_models(
    coarse_model: object,
    fine_model: object,
    averaging_weight: str = "volume",
    tol: float = 1e-8,
) -> tuple[float, float, float, float, float, float]:

    # 2D matrix grid.
    sd_H = coarse_model.mdg.subdomains(dim=2)[0]
    sd_h = fine_model.mdg.subdomains(dim=2)[0]

    # Cell-centered displacement on the 2D matrix grid.
    uH_cc = extract_matrix_displacement_cc(coarse_model)
    uh_cc = extract_matrix_displacement_cc(fine_model)

    matrix_num_sq, matrix_den_sq = (
        matrix_overlap_l2_components_p1_quad(
            sd_H=sd_H,
            sd_h=sd_h,
            uH_cc=uH_cc,
            uh_cc=uh_cc,
            weight=averaging_weight,
            tol=tol,
        )
    )

    matrix_num_sq2, matrix_den_sq2 = (
        matrix_overlap_l2_components_p1(
            sd_H=sd_H,
            sd_h=sd_h,
            uH_cc=uH_cc,
            uh_cc=uh_cc,
            weight=averaging_weight,
            tol=tol,
        )
    )

    print("matrix error", abs(matrix_num_sq - matrix_num_sq2) / abs(matrix_num_sq2))
    print("matrix error", abs(matrix_den_sq - matrix_den_sq2) / abs(matrix_den_sq2))

    # 1D fracture displacement jump
    gamma_H = coarse_model.mdg.subdomains(dim=1)[0]
    gamma_h = fine_model.mdg.subdomains(dim=1)[0]

    wH_cc = extract_fracture_displacement_jump_cc(coarse_model)
    wh_cc = extract_fracture_displacement_jump_cc(fine_model)

    fracture_num_sq, fracture_den_sq = (
        fracture_overlap_l2_components_p1_quad(
            sd_H=gamma_H,
            sd_h=gamma_h,
            wH_cc=wH_cc,
            wh_cc=wh_cc,
            weight=averaging_weight,
            tol=tol,
        )
    )

    fracture_num_sq2, fracture_den_sq2 = (
        fracture_overlap_l2_components_p1(
            sd_H=gamma_H,
            sd_h=gamma_h,
            wH_cc=wH_cc,
            wh_cc=wh_cc,
            weight=averaging_weight,
            tol=tol,
        )
    )

    print("fracture error", abs(fracture_num_sq - fracture_num_sq2) / abs(fracture_num_sq2))
    print("fracture error", abs(fracture_den_sq - fracture_den_sq2) / abs(fracture_den_sq2))

    # 1D interface displacement
    interface_H = coarse_model.mdg.interfaces()[0]
    interface_h = fine_model.mdg.interfaces()[0]

    mH_cc = extract_interface_displacement_cc(coarse_model)
    mh_cc = extract_interface_displacement_cc(fine_model)

    interface_num_sq, interface_den_sq = (
        interface_overlap_l2_components_p1_quad(
            sd_H=gamma_H,
            sd_h=gamma_h,
            mH_cc=mH_cc,
            mh_cc=mh_cc,
            weight=averaging_weight,
            tol=tol,
        )
    )

    interface_num_sq2, interface_den_sq2 = (
        interface_overlap_l2_components_p1(
            sd_H=gamma_H,
            sd_h=gamma_h,
            mH_cc=mH_cc,
            mh_cc=mh_cc,
            weight=averaging_weight,
            tol=tol,
        )
    )

    print("interface error", abs(interface_num_sq - interface_num_sq2) / abs(interface_num_sq2))
    print("interface error", abs(interface_den_sq - interface_den_sq2) / abs(interface_den_sq2))

    matrix_error = np.sqrt(matrix_num_sq / matrix_den_sq)
    fracture_error = np.sqrt(fracture_num_sq / fracture_den_sq)
    interface_error = np.sqrt(interface_num_sq / interface_den_sq)

    matrix_error2 = np.sqrt(matrix_num_sq2 / matrix_den_sq2)
    fracture_error2 = np.sqrt(fracture_num_sq2 / fracture_den_sq2)
    interface_error2 = np.sqrt(interface_num_sq2 / interface_den_sq2)

    # print("mH_cc shape:", mH_cc.shape)
    # print("fracture cells:", gamma_H.num_cells)
    # print("mH side 0 first:", mH_cc[:, :3])
    # print("mH side 1 first:", mH_cc[:, gamma_H.num_cells:gamma_H.num_cells+3])

    return (
        float(matrix_error),
        float(fracture_error),
        float(interface_error),
        float(matrix_error2),
        float(fracture_error2),
        float(interface_error2),
    )


# Main convergence analysis
if __name__ == "__main__":
    cell_sizes = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    # cell_sizes = [0.1, 0.05, 0.025]

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
            relative_l2_error_mixed_dimensional_overlap_between_models(
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
    csv_path = os.path.join(save_csv, "selfconv_mpsa_so.csv")
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
        os.path.join(save_dir, "self_convergence_so_mpsa_5.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
