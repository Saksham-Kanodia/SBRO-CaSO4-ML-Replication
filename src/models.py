from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern,
    ConstantKernel,
    WhiteKernel
)


def build_m52_gpr():
    """
    Matern 5/2 Gaussian Process Regression.

    This is the Python implementation of the M5/2GPR
    model family used in the paper.

    The paper reports M5/2GPR as:
        5-fold CV:
            R2   = 0.998
            RMSE = 0.003
            MAE  = 0.002

    Note:
    The paper does not publish all exact MATLAB
    hyperparameters, so these are explicit Python
    equivalents rather than claimed MATLAB internals.
    """

    kernel = (
        ConstantKernel(
            constant_value=1.0,
            constant_value_bounds=(1e-3, 1e3)
        )
        *
        Matern(
            length_scale=1.0,
            length_scale_bounds=(1e-3, 1e3),
            nu=2.5
        )
        +
        WhiteKernel(
            noise_level=1e-5,
            noise_level_bounds=(1e-8, 1e-1)
        )
    )

    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,

        # Let the GPR optimize kernel parameters.
        n_restarts_optimizer=2,

        random_state=42
    )

    return model