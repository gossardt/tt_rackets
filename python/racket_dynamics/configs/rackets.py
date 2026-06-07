import numpy as np


DEFAULT_RACKETS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

RACKET_PARAMS = {
    "01": {
        "cor_coeffs": np.array([0, 0.021704342, 0.91721153]),
        "mean_cor": np.array([0, 0, 0.76815513]),
        "friction_model": 'elastic',
        "friction_coeffs": np.array([-1.9773528e-05, 0, 0.0019338845]),
        "mean_fr": np.array([0, 0, 0.0018398728]),
    },
    "02": {
        "cor_coeffs": np.array([0, 0.017581417, 0.85852668]),
        "mean_cor": np.array([0, 0, 0.73826114]),
        "friction_model": 'elastic',
        "friction_coeffs": np.array([-1.5171789e-05, 0, 0.0018832439]),
        "mean_fr": np.array([0, 0, 0.0018083808]),
    },
    "03": {
        "cor_coeffs": np.array([0, 0.023409582, 0.88561455]),
        "mean_cor": np.array([0, 0, 0.72409358]),
        "friction_model": 'elastic',
        "friction_coeffs": np.array([-2.2978036e-05, 0, 0.0019700633]),
        "mean_fr": np.array([0, 0, 0.001863152]),
    },
    "04": {
        "cor_coeffs": np.array([0, 0.017944046, 0.81310656]),
        "mean_cor": np.array([0, 0, 0.69239787]),
        "friction_model": 'elastic',
        "friction_coeffs": np.array([-1.5371481e-05, 0, 0.0018840398]),
        "mean_fr": np.array([0, 0, 0.0018121907]),
    },
    "05": {
        "cor_coeffs": np.array([0, 0.0073393645, 0.74882337]),
        "mean_cor": np.array([0, 0, 0.69828695]),
        "friction_model": 'elastic',
        "friction_coeffs": np.array([5.6048014e-05, 0, 0.0005603093]),
        "mean_fr": np.array([0, 0, 0.0008034351]),
    },
    "06": {
        "cor_coeffs": np.array([0, 0.0054736023, 0.72887067]),
        "mean_cor": np.array([0, 0, 0.6905475]),
        "friction_model": 'elastic',
        "friction_coeffs": np.array([3.6831622e-05, 0, 0.00049453194]),
        "mean_fr": np.array([0, 0, 0.00066772605]),
    },
    "07": {
        "cor_coeffs": np.array([0, 0.013813934, 0.77220973]),
        "mean_cor": np.array([0, 0, 0.67847925]),
        "friction_model": 'elastic',
        "friction_coeffs": np.array([-2.0600813e-05, 0, 0.0014951701]),
        "mean_fr": np.array([0, 0, 0.0013977738]),
    },
    "08": {
        "cor_coeffs": np.array([0, 0.016310588, 0.86619602]),
        "mean_cor": np.array([0, 0, 0.75617885]),
        "friction_model": 'elastic',
        "friction_coeffs": np.array([-2.1950601e-05, 0, 0.0019677179]),
        "mean_fr": np.array([0, 0, 0.0018698532]),
    },
    "09": {
        "cor_coeffs": np.array([0, 0.019232049, 0.88442881]),
        "mean_cor": np.array([0, 0, 0.7546967]),
        "friction_model": 'elastic',
        "friction_coeffs": np.array([-2.0892639e-05, 0, 0.0019590403]),
        "mean_fr": np.array([0, 0, 0.0018642255]),
    },
    "10": {
        "cor_coeffs": np.array([0, 0.00045103693, 0.53577745]),
        "mean_cor": np.array([0, 0, 0.53272412]),
        "friction_model": 'coulomb',
        "friction_coeffs": np.array([0.1887255, 2.2547101, 1.1491754]),
        "mean_fr": np.array([0.18859449, 2.2536394, 1.1532996]),
    },
}
