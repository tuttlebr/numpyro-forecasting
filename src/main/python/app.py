import time
from logging import basicConfig, info
from os import getenv

import jax.numpy as jnp
from multiprocessing import cpu_count
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import random, local_device_count
from numpyro.contrib.control_flow import scan
from numpyro.diagnostics import autocorrelation
from numpyro.infer import MCMC, NUTS, Predictive


def make_data():
    a = 7.5
    b = 7.3
    key = random.PRNGKey(3)

    weekly_trend = (b - a) * random.normal(key=key, shape=[7]) + a
    monthly_trend = (b - a) * random.normal(key=key, shape=[5]) + a
    annual_trend = (b - a) * random.normal(key=key, shape=[12]) + a

    y = []
    for yt in sorted(annual_trend):
        for mt in sorted(monthly_trend):
            y.extend(weekly_trend * yt * mt)
    periods = len(y)

    df = pd.DataFrame(columns=["ds", "y"])
    df.ds = pd.date_range(start="2022-01-01", periods=periods, freq="D")
    df.y = y
    return df


def sgt(y, seasonality, future=0):
    # heuristically, standard derivation of Cauchy prior depends on
    # the max value of data
    cauchy_sd = jnp.max(y) / 150

    # NB: priors' parameters are taken from
    # https://github.com/cbergmeir/Rlgt/blob/master/Rlgt/R/rlgtcontrol.R
    nu = numpyro.sample("nu", dist.Uniform(2, 20))
    powx = numpyro.sample("powx", dist.Uniform(0, 1))
    sigma = numpyro.sample("sigma", dist.HalfCauchy(cauchy_sd))
    offset_sigma = numpyro.sample(
        "offset_sigma",
        dist.TruncatedCauchy(low=1e-10, loc=1e-10, scale=cauchy_sd),
    )

    coef_trend = numpyro.sample("coef_trend", dist.Cauchy(0, cauchy_sd))
    pow_trend_beta = numpyro.sample("pow_trend_beta", dist.Beta(1, 1))
    # pow_trend takes values from -0.5 to 1
    pow_trend = 1.5 * pow_trend_beta - 0.5
    pow_season = numpyro.sample("pow_season", dist.Beta(1, 1))

    level_sm = numpyro.sample("level_sm", dist.Beta(1, 2))
    s_sm = numpyro.sample("s_sm", dist.Uniform(0, 1))
    init_s = numpyro.sample("init_s", dist.Cauchy(0, y[:seasonality] * 0.3))

    def transition_fn(carry, t):
        level, s, moving_sum = carry
        season = s[0] * level ** pow_season
        exp_val = level + coef_trend * level ** pow_trend + season
        exp_val = jnp.clip(exp_val, a_min=0)
        # use expected vale when forecasting
        y_t = jnp.where(t >= N, exp_val, y[t])

        moving_sum = (
            moving_sum + y[t] - jnp.where(t >= seasonality, y[t - seasonality], 0.0)
        )
        level_p = jnp.where(t >= seasonality, moving_sum / seasonality, y_t - season)
        level = level_sm * level_p + (1 - level_sm) * level
        level = jnp.clip(level, a_min=0)

        new_s = (s_sm * (y_t - level) / season + (1 - s_sm)) * s[0]
        # repeat s when forecasting
        new_s = jnp.where(t >= N, s[0], new_s)
        s = jnp.concatenate([s[1:], new_s[None]], axis=0)

        omega = sigma * exp_val ** powx + offset_sigma
        y_ = numpyro.sample("y", dist.StudentT(nu, exp_val, omega))

        return (level, s, moving_sum), y_

    N = y.shape[0]
    level_init = y[0]
    s_init = jnp.concatenate([init_s[1:], init_s[:1]], axis=0)
    moving_sum = level_init
    with numpyro.handlers.condition(data={"y": y[1:]}):
        _, ys = scan(
            transition_fn,
            (level_init, s_init, moving_sum),
            jnp.arange(1, N + future),
        )
    if future > 0:
        numpyro.deterministic("y_forecast", ys[-future:])


basicConfig(
    format="%(asctime)s %(message)s",
    level="INFO",
    datefmt="%Y-%m-%d %H:%M:%S",
)

platform = getenv("PLATFORM")
num_devices = cpu_count()

numpyro.set_host_device_count(num_devices)
numpyro.set_platform(platform=platform)

df = make_data()
data = df["y"].values
info("Length of time series: {:,}".format(data.shape[0]))

start = time.time()
y_train, y_test = jnp.array(data[:-30], dtype=jnp.float32), data[-30:]
info("Lag values sorted according to their autocorrelation values:\n")
info(jnp.argsort(autocorrelation(y_train))[::-1])

kernel = NUTS(sgt)
mcmc = MCMC(kernel, num_warmup=5000, num_samples=5000, num_chains=num_devices)
mcmc.run(random.PRNGKey(0), y_train, seasonality=7)
mcmc.print_summary()
samples = mcmc.get_samples()

predictive = Predictive(sgt, samples, return_sites=["y_forecast"])
forecast_marginal = predictive(random.PRNGKey(1), y_train, seasonality=7, future=30)[
    "y_forecast"
]
end = time.time()

y_pred = jnp.mean(forecast_marginal, axis=0)
sMAPE = jnp.mean(jnp.abs(y_pred - y_test) / (y_pred + y_test)) * 200
msqrt = jnp.sqrt(jnp.mean((y_pred - y_test) ** 2))
info("sMAPE: {:.2f}, rmse: {:.2f}".format(sMAPE, msqrt))
info(f"{platform} runtime is {end - start}")
