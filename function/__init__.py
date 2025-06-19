import os

if "darwin" == os.uname():
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    from jax import config

    config.update("jax_enable_x64", False)
