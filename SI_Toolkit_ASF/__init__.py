from yaml import load, FullLoader

config = load(open("config.yml", "r"), Loader=FullLoader)


GLOBALLY_DISABLE_COMPILATION = config["debug"]
USE_JIT_COMPILATION = config["use_jit_compilation"]