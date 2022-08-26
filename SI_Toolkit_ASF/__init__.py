from yaml import load, FullLoader

config = load(open("config.yml", "r"), Loader=FullLoader)


GLOBALLY_DISABLE_COMPILATION = config["1_data_generation"]["debug"]
USE_JIT_COMPILATION = config["1_data_generation"]["use_jit_compilation"]