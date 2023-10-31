from SI_Toolkit.load_and_normalize import load_yaml

config = load_yaml("config.yml", "r")


GLOBALLY_DISABLE_COMPILATION = config["debug"]
USE_JIT_COMPILATION = config["use_jit_compilation"]