from pathlib import Path


def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        # 必须用完整的 namespace/name
        "datasource": "Helsinki-NLP/opus_books",
        "lang_src": "en",
        "lang_tgt": "it",
        # 权重保存目录，不再拼接 datasource
        "model_folder": "weights",
        "model_basename": "tmodel_",
        # 第一次训练时没有权重，preload 设为 latest 也没关系，会返回 None
        "preload": "latest",
        # 建议放到 tokenizers/ 目录，记得提前 mkdir
        "tokenizer_file": "tokenizers/tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
    }


def get_weights_file_path(config, epoch: str):
    # 直接用 model_folder，不再加 datasource 前缀
    model_folder = config["model_folder"]
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)


def latest_weights_file_path(config):
    model_folder = config["model_folder"]
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
