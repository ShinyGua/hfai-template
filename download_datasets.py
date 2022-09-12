import hfai_env


hfai_env.set_env('diff_hfai')

# import hf_env
# hf_env.set_env('diff_hfai')
from pathlib import Path
import hfai

if __name__ == '__main__':
    datasets_name = "CIFAR10"
    out_path = Path(f"datasets")
    out_path.mkdir(parents=True, exist_ok=True)

    hfai.datasets.set_data_dir(out_path)
    hfai.datasets.download(datasets_name, miniset=False)
    print("Done!")
