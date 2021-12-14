import os
import subprocess
import hydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf, DictConfig


def create_slurm_script(slurm_name, run_name, working_dir, time, n_gpus, script_name, constraint=None, nodelist=None):
    with open(slurm_name, "w") as f:
        f.write(f"""#!/bin/bash -x
#SBATCH --job-name={run_name}
#SBATCH --output={os.path.join(working_dir, "slurm.out")}
#SBATCH --error={os.path.join(working_dir, "slurm.out")}
#SBATCH --time={time}
#SBATCH --signal=USR1@120
#SBATCH --partition="killable"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=20000
#SBATCH --cpus-per-task=4
""")
        if constraint:
            f.write(f"#SBATCH --constraint=\"{constraint}\"\n")
        if nodelist:
            f.write(f"#SBATCH --nodelist={nodelist}\n")
        f.write(f"#SBATCH --gpus={n_gpus}\n")
        f.write(f"\nsrun sh {script_name}\n")


def create_run_script_and_config(run_working_dir, config, run_name):
    config_name = "run_config"
    cfg_file_name = os.path.join(run_working_dir, f"{config_name}.yaml")
    # config = config[:config.find("slurm")]
    with open(cfg_file_name, "w") as f:
        f.write(config)

    with open(run_name, "w") as f:
        f.write(f"python {os.path.join(get_original_cwd(), 'train.py')}  \
 --config-dir {run_working_dir} \
 --config-name {config_name} \
 hydra.run.dir={run_working_dir}")  #\
 #checkpoint.save_dir={os.path.join(run_working_dir, 'checkpoints')}")


def send_job_and_report(slurm_name):
    print(f"sending {slurm_name}")
    process = subprocess.Popen(["sbatch", slurm_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print("output:")
    print(stdout.decode("utf-8"))
    print("err:")
    print(stderr.decode("utf-8"))


@hydra.main(config_path="configs", config_name="config")
def send(cfg: DictConfig) -> None:
    assert "slurm" in cfg, "add slurm.run_name, slurm.n_gpus, slurm.time"

    working_dir = os.getcwd()
    slurm_name = os.path.join(working_dir, "slurm.sh")
    run_name = os.path.join(working_dir, "run.sh")
    create_slurm_script(slurm_name, cfg.slurm.run_name, working_dir,
                        cfg.slurm.time, cfg.slurm.n_gpus, run_name, cfg.slurm.constraint, cfg.slurm.nodelist)

    yaml = OmegaConf.to_yaml(cfg)

    run_working_dir = os.path.join(working_dir, "run")
    os.mkdir(run_working_dir)
    create_run_script_and_config(run_working_dir, yaml, run_name)
    print(slurm_name)
    # subprocess.Popen(["chmod", "ug+rx", run_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    send_job_and_report(slurm_name)


if __name__ == '__main__':
    send()
