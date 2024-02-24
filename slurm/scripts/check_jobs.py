"""Check progress of slurm batch experiments"""
import os
import json
import argparse

SLURM_DN = "slurm_logs"
TRANSFER_TO_PREFIX = "Moving input data to the compute node's scratch space: "
RUNNING_PREFIX = 'Running provided command: '
FAILED_PROMPT = 'Command failed!'
TRANSFER_FROM_PROMPT = "Moving output data back to DFS"
DELETE_PROMPT = "Deleting output files in scratch space"
FINISHED_PROMPT = 'Job finished successfully!'
TIMEOUT_PROMPT = 'CANCELLED'

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError('Config file not found!')
    with open(config_path, 'r') as f:
        return json.load(f)

def get_file_paths(cfg):
    slurm_log_path = os.path.join(cfg['EDI']['HOME'], cfg['EDI']['USER'], SLURM_DN)
    project_path = os.path.join(cfg['EDI']['HOME'], cfg['EDI']['USER'], cfg['EDI']['PROJECT'])
    slurm_path = os.path.join(project_path, cfg['SLURM_DN'], 'train')
    exp_fp = os.path.join(slurm_path, cfg['EXP']['CSV']['DEFAULT_FN'])
    exp_fail_fp = os.path.join(slurm_path, cfg['EXP']['CSV']['FAILED_FN'])
    exp_timeout_fp = os.path.join(slurm_path, cfg['EXP']['CSV']['TIMEOUT_FN'])
    return slurm_log_path, exp_fp, exp_fail_fp, exp_timeout_fp

def check_status(log_path, lines, job_id):
    queuing_ids, transferTo_ids, running_ids, transferFrom_ids, delete_ids = [], [], [], [], []
    finished_ids, failed_ids, timeout_ids = [], [], []

    for id, line in enumerate(lines[1:], start=1):
        slurm_log_fp = os.path.join(log_path, f'slurm-{job_id}_{id}.out') 
        if not os.path.exists(slurm_log_fp):
            queuing_ids.append(id)
            continue

        process_flags = [False for _ in range(7)]
        with open(slurm_log_fp, 'r') as f:
            log_line = f.readline()
            while log_line:
                if TRANSFER_TO_PREFIX in log_line:
                    process_flags[0] = True
                elif (RUNNING_PREFIX + line) in log_line:
                    process_flags[1] = True
                elif TRANSFER_FROM_PROMPT in log_line:
                    process_flags[2] = True
                elif DELETE_PROMPT in log_line:
                    process_flags[3] = True
                elif FINISHED_PROMPT in log_line:
                    process_flags[4] = True
                    break
                elif FAILED_PROMPT in log_line:
                    process_flags[5] = True
                    break
                elif TIMEOUT_PROMPT in log_line:
                    process_flags[6] = True
                    break
                log_line = f.readline()

        # sort flags
        if process_flags[6]:
            timeout_ids.append(id)
        elif process_flags[5]:
            failed_ids.append(id)
        elif process_flags[4]:
            finished_ids.append(id)
        elif process_flags[3]:
            delete_ids.append(id)
        elif process_flags[2]:
            transferFrom_ids.append(id)
        elif process_flags[1]:
            running_ids.append(id)
        elif process_flags[0]:
            transferTo_ids.append(id)

    return queuing_ids, transferTo_ids, running_ids, transferFrom_ids, delete_ids, \
           finished_ids, failed_ids, timeout_ids

def save_experiment_details(lines, ids, output_fp):
    with open(output_fp, 'w') as f:
        f.write(lines[0])  # header
        for id in ids:
            f.write(lines[id])
    print(f'Saved details in: {output_fp}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Slurm Experiment Checker")
    parser.add_argument("-j", "--job", required=True, metavar="INT", help="Slurm Job ID", type=int)
    parser.add_argument("-c", "--config", required=True, metavar="PATH", help="Absolute path to path config file", type=str)
    args = parser.parse_args()

    if args.job < 1:
        raise ValueError('Job ID is not positive!')

    cfg = load_config(args.config)
    slurm_log_path, exp_fp, exp_fail_fp, exp_timeout_fp = get_file_paths(cfg)

    lines = []
    with open(exp_fp, 'r') as f:
        lines = f.readlines()

    queuing_ids, transferTo_ids, running_ids, transferFrom_ids, delete_ids, \
    finished_ids, failed_ids, timeout_ids = check_status(slurm_log_path, lines, args.job)

    print('QUEUING/LOG_IN_OTHER_DIR ----------')
    print(queuing_ids)
    print()

    print('TRANSFER TO GPU NODE ----------')
    print(transferTo_ids)
    print()

    print('RUNNING ----------')
    print(running_ids)
    print()

    print('TRANSFER FROM GPU_NODE ----------')
    print(transferFrom_ids)
    print()

    print('DELETING OUTPUT FROM GPU_NODE ----------')
    print(delete_ids)
    print()

    print('FINISHED ----------')
    print(finished_ids)
    print()

    print('FAILED ----------')
    print(failed_ids)
    print()

    print('CANCELLED ----------')
    print(timeout_ids)
    print()

    any_fails = len(failed_ids) > 0
    any_timeouts = len(timeout_ids) > 0

    if any_fails or any_timeouts:
        # saving details
        if any_fails:
            save_experiment_details(lines, failed_ids, exp_fail_fp)

        if any_timeouts:
            save_experiment_details(lines, timeout_ids, exp_timeout_fp)
