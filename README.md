# Object Recognition - MAI Spring 2025

## Install pre-commit hooks

Installing the pre-commit hooks is recommended but not required.
It helps keep the codebase clean and consistent.

```bash
pip install pre-commit
pre-commit install
```

## Development Setup

### Using VS Code with Remote SSH

Developing directly on a remote server (like the BSC login nodes) via SSH is highly recommended for this project. VS Code provides excellent support for this.

**Prerequisites:**
*   An OpenSSH compatible SSH client installed on your local machine.
*   Your SSH key configured for passwordless login to the remote server (see `ssh-copy-id` or server-specific instructions).

**Setup:**
1.  Install the **Remote - SSH** extension by Microsoft in your local VS Code.
2.  Open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P` on Mac).
3.  Type and select **Remote-SSH: Connect to Host...**.
4.  Choose the host alias you have configured in your `~/.ssh/config` file (e.g., `alogin1.bsc.es`). If you haven't configured one, you can type `user@hostname` (e.g., `nctXX@alogin1.bsc.es`).
5.  VS Code will open a new window and automatically install the necessary VS Code Server components on the remote machine (this happens only the first time).
6.  Once connected, use **File > Open Folder...** to open your project directory on the remote server (e.g., `~/dl`). You can now edit files, use the terminal, and run code as if it were local.

### Synchronizing Files with `rsync`

We provide a script to synchronize directories between your local machine and the remote server using `rsync`, which is useful for sending source code or receiving data/results. It automatically excludes common virtual environment directories (`*venv/`).

**Script:** `scripts/rsync.py`

**Prerequisites:**
*   `rsync` installed on both local and remote machines.
*   Passwordless SSH access configured (as mentioned above).

**Usage:**

```bash
# General format
./scripts/rsync.py <send|receive> [--remote_host <remote_host_alias>] [--local_dir <local_directory>] [--remote_dir <remote_directory>] [--exclude PATTERN]

# Default values (uses the top level directory, and the alogin1.bsc.es remote host)
./scripts/rsync.py <send|receive>

# Example: Send local practicals/p1/src to remote ~/dl/practicals/p1/src
./scripts/rsync.py send --remote_host my_bsc_server --local_dir practicals/p1/src --remote_dir 'dl/practicals/p1/src'

# Example: Receive remote ~/dl/practicals/p1/data to local ./data
./scripts/rsync.py receive --remote_host my_bsc_server --remote_dir 'dl/practicals/p1/data' --local_dir practicals/p1/data
```

Replace `<remote_host_alias>` with the name defined in your `~/.ssh/config`. The script ensures directories like `*venv/` are ignored by default. Use the `--exclude` flag (multiple times if needed) to specify additional patterns to ignore during synchronization.

## Run the code on the BSC cluster

The code can be run on the BSC cluster using the `launcher-main.sh` script.

```bash
cd practicals/p3/src
sbatch -A nct_328 -q acc_training launcher-main.sh
```

You can monitor the queue with:

```bash
squeue
```

You can cancel a job with:

```bash
scancel <job_id>
```

The logs will be saved in the `data/03_logs` directory.
