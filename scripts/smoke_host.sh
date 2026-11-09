#!/usr/bin/env bash
# READ-ONLY host inventory for Step 02. Installs nothing, kills nothing, changes nothing.
# Run on the shared workstation only with the administrator's agreement.
set -u
section() { printf '\n===== %s =====\n' "$1"; }
section "OS";        cat /etc/os-release 2>/dev/null; uname -a
section "User";      id
section "CPU/RAM";   lscpu | head -20; free -h
section "GPU";       nvidia-smi 2>/dev/null || echo "nvidia-smi not available"
section "GPU procs"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null
section "Disk";      df -h; df -i
section "Mounts";    lsblk -o NAME,SIZE,TYPE,MOUNTPOINT 2>/dev/null
section "Tools";     for t in python3 git docker podman apptainer singularity sbatch squeue nvcc; do
                       printf '%-12s %s\n' "$t" "$(command -v "$t" || echo missing)"; done
section "Slurm";     sinfo 2>/dev/null || echo "Slurm not available"
section "Load";      uptime; ps -eo user,pid,%cpu,%mem,etime,comm --sort=-%cpu | head -15
