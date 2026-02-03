#!/usr/bin/env python3
# Literal translation of the Perl script https://github.com/andrewjanke/volgenmodel
# to Python and using Nipype interfaces where possible.

# Author: Carlo Hamalainen <carlo@carlo-hamalainen.net>
# Minor Edits: Isshaa Aarya and Steffen Bollmann <Steffen.Bollmann@live.de>

#from nipype import config
#config.enable_debug_mode()

import os
import os.path
import subprocess
import json
import nipype.pipeline.engine as pe
from nipype import config as nipype_config
import nipype.interfaces.io as nio
import nipype.interfaces.utility as utils
from copy import deepcopy
import argparse
import sys

from nipype.interfaces.minc import  \
        Volcentre,      \
        Norm,           \
        Volpad,         \
        Voliso,         \
        Math,           \
        Pik,            \
        Blur,           \
        Gennlxfm,       \
        XfmConcat,      \
        BestLinReg,     \
        NlpFit,         \
        XfmAvg,         \
        XfmInvert,      \
        Resample,       \
        BigAverage,     \
        Reshape,        \
        VolSymm
from nipype.interfaces.utility import Rename
import glob

import pickle
import gzip
import shutil


def check_minc_on_path():
    """
    Check if MINC tools are available on the PATH.
    If not, try to load them via 'ml minc' (module load).
    Returns True if MINC is available, exits with error otherwise.
    """
    # Check if mincinfo (a basic MINC tool) is on the PATH
    if shutil.which('mincinfo') is not None:
        print("+++ MINC tools found on PATH")
        return True
    
    print("+++ MINC tools not found on PATH, attempting to load via 'ml minc'...")
    
    try:
        # Try to load the minc module
        result = subprocess.run(
            ['bash', '-c', 'source /etc/profile.d/modules.sh 2>/dev/null || true; ml minc && which mincinfo'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            # Module loaded successfully, now we need to update the current environment
            # Get the updated PATH from loading the module
            env_result = subprocess.run(
                ['bash', '-c', 'source /etc/profile.d/modules.sh 2>/dev/null || true; ml minc && echo $PATH'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if env_result.returncode == 0:
                new_path = env_result.stdout.strip()
                os.environ['PATH'] = new_path
                print(f"+++ Successfully loaded MINC module, updated PATH")
                
                # Verify mincinfo is now available
                if shutil.which('mincinfo') is not None:
                    print("+++ MINC tools now available on PATH")
                    return True
        
        # If we get here, module load didn't work as expected
        print("!!! Failed to load MINC module via 'ml minc'")
        print(f"!!! stdout: {result.stdout}")
        print(f"!!! stderr: {result.stderr}")
        
    except subprocess.TimeoutExpired:
        print("!!! Timeout while trying to load MINC module")
    except Exception as e:
        print(f"!!! Error while trying to load MINC module: {e}")
    
    print("!!! MINC tools are required but not available.")
    print("!!! Please ensure MINC is installed and on your PATH, or that 'ml minc' works on your system.")
    sys.exit(1)


def get_slurm_cpu_count():
    """
    Detect the CPU count allocated by SLURM.
    Checks SLURM environment variables in order of preference.
    Returns the CPU count, or None if not running under SLURM.
    """
    import os
    
    # Check various SLURM environment variables for CPU allocation
    # SLURM_CPUS_PER_TASK: CPUs allocated per task (set with --cpus-per-task)
    # SLURM_JOB_CPUS_PER_NODE: CPUs allocated per node for the job
    # SLURM_CPUS_ON_NODE: Number of CPUs on the allocated node(s)
    
    for var in ['SLURM_CPUS_PER_TASK', 'SLURM_JOB_CPUS_PER_NODE', 'SLURM_CPUS_ON_NODE']:
        value = os.environ.get(var)
        if value:
            try:
                # SLURM_JOB_CPUS_PER_NODE can have format like "8(x2)" for 2 nodes with 8 CPUs each
                # We just take the first number
                cpu_count = int(value.split('(')[0])
                print(f"+++ Detected SLURM CPU allocation: {cpu_count} CPUs (from {var})")
                return cpu_count
            except (ValueError, IndexError):
                continue
    
    return None


def get_available_cpus(default=None):
    """
    Get available CPU count, checking in order:
    1. SLURM environment variables
    2. PBS/Torque NCPUS environment variable
    3. Cgroups CPU quota
    4. System CPU count
    5. Fallback to default (if provided) or system count
    """
    import os
    
    # First try SLURM
    slurm_cpus = get_slurm_cpu_count()
    if slurm_cpus is not None:
        return slurm_cpus
    
    # Try PBS/Torque NCPUS
    ncpus = os.environ.get('NCPUS')
    if ncpus:
        try:
            cpu_count = int(ncpus)
            print(f"+++ Detected PBS CPU allocation: {cpu_count} CPUs (from NCPUS)")
            return cpu_count
        except ValueError:
            pass
    
    # Try cgroups CPU quota (for containers/cgroups v2)
    cgroup_cpus = get_cgroup_cpu_quota()
    if cgroup_cpus is not None:
        return cgroup_cpus
    
    # Fall back to system CPU count
    system_cpus = os.cpu_count() or 1
    print(f"+++ Using system CPU count: {system_cpus}")
    return default if default is not None else system_cpus


def get_cgroup_cpu_quota():
    """
    Detect the CPU quota set by cgroups (v1 or v2).
    Returns the effective number of CPUs, or None if not available/unlimited.
    """
    import os
    
    # Try cgroups v2 first
    cgroup_v2_max = '/sys/fs/cgroup/cpu.max'
    # Try cgroups v1
    cgroup_v1_quota = '/sys/fs/cgroup/cpu/cpu.cfs_quota_us'
    cgroup_v1_period = '/sys/fs/cgroup/cpu/cpu.cfs_period_us'
    
    try:
        if os.path.exists(cgroup_v2_max):
            with open(cgroup_v2_max, 'r') as f:
                value = f.read().strip()
                parts = value.split()
                if len(parts) >= 2 and parts[0] != 'max':
                    quota = int(parts[0])
                    period = int(parts[1])
                    cpu_count = max(1, quota // period)
                    print(f"+++ Detected cgroups v2 CPU quota: {cpu_count} CPUs")
                    return cpu_count
        elif os.path.exists(cgroup_v1_quota) and os.path.exists(cgroup_v1_period):
            with open(cgroup_v1_quota, 'r') as f:
                quota = int(f.read().strip())
            if quota > 0:  # -1 means unlimited
                with open(cgroup_v1_period, 'r') as f:
                    period = int(f.read().strip())
                cpu_count = max(1, quota // period)
                print(f"+++ Detected cgroups v1 CPU quota: {cpu_count} CPUs")
                return cpu_count
    except (IOError, ValueError, PermissionError):
        pass
    
    return None


def get_cgroup_memory_limit_gb():
    """
    Detect the memory limit set by cgroups (v1 or v2).
    Returns the memory limit in GB, or None if not available/unlimited.
    """
    import os

    # Try cgroups v2 first
    cgroup_v2_path = '/sys/fs/cgroup/memory.max'
    # Try cgroups v1
    cgroup_v1_path = '/sys/fs/cgroup/memory/memory.limit_in_bytes'

    memory_bytes = None

    try:
        if os.path.exists(cgroup_v2_path):
            with open(cgroup_v2_path, 'r') as f:
                value = f.read().strip()
                if value != 'max':  # 'max' means unlimited
                    memory_bytes = int(value)
        elif os.path.exists(cgroup_v1_path):
            with open(cgroup_v1_path, 'r') as f:
                value = int(f.read().strip())
                # Check if it's effectively unlimited (very large value close to max int)
                if value < 9223372036854771712:  # Common "unlimited" value
                    memory_bytes = value
    except (IOError, ValueError, PermissionError):
        pass

    if memory_bytes is not None:
        # Convert bytes to GB, leave some headroom (use 90% of limit)
        return int(memory_bytes / (1024 ** 3) * 0.9)

    return None


# Per-node memory requirements in MB, based on profiling data
# These are peak RSS values observed during a typical run
# Nodes not listed here use default Nipype memory settings
NODE_MEMORY_MB = {
    # High memory nodes (registration/fitting)
    'nlpfit': 5000,           # Non-linear fitting: ~4.7 GB peak
    'bigaverage': 2500,       # Averaging: ~2.2 GB peak
    'resample': 2500,         # Resampling: ~2.2 GB peak
    'bestlinreg': 1500,       # Linear registration: ~1.5 GB peak
    'xfmconcat': 1500,        # Transform concatenation: ~1.3 GB peak
    
    # Medium memory nodes
    'volsymm': 1000,          # Symmetry operations: ~800 MB peak
    'blur': 800,              # Blurring: ~700 MB peak
    'mincmath': 600,          # Math operations: ~500 MB peak
    
    # Low memory nodes (preprocessing)
    'preprocess': 500,        # General preprocessing: ~400 MB peak
    'volcentre': 300,         # Volume centering: ~250 MB peak
    'volpad': 300,            # Volume padding: ~250 MB peak
    'voliso': 300,            # Isotropic resampling: ~250 MB peak
    'pik': 200,               # Picture generation: ~150 MB peak
}

# Nodes that are lightweight and can run without submitting to scheduler
# These run on the submit node or within a parent job
LIGHTWEIGHT_NODES = [
    'datasource', 'datasink', 'select_first', 'merge_', 'rename',
    'nii_to_mnc', 'identity', 'write_conf', 'calc_threshold', 'calc_initial',
    'preprocess_volcentre', 'preprocess_threshold_blur', 'preprocess_normalise',
    'preprocess_volpad', 'preprocess_voliso', 'preprocess_pik', 'preprocess_iso',
    'combined_preprocess',
]


def mark_nodes_run_locally(workflow, patterns=None):
    """
    Mark nodes matching certain patterns to run without submitting as separate jobs.
    This reduces SLURM/PBS job count by running lightweight nodes on the submit node.
    
    Args:
        workflow: Nipype workflow object
        patterns: List of node name patterns to mark. If None, uses LIGHTWEIGHT_NODES.
    """
    if patterns is None:
        patterns = LIGHTWEIGHT_NODES
    
    nodes_marked = 0
    
    for node in workflow._get_all_nodes():
        node_name = node.name.lower()
        
        for pattern in patterns:
            if pattern.lower() in node_name:
                node.run_without_submitting = True
                nodes_marked += 1
                break
    
    print(f"+++ Marked {nodes_marked} lightweight nodes to run locally (no separate jobs)")
    return nodes_marked


def set_node_memory_requirements(workflow, scale=1.0):
    """
    Set memory requirements for workflow nodes based on profiling data.
    
    Args:
        workflow: Nipype workflow object
        scale: Memory scaling factor (default 1.0). Increase for larger datasets.
               e.g., scale=2.0 doubles all memory allocations
    """
    nodes_configured = 0
    
    for node in workflow._get_all_nodes():
        node_name = node.name.lower()
        
        # Find matching memory requirement
        mem_mb = None
        for pattern, base_mem in NODE_MEMORY_MB.items():
            if pattern in node_name:
                mem_mb = int(base_mem * scale)
                break
        
        if mem_mb is not None:
            # Convert to GB for Nipype
            node.mem_gb = mem_mb / 1024.0
            nodes_configured += 1
    
    print(f"+++ Configured memory for {nodes_configured} nodes (scale={scale:.1f}x)")
    if scale != 1.0:
        print(f"    Memory values scaled by {scale:.1f}x from baseline")


def get_available_memory_gb(default=80):
    """
    Get available memory in GB, checking in order:
    1. Cgroups limit (for containers/HPC jobs)
    2. System total memory
    3. Fallback to default
    """
    # First try cgroups
    cgroup_mem = get_cgroup_memory_limit_gb()
    if cgroup_mem is not None:
        print(f"+++ Detected cgroups memory limit: {cgroup_mem} GB (with 10% headroom)")
        return cgroup_mem

    # Fall back to system memory
    try:
        import os
        if hasattr(os, 'sysconf'):
            pages = os.sysconf('SC_PHYS_PAGES')
            page_size = os.sysconf('SC_PAGE_SIZE')
            if pages > 0 and page_size > 0:
                total_bytes = pages * page_size
                total_gb = int(total_bytes / (1024 ** 3) * 0.9)  # 90% of total
                print(f"+++ Using system memory: {total_gb} GB (with 10% headroom)")
                return total_gb
    except (ValueError, OSError):
        pass

    print(f"+++ Using default memory limit: {default} GB")
    return default


def analyze_proc_files(search_dir='.'):
    """
    Parse Nipype .proc-* files (raw resource monitor logs) and summarize usage.
    These files are created by nipype's resource_monitor during execution.
    
    Format: timestamp,elapsed/cpu,rss_mb,vms_mb (comma-separated, no header)
    
    Args:
        search_dir: Directory to search for .proc-* files (default: current directory)
    """
    import glob
    
    # Find all .proc-* files
    proc_files = glob.glob(os.path.join(search_dir, '.proc-*'))
    
    if not proc_files:
        # Also check parent directory and work subdirectory
        for alt_dir in [os.path.dirname(search_dir), os.path.join(search_dir, 'work')]:
            proc_files = glob.glob(os.path.join(alt_dir, '.proc-*'))
            if proc_files:
                break
    
    if not proc_files:
        print(f"\n=== No .proc-* Files Found ===")
        print(f"Searched in: {search_dir}")
        print("These files are created by Nipype's resource monitor during workflow execution.")
        return []
    
    print(f"\nFound {len(proc_files)} .proc-* file(s)")
    
    results = []
    
    for proc_file in proc_files:
        try:
            with open(proc_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 1:
                continue
            
            peak_rss_mb = 0
            peak_vms_mb = 0
            start_time = None
            end_time = None
            
            for line in lines:
                # Format: timestamp,elapsed/cpu,rss_mb,vms_mb
                fields = line.strip().split(',')
                if len(fields) < 4:
                    continue
                
                try:
                    timestamp = float(fields[0])
                    rss_mb = float(fields[2])
                    vms_mb = float(fields[3])
                    
                    peak_rss_mb = max(peak_rss_mb, rss_mb)
                    peak_vms_mb = max(peak_vms_mb, vms_mb)
                    
                    if start_time is None:
                        start_time = timestamp
                    end_time = timestamp
                except (ValueError, IndexError):
                    continue
            
            # Calculate elapsed time
            elapsed_sec = (end_time - start_time) if (start_time and end_time) else 0
            
            # Extract PID from filename (.proc-PID_time-...)
            filename = os.path.basename(proc_file)
            pid = filename.replace('.proc-', '').split('_')[0]
            
            results.append({
                'file': filename,
                'pid': pid,
                'peak_rss_mb': peak_rss_mb,
                'peak_vms_mb': peak_vms_mb,
                'elapsed_sec': elapsed_sec,
            })
            
        except Exception as e:
            print(f"Warning: Could not parse {proc_file}: {e}")
            continue
    
    if not results:
        print("\n=== No Valid Data in .proc Files ===")
        return []
    
    # Sort by peak memory (highest first)
    results.sort(key=lambda x: x['peak_rss_mb'], reverse=True)
    
    print("\n" + "=" * 50)
    print("RESOURCE USAGE (sorted by peak RSS)")
    print("=" * 50)
    print(f"{'PID':<8} {'RSS MB':>10} {'VMS MB':>10} {'Time':>10}")
    print("-" * 50)
    
    overall_peak_rss_mb = 0
    overall_peak_vms_mb = 0
    
    for r in results:
        elapsed = f"{r['elapsed_sec']:.0f}s"
        print(f"{r['pid']:<8} {r['peak_rss_mb']:>10.1f} {r['peak_vms_mb']:>10.1f} {elapsed:>10}")
        overall_peak_rss_mb = max(overall_peak_rss_mb, r['peak_rss_mb'])
        overall_peak_vms_mb = max(overall_peak_vms_mb, r['peak_vms_mb'])
    
    print("-" * 50)
    print(f"{'PEAK:':<8} {overall_peak_rss_mb:>10.1f} {overall_peak_vms_mb:>10.1f}")
    print("=" * 50)
    
    recommended_gb = int(overall_peak_rss_mb / 1024 * 1.2) + 1
    print(f"Recommend --memory_gb >= {recommended_gb}\n")
    
    return results


def analyze_resource_reports(work_dir):
    """
    Parse Nipype resource reports and summarize memory usage per node.
    Also checks for .proc-* files if no report.json files are found.
    Requires resource monitoring to be enabled before workflow execution.
    """
    import glob
    import json
    
    results = []
    
    # Find all report.json files in the work directory
    for report_file in glob.glob(f'{work_dir}/**/result_*.pklz', recursive=True):
        # Resource reports are stored alongside result files
        report_dir = os.path.dirname(report_file)
        resource_file = os.path.join(report_dir, '_report', 'report.json')
        
        if os.path.exists(resource_file):
            try:
                with open(resource_file) as f:
                    data = json.load(f)
                
                # Extract node name from path
                node_name = os.path.basename(report_dir)
                
                results.append({
                    'node': node_name,
                    'memory_gb': data.get('runtime_memory_gb', 0),
                    'peak_memory_gb': data.get('peak_memory_gb', data.get('runtime_memory_gb', 0)),
                    'runtime_sec': data.get('runtime_seconds', 0),
                    'cpu_percent': data.get('cpu_percent', 0),
                })
            except (json.JSONDecodeError, KeyError, IOError) as e:
                print(f"Warning: Could not parse {resource_file}: {e}")
                continue
    
    if not results:
        print("\n=== No report.json Files Found ===")
        print(f"Searched in: {work_dir}")
        print("\nFalling back to .proc-* files...")
        return analyze_proc_files(work_dir)
    
    # Sort by memory usage (highest first)
    results.sort(key=lambda x: x['memory_gb'], reverse=True)
    
    print("\n" + "=" * 80)
    print("RESOURCE USAGE SUMMARY (sorted by memory, highest first)")
    print("=" * 80)
    print(f"{'Node':<50} {'Memory (GB)':>12} {'Runtime (s)':>12}")
    print("-" * 80)
    
    total_memory_max = 0
    total_runtime = 0
    
    for r in results:
        print(f"{r['node']:<50} {r['memory_gb']:>12.2f} {r['runtime_sec']:>12.1f}")
        total_memory_max = max(total_memory_max, r['memory_gb'])
        total_runtime += r['runtime_sec']
    
    print("-" * 80)
    print(f"{'PEAK MEMORY ACROSS ALL NODES:':<50} {total_memory_max:>12.2f} GB")
    print(f"{'TOTAL RUNTIME (sequential):':<50} {total_runtime:>12.1f} sec")
    print("=" * 80)
    
    # Provide recommendations
    print("\n=== RECOMMENDATIONS ===")
    print(f"Set --memory_gb to at least {int(total_memory_max * 1.2)} GB (peak + 20% headroom)")
    print("\nTo set mem_gb on individual nodes, add lines like:")
    for r in results[:5]:  # Top 5 memory consumers
        if r['memory_gb'] > 1:
            print(f"    {r['node']}.mem_gb = {r['memory_gb']:.1f}")
    print("=" * 80 + "\n")
    
    return results


# <editor-fold desc="Functions">
def identity_file(input_file):
    # Adapted from: http://nipy.org/nipype/users/function_interface.html

    import os
    import shutil

    output_file = 'IdentityFile_copy' + os.path.splitext(input_file)[1]
    shutil.copyfile(input_file, output_file)

    return os.path.abspath(output_file)


def _combined_preprocessing(input_file, normalise, model_norm_thresh, pad, iso, check):
    """
    Combined preprocessing function that performs multiple lightweight steps
    in a single job to reduce SLURM job count.
    
    This combines: NIfTI conversion, volcentre, norm, volpad, voliso, and pik
    into a single function node.
    """
    import os
    import subprocess
    import tempfile
    
    def run_cmd(cmd, desc=""):
        """Run a command and check for errors."""
        print(f"+++ {desc}: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        result = subprocess.run(cmd if isinstance(cmd, list) else cmd, 
                                shell=isinstance(cmd, str),
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"{desc} failed: {result.stderr}")
        return result.stdout
    
    def get_step_sizes(mincfile):
        xstep = float(run_cmd(f'mincinfo -attvalue xspace:step {mincfile}').split()[0])
        ystep = float(run_cmd(f'mincinfo -attvalue yspace:step {mincfile}').split()[0])
        zstep = float(run_cmd(f'mincinfo -attvalue zspace:step {mincfile}').split()[0])
        return (xstep, ystep, zstep)
    
    current_file = input_file
    basename = os.path.basename(input_file)
    
    # Step 1: Convert NIfTI to MINC if needed
    lower_file = input_file.lower()
    if lower_file.endswith('.nii.gz') or lower_file.endswith('.nii'):
        if lower_file.endswith('.nii.gz'):
            output_file = basename[:-7] + '.mnc'
        else:
            output_file = basename[:-4] + '.mnc'
        output_path = os.path.abspath(output_file)
        run_cmd(['nii2mnc', '-float', '-clobber', input_file, output_path], "NIfTI to MINC")
        current_file = output_path
        basename = os.path.basename(current_file)
    
    # Step 2: Volcentre
    volcentre_out = os.path.abspath(basename.replace('.mnc', '_volcentre.mnc'))
    run_cmd(['volcentre', '-clobber', '-zero_dircos', current_file, volcentre_out], "Volcentre")
    current_file = volcentre_out
    
    # Step 3: Normalize (if requested)
    if normalise:
        step_x, step_y, step_z = get_step_sizes(current_file)
        threshold_blur = abs(step_x + step_y + step_z)
        
        norm_out = os.path.abspath(basename.replace('.mnc', '_norm.mnc'))
        run_cmd([
            'mincnorm', '-clobber', 
            '-cutoff', str(model_norm_thresh),
            '-threshold', '-threshold_perc', str(model_norm_thresh),
            '-threshold_blur', str(threshold_blur),
            current_file, norm_out
        ], "Normalize")
        current_file = norm_out
    
    # Step 4: Volpad (if requested)
    if pad > 0:
        volpad_out = os.path.abspath(basename.replace('.mnc', '_volpad.mnc'))
        run_cmd([
            'volpad', '-clobber',
            '-distance', str(pad),
            '-smooth', '-smooth_distance', '5',
            current_file, volpad_out
        ], "Volpad")
        current_file = volpad_out
    
    # Step 5: Voliso (if requested)
    if iso:
        voliso_out = os.path.abspath(basename.replace('.mnc', '_voliso.mnc'))
        run_cmd(['voliso', '-clobber', '-avgstep', current_file, voliso_out], "Voliso")
        current_file = voliso_out
    
    # Step 6: Pik check image (if requested)
    pik_output = None
    if check:
        pik_out = os.path.abspath(basename.replace('.mnc', '_check.jpg'))
        run_cmd([
            'mincpik', '-clobber',
            '-triplanar', '-sagittal_offset', '10',
            current_file, pik_out
        ], "Pik check")
        pik_output = pik_out
    
    return current_file, pik_output


combined_preprocessing = utils.Function(
                            input_names=['input_file', 'normalise', 'model_norm_thresh', 'pad', 'iso', 'check'],
                            output_names=['output_file', 'pik_output'],
                            function=_combined_preprocessing)


def _is_nifti_file(input_file):
    """
    Check if a file is a NIfTI file based on its extension.
    Returns True for .nii and .nii.gz files.
    """
    import os
    lower_file = input_file.lower()
    return lower_file.endswith('.nii') or lower_file.endswith('.nii.gz')


def _convert_nii_to_mnc(input_file):
    """
    Convert a NIfTI file to MINC format using nii2mnc.
    If the file is already MINC, return it as-is.
    """
    import os
    import subprocess
    
    lower_file = input_file.lower()
    
    # Check if it's a NIfTI file
    if lower_file.endswith('.nii.gz'):
        output_file = os.path.basename(input_file)[:-7] + '.mnc'
    elif lower_file.endswith('.nii'):
        output_file = os.path.basename(input_file)[:-4] + '.mnc'
    else:
        # Not a NIfTI file, return as-is (assume it's already MINC)
        return input_file
    
    output_path = os.path.abspath(output_file)
    
    # Run nii2mnc conversion
    cmd = ['nii2mnc', '-float', '-clobber', input_file, output_path]
    print(f"+++ Converting NIfTI to MINC: {input_file} -> {output_path}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"nii2mnc conversion failed: {result.stderr}")
    
    return output_path


convert_nii_to_mnc = utils.Function(
                            input_names=['input_file'],
                            output_names=['output_file'],
                            function=_convert_nii_to_mnc)


def _convert_mnc_to_nii(input_file):
    """
    Convert a MINC file to NIfTI format using mnc2nii.
    This is used as a post-processing step to output NIfTI files.
    """
    import os
    import subprocess
    
    lower_file = input_file.lower()
    
    # Check if it's a MINC file
    if not lower_file.endswith('.mnc'):
        # Not a MINC file, return as-is
        return input_file
    
    output_file = os.path.basename(input_file)[:-4] + '.nii'
    output_path = os.path.abspath(output_file)
    
    # Run mnc2nii conversion
    cmd = ['mnc2nii', input_file, output_path]
    print(f"+++ Converting MINC to NIfTI: {input_file} -> {output_path}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"mnc2nii conversion failed: {result.stderr}")
    
    return output_path


convert_mnc_to_nii = utils.Function(
                            input_names=['input_file'],
                            output_names=['output_file'],
                            function=_convert_mnc_to_nii)


def load_pklz(f):
    return pickle.load(gzip.open(f))


def _calc_threshold_blur_preprocess(input_file):
    from volgenmodel import get_step_sizes
    (step_x, step_y, step_z) = get_step_sizes(input_file)
    return abs(step_x + step_y + step_z)


calc_threshold_blur_preprocess = utils.Function(
                                        input_names=['input_file'],
                                        output_names=['threshold_blur'],
                                        function=_calc_threshold_blur_preprocess)


def _calc_initial_model_fwhm3d(input_file):
    from volgenmodel import get_step_sizes
    (xstep, ystep, zstep) = get_step_sizes(input_file)
    return (abs(xstep*4), abs(ystep*4), abs(zstep*4))


calc_initial_model_fwhm3d = utils.Function(
                                        input_names=['input_file'],
                                        output_names=['fwhm3d'],
                                        function=_calc_initial_model_fwhm3d)


def _write_stage_conf_file(snum, snum_txt, conf, end_stage):
    assert snum is not None
    assert snum_txt is not None
    assert conf is not None
    assert end_stage is not None

    import os.path
    from volgenmodel import to_perl_syntax

    conf_fname = os.path.join(os.getcwd(), "fit_stage_%02d.conf" % snum)
    # print "    + Creating", conf_fname

    with open(conf_fname, 'w') as CONF:
        CONF.write("# %s -- created by %s\n#\n" % (conf_fname, 'FIXME'))
        CONF.write("# End stage: " + str(end_stage) + "\n")
        CONF.write("# Stage Num: " + snum_txt + "\n\n")

        CONF.write('@conf = ')

        conf_dicts = []
        for s in range(end_stage + 1):
            conf_dicts.append({str('step'): + conf[s][str('step')],
                               str('blur_fwhm'): conf[s][str('blur_fwhm')],
                               str('iterations'): conf[s][str('iterations')]})

        CONF.write(to_perl_syntax(conf_dicts))

        CONF.write("\n")

    return conf_fname


write_stage_conf_file = utils.Function(
                                    input_names=['snum', 'snum_txt', 'conf', 'end_stage'],
                                    output_names=['conf_fname'],
                                    function=_write_stage_conf_file)


def to_perl_syntax(d):
    """
    Convert a list of dictionaries to Perl-style syntax. Uses
    string-replace so rather brittle.
    """

    return str(d).replace(':', ' => ').replace('[', '(').replace(']', ')')


def from_perl_syntax(d):
    """
    Essentially the inverse of to_perl_syntax() but we also nuke the
    '@' prefix on a list.
    """

    return str(d).replace(' => ', ':').replace('(', '[').replace(')', ']').replace('@', '')


def do_cmd(cmd):
    """
    Run a shell command and return all stdout, throwing an error
    if anything appears on stderr. Only used for commands that are
    expected to be short running, e.g. mincinfo.
    """

    print('do_cmd:', cmd)

    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            shell=True)
    stdoutByte, stderrByte = proc.communicate()


    stderr = stderrByte.decode('ascii')
    stdout = stdoutByte.decode('ascii')
    if stderr == '':
        return stdout
    else:
        assert False, 'Stuff on stderr: ' + str(stderr)


def get_step_sizes(mincfile):
    """
    Get the x, y, and z step sizes from a Minc file.
    """

    xcmd = 'mincinfo -attvalue xspace:step ' + mincfile
    ycmd = 'mincinfo -attvalue yspace:step ' + mincfile
    zcmd = 'mincinfo -attvalue zspace:step ' + mincfile

    xstep = float(do_cmd(xcmd).split()[0])
    ystep = float(do_cmd(ycmd).split()[0])
    zstep = float(do_cmd(zcmd).split()[0])
 
    return (xstep, ystep, zstep)
# </editor-fold>


def make_workflow(args, opt, conf):
    # <editor-fold desc="Setup and datasource">

    workflow = pe.Workflow(name='workflow_temp_'+args.name+args.run+str(args.ncpus))
    workflow.base_dir = os.path.abspath(args.work_dir)

    # infiles = sorted(glob.glob(os.path.join(args.input_dir, args.input_pattern)))

    # templates = {'outfiles': 'sub-{subject}/ses-{ses_name}/anat/*nii2mnc.mnc'}
    templates = {'outfiles': args.input_pattern}

    datasource = pe.Node(interface=nio.SelectFiles(templates), name='datasource')
    datasource.inputs.base_directory = os.path.abspath(args.input_dir)
    datasource.inputs.sort_filelist = True
    # datasource.inputs.ses_name = 'T1'
    # datasource.inputs.subject = ['045', '197']
    # datasource.inputs.template = args.input_pattern
    # datasource.inputs.run = [args.input_pattern_run]
    # datasource.inputs.subject = [args.input_pattern_subject]

    # Note: Do NOT call datasource.run() here - it would execute outside the workflow
    # context and store results in /tmp, which causes FileNotFoundError on SLURM nodes.
    # The datasource will run properly when the workflow executes.

    datasink = pe.Node(interface=nio.DataSink(), name="datasink")
    datasink.inputs.base_directory = os.path.abspath(
        os.path.join(args.output_dir, str('workflow_output_')+args.name+args.run+str(args.ncpus)))


    # </editor-fold>

    # <editor-fold desc="check for infiles and create files array">
    def eval_to_int(x):
        try:
            return int(x)
        except:
            return x

    # setup the fit stages
    fit_stages = opt['fit_stages'].split(',')
    fit_stages = list(map(eval_to_int, fit_stages))
    #
    # if opt['verbose']: print("+++ INFILES\n")
    #
    # dirs = [None] * len(infiles)
    # files = [None] * len(infiles)
    # fileh = {}
    # sub_id = []
    #
    # c = 0
    #
    # for z in infiles:
    #     dir = None
    #     f = None
    #
    #     c_txt = '%04d' % c
    #
    #     # check
    #     assert os.path.exists(z)
    #
    #     # set up arrays
    #     dirs[c] = os.path.split(z)[0] # &dirname($_);
    #     files[c] = c_txt + '-' + os.path.basename(z) # "$c_txt-" . &basename($_);
    #     files[c] = files[c].replace('.mnc', '') # =~ s/\.mnc$//;
    #     fileh[files[c]] = c
    #     sub_id.append(c)
    #
    #     if opt['verbose']:
    #         print("  | [{c_txt}] {d} / {f}".format(c_txt=c_txt, d=dirs[c], f=files[c]))
    #     c += 1

    if fit_stages[-1] > (len(conf) - 1):
       assert False, ( "Something is amiss with fit config, requested a "
                       "fit step ($fit_stages[-1]) beyond what is defined in the "
                       "fitting protocol (size: $#conf)\n\n")

    #rename
    #renameFiles = pe.MapNode(interface=Rename(format_string="importDcm2Mnc%(sd)04d_normStepSize_", keep_ext=True),
                             #iterfield=['in_file', 'sd'], name='RenameFile')
    #renameFiles.inputs.sd = sub_id

    #workflow.connect(datasource, 'outfiles', renameFiles, 'in_file')  
    # </editor-fold>

    # <editor-fold desc="Combined preprocessing mode to reduce SLURM jobs">
    # When combine_jobs is True, we use a single Function node that performs
    # all preprocessing steps in one SLURM job instead of many separate jobs.
    if opt.get('combine_jobs', False):
        print("+++ Using COMBINED preprocessing mode (reduces SLURM job count)")
        
        combined_preprocess = pe.MapNode(
                        interface=deepcopy(combined_preprocessing),
                        name='combined_preprocess',
                        iterfield=['input_file'])
        
        # Set constant inputs
        combined_preprocess.inputs.normalise = opt['normalise']
        combined_preprocess.inputs.model_norm_thresh = opt['model_norm_thresh']
        combined_preprocess.inputs.pad = opt['pad']
        combined_preprocess.inputs.iso = opt['iso']
        combined_preprocess.inputs.check = opt['check']
        
        workflow.connect(datasource, 'outfiles', combined_preprocess, 'input_file')
        
        # Create passthrough nodes that just return the combined output
        # This maintains compatibility with the rest of the workflow
        preprocess_voliso_passthrough = pe.MapNode(
                                interface=utils.Function(
                                    input_names=['input_file'],
                                    output_names=['output_file'],
                                    function=identity_file),
                                name='preprocess_voliso',
                                iterfield=['input_file'])
        
        workflow.connect(combined_preprocess, 'output_file', preprocess_voliso_passthrough, 'input_file')
        
        # For the normalise output (used by resample later), we need to track intermediate
        # Since combined mode doesn't preserve intermediates, use the final output
        preprocess_normalise_passthrough = pe.MapNode(
                                interface=utils.Function(
                                    input_names=['input_file'],
                                    output_names=['output_file'],
                                    function=identity_file),
                                name='preprocess_normalise',
                                iterfield=['input_file'])
        
        workflow.connect(combined_preprocess, 'output_file', preprocess_normalise_passthrough, 'input_file')
        
        # For volpad output (used by initial model)
        preprocess_volpad_passthrough = pe.MapNode(
                                interface=utils.Function(
                                    input_names=['input_file'],
                                    output_names=['output_file'],
                                    function=identity_file),
                                name='preprocess_volpad',
                                iterfield=['input_file'])
        
        workflow.connect(combined_preprocess, 'output_file', preprocess_volpad_passthrough, 'input_file')
        
        # Assign to the standard variable names for downstream compatibility
        preprocess_voliso = preprocess_voliso_passthrough
        preprocess_normalise = preprocess_normalise_passthrough
        preprocess_volpad = preprocess_volpad_passthrough
        
        # Skip the separate preprocessing nodes - they're all combined above
        # Jump directly to initial model setup
        
    else:
        # Standard mode: use individual nodes for each preprocessing step
        # </editor-fold>

        # <editor-fold desc="convert NIfTI to MINC if needed">
        # Check if input files might be NIfTI format and convert them to MINC
        # This handles .nii and .nii.gz files automatically
        nii_to_mnc_converter = pe.MapNode(
                        interface=deepcopy(convert_nii_to_mnc),
                        name='nii_to_mnc_converter',
                        iterfield=['input_file'])
        
        workflow.connect(datasource, 'outfiles', nii_to_mnc_converter, 'input_file')
        # </editor-fold>

        # <editor-fold desc="do pre-processing nad normalise">
        preprocess_volcentre = pe.MapNode(
                        interface=Volcentre(zero_dircos=True),
                        name='preprocess_volcentre',
                        iterfield=['input_file'])

        #workflow.connect(renameFiles, 'out_file', preprocess_volcentre, 'input_file')
        workflow.connect(nii_to_mnc_converter, 'output_file', preprocess_volcentre, 'input_file')

        if opt['normalise']:
            preprocess_threshold_blur = pe.MapNode(
                                            interface=deepcopy(calc_threshold_blur_preprocess), # Beware! Need deepcopy since calc_threshold_blur_preprocess is not a constructor!
                                            name='preprocess_threshold_blur',
                                            iterfield=['input_file'])

            workflow.connect(preprocess_volcentre, 'output_file', preprocess_threshold_blur, 'input_file')

            preprocess_normalise = pe.MapNode(
                                        interface=Norm(
                                                    cutoff=opt['model_norm_thresh'],
                                                    threshold=True,
                                                    threshold_perc=opt['model_norm_thresh']),
                                                    # output_file=nrmfile),
                                        name='preprocess_normalise',
                                        iterfield=['input_file', 'threshold_blur'])

            workflow.connect(preprocess_threshold_blur, 'threshold_blur', preprocess_normalise, 'threshold_blur')

            # do_cmd('mv -f %s %s' % (nrmfile, resfiles[f],))
        else:
            preprocess_normalise_id = utils.Function(
                                                input_names=['input_file'],
                                                output_names=['output_file'],
                                                function=identity_file,
                                                )

            preprocess_normalise = pe.MapNode(
                                        interface=preprocess_normalise_id,
                                        name='preprocess_normalise',
                                        iterfield=['input_file'])

        workflow.connect(preprocess_volcentre, 'output_file', preprocess_normalise, 'input_file')
        # </editor-fold>

        # <editor-fold desc="extend/pad">
        if opt['pad'] > 0:
            #smoothPadValue = 2
            preprocess_volpad = pe.MapNode(
                                    interface=Volpad(
                                                distance=opt['pad'],
                                                smooth=True,
                                                smooth_distance=5), 
                                                # output_file=fitfiles[f]),
                                    name='preprocess_volpad',
                                    iterfield=['input_file'])
        else:
            preprocess_volpad_id = utils.Function(
                                                input_names=['input_file'],
                                                output_names=['output_file'],
                                                function=identity_file,
                                                )

            preprocess_volpad = pe.MapNode(
                                        interface=preprocess_volpad_id,
                                        name='preprocess_volpad',
                                        iterfield=['input_file'])
            if args.run == 'PBSGraph':
                preprocess_volpad.plugin_args = {'qsub_args': '-A UQ-CAI -l nodes=1:ppn=10,mem=10gb,vmem=10gb,walltime=04:10:00',
                                              'overwrite': True}
            if args.run == 'SLURMGraph':
                preprocess_volpad.plugin_args = {'sbatch_args': '--time=04:10:00 --mem=10G --cpus-per-task=10',
                                              'overwrite': True}

        workflow.connect(preprocess_normalise, 'output_file', preprocess_volpad, 'input_file')
        # </editor-fold>

        # <editor-fold desc="isotropic resampling">
        if opt['iso']:
            preprocess_voliso = pe.MapNode(
                                        interface=Voliso(avgstep=True), # output_file=isofile),
                                        name='preprocess_voliso',
                                        iterfield=['input_file'])
        else:
            preprocess_voliso_id = utils.Function(
                                                input_names=['input_file'],
                                                output_names=['output_file'],
                                                function=identity_file,
                                                )

            preprocess_voliso = pe.MapNode(
                                        interface=preprocess_voliso_id,
                                        name='preprocess_iso',
                                        iterfield=['input_file'])

        workflow.connect(preprocess_volpad, 'output_file', preprocess_voliso, 'input_file')
        # </editor-fold>

        # <editor-fold desc="checkfile">
        if opt['check']:
            preprocess_pik = pe.MapNode(
                                    interface=Pik(
                                                triplanar=True,
                                                sagittal_offset=10), # output_file=chkfile),
                                    name='preprocess_pik',
                                    iterfield=['input_file'])
        else:
            preprocess_pik_id = utils.Function(
                                        input_names=['input_file'],
                                        output_names=['output_file'],
                                        function=identity_file,
                                        )

            preprocess_pik = pe.MapNode(
                                    interface=preprocess_pik_id,
                                    name='preprocess_pik',
                                    iterfield=['input_file'])

        workflow.connect(preprocess_volpad, 'output_file', preprocess_pik, 'input_file')
        # </editor-fold>

    # <editor-fold desc="setup the initial model">
    if opt['init_model'] is not None:
        # cmodel = opt['init_model']
        raise NotImplemented
        # To do this, make a data grabber that sends the MNC file to
        # the identity_transformation node below.
    else:
        # Select the 'first' output file from volpad (fitfiles[] in the original volgenmodel).
        select_first_volpad = pe.Node(interface=utils.Select(index=[0]), name='select_first_volpad')
        workflow.connect(preprocess_volpad, 'output_file', select_first_volpad, 'inlist')

        # Select the 'first' input file to calculate the fhwm3d parameter (infiles[] in the original volgenmodel).
        select_first_datasource = pe.Node(interface=utils.Select(index=[0]), name='select_first_datasource')
        workflow.connect(datasource, 'outfiles', select_first_datasource, 'inlist')

        # Calculate the fhwm3d parameter using the first datasource.
        initial_model_fwhm3d = pe.Node(interface=deepcopy(calc_initial_model_fwhm3d), name='initial_model_fwhm3d') # Beware! Need deepcopy since calc_initial_model_fwhm3d is not a constructor!
        workflow.connect(select_first_datasource, 'out', initial_model_fwhm3d, 'input_file')

        initial_model = pe.Node(
                            interface=Blur(), # output_file_base=os.path.join(opt['workdir'], '00-init-model')),
                            name='initial_model')

        workflow.connect(select_first_volpad,  'out',    initial_model, 'input_file')
        workflow.connect(initial_model_fwhm3d, 'fwhm3d', initial_model, 'fwhm3d')

    # Current model starts off as the initial model.
    cmodel = initial_model

    identity_transformation = pe.Node(
                                    interface=Gennlxfm(step=conf[0]['step']), # output_file=initxfm, also output_grid!
                                    name='identity_transformation')

    workflow.connect(initial_model, 'output_file', identity_transformation, 'like')
    # </editor-fold>

    # <editor-fold desc="get last linear stage from fit config">
    s = None
    end_stage = None

    snum = 0
    lastlin = 0
    for snum in range(len(fit_stages)): # for($snum = 0; $snum <= $#fit_stages; $snum++){
        if fit_stages[snum] == 'lin':
            lastlin = snum # "%02d" % snum

    print("+++ Last Linear stage:", lastlin)

    # Foreach end stage in the fitting profile
    print("+++ Fitting")

    last_linear_stage_xfm_node = None
    # </editor-fold>

    for snum in range(len(fit_stages)):
        # <editor-fold desc="Preprocessing">
        snum_txt = None
        end_stage = None
        # f = None
        # cworkdir = None
        # conf_fname = None
        # modxfm = [None] * len(files)
        # rsmpl = [None] * len(files)

        end_stage = fit_stages[snum]
        snum_txt = "%02d_" % snum
        print("  + [Stage: {snum_txt}] End stage: {end_stage}".format(snum_txt=snum_txt, end_stage=end_stage))

        # make subdir in working dir for files
        # cworkdir = os.path.join(opt['workdir'], snum_txt)
        # if not os.path.exists(cworkdir):
        #     do_cmd('mkdir ' + cworkdir)

        # set up model and xfm names
        # avgxfm = os.path.join(cworkdir, "avgxfm.xfm")
        # iavgfile = os.path.join(cworkdir, "model.iavg.mnc")
        # istdfile = os.path.join(cworkdir, "model.istd.mnc")
        # stage_model = os.path.join(cworkdir, "model.avg.mnc")
        # iavgfilechk = os.path.join(cworkdir, "model.iavg.jpg")
        # istdfilechk = os.path.join(cworkdir, "model.istd.jpg")
        # stage_modelchk = os.path.join(cworkdir, "model.avg.jpg")

        # create the ISO model
        # isomodel_base = os.path.join(cworkdir, "fit-model-iso")
        if end_stage == 'lin':
            _idx = 0
        else:
            _idx = end_stage
        modelmaxstep = conf[_idx][ 'step']/4

        # check that the resulting model won't be too large
        # this seems confusing but it actually makes sense...
        if float(modelmaxstep) < float(opt['model_min_step']):
            modelmaxstep = opt['model_min_step']

        print("   -- Model Max step:", modelmaxstep)

        norm = pe.Node(
                    interface=Norm(
                                cutoff=opt['model_norm_thresh'],
                                threshold=True,
                                threshold_perc=opt['model_norm_thresh'],
                                threshold_blur=3),
                                # output_threshold_mask=isomodel_base + ".msk.mnc"),
                                # input_file=cmodel,
                                # output_file=isomodel_base + ".nrm.mnc"),
                    name='norm_' + snum_txt)

        workflow.connect(cmodel, 'output_file', norm, 'input_file')
        voliso = pe.Node(
                        interface=Voliso(maxstep=modelmaxstep),
                                    # input_file=isomodel_base + ".nrm.mnc",
                                    # output_file=isomodel_base + ".mnc"),
                        name='voliso_' + snum_txt)
        workflow.connect(norm, 'output_file', voliso, 'input_file')
        if opt['check']:
            pik = pe.Node(
                        interface=Pik(
                                    triplanar=True,
                                    horizontal_triplanar_view=True,
                                    scale=4,
                                    tile_size=400,
                                    sagittal_offset=10),
                                    # input_file=isomodel_base + ".mnc",
                                    # output_file=isomodel_base + ".jpg"),
                        name='pik_check_voliso' + snum_txt)

            workflow.connect(voliso, 'output_file', pik, 'input_file')
        # create the isomodel fit mask
        #chomp($step_x = `mincinfo -attvalue xspace:step $isomodel_base.msk.mnc`);
        step_x = 1
        blur = pe.Node(
                    interface=Blur(fwhm=step_x*15), # input_file=isomodel_base + ".msk.mnc",
                                                    # output_file_base=isomodel_base + ".msk"),
                    name='blur_' + snum_txt)

        workflow.connect(norm, 'output_threshold_mask', blur, 'input_file')

        mincmath = pe.Node(
                        interface=Math(test_gt=0.1),
                                    # input_files=[isomodel_base + ".msk_blur.mnc"],
                                    # output_file=isomodel_base + ".fit-msk.mnc"),
                        name='mincmath_' + snum_txt)

        workflow.connect(blur, 'output_file', mincmath, 'input_files')
        # </editor-fold>

        # <editor-fold desc="linear or nonlinear fit">
        if end_stage == 'lin':
            print("---Linear fit---")
        else:
            print("---Non Linear fit---")

            # create nlin fit config
            if end_stage != 'lin':
                write_conf = pe.Node(interface=deepcopy(write_stage_conf_file),
                                     name='write_conf_' + snum_txt)
                # Beware! Need deepcopy since write_stage_conf_file is not a constructor!

                write_conf.inputs.snum = snum
                write_conf.inputs.snum_txt = snum_txt
                write_conf.inputs.conf = conf
                write_conf.inputs.end_stage = end_stage
                write_conf.run_without_submitting = True
        # </editor-fold>

        # <editor-fold desc="register each file in the input series">
        if end_stage == 'lin':
            assert opt['linmethod'] == 'bestlinreg'
            bestlinreg = pe.MapNode(
                                interface=BestLinReg(),
                                                # source=isomodel_base + ".mnc",
                                                # target=fitfiles[f],
                                                # output_xfm=modxfm[f]),
                                name='register_' + snum_txt,
                                iterfield=['target'])

            workflow.connect(voliso,            'output_file', bestlinreg, 'source')
            workflow.connect(preprocess_voliso, 'output_file', bestlinreg, 'target')

            if snum == lastlin:
                last_linear_stage_xfm_node = bestlinreg

            modxfm = bestlinreg
        else:
            xfmconcat = pe.MapNode(
                                interface=XfmConcat(),
                                            # input_files=[os.path.join(opt['workdir'], lastlin, files[f] + ".xfm"), initxfm],
                                            # output_file=initcnctxfm),
                                name='xfmconcat_for_nlpfit_' + snum_txt,
                                iterfield=['input_files'])

            merge_lastlin_initxfm = pe.MapNode(
                            interface=utils.Merge(2),
                            name='merge_lastlin_initxfm_' + snum_txt,
                            iterfield=['in1'])

            workflow.connect(last_linear_stage_xfm_node, 'output_xfm',  merge_lastlin_initxfm, 'in1')
            workflow.connect(identity_transformation,    'output_file', merge_lastlin_initxfm, 'in2')

            workflow.connect(merge_lastlin_initxfm, 'out', xfmconcat, 'input_files')

            workflow.connect(identity_transformation, 'output_grid', xfmconcat, 'input_grid_files')

            nlpfit = pe.MapNode(
                            interface=NlpFit(),
                                        # init_xfm=initcnctxfm,
                                        # config_file=conf_fname),
                                        # source_mask=isomodel_base + ".fit-msk.mnc",
                                        # source=isomodel_base + ".mnc",
                                        # target=fitfiles[f],
                                        # output_xfm=modxfm[f]),
                            name='nlpfit_' + snum_txt,
                            iterfield=['target', 'init_xfm'])

            if args.run == 'PBSGraph':
                nlpfit.plugin_args = {'qsub_args': '-A UQ-CAI -l nodes=1:ppn=1,mem=10gb,vmem=10gb,walltime=04:10:00',
                                      'overwrite': True}
            if args.run == 'SLURMGraph':
                nlpfit.plugin_args = {'sbatch_args': '--time=04:10:00 --mem=10G --cpus-per-task=1',
                                      'overwrite': True}

            workflow.connect(write_conf, 'conf_fname', nlpfit, 'config_file')

            workflow.connect(xfmconcat,         'output_file', nlpfit, 'init_xfm')
            workflow.connect(mincmath,          'output_file', nlpfit, 'source_mask')
            workflow.connect(voliso,            'output_file', nlpfit, 'source')
            workflow.connect(preprocess_voliso, 'output_file', nlpfit, 'target') # Make sure that fitfiles[f] is preprocess_voliso at this point in the program.

            workflow.connect(xfmconcat, 'output_grids', nlpfit, 'input_grid_files')

            modxfm = nlpfit
        # </editor-fold>

        # <editor-fold desc="average xfms">
        xfmavg = pe.Node(
                        interface=XfmAvg(),
                                    # input_files=modxfm,
                                    # output_file=avgxfm),
                        name='xfmavg_' + snum_txt)

        if end_stage != 'lin':
            workflow.connect(nlpfit, 'output_grid', xfmavg, 'input_grid_files')

        workflow.connect(modxfm, 'output_xfm', xfmavg, 'input_files') # check that this works - multiple outputs of MapNode going into single list of xfmavg.

        if end_stage == 'lin':
            xfmavg.interface.inputs.ignore_nonlinear = True
        else:
            xfmavg.interface.inputs.ignore_linear = True

        # invert model xfm 
        xfminvert = pe.MapNode(
                            interface=XfmInvert(),
                                        # input_file=modxfm[f],
                                        # output_file=invxfm),
                            name='xfminvert_' + snum_txt,
                            iterfield=['input_file'])

        workflow.connect(modxfm, 'output_xfm', xfminvert, 'input_file')

        # concat: invxfm, avgxfm
        merge_xfm = pe.MapNode(
                        interface=utils.Merge(2),
                        name='merge_xfm_' + snum_txt,
                        iterfield=['in1'])

        workflow.connect(xfminvert, 'output_file', merge_xfm, 'in1')
        workflow.connect(xfmavg,    'output_file', merge_xfm, 'in2')
        # </editor-fold>

        # <editor-fold desc="Collect grid files of xfminvert and xvmavg. This is in two steps.">
        # 1. Merge MapNode results. 
        merge_xfm_mapnode_result = pe.Node(
                            interface=utils.Merge(1),
                            name='merge_xfm_mapnode_result_' + snum_txt)
        workflow.connect(xfminvert, 'output_grid', merge_xfm_mapnode_result, 'in1')

        # 2. Merge xfmavg's single output with the result from step 1.
        merge_xfmavg_and_step1 = pe.Node(
                        interface=utils.Merge(2),
                        name='merge_xfmavg_and_step1' + snum_txt)
        workflow.connect(merge_xfm_mapnode_result,  'out',         merge_xfmavg_and_step1, 'in1')
        workflow.connect(xfmavg,                    'output_grid', merge_xfmavg_and_step1, 'in2')

        xfmconcat = pe.MapNode(
                            interface=XfmConcat(),
                                            # input_files=[invxfm, avgxfm],
                                            # output_file=resxfm),
                            name='xfmconcat_' + snum_txt,
                            iterfield=['input_files'])

        workflow.connect(merge_xfm, 'out', xfmconcat, 'input_files')

        workflow.connect(merge_xfmavg_and_step1, 'out', xfmconcat, 'input_grid_files')
        # </editor-fold>

        # <editor-fold desc="Resample. The first stage (snum == 0) does not involve grid files.">
        if snum == 0:
            resample = pe.MapNode(
                                interface=Resample(sinc_interpolation=True),
                                name='resample_' + snum_txt,
                                iterfield=['input_file', 'transformation'])
        else:
            resample = pe.MapNode(
                                interface=Resample(sinc_interpolation=True),
                                name='resample_' + snum_txt,
                                iterfield=['input_file', 'transformation', 'input_grid_files'])

        workflow.connect(preprocess_normalise, 'output_file',  resample, 'input_file')
        workflow.connect(xfmconcat,            'output_file',  resample, 'transformation')

        if snum > 0:
            workflow.connect(xfmconcat, 'output_grids', resample, 'input_grid_files')

        workflow.connect(voliso,               'output_file', resample, 'like')

        if opt['check']:
            pik_check_resample = pe.MapNode(
                                        interface=Pik(
                                                    triplanar=True,
                                                    sagittal_offset=10),
                                                    # input_file=rsmpl[f],
                                                    # output_file=chkfile),
                                        name='pik_check_resample_' + snum_txt,
                                        iterfield=['input_file'])

            workflow.connect(resample, 'output_file', pik_check_resample, 'input_file')

        # create model
        bigaverage = pe.Node(
                            interface=BigAverage(
                                            output_float=True,
                                            robust=False),
                                            # tmpdir=os.path.join(opt['workdir'], 'tmp'),
                                            # sd_file=istdfile,
                                            # input_files=rsmpl,
                                            # output_file=iavgfile),
                            name='bigaverage_' + snum_txt,
                            iterfield=['input_file'])

        workflow.connect(resample, 'output_file', bigaverage, 'input_files')

        if opt['check']:
            pik_check_iavg = pe.Node(
                                    interface=Pik(
                                                triplanar=True,
                                                horizontal_triplanar_view=True,
                                                scale=4,
                                                tile_size=400,
                                                sagittal_offset=10),
                                                # input_file=iavgfile,
                                                # output_file=iavgfilechk),
                                    name='pik_check_iavg_' + snum_txt)

            workflow.connect(bigaverage, 'output_file', pik_check_iavg, 'input_file')
        # </editor-fold>

        # <editor-fold desc="Do symmetric averaging if required">
        if opt['symmetric']:
            # symxfm = os.path.join(cworkdir, 'model.sym.xfm')
            # symfile = os.path.join(cworkdir, 'model.iavg-short.mnc')

            # convert double model to short
            resample_to_short = pe.Node(
                                        interface=Reshape(write_short=True),
                                                    # input_file=iavgfile,
                                                    # output_file=symfile),
                                        name='resample_to_short_' + snum_txt)

            workflow.connect(bigaverage, 'output_file', resample_to_short, 'input_file')


            assert opt['symmetric_dir'] == 'x' #  handle other cases
            volsymm_on_short = pe.Node(
                                    interface=VolSymm(x=True),
                                                    # input_file=symfile,
                                                    # trans_file=symxfm, # This is an output!
                                                    # output_file=stage_model),
                                    name='volsymm_on_short_' + snum_txt)

            workflow.connect(resample_to_short, 'output_file', volsymm_on_short, 'input_file')

            # set up fit args
            if end_stage == 'lin':
                volsymm_on_short.interface.inputs.fit_linear = True
            else:
                volsymm_on_short.interface.inputs.fit_nonlinear = True
                workflow.connect(write_conf, 'conf_fname', volsymm_on_short, 'config_file')

        else:
            # do_cmd('ln -s -f %s %s' % (os.path.basename(iavgfile), stage_model,))
            volsymm_on_short_id = utils.Function(
                                    input_names=['input_file'],
                                    output_names=['output_file'],
                                    function=identity_file,
                                    )

            volsymm_on_short = pe.Node(
                                    interface=volsymm_on_short_id,
                                    name='volsymm_on_short_' + snum_txt)

            workflow.connect(bigaverage, 'output_file', volsymm_on_short, 'input_file')
        # </editor-fold>

        # <editor-fold desc="We finally have the stage model.">
        stage_model = volsymm_on_short

        if opt['check']:
            pik_on_stage_model = pe.Node(
                                        interface=Pik(
                                                    triplanar=True,
                                                    horizontal_triplanar_view=True,
                                                    scale=4,
                                                    tile_size=400,
                                                    sagittal_offset=10),
                                                    # input_file=stage_model,
                                                    # output_file=stage_modelchk),
                                        name='pik_on_stage_model_' + snum_txt)

            workflow.connect(stage_model, 'output_file', pik_on_stage_model, 'input_file')
        # </editor-fold>

        # <editor-fold desc="if on last step, copy model to $opt{'output_model'}">
        if snum == len(fit_stages) - 1:
            workflow.connect(stage_model, 'output_file', datasink, 'model')

            # create and output standard deviation file if requested
            if opt['output_stdev'] is not None:
                if opt['symmetric']:
                    assert opt['symmetric_dir'] == 'x' # handle other cases
                    volsymm_final_model = pe.Node(
                                                interface=VolSymm(
                                                                x=True,
                                                                nofit=True),
                                                                # input_file=istdfile,
                                                                # trans_file=symxfm, # This is an output!
                                                                # output_file=opt['output_stdev']),
                                                name='volsymm_final_model_' + snum_txt)

                    workflow.connect(bigaverage,        'sd_file',      volsymm_final_model, 'input_file')
                    workflow.connect(volsymm_on_short,  'trans_file',   volsymm_final_model, 'trans_file')
                    workflow.connect(volsymm_on_short,  'output_grid',  volsymm_final_model, 'input_grid_files')
                    workflow.connect(volsymm_final_model, 'output_file', datasink, 'stdev') # we ignore opt['output_stdev']
                else:
                    # do_cmd('cp -f %s %s' % (istdfile, opt['output_stdev'],))
                    workflow.connect(bigaverage, 'sd_file', datasink, 'stdev') # we ignore opt['output_stdev']
            
            # Post-processing: convert final model and stdev to NIfTI if requested
            if opt['output_nifti']:
                # Convert final model to NIfTI
                model_to_nifti = pe.Node(
                                        interface=deepcopy(convert_mnc_to_nii),
                                        name='model_to_nifti_' + snum_txt)
                workflow.connect(stage_model, 'output_file', model_to_nifti, 'input_file')
                workflow.connect(model_to_nifti, 'output_file', datasink, 'model_nifti')
                
                # Convert stdev to NIfTI if it exists
                if opt['output_stdev'] is not None:
                    stdev_to_nifti = pe.Node(
                                            interface=deepcopy(convert_mnc_to_nii),
                                            name='stdev_to_nifti_' + snum_txt)
                    if opt['symmetric']:
                        workflow.connect(volsymm_final_model, 'output_file', stdev_to_nifti, 'input_file')
                    else:
                        workflow.connect(bigaverage, 'sd_file', stdev_to_nifti, 'input_file')
                    workflow.connect(stdev_to_nifti, 'output_file', datasink, 'stdev_nifti')
        cmodel = stage_model
        # </editor-fold>

    return workflow


if __name__ == '__main__':
    # Check that MINC tools are available before doing anything else
    check_minc_on_path()
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='workflow',
                        help='The workflow name')
    parser.add_argument('--run', type=str, default='MultiProc', choices=['MultiProc', 'PBSGraph', 'SLURMGraph'],
                        help='The execution plugin to use')
    parser.add_argument('--ncpus', type=int, default=1,
                        help='The amount of CPUs used in MultiProc mode')
    parser.add_argument('--input_dir', type=str, default='../fast-example',
                        help='The input directory')
    parser.add_argument('--input_pattern', type=str, default='*mouse*.mnc',
                        help='The regular expression to find input files in the input directory')
    parser.add_argument('--input_pattern_run', type=str, default='*',
                        help='The list of runs to be used')
    parser.add_argument('--input_pattern_subject', type=str, default='*',
                        help='The list of subjects to be used')
    parser.add_argument('--work_dir', type=str, default='.',
                        help='The work directory (for temporary workflow files)')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='The output directory (for final models)')
    parser.add_argument('--symmetric', type=bool, default=1, choices=[0, 1],
                        help='Symmetric averaging on? Will flip template at every level and repeat fit')
    parser.add_argument('--symmetric_dir', type=str, default='x', choices=['x', 'y', 'z'],
                        help='Direction for flipping template')
    parser.add_argument('--check', type=bool, default=0, choices=[0, 1],
                        help='Write out jpg files to check during model building')
    parser.add_argument('--normalise', type=bool, default=1, choices=[0, 1],
                        help='normalise input data via histogram clamping')
    parser.add_argument('--model_norm_thresh', type=float, default=0.1,
                        help='thresholding of normalized image to remove background noise')
    parser.add_argument('--model_min_step', type=float, default=0.7,
                        help='the mininmal step size of the final model in mm')
    parser.add_argument('--pad', type=int, default=5,
                        help='zero padding around image')
    parser.add_argument('--iso', type=bool, default=1, choices=[0, 1],
                        help='resample image to be isometric')
    parser.add_argument('--fit_stages', type=str, default='lin,0,1,2,3,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11',
                        help='fit stages to be run. Use presets: "fast" (lin,0,2,4,6,8,10), '
                             '"medium" (lin,0,1,2,3,4,5,6,7,8,9,10,11), or custom comma-separated values')
    parser.add_argument('--output_nifti', type=bool, default=1, choices=[0, 1],
                        help='Convert final model and stdev outputs to NIfTI format (post-processing)')

    # SLURM-specific arguments
    parser.add_argument('--slurm_partition', type=str, default='normal',
                        help='SLURM partition/queue name')
    parser.add_argument('--slurm_account', type=str, default=None,
                        help='SLURM account for job submission')
    parser.add_argument('--slurm_time', type=str, default='04:00:00',
                        help='SLURM walltime limit (HH:MM:SS)')
    parser.add_argument('--slurm_mem', type=str, default='10G',
                        help='SLURM memory per job (e.g., 10G, 4000M)')
    parser.add_argument('--slurm_cpus_per_task', type=int, default=1,
                        help='SLURM CPUs per task')
    parser.add_argument('--memory_gb', type=int, default=None,
                        help='Memory limit in GB for MultiProc mode (auto-detected from cgroups if not specified)')
    parser.add_argument('--memory_scale', type=float, default=1.0,
                        help='Memory scaling factor for per-node allocations (default: 1.0). '
                             'Increase for larger input data (e.g., 2.0 for high-res scans)')
    parser.add_argument('--profile', action='store_true', default=False,
                        help='Enable resource monitoring to profile memory/CPU usage per node')
    parser.add_argument('--combine_jobs', action='store_true', default=False,
                        help='Combine lightweight preprocessing steps into single SLURM jobs. '
                             'Reduces total job count significantly for SLURMGraph execution. '
                             'Preprocessing (volcentre, norm, volpad, voliso) runs locally within jobs.')
    parser.add_argument('--run_preproc_locally', action='store_true', default=False,
                        help='Mark all preprocessing nodes to run without submitting separate SLURM jobs. '
                             'These will execute on the submit node or within a parent SLURM job.')

    cli_args, unparsed = parser.parse_known_args()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    # Handle fit_stages presets to reduce job count
    FIT_STAGES_PRESETS = {
        'fast': 'lin,0,2,4,6,8,10',           # 7 stages, ~65% fewer jobs
        'medium': 'lin,0,1,2,3,4,5,6,7,8,9,10,11',  # 13 stages, ~35% fewer jobs
        'full': 'lin,0,1,2,3,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11',  # default, 20 stages
    }
    
    fit_stages_input = cli_args.fit_stages.lower()
    if fit_stages_input in FIT_STAGES_PRESETS:
        cli_args.fit_stages = FIT_STAGES_PRESETS[fit_stages_input]
        print(f"+++ Using fit_stages preset '{fit_stages_input}': {cli_args.fit_stages}")

    options = dict()
    options['symmetric'] = cli_args.symmetric
    options['symmetric_dir'] = cli_args.symmetric_dir
    options['check'] = cli_args.check
    options['normalise'] = cli_args.normalise
    options['model_norm_thresh'] = cli_args.model_norm_thresh
    options['model_min_step'] = cli_args.model_min_step
    options['pad'] = cli_args.pad
    options['iso'] = cli_args.iso
    options['linmethod'] = 'bestlinreg'
    options['init_model'] = None
    options['config_file'] = None
    options['fit_stages'] = cli_args.fit_stages
    options['output_model'] = 'model.mnc'
    options['output_stdev'] = 'stdev.mnc'
    options['output_nifti'] = cli_args.output_nifti
    options['combine_jobs'] = cli_args.combine_jobs
    options['run_preproc_locally'] = cli_args.run_preproc_locally
    # opt['workdir'] = '/scratch/volgenmodel-fast-example/work'
    options['verbose'] = 1
    options['clobber'] = 1
    options['fake'] = 0
    options['clean'] = 0
    options['keep_tmp'] = 0

    # Enable resource monitoring if --profile flag is set
    if cli_args.profile:
        print("+++ Resource monitoring ENABLED - will profile memory/CPU usage per node")
        nipype_config.enable_resource_monitor()
    
    configuration = [{str('step'): 32, str('blur_fwhm'): 16, str('iterations'): 20},        # 0
                     {str('step'): 16, str('blur_fwhm'): 8, str('iterations'): 20},         # 1
                     {str('step'): 12, str('blur_fwhm'): 6, str('iterations'): 20},         # 2
                     {str('step'): 8, str('blur_fwhm'): 4, str('iterations'): 20},          # 3
                     {str('step'): 6, str('blur_fwhm'): 3, str('iterations'): 20},          # 4
                     {str('step'): 4, str('blur_fwhm'): 2, str('iterations'): 10},          # 5
                     {str('step'): 2, str('blur_fwhm'): 1, str('iterations'): 10},          # 6
                     {str('step'): 1.5, str('blur_fwhm'): 0.75, str('iterations'): 10},     # 7
                     {str('step'): 1, str('blur_fwhm'): 0.5, str('iterations'): 5},         # 8
                     {str('step'): 0.9, str('blur_fwhm'): 0.45, str('iterations'): 5},      # 9
                     {str('step'): 0.8, str('blur_fwhm'): 0.4, str('iterations'): 5},       # 10
                     {str('step'): 0.7, str('blur_fwhm'): 0.35, str('iterations'): 5}]      # 11

    wf = make_workflow(cli_args, options, configuration)

    # Set per-node memory requirements based on profiling data
    set_node_memory_requirements(wf, scale=cli_args.memory_scale)
    
    # Mark lightweight nodes to run locally (not as separate SLURM jobs)
    if cli_args.run_preproc_locally or cli_args.run == 'SLURMGraph':
        # Always mark some nodes as local for SLURM to reduce job count
        mark_nodes_run_locally(wf)

    os.makedirs(os.path.abspath(args.work_dir), exist_ok=True)
    os.makedirs(os.path.abspath(args.output_dir), exist_ok=True)

    if cli_args.run == 'MultiProc':
        # Determine memory limit: CLI arg > cgroups > system memory > default
        if cli_args.memory_gb is not None:
            memory_gb = cli_args.memory_gb
            print(f"+++ Using user-specified memory limit: {memory_gb} GB")
        else:
            memory_gb = get_available_memory_gb(default=80)

        # Determine CPU count: CLI arg > SLURM > PBS > cgroups > system count
        if cli_args.ncpus != 1:  # User explicitly set --ncpus
            n_procs = cli_args.ncpus
            print(f"+++ Using user-specified CPU count: {n_procs}")
        else:
            n_procs = get_available_cpus()

        wf.run(
            plugin='MultiProc',
            plugin_args={
                'n_procs': n_procs,
                'memory_gb': memory_gb,
            }
        )
    if cli_args.run == 'PBSGraph':
        wf.run(
            plugin='PBSGraph',
            plugin_args={
                'qsub_args': '-A UQ-CAI -l nodes=1:ppn=1,mem=1gb,vmem=1gb,walltime=00:10:00',
                #'max_jobs': '10',
                'dont_resubmit_completed_jobs': True
            }
        )

    if cli_args.run == 'SLURMGraph':
        # Build sbatch_args from CLI parameters
        sbatch_args = f'--time={cli_args.slurm_time} --mem={cli_args.slurm_mem} --cpus-per-task={cli_args.slurm_cpus_per_task}'
        if cli_args.slurm_partition:
            sbatch_args += f' --partition={cli_args.slurm_partition}'
        if cli_args.slurm_account:
            sbatch_args += f' --account={cli_args.slurm_account}'

        # Add PYTHONPATH so SLURM workers can import volgenmodel module
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sbatch_args += f' --export=ALL,PYTHONPATH="{script_dir}:$PYTHONPATH"'

        wf.run(
            plugin='SLURMGraph',
            plugin_args={
                'sbatch_args': sbatch_args,
                'dont_resubmit_completed_jobs': True,
                'dont_wait': True
            }
        )

    # Print resource usage summary if profiling was enabled
    if cli_args.profile:
        print("\n+++ Analyzing resource usage...")
        analyze_resource_reports(os.path.abspath(args.work_dir))
    
    print('done')
