import subprocess

def get_reproducibility_info():
    # 1. Get the current hash
    hash_cmd = ['git', 'rev-parse', 'HEAD']
    git_hash = subprocess.check_output(hash_cmd).decode('ascii').strip()
    
    # 2. Check for uncommitted changes (Dirty State)
    # Returns output if there are changes, empty otherwise
    status_cmd = ['git', 'status', '--porcelain']
    is_dirty = subprocess.check_output(status_cmd).decode('ascii').strip() != ""
    
    return {
        "git_hash": git_hash,
        "is_dirty": is_dirty, # WARNING: If True, results may not be reproducible!
        "branch": subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
    }


if __name__ == '__main__':
    print(get_reproducibility_info())