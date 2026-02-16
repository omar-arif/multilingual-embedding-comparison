import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix OpenMP conflict

import subprocess
import sys
import urllib.request
import shutil
from colorama import init, Fore, Style

init(autoreset=True)


def run_command(cmd, description):
    """Execute shell command with colored output"""
    try:
        subprocess.run(cmd, check=True, shell=False)
        print(f"{Fore.GREEN}[SUCCESS] {description} completed{Style.RESET_ALL}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}[FAILURE] {description} failed{Style.RESET_ALL}")
        return False


def cleanup_aligned_folder(emb_dir):
    """Move all files from versioned subfolder to main aligned/ directory"""
    aligned_dir = os.path.join(emb_dir, 'aligned')
    
    if not os.path.exists(aligned_dir):
        return
    
    # Find subfolders
    subfolders = [f for f in os.listdir(aligned_dir) 
                  if os.path.isdir(os.path.join(aligned_dir, f))]
    
    for subfolder in subfolders:
        source_dir = os.path.join(aligned_dir, subfolder)
        
        # Move all files
        for filename in os.listdir(source_dir):
            source_file = os.path.join(source_dir, filename)
            dest_file = os.path.join(aligned_dir, filename)
            
            if os.path.isfile(source_file):
                shutil.move(source_file, dest_file)
        
        # Remove subfolder
        shutil.rmtree(source_dir)
    
    print(f"{Fore.GREEN}[OK] Cleaned up aligned directory{Style.RESET_ALL}")


def setup_muse():
    """Clone MUSE, patch for PyTorch 2.6, download dictionaries"""
    
    # Clone MUSE repo
    if not os.path.exists('MUSE'):
        print(f"{Fore.CYAN}Cloning MUSE repository...{Style.RESET_ALL}")
        subprocess.run(['git', 'clone', 'https://github.com/facebookresearch/MUSE.git'])
    
    os.chdir('MUSE')
    
    
    # Patch for PyTorch 2.6 compatibility
    # fix trainer
    trainer_file = 'src/trainer.py'
    print(f"{Fore.YELLOW}Patching trainer.py for PyTorch 2.6...{Style.RESET_ALL}")
    
    with open(trainer_file, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace(
        'to_reload = torch.from_numpy(torch.load(path))',
        'to_reload = torch.from_numpy(torch.load(path, weights_only=False))'
    )
    with open(trainer_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"{Fore.GREEN}[OK] Patched trainer.py{Style.RESET_ALL}")
    
   # fix evaluator
    evaluate_file = 'src/evaluator.py'
    print(f"{Fore.YELLOW}Patching evaluator.py for PyTorch 2.6...{Style.RESET_ALL}")
    with open(evaluate_file, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace(
        'torch.load(path)',
        'torch.load(path, weights_only=False)'
    )

    with open(evaluate_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"{Fore.GREEN}[OK] Patched evaluator.py{Style.RESET_ALL}")
    
    
    
    # Download bilingual dictionaries
    dict_dir = 'data/crosslingual/dictionaries'
    if not os.path.exists(dict_dir):
        print(f"{Fore.CYAN}Downloading evaluation dictionaries...{Style.RESET_ALL}")
        
        bash_success = False
        
        # Try bash script if available
        if shutil.which('bash'):
            print(f"{Fore.YELLOW}Using bash script from repo...{Style.RESET_ALL}")
            os.chdir('data')
            try:
                subprocess.run(['bash', 'get_evaluation.sh'], check=True)
                os.chdir('..')
                bash_success = True
                print(f"{Fore.GREEN}[OK] Dictionaries downloaded{Style.RESET_ALL}")
            except subprocess.CalledProcessError:
                print(f"{Fore.RED}[ERROR] Bash script failed, trying manual download...{Style.RESET_ALL}")
                os.chdir('..')
        
        # Fallback to manual download
        if not bash_success:
            print(f"{Fore.YELLOW}Downloading manually...{Style.RESET_ALL}")
            os.makedirs(dict_dir, exist_ok=True)
            
            dict_url = 'https://dl.fbaipublicfiles.com/arrival/dictionaries/en-fr.txt'
            dict_file = os.path.join(dict_dir, 'en-fr.txt')
            
            try:
                urllib.request.urlretrieve(dict_url, dict_file)
                
                with open(dict_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Split into train and test sets
                with open(os.path.join(dict_dir, 'en-fr.0-5000.txt'), 'w', encoding='utf-8') as f:
                    f.writelines(lines[:5000])
                
                with open(os.path.join(dict_dir, 'en-fr.5000-6500.txt'), 'w', encoding='utf-8') as f:
                    f.writelines(lines[5000:6500])
                    
                print(f"{Fore.GREEN}[OK] Dictionaries downloaded{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}[ERROR] Dictionary download failed: {e}{Style.RESET_ALL}")
    
    print(f"{Fore.GREEN}MUSE setup complete{Style.RESET_ALL}")


def align_embeddings(embeddings, project_root):
    """Run MUSE supervised alignment for all embedding types"""
    
    for emb_type, en_file, fr_file, emb_dim in embeddings:
        emb_dir = os.path.join(project_root, 'embeddings', emb_type)
        
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Aligning {emb_type.upper()} embeddings...{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
        
        success = run_command([
            'python', 'supervised.py',
            '--src_lang', 'en',
            '--tgt_lang', 'fr',
            '--src_emb', os.path.join(emb_dir, en_file),
            '--tgt_emb', os.path.join(emb_dir, fr_file),
            '--n_refinement', '5',
            '--dico_train', 'default',
            '--exp_path', emb_dir,
            '--exp_name', 'aligned',
            '--export', 'pth',
            '--cuda', 'False',
            '--emb_dim', str(emb_dim)
        ], f"Aligning {emb_type}")
        
        # Clean up versioned subfolders
        if success:
            cleanup_aligned_folder(emb_dir)


def evaluate_embeddings(embeddings, project_root):
    """Evaluate alignment quality with precision@k metrics"""
    
    for emb_type, _, _, _ in embeddings:
        emb_dir = os.path.join(project_root, 'embeddings', emb_type)
        
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Evaluating {emb_type.upper()} embeddings...{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
        
        run_command([
            'python', 'evaluate.py',
            '--src_lang', 'en',
            '--tgt_lang', 'fr',
            '--src_emb', os.path.join(emb_dir, 'aligned', 'vectors-en.pth'),
            '--tgt_emb', os.path.join(emb_dir, 'aligned', 'vectors-fr.pth'),
            '--max_vocab', '200000',
            '--cuda', 'False'
        ], f"Evaluating {emb_type}")


def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else 'both'
    project_root = os.path.abspath(os.getcwd())
    
    setup_muse()
    
    # Define embeddings: (type, en_file, fr_file, dimension)
    embeddings = [
        ('ohe', 'english_onehot.vec', 'french_onehot.vec', 1700),
        ('tfidf', 'english_tfidf.vec', 'french_tfidf.vec', 1700),
        ('w2v', 'english_word2vec.vec', 'french_word2vec.vec', 300),
        ('ft', 'english_fasttext.vec', 'french_fasttext.vec', 300),
        ('glv', 'english_glove.vec', 'french_glove.vec', 300)
    ]
    
    if mode in ['align', 'both']:
        align_embeddings(embeddings, project_root)
    
    if mode in ['eval', 'both']:
        evaluate_embeddings(embeddings, project_root)
    
    os.chdir(project_root)
    print(f"\n{Fore.GREEN}Done!{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
