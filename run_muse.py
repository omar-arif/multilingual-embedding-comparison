import subprocess
import os
import sys


def run_command(cmd, description):
    """Execute a shell command and handle errors"""

    try:
        subprocess.run(cmd, check=True, shell=False)
        print(f"{description} completed")
        return True
    except subprocess.CalledProcessError:
        print(f"{description} failed")
        return False


def setup_muse():
    """Setup MUSE repository and data"""
    
    if not os.path.exists('MUSE'):
        print("Cloning MUSE repository...")
        subprocess.run(['git', 'clone', 'https://github.com/facebookresearch/MUSE.git'])
    
    os.chdir('MUSE')
    
    if not os.path.exists('data/crosslingual/dictionaries'):
        print("Downloading evaluation dictionaries...")
        os.chdir('data')
        subprocess.run(['bash', 'get_evaluation.sh'])
        os.chdir('..')


def align_embeddings(embeddings):
    """Run MUSE alignment for all embeddings"""
    for emb_type, en_file, fr_file in embeddings:
        emb_dir = f"../embeddings/{emb_type}"
        
        run_command([
            'python', 'supervised.py',
            '--src_lang', 'en',
            '--tgt_lang', 'fr',
            '--src_emb', f'{emb_dir}/{en_file}',
            '--tgt_emb', f'{emb_dir}/{fr_file}',
            '--n_refinement', '5',
            '--dico_train', 'default',
            '--exp_path', emb_dir,
            '--exp_name', 'aligned',
            '--export', 'pth'
        ], f"Aligning {emb_type}")


def evaluate_embeddings(embeddings):
    """Run MUSE evaluation for all embeddings"""
    for emb_type, _, _ in embeddings:
        emb_dir = f"../embeddings/{emb_type}"
        
        run_command([
            'python', 'evaluate.py',
            '--src_lang', 'en',
            '--tgt_lang', 'fr',
            '--src_emb', f'{emb_dir}/aligned/vectors-en.pth',
            '--tgt_emb', f'{emb_dir}/aligned/vectors-fr.pth',
            '--max_vocab', '200000'
        ], f"Evaluating {emb_type}")


def main():
    # Parse command line argument
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else 'both'
    
    # Setup
    setup_muse()
    
    # Define embeddings
    embeddings = [
        ('ohe', 'english_onehot.vec', 'french_onehot.vec'),
        ('tfidf', 'english_tfidf.vec', 'french_tfidf.vec'),
        ('w2v', 'english_word2vec.vec', 'french_word2vec.vec'),
        ('ft', 'english_fasttext.vec', 'french_fasttext.vec'),
        ('glv', 'english_glove.vec', 'french_glove.vec')
    ]
    
    # Run requested operations
    if mode in ['align', 'both']:
        align_embeddings(embeddings)
    
    if mode in ['eval', 'both']:
        evaluate_embeddings(embeddings)
    
    os.chdir('..')


if __name__ == "__main__":
    main()
