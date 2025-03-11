import subprocess

main_url = 'https://physionet.org/files/ptb-xl/1.0.3/records500/'
all_urls = [main_url+str(int(k))+'000/' for k in range(12, 22)]
target_dir = '/lustre/fsn1/projects/rech/vpd/udw33dp'

for url in all_urls:
    subprocess.call(['sbatch',
                     'download.slurm',
                     str(url),
                     str(target_dir)
                     ])
print('done')