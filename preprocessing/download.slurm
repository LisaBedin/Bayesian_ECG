#!/bin/bash
#SBATCH --job-name=download          # nom du job
#SBATCH --partition=prepost          # decommenter pour la partition gpu_p2 (GPU V100 32 Go)
#SBATCH -A lky@v100
#SBATCH --qos=qos_gpu-dev
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --cpus-per-task=12           # nombre de CPU par tache pour gpu_p5 (1/8 des CPU du noeud 8-GPU A100)
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=00:30:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=/lustre/fsn1/projects/rech/vpd/udw33dp/job_output/download%j.out      # nom du fichier de sortie
#SBATCH --error=/lustre/fsn1/projects/rech/vpd/udw33dp/job_output/download%j.out       # nom du fichier d'erreur (ici commun avec la sortie)

# Nettoyage des modules charges en interactif et herites par defaut
module purge

# Chargement des modules
module load pytorch-gpu/py3/2.3.0

URL=$1
TARGETDIR=$2

# Echo des commandes lancees
set -x
cd $TARGETDIR
pwd

wget -r -N -c -np $URL