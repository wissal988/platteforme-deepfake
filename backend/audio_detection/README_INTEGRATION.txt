Integration detection cross-attack
==================================

Fichier principal:
- detect_cross_attack.py

Commande de test:
python detect_cross_attack.py --audio chemin/vers/audio.wav

Commande avec chemins explicites:
python detect_cross_attack.py --audio chemin/vers/audio.wav --config configs/debug_cpu_lcnn_logmel.yaml --checkpoint_dir trained_models/cross_attack

Dependances Python minimales:
- torch
- torchaudio
- numpy
- scikit-learn
- pyyaml
- tqdm
- soundfile utile si le backend gere plusieurs formats audio

Format sortie:
- Le script imprime le rapport dans le terminal.
- Il sauvegarde aussi un JSON dans trained_models/cross_attack/detection_ensemble_<nom_audio>.json.

Important:
- Garder la meme structure des dossiers: configs/, src/, trained_models/cross_attack/.
- Les fichiers .pth sont les checkpoints du modele, il faut les envoyer avec le script.
