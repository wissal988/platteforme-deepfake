Integration detection audio - version CLAUDE CODE
=================================================

Ce paquet vient du dossier:
  C:\pfe\pfecode\claude code

Ne pas utiliser l'ancien paquet:
  a_envoyer_detection_cross_attack


Contenu important
-----------------

- audio_engine_claude_code.py
  Adaptateur Python pour backend FastAPI. C'est le plus simple a importer.

- rapport_conformite_audio.py
  Script CLI pour tester un fichier audio ou un dossier REAL/FAKE.

- configs/attack_agnostic_lcnn_logmel.yaml
  Configuration du modele courant.

- src/
  Code modele + preprocessing audio.

- trained_models/attack_agnostic/
  Checkpoints .pth et fichiers *_results.json avec les seuils.

- requirements_backend_audio.txt
  Dependances minimales pour le backend.

- requirements.txt
  Dependances completes du projet original.


Installation
------------

Depuis ce dossier:

  pip install -r requirements_backend_audio.txt

Dependances minimales cote backend:

  torch
  torchaudio
  numpy
  pyyaml
  scikit-learn
  scipy
  soundfile

Pour certains formats audio, installer aussi ffmpeg sur la machine.


Test en ligne de commande
-------------------------

Depuis ce dossier:

  python rapport_conformite_audio.py --audio chemin/vers/audio.wav --config configs/attack_agnostic_lcnn_logmel.yaml --checkpoint_dir trained_models/attack_agnostic --max_segments 5

Le script affiche un rapport dans le terminal et sauvegarde un JSON dans:

  trained_models/attack_agnostic/detection_ensemble_<nom_audio>.json


Integration FastAPI recommandee
-------------------------------

Dans le backend de ta binome, copier ce dossier complet, puis importer:

  from audio_engine_claude_code import load_audio_engine, run_audio_inference

Au demarrage du backend:

  load_audio_engine()

Dans la route upload audio:

  data = await file.read()
  result = run_audio_inference(data, file.filename or "audio.wav")
  verdict = result["verdict"] if result["verdict"] in ("fake", "real") else None
  confidence = result["confidence"]

Exemple de reponse API:

  {
    "verdict": verdict,
    "confidence": confidence,
    "details": result
  }


Formats acceptes
----------------

  .wav, .mp3, .flac, .ogg, .m4a, .aac


Important
---------

- Garder la meme structure des dossiers.
- Ne pas renommer trained_models/attack_agnostic.
- Les fichiers .pth sont obligatoires.
- Les fichiers *_results.json sont utiles pour recuperer les seuils.
- Les dossiers cache/, notebooks/, manifests/ et archive_anciens_scripts/ ne sont pas necessaires pour le backend.
