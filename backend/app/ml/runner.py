import json
import sys
from pathlib import Path
from image_model import predict_image
from video_model import predict_video


def main():
    if len(sys.argv) < 3:
        print(json.dumps({
            "success": False,
            "error": "Usage: python runner.py <IMAGE|VIDEO> <FILE_PATH>"
        }))
        return

    media_type = sys.argv[1].upper()
    file_path = sys.argv[2]

    if not Path(file_path).exists():
        print(json.dumps({
            "success": False,
            "error": f"Fichier introuvable: {file_path}"
        }))
        return

    try:
        if media_type == "IMAGE":
            result = predict_image(file_path)
        elif media_type == "VIDEO":
            result = predict_video(file_path)
        else:
            raise ValueError("Type de média non supporté")

        print(json.dumps({
            "success": True,
            "result": result
        }))
    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": str(e)
        }))


if __name__ == "__main__":
    main()