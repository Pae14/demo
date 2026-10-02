import os
import glob
import cv2
import csv
import argparse
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import config
    from config import *
    from utils_vision import _compute_metrics, _get_status
    from ultralytics import YOLO
except ImportError as e:
    print(f"Error al importar módulos: {e}")
    sys.exit(1)

def get_images_recursive(base_folder, max_depth=2):
    img_extensions = ("*.png", "*.jpg", "*.jpeg")
    all_imgs = []
    
    base_folder = os.path.abspath(base_folder)
    prefix_len = len(base_folder) + 1

    for root, dirs, files in os.walk(base_folder):
        # Calcular profundidad relativa
        rel_path = root[prefix_len:]
        # depth 0 = root, depth 1 = subfolder, depth 2 = sub-subfolder
        depth = rel_path.count(os.sep) + 1 if rel_path else 0
        
        if depth > max_depth:
            continue
            
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_imgs.append(os.path.join(root, f))
            
    return sorted(all_imgs)

def main():
    parser = argparse.ArgumentParser(description="Exporta métricas de defectos a CSV.")
    parser.add_argument("--folder", type=str, default=IMG_DIR, help="Carpeta raíz de imágenes.")
    parser.add_argument("--output", type=str, default="analisis_defectos.csv", help="Archivo CSV de salida.")
    parser.add_argument("--append", action="store_true", help="Si se activa, añade los datos al CSV en lugar de sobreescribir.")
    parser.add_argument("--conf", type=float, default=0.25, help="Umbral de confianza.")
    parser.add_argument("--model", type=str, default=YOLO12_PATH, help="Modelo YOLO.")
    
    args = parser.parse_args()

    if not os.path.exists(args.folder):
        print(f"Error: La carpeta '{args.folder}' no existe.")
        return

    print(f"Cargando modelo: {args.model}")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    # Búsqueda limitada a 2 niveles de subcarpetas
    img_paths = get_images_recursive(args.folder, max_depth=2)

    if not img_paths:
        print(f"No se encontraron imágenes en {args.folder} (hasta 2 niveles)")
        return

    print(f"Procesando {len(img_paths)} imágenes...")

    headers = [
        "filename", "relative_path", "class_id", "class_name", "confidence", 
        "repairable", "status", "area_mm2", "perimeter_mm", 
        "eccentricity", "compactation", "glcm_homogeneity", 
        "glcm_energy", "glcm_entropy"
    ]

    # Determinar modo de apertura y si escribir cabecera
    file_exists = os.path.isfile(args.output)
    mode = 'a' if (args.append and file_exists) else 'w'
    write_header = not (args.append and file_exists)

    try:
        with open(args.output, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if write_header:
                writer.writeheader()

            for ip in img_paths:
                filename = os.path.basename(ip)
                rel_path = os.path.relpath(ip, args.folder)
                
                bgr = cv2.imread(ip)
                if bgr is None: continue
                
                h, w = bgr.shape[:2]
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                pred = model.predict(ip, conf=args.conf, imgsz=640, verbose=False)[0]
                
                detections = []
                is_image_repairable = True

                if pred.masks is None:
                    writer.writerow({
                        "filename": filename, "relative_path": rel_path,
                        "class_id": 4, "class_name": "No Defect", "confidence": 1.0,
                        "repairable": "YES", "status": "NO DEFECTS",
                        "area_mm2": 0, "perimeter_mm": 0, "eccentricity": 0,
                        "compactation": 0, "glcm_homogeneity": 0, "glcm_energy": 0, "glcm_entropy": 0
                    })
                    continue

                for i in range(len(pred.boxes)):
                    cid = int(pred.boxes.cls[i])
                    conf = float(pred.boxes.conf[i])
                    if cid == 4: continue
                        
                    mask_data = cv2.resize(pred.masks.data[i].cpu().numpy(), (w, h))
                    metrics = _compute_metrics(mask_data > 0.5, gray)
                    
                    # Lógica de reparabilidad basada estrictamente en parámetros estadísticos (Medias)
                    defect_repairable = True
                    area_mm2 = metrics["area_mm2"]
                    perim_mm = metrics["perimeter_mm"]
                    ener = metrics["glcm_energy"]
                    entro = metrics["glcm_entropy"]

                    if cid == 3: # Material loss - Holes
                        # No reparable si supera los máximos estadísticos
                        if area_mm2 > 20.5 or perim_mm > 18.6:
                            defect_repairable = False
                    elif cid == 1: # Crack
                        # No reparable si supera medias de tamaño o si la forma/textura es compleja
                        if area_mm2 > 6.67 or perim_mm > 14.13 or metrics["eccentricity"] < 0.98 or entro > 9.28:
                            defect_repairable = False
                    elif cid == 0: # Scratch-Abrasion
                        if ener < 0.12 or entro > 8.46:
                            defect_repairable = False
                    elif cid == 2: # Scratch-Friction
                        if ener < 0.06 or entro > 8.75:
                            defect_repairable = False

                    if not defect_repairable: is_image_repairable = False

                    detections.append({
                        "filename": filename, "relative_path": rel_path,
                        "class_id": cid, "class_name": CLASS_NAMES.get(cid, "Unknown"),
                        "confidence": f"{conf:.4f}",
                        "repairable": "YES" if defect_repairable else "NO",
                        "area_mm2": f"{area_mm2:.4f}",
                        "perimeter_mm": f"{perim_mm:.4f}",
                        "eccentricity": f"{metrics['eccentricity']:.4f}",
                        "compactation": f"{metrics['compactation']:.4f}",
                        "glcm_homogeneity": f"{metrics['glcm_homogeneity']:.4f}",
                        "glcm_energy": f"{ener:.4f}",
                        "glcm_entropy": f"{entro:.4f}"
                    })

                st_text, _ = _get_status(len(detections) > 0, is_image_repairable)
                for det in detections:
                    det["status"] = st_text
                    writer.writerow(det)

        print(f"✓ Éxito: Resultados guardados en {args.output} ({'añadidos' if args.append else 'nuevo archivo'})")
    except Exception as e:
        print(f"Error al escribir CSV: {e}")

if __name__ == "__main__":
    main()
