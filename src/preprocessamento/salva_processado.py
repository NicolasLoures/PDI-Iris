# Ajusta sys.path para incluir a raiz do projeto
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent  # assume este script está em notebooks/
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import logging
import cv2
from src.preprocessamento.filtros import filtro_mediana, equalizacao_histograma

# Configura logging básico
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def processa_e_salva(
    input_path: Path,
    output_dir: Path,
    ksize: int = 5
) -> tuple[Path, Path]:
    """
    Processa uma imagem aplicando filtro de mediana e equalização de histograma,
    salvando duas versões:
      1) original com equalização de histograma
      2) filtrada (mediana) + equalização

    Args:
        input_path: Caminho da imagem de entrada.
        output_dir: Diretório onde as imagens serão salvas.
        ksize: Tamanho do kernel para filtro mediano (ímpar).

    Returns:
        Tuple com (caminho_da_equalizacao_original, caminho_da_processada).
    """
    # Garante que a imagem existe
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")

    # Cria pasta de saída
    output_dir.mkdir(parents=True, exist_ok=True)

    # Carrega a imagem (BGR) e converte para grayscale se necessário
    img = cv2.imread(str(input_path))
    if img is None:
        raise ValueError(f"Não foi possível ler a imagem: {input_path}")

    # Equalização na imagem original (BGR ou cinza)
    img_eq_original = equalizacao_histograma(img)

    # Aplica filtro de mediana na imagem (mantém o mesmo tipo)
    img_mediana = filtro_mediana(img_eq_original, ksize=ksize)

    # Equalização após filtro de mediana
    img_eq_mediana = equalizacao_histograma(img_mediana)

    # Prepara caminhos de saída
    nome_base = input_path.stem
    ext       = input_path.suffix

    eq_orig_name = f"{nome_base}_equalized{ext}"
    proc_name    = f"{nome_base}_processed{ext}"

    eq_orig_path = output_dir / eq_orig_name
    proc_path    = output_dir / proc_name

    # Salva as duas imagens
    cv2.imwrite(str(eq_orig_path), img_eq_original)
    cv2.imwrite(str(proc_path), img_eq_mediana)

    logger.info(f"Imagens salvas: equalized -> {eq_orig_path}, processed -> {proc_path}")
    return eq_orig_path, proc_path


if __name__ == "__main__":
    RAW_DIR = project_root / "data" / "raw"
    OUT_DIR = project_root / "data" / "preprocessed"

    for img_file in RAW_DIR.iterdir():
        if img_file.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.bmp'}:
            continue
        try:
            eq_path, proc_path = processa_e_salva(img_file, OUT_DIR, ksize=5)
        except Exception as e:
            logger.error(f"Erro ao processar {img_file.name}: {e}")
        else:
            logger.info(f"[✓] {img_file.name} -> {proc_path.name}")