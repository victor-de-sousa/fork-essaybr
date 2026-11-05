import io
import warnings
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError

FORMATS = {'PNG':('.png','image/png'), 'JPEG':('.jpg','image/jpeg'), 'WEBP':('.webp','image/webp')}


def validate_image(data: bytes, filename: str, content_type: str, settings):
    if len(data) > settings.max_image_mb * 1024 * 1024:
        raise ValueError(f'Cada imagem pode ter até {settings.max_image_mb} MB.')
    extension = Path(filename).suffix.lower()
    if extension not in ('.png','.jpg','.jpeg','.webp'):
        raise ValueError('Envie imagens PNG, JPEG ou WEBP.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as check:
                fmt = check.format
                if fmt not in FORMATS or check.n_frames != 1:
                    raise ValueError('Formato inválido ou imagem animada não permitida.')
                suffix, mime = FORMATS[fmt]
                if extension != suffix and not (fmt == 'JPEG' and extension == '.jpeg'):
                    raise ValueError('A extensão não corresponde ao conteúdo da imagem.')
                if content_type != mime:
                    raise ValueError('O tipo declarado não corresponde ao conteúdo da imagem.')
                if min(check.size)<32 or max(check.size)>20000 or check.width*check.height > settings.max_image_pixels:
                    raise ValueError('Dimensões de imagem fora dos limites permitidos.')
                check.verify()
            with Image.open(io.BytesIO(data)) as source:
                picture = ImageOps.exif_transpose(source).convert('RGB')
                output = io.BytesIO()
                # Lossless storage avoids introducing new handwriting artifacts; strips metadata.
                picture.save(output, format='PNG')
                return output.getvalue(), '.png'
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError('Imagem inválida ou corrompida.') from exc
