import hashlib
import io

from PIL import Image

from shar.schemas import BBox, FrameRecord, Transform


def prepare(
    image: Image.Image,
    index: int,
    pts: int,
    timestamp: float,
    longest_edge: int = 640,
    quality: int = 85,
    grayscale: bool = False,
) -> FrameRecord:
    image = image.convert("RGB")
    width, height = image.size
    image.thumbnail((longest_edge, longest_edge), Image.Resampling.LANCZOS)
    if grayscale:
        image = image.convert("L").convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    data = buf.getvalue()
    return FrameRecord(
        frame_id=f"pts_{pts}",
        frame_index=index,
        pts=pts,
        timestamp_sec=timestamp,
        jpeg=data,
        image_hash=hashlib.sha256(data).hexdigest(),
        transform=Transform(
            source_width=width,
            source_height=height,
            prepared_width=image.width,
            prepared_height=image.height,
        ),
    )


def invert_box(box: BBox, transform: Transform) -> BBox:
    return BBox(
        x=transform.crop_x + box.x * transform.crop_width,
        y=transform.crop_y + box.y * transform.crop_height,
        width=box.width * transform.crop_width,
        height=box.height * transform.crop_height,
    )
