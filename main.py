#!/usr/bin/env python3
import click
import os
import cv2
from typing import List
from build_panoramic_image import RgbImg, build_panoramic_image


def _read_rgb_images(directory_path: str) -> List[RgbImg]:
    images_bgr = [cv2.imread(os.path.join(directory_path, file_name))
                  for file_name in sorted(os.listdir(directory_path))]
    return [cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            for img_bgr in images_bgr]


def _write_rgb_image(img: RgbImg, output_path: str) -> bool:
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(output_path, img_bgr)


@click.command()
@click.argument('directory_path', type=click.Path(exists=True, dir_okay=True, file_okay=False, readable=True))
@click.argument('output_path', type=click.Path(exists=False, dir_okay=False, writable=True))
def main(directory_path: str, output_path: str):
    imgs = _read_rgb_images(directory_path)
    panoramic_image = build_panoramic_image(imgs)
    if not _write_rgb_image(panoramic_image, output_path):
        click.echo(f'failed to write result image to "{output_path}"')
        raise click.Abort()


if __name__ == '__main__':
    main()
