import main
import os


_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
_TMP_OUTPUT_PATH = os.path.join(_ROOT_DIR, 'test_output_image.png')


def test_build_panoramic_works():
    if os.path.isfile(_TMP_OUTPUT_PATH):
        os.remove(_TMP_OUTPUT_PATH)
    main.main(directory_path=os.path.join(_ROOT_DIR, 'test_images', '1'),
              output_path=_TMP_OUTPUT_PATH)
    assert os.path.exists(_TMP_OUTPUT_PATH)
    os.remove(_TMP_OUTPUT_PATH)
