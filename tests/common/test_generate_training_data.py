from hanzi_font_deconstructor.common.generate_training_data import (
    get_training_input_and_mask_tensors,
)


def test_get_training_input_and_mask_tensors():
    input, mask = get_training_input_and_mask_tensors(size_px=256)
    assert input.shape == (1, 256, 256)
    assert mask.shape == (256, 256)
