import pytest
from ..api.stream import MTMCGeneration


def test_proper_dict_format(dataset_path, dataset_split, scenario_id):
    gen = MTMCGeneration(dataset_path, dataset_split, scenario_id)
    res = next(gen.mtmc_generator())
    res_sent_data = next(gen.get_current_data_dict())
    assert isinstance(res, dict) and len(res) == 4 and isinstance(res_sent_data, dict) and len(res_sent_data) == 4