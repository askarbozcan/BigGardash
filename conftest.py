
import pytest

def pytest_addoption(parser):
    parser.addoption("--dataset_path", action="store")
    parser.addoption("--dataset_split", action="store")
    parser.addoption("--scenario_id", action="store")

@pytest.fixture(scope="session", autouse=True)
def dataset_path(request):
    dataset_path = request.config.getoption("--dataset_path")
    return dataset_path

@pytest.fixture(scope="session", autouse=True)
def dataset_split(request):
    dataset_split = request.config.getoption("--dataset_split")
    return dataset_split

@pytest.fixture(scope="session", autouse=True)
def scenario_id(request):
    scenario_id = request.config.getoption("--scenario_id")
    return scenario_id