import json
import importlib.util
from pathlib import Path
import sys

# Load module from file path to avoid import issues with the hyphenated package name
module_path = Path(__file__).parents[2] / "stock-service" / "function_app.py"
# Ensure `stock-service` utils/ is importable as top-level `utils`
stock_service_dir = str(module_path.parent)
if stock_service_dir not in sys.path:
    sys.path.insert(0, stock_service_dir)

spec = importlib.util.spec_from_file_location("function_app", str(module_path))
fa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fa)


def test_health_check_returns_healthy_status():
    # health_check does not depend on the request object, so pass None
    resp = fa.health_check(None)
    assert resp.status_code == 200
    body = resp.get_body().decode()
    data = json.loads(body)
    assert data["status"] == "healthy"
    assert data["service"] == "stock-service"
    assert "timestamp" in data


def test_setup_logger_returns_logger_with_name():
    logger = fa.setup_logger("my_logger")
    assert logger.name == "my_logger"
