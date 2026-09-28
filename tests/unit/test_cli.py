from typer.testing import CliRunner

from shar.cli import app


def test_cli_failure(tmp_path):
    result = CliRunner().invoke(app, ["analyze", str(tmp_path / "missing.mp4")])
    assert result.exit_code == 1
    assert "error" in result.output
